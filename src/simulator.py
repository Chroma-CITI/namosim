import time
import copy
import yaml
import json
import os
import random
import numpy as np
import traceback
import logging
from bidict import bidict
from datetime import datetime
from shapely import affinity
from shapely.geometry import LineString

from src.behaviors.navigation_only_behavior import NavigationOnlyBehavior
# from src.behaviors.wu_levihn_2014_behavior import WuLevihn2014Behavior
from src.behaviors.stilman_2005_behavior import Stilman2005Behavior

import src.behaviors.plan.basic_actions as ba
import src.behaviors.plan.action_result as ar

from src.display.ros_publisher import RosPublisher

from src.worldreps.entity_based.world import World
from src.worldreps.entity_based.robot import Robot
from src.worldreps.entity_based.obstacle import Obstacle

from src.utils import stats_utils, utils, conversion, collision


class Simulator:
    def __init__(self, simulation_file_path):
        # Import YAML world configuration file
        self.sim_start_timestring = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f")

        behavior_yaml_abs_path = os.path.abspath(simulation_file_path)
        self.config = yaml.load(open(behavior_yaml_abs_path))

        # Save general simulation parameters
        self.provide_walls = self.config["provide_walls"]
        self.display_sim_knowledge_only_once = self.config["display_sim_knowledge_only_once"]
        self.reset_after_first_goal = False if not "reset_after_first_goal" in self.config else self.config["reset_after_first_goal"]
        self.human_inflation_radius = 0.55/2.  # [m]
        simulation_file_parent_dirname = os.path.basename(
            os.path.normpath(os.path.abspath(os.path.join(behavior_yaml_abs_path, '..'))))
        self.simulation_filename = os.path.splitext(os.path.basename(behavior_yaml_abs_path))[0]

        rel_path_to_main_sim_logs_dir = os.path.join('../logs/', simulation_file_parent_dirname, self.simulation_filename)
        abs_path_to_main_sim_logs_dir = os.path.join(os.path.dirname(__file__), rel_path_to_main_sim_logs_dir)
        self.abs_path_to_logs_dir = os.path.join(abs_path_to_main_sim_logs_dir, self.sim_start_timestring + "/")
        os.makedirs(self.abs_path_to_logs_dir)
        os.makedirs(self.abs_path_to_logs_dir + "simulation/")

        # Reinitialize rviz display

        agents_names = [a_to_b_config["agent_name"] for a_to_b_config in self.config["agents_behaviors"]]
        self.rp = RosPublisher(top_level_namespaces=['simulation'] + agents_names)
        self.rp.cleanup_all()

        # Create world from world description yaml file
        world_file_path = self.config["files"]["world_file"]
        world_yaml_abs_path = os.path.join(os.path.dirname(behavior_yaml_abs_path), world_file_path)
        self.init_ref_world = World.load_from_yaml(world_yaml_abs_path)
        self.init_ref_world.save_to_files(
            json_filepath=self.abs_path_to_logs_dir + "simulation/" + self.simulation_filename + ".json",
            svg_filepath=self.init_ref_world.init_geometry_filename
        )
        self.ref_world = copy.deepcopy(self.init_ref_world)

        # Associate autonomous agents with goals and behaviors
        self.goals_geometries = {goal.name: goal.pose for goal in self.init_ref_world.goals.values()}
        self.agent_uid_to_goals = self.initialize_agents_goals(self.goals_geometries)

        if self.reset_after_first_goal:
            # Only give first goal if reset after first goal
            agent_uid_to_goals = {
                agent_uid: [goals.pop(0)] for agent_uid, goals in self.agent_uid_to_goals.items() if goals
            }
        else:
            agent_uid_to_goals = self.agent_uid_to_goals
        self.agent_uid_to_behavior = self.initialize_agents_behaviors(agent_uid_to_goals)

        self.rp.cleanup_sim_world()

        if self.display_sim_knowledge_only_once:
            time.sleep(2.0)
            self.rp.cleanup_sim_world()

        # Time stats
        self.agent_uid_to_think_time = {agent_uid: 0. for agent_uid in self.agent_uid_to_behavior.keys()}
        self.agent_uid_to_action_results = {agent_uid: [] for agent_uid in self.agent_uid_to_behavior.keys()}
        self.agent_uid_and_goal_to_action_results = {agent_uid: {} for agent_uid in self.agent_uid_to_behavior.keys()}
        self.run_duration = 0.
        self.agent_uid_and_goal_to_world_snapshot = {agent_uid: [] for agent_uid in self.agent_uid_to_behavior.keys()}

        self.init_nb_cc, self.init_biggest_cc_size, self.init_all_cc_sum_size, self.init_frag_percentage = \
            stats_utils.get_connectivity_stats(
                self.init_ref_world, self.human_inflation_radius,
                [uid for uid, entity in self.ref_world.entities.items() if isinstance(entity, Robot)]
            )

        self.catch_exceptions = False

    def run(self):
        run_start_time = time.time()

        run_active = True
        exceptions_traces_met_during_run = []
        while run_active:

            active_agents = set(self.agent_uid_to_behavior.keys())

            # TODO : REMOVE USE OF AGENT UID FOR SIM WORLD DISPLAY !!!
            agent_uid = self.agent_uid_to_behavior.keys()[0]
            self.rp.publish_sim_world(self.ref_world, agent_uid)

            goal_counter = 0
            trace_polygons = []
            attached_entity_to_robot = bidict()

            while active_agents:
                # Sense loop: update each agent's knowledge of the world
                self.sense(active_agents)

                # Think loop: get each agent to think about their next step
                agent_uid_to_next_action = self.think(
                    active_agents, goal_counter, trace_polygons, exceptions_traces_met_during_run)

                # Act loops: Verify that each action is doable individually, together and if so execute them
                self.act(agent_uid_to_next_action, attached_entity_to_robot, trace_polygons)

                # Once the simulation reference world has been modified, display the modification
                if not self.display_sim_knowledge_only_once:
                    # TODO : REMOVE USE OF AGENT UID FOR SIM WORLD DISPLAY !!!
                    self.rp.publish_sim_world(self.ref_world, agent_uid)

            # If the simulation is set to be reset after all agents have reached their first goal,
            # and there are goals left to reach, reset the simulation world and give the agents their next goal
            goals_left = any([bool(goals) for goals in self.agent_uid_to_goals.values()])
            if self.reset_after_first_goal and goals_left:
                self.ref_world = copy.deepcopy(self.init_ref_world)
                agent_uid_to_goals = {
                    agent_uid: [goals.pop(0)] for agent_uid, goals in self.agent_uid_to_goals.items() if goals
                }
                self.agent_uid_to_behavior = self.initialize_agents_behaviors(agent_uid_to_goals)
                self.rp.cleanup_sim_world()
            else:
                # Otherwise, simply leave and finish up the simulation
                run_active = False

        # Save simulation results
        self.ref_world.save_to_files(
            json_filepath=self.abs_path_to_logs_dir + "simulation/" + self.simulation_filename + "_end" + ".json",
            svg_filepath=utils.append_suffix(self.init_ref_world.init_geometry_filename, "_end")
        )
        self.run_duration = time.time() - run_start_time

        simulation_report = self.create_simulation_report()
        if exceptions_traces_met_during_run:
            simulation_report['Exceptions'] = json.dumps(exceptions_traces_met_during_run)
        simulation_report_json = json.dumps(simulation_report, indent=4, sort_keys=True)

        log_filepath = os.path.join(
                os.path.dirname(self.abs_path_to_logs_dir), "sim_results.json")
        with open(log_filepath, 'w+') as f:
            f.write(simulation_report_json)

        return simulation_report

    def act(self, robot_uid, next_step):
        pass

    def _create_robot_world_from_sim_world(self):
        entities = dict()
        for entity_uid, entity in self.ref_world.entities.items():
            if (isinstance(entity, Robot)
                    or ((isinstance(entity, Obstacle) and entity.type == "wall") if self.provide_walls else True)):
                entities[entity_uid] = copy.deepcopy(entity)

        return World(entities=entities,
                     taboo_zones=copy.deepcopy(self.ref_world.taboo_zones),
                     dd=copy.deepcopy(self.ref_world.dd))

    def create_simulation_report(self):
        all_movable_types = set()
        for entity in self.init_ref_world.entities.values():
            if isinstance(entity, Robot):
                all_movable_types.update(set(entity.movable_whitelist))

        all_movables_uids = tuple({
            entity_uid for entity_uid, entity in self.init_ref_world.entities.items()
            if isinstance(entity, Obstacle) and entity.type in all_movable_types})

        init_abs_social_cost = stats_utils.get_social_costs_stats(self.init_ref_world, tuple(all_movables_uids))

        report = {
            "total_run_time": self.run_duration,
            "number_of_connected_components_initial": self.init_nb_cc,
            "biggest_free_component_size_initial": self.init_biggest_cc_size,
            "free_space_size_initial": self.init_all_cc_sum_size,
            "space_fragmentation_percentage_initial": self.init_frag_percentage,
            "absolute_social_cost_initial": init_abs_social_cost,
            "agents": []
        }

        goal_counter = 1
        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            goals_reports = []

            goal_world_snapshots = self.agent_uid_and_goal_to_world_snapshot[agent_uid]

            for counter, goal_world_snapshot in enumerate(goal_world_snapshots):
                goal = goal_world_snapshot["goal"]
                goal_status = goal_world_snapshot["goal_status"]
                world_snapshot = goal_world_snapshot["world_snapshot"]
                actions_results_to_goal = self.agent_uid_and_goal_to_action_results[agent_uid][goal]
                transit_path_length, transfer_path_length = stats_utils.get_total_path_lengths(actions_results_to_goal)

                # world_snapshot.save_to_files(
                #     json_filepath=self.abs_path_to_logs_dir + "simulation/" + self.simulation_filename + "_after_goal_" + str(
                #         goal_counter) + ".json",
                #     svg_filepath=utils.append_suffix(self.init_ref_world.init_geometry_filename,
                #                                      "_after_goal_" + str(goal_counter))
                # )
                goal_counter += 1

                end_nb_cc, end_biggest_cc_size, end_all_cc_sum_size, end_frag_percentage = stats_utils.get_connectivity_stats(
                    world_snapshot, self.human_inflation_radius,
                    [uid for uid, entity in world_snapshot.entities.items() if isinstance(entity, Robot)]
                )

                end_abs_social_cost = stats_utils.get_social_costs_stats(world_snapshot, all_movables_uids)

                total_path_length = transit_path_length + transfer_path_length

                nb_reallocated_obstacles = stats_utils.get_nb_reallocated_obstacles(self.init_ref_world, world_snapshot)

                goal_report = {
                    "goal": goal,
                    "goal_status": goal_status,
                    "number_of_transferred_obstacles": stats_utils.get_nb_transferred_obstacles(actions_results_to_goal),
                    "number_of_reallocated_obstacles": nb_reallocated_obstacles,
                    "total_path_length": total_path_length,
                    "transit_path_length": transit_path_length,
                    "transfer_path_length": transfer_path_length,
                    "transit_transfer_ratio": (
                        1. if transfer_path_length == 0. else transit_path_length / transfer_path_length),
                    "transfer_percentage": (
                        0. if total_path_length == 0. else transfer_path_length / total_path_length * 100),
                    "number_of_connected_components_after_goal": end_nb_cc,
                    "biggest_free_component_size_after_goal": end_biggest_cc_size,
                    "free_space_size_after_goal": end_all_cc_sum_size,
                    "space_fragmentation_percentage_after_goal": end_frag_percentage,
                    "absolute_social_cost_after_goal": end_abs_social_cost,
                    "number_of_connected_components_relative_change": stats_utils.relative_change(
                        self.init_nb_cc, end_nb_cc),
                    "biggest_free_component_size_relative_change": stats_utils.relative_change(
                        self.init_biggest_cc_size, end_biggest_cc_size),
                    "free_space_size_relative_change": stats_utils.relative_change(
                        self.init_all_cc_sum_size, end_all_cc_sum_size),
                    "space_fragmentation_percentage_relative_change": stats_utils.relative_change(
                        self.init_frag_percentage, end_frag_percentage, False) * 100.,
                    "absolute_social_cost_relative_change": stats_utils.relative_change(
                        init_abs_social_cost, end_abs_social_cost)
                }
                goals_reports.append(goal_report)

            agent_report = {
                "agent_uid": agent_uid,
                "agent_name": self.ref_world.entities[agent_uid].name,
                "agent_behavior_name": behavior.name,
                "total_planning_time": self.agent_uid_to_think_time[agent_uid],
                "goals_reports": goals_reports
            }

            report["agents"].append(agent_report)

        return report

    def create_simulation_light_report(self):
        report = {
            "total_run_time": self.run_duration,
            "agents": []
        }

        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            agent_report = {
                "agent_uid": agent_uid,
                "agent_name": self.ref_world.entities[agent_uid].name,
                "agent_behavior_name": behavior.name,
                "total_planning_time": self.agent_uid_to_think_time[agent_uid]
            }
            report["agents"].append(agent_report)

        return report

    @staticmethod
    def sample_poses_uniform(world, agent_uid, nb_poses=1):
        map_min_x, map_min_y, map_max_x, map_max_y = world.get_map_bounds()
        agent = world.entities[agent_uid]
        other_entities = [entity for entity in world.entities if entity.uid != agent_uid]

        generated_poses = []

        while len(generated_poses) < nb_poses:
            pose_collides = True
            while pose_collides:
                rand_pose = (
                    random.uniform(map_min_x, map_max_x),
                    random.uniform(map_min_y, map_max_y),
                    random.uniform(0., 360.)
                )
                translation, rotation = utils.get_translation_and_rotation(agent.pose, rand_pose)
                expected_polygon = affinity.rotate(
                        affinity.translate(agent.polygon, translation[0], translation[1]), rotation
                )
                pose_collides = utils.polygon_collides_with_entities(expected_polygon, other_entities)
                if not pose_collides:
                    generated_poses.append(rand_pose)
        return generated_poses

    @staticmethod
    def sample_poses_on_grid(world, agent_uid, nb_poses):
        agent = world.entities[agent_uid]
        bin_inf_occ_grid = BinaryInflatedOccupancyGrid(
            world.dd.d_width, world.dd.d_height, world.dd.res,
            world.dd.grid_pose, agent.min_inflation_radius, world.entities, entities_to_ignore=(agent_uid,))
        grid = bin_inf_occ_grid.get_grid()
        free_cells = zip(*np.where(grid == 0))

        generated_poses = []

        while free_cells and len(generated_poses) < nb_poses:
            random_free_cell = random.choice(free_cells)
            free_cells.remove(random_free_cell)
            random_theta = random.uniform(0., 360.)
            rand_pose = utils.grid_pose_to_real_pose(
                (random_free_cell[0], random_free_cell[1], random_theta), world.dd.res, world.dd.grid_pose
            )
            generated_poses.append(rand_pose)
        return generated_poses

    def initialize_agents_goals(self, goals_geometries):
        agent_uid_to_goals = {}
        for agent_to_behavior_config in self.config["agents_behaviors"]:
            agent_name = agent_to_behavior_config["agent_name"]
            agent_uid = self.ref_world.get_entity_uid_from_name(agent_name)
            if agent_name in agent_uid_to_goals:
                raise RuntimeError("You can only associate a single behavior with entity: {entity_name}.".format(
                    entity_name=agent_name
                ))
            else:
                behavior_config = agent_to_behavior_config["behavior"]
                agent_navigation_goals = []

                if "navigation_goals" in behavior_config:
                    for config_goal in behavior_config["navigation_goals"]:
                        if config_goal["name"] in goals_geometries:
                            agent_navigation_goals.append(goals_geometries[config_goal["name"]])

                if "randomization" in behavior_config:
                    randomization_config = behavior_config["randomization"]
                    if "randomize_existing_navigation_goals" in randomization_config:
                        if "goal_multiplier" in randomization_config:
                            agent_navigation_goals *= randomization_config["goal_multiplier"]
                        random.shuffle(agent_navigation_goals)
                    elif "generate_random_goals" in randomization_config:
                        nb_goals_to_generate = 1
                        if "nb_goals_to_generate" in randomization_config:
                            nb_goals_to_generate = randomization_config["nb_goals_to_generate"]

                        randomization_types = ["discrete", "uniform"]
                        sampling_function = self.sample_poses_on_grid
                        if "randomization_type" in randomization_config:
                            randomization_type = randomization_config["randomization_type"]
                            if randomization_type not in randomization_types:
                                raise ValueError("Randomization can only be one of : {}".format(randomization_types))
                            if randomization_type == "discrete":
                                sampling_function = self.sample_poses_on_grid
                            elif randomization_type == "uniform":
                                sampling_function = self.sample_poses_uniform

                        agent_navigation_goals = sampling_function(self.ref_world, agent_uid, nb_goals_to_generate)
                        # TODO: Add the generated goals to world !

                agent_uid_to_goals[agent_uid] = agent_navigation_goals

        return agent_uid_to_goals

    def initialize_agents_behaviors(self, agents_navigation_goals):
        agent_uid_to_behavior = dict()

        for agent_to_behavior_config in self.config["agents_behaviors"]:
            agent_name = agent_to_behavior_config["agent_name"]
            agent_uid = self.ref_world.get_entity_uid_from_name(agent_name)
            agent_navigation_goals = agents_navigation_goals[agent_uid]
            if agent_name in agent_uid_to_behavior:
                raise RuntimeError("You can only associate a single behavior with entity: {entity_name}.".format(
                    entity_name=agent_name
                ))
            else:
                behavior_config = agent_to_behavior_config["behavior"]
                agent_behavior_name = behavior_config["name"]

                if agent_behavior_name == "navigation_only_behavior":
                    agent_world = self._create_robot_world_from_sim_world()
                    self.rp.cleanup_robot_world()
                    agent_uid_to_behavior[agent_uid] = NavigationOnlyBehavior(
                        agent_world, agent_uid, agent_navigation_goals, behavior_config, self.abs_path_to_logs_dir)
                elif agent_behavior_name == "wu_levihn_2014_behavior":
                    agent_world = self._create_robot_world_from_sim_world()
                    self.rp.cleanup_robot_world()
                    agent_uid_to_behavior[agent_uid] = WuLevihn2014Behavior(
                        agent_world, agent_uid, agent_navigation_goals, behavior_config, self.abs_path_to_logs_dir)
                elif agent_behavior_name == "stilman_2005_behavior":
                    agent_world = copy.deepcopy(self.ref_world)
                    self.rp.cleanup_robot_world()
                    agent_uid_to_behavior[agent_uid] = Stilman2005Behavior(
                        agent_world, agent_uid, agent_navigation_goals, behavior_config, self.abs_path_to_logs_dir)
                else:
                    raise NotImplementedError("You tried to associate entity '{agent_name}' with a behavior named"
                                              "'{b_name}' that is not implemented yet."
                                              "Maybe you mispelled something ?".format(
                        agent_name=agent_name, b_name=agent_behavior_name))
        return agent_uid_to_behavior

    def save_world_snapshot(self, agent_uid, action, goal_counter, trace_polygons):
        world_snapshot = copy.deepcopy(self.ref_world)
        self.agent_uid_and_goal_to_world_snapshot[agent_uid].append({
            "goal": action.goal,
            "goal_status": str(action),
            "world_snapshot": copy.deepcopy(self.ref_world)
        })

        json_filepath = self.abs_path_to_logs_dir + "simulation/" + self.simulation_filename + "_after_goal_" + str(
            goal_counter) + ".json"
        svg_filepath = utils.append_suffix(self.init_ref_world.init_geometry_filename,
                                           "_after_goal_" + str(goal_counter))
        svg_data = world_snapshot.to_svg()

        conversion.add_shapely_geometry_to_svg(
            utils.set_polygon_pose(
                self.ref_world.entities[agent_uid].polygon, self.ref_world.entities[agent_uid].pose, action.goal
            ),
            self.ref_world.scaling_value,
            self.ref_world.dd.width,
            self.ref_world.dd.height,
            'goal_generated_' + str(goal_counter),
            conversion.GOAL_STYLE,
            svg_data
        )

        # TODO ADD ORIENTATION GEOMETRY
        # conversion.add_shapely_geometry_to_svg(
        #     utils.set_polygon_pose(
        #         self.ref_world[agent_uid].polygon, self.ref_world[agent_uid].pose, action.goal
        #     ),
        #     self.ref_world.scaling_value,
        #     self.ref_world.dd.grid_pose,
        #     'goal_generated_' + str(goal_counter) + '_dir',
        #     conversion.GOAL_STYLE,
        #     svg_data
        # )

        for polygon in trace_polygons:
            conversion.add_shapely_geometry_to_svg(
                polygon,
                self.ref_world.scaling_value,
                self.ref_world.dd.width,
                self.ref_world.dd.height,
                'goal_generated_' + str(goal_counter),
                conversion.OBSTACE_TRACE_STYLE,
                svg_data
            )
        del trace_polygons[:len(trace_polygons)]

        json_data = world_snapshot.to_json(svg_filepath)
        world_snapshot.save_to_files(
            json_data=json_data,
            svg_data=svg_data,
            json_filepath=json_filepath,
            svg_filepath=svg_filepath
        )

    def sense(self, active_agents):
        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            if agent_uid in active_agents:
                last_action_result = (self.agent_uid_to_action_results[agent_uid][-1]
                                      if self.agent_uid_to_action_results[agent_uid]
                                      else ar.ActionSuccess)
                behavior.sense(self.ref_world, last_action_result)

    def think(self, active_agents, goal_counter, trace_polygons, exceptions_traces_met_during_run):
        agent_uid_to_next_action = {}
        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            if agent_uid in active_agents:
                planning_start_time = time.time()
                try:
                    agent_next_action = behavior.think()

                    if isinstance(agent_next_action, ba.GoalsFinished):
                        # If the agent has executed all of its goals, remove it from the active agents
                        active_agents.remove(agent_uid)
                    elif isinstance(agent_next_action, ba.GoalFailed):
                        goal_counter += 1
                        self.save_world_snapshot(agent_uid, agent_next_action, goal_counter, trace_polygons)
                        if agent_next_action.goal not in self.agent_uid_and_goal_to_action_results[agent_uid]:
                            self.agent_uid_and_goal_to_action_results[agent_uid][agent_next_action.goal] = []
                    elif isinstance(agent_next_action, ba.GoalSuccess):
                        # If the agent reached its current goal
                        goal_counter += 1
                        self.save_world_snapshot(agent_uid, agent_next_action, goal_counter, trace_polygons)
                    else:
                        # If the agent could think of a plan and its step
                        agent_uid_to_next_action[agent_uid] = agent_next_action
                except:
                    exceptions_traces_met_during_run.append(traceback.format_exc())
                    traceback.print_exc()
                    continue
                self.agent_uid_to_think_time[agent_uid] += time.time() - planning_start_time
        return agent_uid_to_next_action

    def act(self, agent_uid_to_next_action, attached_entity_to_robot, trace_polygons):
        # TODO ADD VERIFICATION IF OBSTACLE IS STILL MANIPULABLE ?

        agent_uid_to_doable_action = {
            uid: action for uid, action in agent_uid_to_next_action.items() if isinstance(action, ba.Wait)
        }
        agent_uid_to_impossible_action_result = {}

        # 1. Check static geometrical validity of MOVEMENT actions:
        #    (GOTOPOSE, TRANSLATION, ROTATION, GRAB, RELEASE)
        transition_polygons = {}
        agent_uid_to_movement_actions = {
            uid: action for uid, action in agent_uid_to_next_action.items()
            if isinstance(action, (ba.GoToPose, ba.Translation, ba.Rotation))
        }
        agent_uid_to_statically_doable_action = {}
        for agent_uid, next_action in agent_uid_to_movement_actions.items():
            # If there is an actual action to be executed
            attached_entity_uid = (
                None if agent_uid not in attached_entity_to_robot.inverse
                else attached_entity_to_robot.inverse[agent_uid]
            )
            to_be_attached_entity_uid = (
                None if not isinstance(next_action, (ba.Grab, ba.Release)) else next_action.entity_uid
            )
            other_entities = {
                uid: entity
                for uid, entity in self.ref_world.entities.items()
                if uid != agent_uid and uid != attached_entity_uid and uid != to_be_attached_entity_uid
            }
            other_polygons = {uid: e.polygon for uid, e in other_entities.items()}
            other_entities_aabb_tree = collision.polygons_to_aabb_tree(other_polygons)
            agent = self.ref_world.entities[agent_uid]

            if isinstance(next_action, ba.GoToPose):
                collision_polygon = LineString([agent.pose, next_action.pose]).buffer(
                    utils.get_circumscribed_radius(agent.polygon))
                collision_aabb = collision.polygon_to_aabb(collision_polygon)
                potential_collision_uids = other_entities_aabb_tree.overlap_values(collision_aabb)
                agent_collides = False
                for uid in potential_collision_uids:
                    if other_polygons[uid].intersects(collision_polygon):
                        agent_collides = True
                        agent_uid_to_impossible_action_result[agent_uid] = ar.StaticCollisionFailure(
                            next_action, agent_uid, uid, other_polygons[uid].intersection(collision_polygon)
                        )
                        logging.warning('Simulation static collision: Entity {} ({}) collides with entity {} ({}).'.format(
                            agent_uid, self.ref_world.entities[agent_uid].name, uid, self.ref_world.entities[uid].name
                        ))
                        break
                if not agent_collides:
                    agent_uid_to_statically_doable_action[agent_uid] = next_action
                    transition_polygons[agent_uid] = collision_polygon
            elif isinstance(next_action, (ba.Rotation, ba.Translation)):
                converted_action = ba.convert_action(next_action, agent.pose)
                agent_polygon_sequence = [agent.polygon, converted_action.apply(agent.polygon)]
                agent_collides, agent_collision_data, _ = collision.csv_check_collisions(
                    other_polygons, agent_polygon_sequence, [converted_action],
                    'minimum_rotated_rectangle', other_entities_aabb_tree, [0, 1]
                )
                if agent_collides:
                    agent_uid_to_impossible_action_result[agent_uid] = ar.StaticCollisionFailure(
                        next_action, agent_uid,
                        agent_collision_data[(0, 1)]['colliding_polygon_uid'],
                        agent_collision_data[(0, 1)]['intersection_polygon']
                    )
                    logging.warning('Simulation static collision: Entity {} ({}) collides with entity {} ({}).'.format(
                        agent_uid, self.ref_world.entities[agent_uid].name,
                        agent_collision_data[(0,1)]['colliding_polygon_uid'],
                        self.ref_world.entities[agent_collision_data[(0,1)]['colliding_polygon_uid']].name
                    ))
                else:
                    if attached_entity_uid:
                        att_entity = self.ref_world.entities[attached_entity_uid]
                        att_entity_polygon_sequence = [
                            att_entity.polygon, converted_action.apply(att_entity.polygon)]
                        att_entity_collides, att_entity_collision_data, _ = collision.csv_check_collisions(
                            other_polygons, att_entity_polygon_sequence, [converted_action],
                            'minimum_rotated_rectangle', other_entities_aabb_tree, [0, 1]
                        )
                        if att_entity_collides:
                            agent_uid_to_impossible_action_result[agent_uid] = ar.StaticCollisionFailure(
                                next_action, attached_entity_uid,
                                att_entity_collision_data[(0, 1)]['colliding_polygon_uid'],
                                att_entity_collision_data[(0, 1)]['intersection_polygon']
                            )
                            logging.warning(
                                'Simulation static collision: Entity {} ({}) collides with entity {} ({}).'.format(
                                    attached_entity_uid, self.ref_world.entities[attached_entity_uid].name,
                                    att_entity_collision_data[(0, 1)]['colliding_polygon_uid'],
                                    self.ref_world.entities[att_entity_collision_data[(0, 1)]['colliding_polygon_uid']].name
                            ))
                        else:
                            transition_polygons[agent_uid] = agent_collision_data[(0, 1)]['csv_polygon']
                            transition_polygons[attached_entity_uid] = att_entity_collision_data[(0, 1)]['csv_polygon']
                            agent_uid_to_statically_doable_action[agent_uid] = next_action
                    else:
                        transition_polygons[agent_uid] = agent_collision_data[(0, 1)]['csv_polygon']
                        agent_uid_to_statically_doable_action[agent_uid] = next_action
            else:
                raise TypeError('Provided action type is not supported.')

        # 2. Check dynamic (transitionnal) geometrical validity of MOVEMENT actions:
        #    (GOTOPOSE, TRANSLATION, ROTATION, GRAB, RELEASE)
        for agent_uid, action in agent_uid_to_statically_doable_action.items():
            attached_entity_uid = (
                None if agent_uid not in attached_entity_to_robot.inverse
                else attached_entity_to_robot.inverse[agent_uid]
            )
            to_be_attached_entity_uid = (
                None if not isinstance(action, (ba.Grab, ba.Release)) else action.entity_uid
            )

            other_transition_polygons = {
                uid: p for uid, p in transition_polygons.items()
                if uid != agent_uid and uid != attached_entity_uid and uid != to_be_attached_entity_uid
            }
            other_transitions_aabb_tree = collision.polygons_to_aabb_tree(other_transition_polygons)
            agent_transition_polygon = transition_polygons[agent_uid]
            agent_transition_aabb = collision.polygon_to_aabb(agent_transition_polygon)

            agent_potential_collision_uids = other_transitions_aabb_tree.overlap_values(agent_transition_aabb)
            agent_transitionnally_collides = False
            for uid in agent_potential_collision_uids:
                if other_transition_polygons[uid].intersects(agent_transition_polygon):
                    agent_transitionnally_collides = True
                    agent_uid_to_impossible_action_result[agent_uid] = ar.DynamicCollisionFailure(
                        action, agent_uid, uid,
                        other_transition_polygons[uid].intersection(agent_transition_polygon)
                    )
                    logging.warning(
                        'Simulation static collision: Entity {} ({}) collides with entity {} ({}).'.format(
                            agent_uid, self.ref_world.entities[agent_uid].name,
                            uid, self.ref_world.entities[uid].name
                    ))
                    break

            if not agent_transitionnally_collides:
                if attached_entity_uid:
                    att_entity_transition_polygon = transition_polygons[attached_entity_uid]
                    att_entity_transition_aabb = collision.polygon_to_aabb(att_entity_transition_polygon)

                    att_entity_potential_collision_uids = other_transitions_aabb_tree.overlap_values(
                        agent_transition_aabb)
                    att_entity_transitionnally_collides = False
                    for uid in att_entity_potential_collision_uids:
                        if other_transition_polygons[uid].intersects(att_entity_transition_polygon):
                            att_entity_transitionnally_collides = True
                            agent_uid_to_impossible_action_result[agent_uid] = ar.DynamicCollisionFailure(
                                action, attached_entity_uid, uid,
                                other_transition_polygons[uid].intersection(att_entity_transition_polygon)
                            )
                            logging.warning(
                                'Simulation static collision: Entity {} ({}) collides with entity {} ({}).'.format(
                                    attached_entity_uid, self.ref_world.entities[attached_entity_uid].name,
                                    uid, self.ref_world.entities[uid].name
                            ))
                            break
                    if not att_entity_transitionnally_collides:
                        agent_uid_to_doable_action[agent_uid] = action
                else:
                    agent_uid_to_doable_action[agent_uid] = action

        # 3. Check conceptual validity of RELEASE actions
        agent_uid_to_release_action = {
            uid: action for uid, action in agent_uid_to_doable_action.items() if
            isinstance(action, ba.Release)
        }
        about_to_be_released_uids = set()
        for agent_uid, release_action in agent_uid_to_release_action.items():
            if release_action.entity_uid in attached_entity_to_robot:
                agent_that_already_grabbed_entity = attached_entity_to_robot[
                    release_action.entity_uid]
                if agent_that_already_grabbed_entity == agent_uid:
                    about_to_be_released_uids.add(release_action.entity_uid)
                else:
                    agent_uid_to_impossible_action_result[agent_uid] = ar.GrabbedByOtherFailure(
                        release_action, agent_that_already_grabbed_entity
                    )
                    logging.warning(
                        'Entity {} ({}) could not release entity {} ({}) because grabbed by other.'.format(
                            agent_uid, self.ref_world.entities[agent_uid].name,
                            release_action.entity_uid, self.ref_world.entities[release_action.entity_uid].name
                    ))
                    del agent_uid_to_doable_action[agent_uid]
            else:
                agent_uid_to_impossible_action_result[agent_uid] = ar.NotGrabbedFailure(
                    release_action)
                logging.warning(
                    'Entity {} ({}) could not release entity {} ({}) because not grabbed.'.format(
                        agent_uid, self.ref_world.entities[agent_uid].name,
                        release_action.entity_uid, self.ref_world.entities[release_action.entity_uid].name
                    ))
                del agent_uid_to_doable_action[agent_uid]

        # 4. Check conceptual validity of GRAB actions
        agent_uid_to_grab_action = {
            uid: action for uid, action in agent_uid_to_doable_action.items() if
            isinstance(action, ba.Grab)
        }
        for agent_uid, grab_action, in agent_uid_to_grab_action.items():
            if grab_action.entity_uid in attached_entity_to_robot:
                if grab_action.entity_uid not in about_to_be_released_uids:
                    agent_that_already_grabbed_entity = attached_entity_to_robot[
                        grab_action.entity_uid]
                    agent_uid_to_impossible_action_result[agent_uid] = ar.AlreadyGrabbedFailure(
                        grab_action, agent_that_already_grabbed_entity
                    )
                    logging.warning(
                        'Entity {} ({}) could not grab entity {} ({}) because already grabbed.'.format(
                            agent_uid, self.ref_world.entities[agent_uid].name,
                            grab_action.entity_uid, self.ref_world.entities[grab_action.entity_uid].name
                    ))
                    del agent_uid_to_doable_action[agent_uid]
            elif agent_uid in attached_entity_to_robot.inverse:
                agent_uid_to_impossible_action_result[agent_uid] = ar.GrabMoreThanOneFailure(
                    grab_action)
                logging.warning(
                    'Entity {} ({}) could not grab entity {} ({}) because trying to grab more than one.'.format(
                        agent_uid, self.ref_world.entities[agent_uid].name,
                        grab_action.entity_uid, self.ref_world.entities[grab_action.entity_uid].name
                    ))
                del agent_uid_to_doable_action[agent_uid]

        # SECOND Act loop: execute all doable actions in reference simulation world, save results
        for agent_uid, action in agent_uid_to_doable_action.items():
            agent = self.ref_world.entities[agent_uid]
            if isinstance(action, ba.Wait):
                action_result = ar.ActionSuccess(action, agent.pose)
            elif isinstance(action, ba.GoToPose):
                agent.polygon = utils.set_polygon_pose(agent.polygon, agent.pose, action.pose)
                agent.pose = action.pose
                trace_polygons.append(agent.polygon)
                action_result = ar.ActionSuccess(action, agent.pose)
            elif isinstance(action, (ba.Translation, ba.Rotation)):
                if isinstance(action, ba.Release):
                    # If release, only do movement after release is done
                    del attached_entity_to_robot[action.entity_uid]

                attached_entity_uid = (
                    None if agent_uid not in attached_entity_to_robot.inverse
                    else attached_entity_to_robot.inverse[agent_uid]
                )

                agent.polygon = action.apply(agent.polygon, agent.pose)
                if isinstance(action, ba.Rotation):
                    agent.pose = action.predict_pose(agent.pose, (agent.pose[0], agent.pose[1]))
                else:
                    agent.pose = action.predict_pose(agent.pose)
                trace_polygons.append(agent.polygon)

                if attached_entity_uid:
                    attached_entity = self.ref_world.entities[attached_entity_uid]

                    attached_entity.polygon = action.apply(attached_entity.polygon, agent.pose)
                    if isinstance(action, ba.Rotation):
                        attached_entity.pose = action.predict_pose(attached_entity.pose, (agent.pose[0], agent.pose[1]))
                    else:
                        attached_entity.pose = action.predict_pose(attached_entity.pose)
                    trace_polygons.append(attached_entity.polygon)

                if isinstance(action, ba.Grab):
                    # If grab, only attach entity after grab movement is done
                    attached_entity_to_robot[action.entity_uid] = agent_uid

                action_result = ar.ActionSuccess(action, agent.pose, bool(attached_entity_uid), attached_entity_uid)
            else:
                raise TypeError('Action must be of type Wait, Translation, Rotation, GoToPose, Grab or Release')

            # Save Action Result in action result history
            self.agent_uid_to_action_results[agent_uid].append(action_result)

            agent_current_goal = self.agent_uid_to_behavior[agent_uid].get_current_goal()
            if agent_current_goal in self.agent_uid_and_goal_to_action_results[agent_uid]:
                self.agent_uid_and_goal_to_action_results[agent_uid][agent_current_goal].append(action_result)
            else:
                self.agent_uid_and_goal_to_action_results[agent_uid][agent_current_goal] = [action_result]

        # Third Act loop: Save results of impossible actions
        for agent_uid, action_result in agent_uid_to_impossible_action_result.items():
            self.agent_uid_to_action_results[agent_uid].append(action_result)

            agent_current_goal = self.agent_uid_to_behavior[agent_uid].get_current_goal()
            if agent_current_goal in self.agent_uid_and_goal_to_action_results[agent_uid]:
                self.agent_uid_and_goal_to_action_results[agent_uid][agent_current_goal].append(action_result)
            else:
                self.agent_uid_and_goal_to_action_results[agent_uid][agent_current_goal] = [action_result]
