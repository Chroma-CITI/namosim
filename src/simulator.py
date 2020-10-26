import time
import copy
import yaml
import json
import os
import random
import numpy as np
import traceback
from datetime import datetime
from shapely import affinity

from src.behaviors.navigation_only_behavior import NavigationOnlyBehavior
from src.behaviors.wu_levihn_2014_behavior import WuLevihn2014Behavior
from src.behaviors.stilman_2005_behavior import Stilman2005Behavior
from src.behaviors.new_stilman_2005_behavior import NewStilman2005Behavior

from src.behaviors.plan.basic_actions import ActionGoalsFinished, ActionGoalResult
from src.behaviors.plan.action_result import IntersectionFailure, UnmanipulableFailure, ActionSuccess
from src.worldreps.entity_based.custom_exceptions import IntersectionError

from src.display.ros_publisher import RosPublisher

from src.worldreps.entity_based.world import World
from src.worldreps.entity_based.robot import Robot
from src.worldreps.entity_based.obstacle import Obstacle
from src.worldreps.occupation_based.binary_inflated_occupancy_grid import BinaryInflatedOccupancyGrid

from src.utils import stats_utils, utils, conversion


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
            stats_utils.get_connectivity_stats(self.init_ref_world, self.human_inflation_radius, tuple())

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

            while active_agents:
                # Sense loop: update each agent's knowledge of the world
                for agent_uid, behavior in self.agent_uid_to_behavior.items():
                    if agent_uid in active_agents:
                        last_action_result = (self.agent_uid_to_action_results[agent_uid][-1]
                                              if self.agent_uid_to_action_results[agent_uid]
                                              else ActionSuccess)
                        behavior.sense(self.ref_world, last_action_result)

                # Think loop: get each agent to think about their next step
                agent_uid_to_next_action = {}
                for agent_uid, behavior in self.agent_uid_to_behavior.items():
                    planning_start_time = time.time()
                    try:
                        agent_uid_to_next_action[agent_uid] = behavior.think()
                    except:
                        exceptions_traces_met_during_run.append(traceback.format_exc())
                        traceback.print_exc()
                        continue
                    self.agent_uid_to_think_time[agent_uid] += time.time() - planning_start_time

                # Act loop: try to execute each agent's next step 'at the same time',
                for agent_uid, behavior in self.agent_uid_to_behavior.items():
                    action = agent_uid_to_next_action[agent_uid]

                    if isinstance(action, ActionGoalsFinished):
                        # If the agent signals it has executed all of its goals, remove it from the active agents
                        active_agents.remove(agent_uid)
                    elif isinstance(action, ActionGoalResult):
                        # If the agent signals whether it reached its current goal or could not reach it
                        goal_counter += 1
                        self.save_world_snapshot(agent_uid, action, goal_counter, trace_polygons)
                        if action.goal not in self.agent_uid_and_goal_to_action_results[agent_uid]:
                            self.agent_uid_and_goal_to_action_results[agent_uid][action.goal] = []
                    else:
                        # If there is an actual action to be executed
                        action_result = self.act(agent_uid, action)

                        trace_polygons.append(self.ref_world.entities[agent_uid].polygon)
                        if action.is_transfer:
                            trace_polygons.append(self.ref_world.entities[action.obstacle_uid].polygon)

                        self.agent_uid_to_action_results[agent_uid].append(action_result)
                        if action.goal in self.agent_uid_and_goal_to_action_results[agent_uid]:
                            self.agent_uid_and_goal_to_action_results[agent_uid][action.goal].append(action_result)
                        else:
                            self.agent_uid_and_goal_to_action_results[agent_uid][action.goal] = [action_result]

                # Once the simulation reference world has been modified, display the modification
                if not self.display_sim_knowledge_only_once:
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

        self.ref_world.save_to_files(
            json_filepath=self.abs_path_to_logs_dir + "simulation/" + self.simulation_filename + "_end" + ".json",
            svg_filepath=utils.append_suffix(self.init_ref_world.init_geometry_filename, "_end")
        )

        # Print simulation results
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
        if next_step is None:
            return True

        robot = self.ref_world.entities[robot_uid]

        target_trans = [next_step.target_pose[0] - robot.pose[0], next_step.target_pose[1] - robot.pose[1]]
        target_rot = (next_step.target_pose[2] - robot.pose[2])
        if -180. <= target_rot <= 180.:
            fixed_target_rot = target_rot
        elif 180. < target_rot:
            fixed_target_rot = -(360. - target_rot)
        else:  # i.e. if target_rot < -180.
            fixed_target_rot = 360 + target_rot

        try:
            other_entities = [entity for entity_uid, entity in self.ref_world.entities.items()
                if entity_uid != robot_uid and entity_uid != next_step.obstacle_uid]
            if next_step.is_transfer:
                obstacle = self.ref_world.entities[next_step.obstacle_uid]
                if robot.deduce_movability(obstacle.type) == "unmovable":
                    return UnmanipulableFailure(next_step, next_step.obstacle_uid)
                obstacle.rotate(fixed_target_rot, robot.pose, other_entities, 5.)
                obstacle.translate(target_trans[0], target_trans[1], self.ref_world.dd.res, other_entities)
            robot.translate(target_trans[0], target_trans[1], self.ref_world.dd.res, other_entities)
            robot.rotate(fixed_target_rot, 'centroid', other_entities, 5.)
        except IntersectionError as e:
            return IntersectionFailure(next_step, robot_uid, 0)  # TODO Raise proper entity id

        return ActionSuccess(next_step)

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
            if isinstance(entity, Robot) or (isinstance(entity, Obstacle) and entity.type in all_movable_types)})

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
                    world_snapshot, self.human_inflation_radius, tuple()
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
                    agent_uid_to_behavior[agent_uid] = NewStilman2005Behavior(
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