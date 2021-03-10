import time
import copy
import json
import os
import traceback
from bidict import bidict

from snamosim.behaviors.stilman_2005_behavior import Stilman2005Behavior

import snamosim.behaviors.plan.basic_actions as ba
import snamosim.behaviors.plan.action_result as ar

from snamosim.display.ros_publisher import RosPublisher

from snamosim.worldreps.entity_based.world import World
from snamosim.worldreps.entity_based.robot import Robot
from snamosim.worldreps.entity_based.obstacle import Obstacle

from snamosim.utils import stats_utils, utils, conversion, b2_collision


class Simulator:
    def __init__(self, simulation_file_path, goals=None, timestring=None):
        # Load simulation file and initialize logs
        if timestring:
            self.sim_start_timestring = timestring
        else:
            self.sim_start_timestring = utils.timestamp_string()
        simulation_file_abs_path = os.path.abspath(simulation_file_path)
        with open(simulation_file_abs_path) as f:
            self.config = json.load(f)
        sim_file_parent_dirname = os.path.basename(
            os.path.normpath(os.path.abspath(os.path.join(simulation_file_abs_path, '..'))))
        self.simulation_filename = os.path.splitext(os.path.basename(simulation_file_abs_path))[0]

        rel_path_to_main_sim_logs_dir = os.path.join('../logs/', sim_file_parent_dirname, self.simulation_filename)
        abs_path_to_main_sim_logs_dir = os.path.join(os.path.dirname(__file__), rel_path_to_main_sim_logs_dir)
        self.abs_path_to_logs_dir = os.path.join(abs_path_to_main_sim_logs_dir, self.sim_start_timestring + "/")
        os.makedirs(self.abs_path_to_logs_dir)
        os.makedirs(self.abs_path_to_logs_dir + "simulation/")
        self.log_filepath = os.path.join(os.path.dirname(self.abs_path_to_logs_dir), "sim_results.json")
        self.simulation_log = utils.CustomLogger()

        self.simulation_log.append(utils.BasicLog("Simulation file successfully loaded", 0))

        # Save general simulation parameters
        self.provide_walls = self.config["provide_walls"]
        self.display_sim_knowledge_only_once = self.config["display_sim_knowledge_only_once"]
        self.reset_after_first_goal = (
            False if "reset_after_first_goal" not in self.config else self.config["reset_after_first_goal"]
        )
        self.human_inflation_radius = 0.55/2.  # [m]

        self.simulation_log.append(
            utils.BasicLog("Created log folders at:{}".format(str(self.abs_path_to_logs_dir)), 0)
        )

        # Reinitialize rviz display

        agents_names = [a_to_b_config["agent_name"] for a_to_b_config in self.config["agents_behaviors"]]
        self.rp = RosPublisher(top_level_namespaces=['simulation'] + agents_names)
        self.rp.cleanup_all()

        self.simulation_log.append(utils.BasicLog("Display backend initialized.", 0))

        # Create world from world description json file
        world_file_path = self.config["files"]["world_file"]
        world_abs_path = os.path.join(os.path.dirname(simulation_file_abs_path), world_file_path)
        self.init_ref_world = World.load_from_json(world_abs_path)

        self.simulation_log.append(utils.BasicLog("World file successfully loaded.", 0))

        self.init_ref_world.save_to_files(
            json_filepath=self.abs_path_to_logs_dir + "simulation/" + self.simulation_filename + ".json",
            svg_filepath=self.init_ref_world.init_geometry_filename
        )
        self.ref_world = copy.deepcopy(self.init_ref_world)

        # Associate autonomous agents with goals and behaviors
        self.goals_geometries = {goal.name: goal.pose for goal in self.init_ref_world.goals.values()}
        if goals:
            self.saved_goals = goals
            self.agent_uid_to_goals = {
                self.ref_world.get_entity_uid_from_name(agent_name): gls for agent_name, gls in goals.items()
            }
        else:
            self.agent_uid_to_goals = self.initialize_agents_goals(self.goals_geometries)
            self.saved_goals = {
                self.ref_world.entities[agent_uid].name: copy.deepcopy(goals)
                for agent_uid, goals in self.agent_uid_to_goals.items()
            }

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

        self.simulation_log.append(utils.BasicLog("Simulation successfully loaded.", 0))
        self.log_filepath = os.path.join(os.path.dirname(self.abs_path_to_logs_dir), "sim_results.json")

        self.b2_sim = b2_collision.B2Sim(self.ref_world.entities)

    def run(self):
        simulation_report = {}
        with open(self.log_filepath, 'w+') as f:
            json.dump(simulation_report, f, default=lambda o: o.__dict__, indent=4, sort_keys=True)

        run_start_time = time.time()

        run_active = True

        run_exceptions_traces = []

        step_count = 0

        while run_active:

            active_agents = set(self.agent_uid_to_behavior.keys())

            self.rp.publish_sim_world(self.ref_world)

            trace_polygons = []
            attached_entity_to_robot = bidict()

            step_count = 0

            self.simulation_log.append(utils.BasicLog("Starting run.", step_count))

            while active_agents:
                # try:
                # Increment simulation step count
                step_count += 1

                # Sense loop: update each agent's knowledge of the world
                self.sense(active_agents, step_count)

                # Think loop: get each agent to think about their next step
                agent_uid_to_next_action = self.think(active_agents, trace_polygons, step_count)

                # Act loops: Verify that each action is doable individually and together, if so, execute them
                self.act(agent_uid_to_next_action, attached_entity_to_robot, trace_polygons, step_count)

                # Once the simulation reference world has been modified, display the modification
                if not self.display_sim_knowledge_only_once:
                    self.rp.publish_sim_world(self.ref_world)
                # except Exception as e:
                #     if self.catch_exceptions:
                #         tb = traceback.format_exc()
                #         run_exceptions_traces.append(tb)
                #         self.simulation_log.append(utils.BasicLog(tb, step_count))
                #     else:
                #         self.simulation_log.append(utils.BasicLog("MET A RUNTIME EXCEPTION, EXITING !", step_count))
                #         raise e

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

                self.simulation_log.append(utils.BasicLog("Reset world and executing next goal.", step_count))
            else:
                # Otherwise, simply leave and finish up the simulation
                run_active = False

        # Save simulation results
        self.ref_world.save_to_files(
            json_filepath=self.abs_path_to_logs_dir + "simulation/" + self.simulation_filename + "_end" + ".json",
            svg_filepath=utils.append_suffix(self.init_ref_world.init_geometry_filename, "_end")
        )
        self.run_duration = time.time() - run_start_time
        self.simulation_log.append(utils.BasicLog("Saved simulation final state.", step_count))

        simulation_report = self.create_simulation_report()
        if run_exceptions_traces:
            simulation_report['Exceptions'] = json.dumps(run_exceptions_traces)

        simulation_report["agents_logs"] = {}
        for uid, behavior in self.agent_uid_to_behavior.items():
            simulation_report["agents_logs"][self.ref_world.entities[uid].name] = behavior.simulation_log

        # TODO Remove this temporary measure for a better separation between scenario generation and execution
        simulation_report["temp_goals"] = self.saved_goals
        self.simulation_log.append(
            utils.BasicLog("Simulation report saved at: {}".format(self.log_filepath), step_count)
        )
        simulation_report["simulation_log"] = self.simulation_log
        simulation_report_json = json.dumps(simulation_report, default=lambda o: o.__dict__, indent=4, sort_keys=True)
        with open(self.log_filepath, 'w+') as f:
            f.write(simulation_report_json)

        return simulation_report

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

                goal_counter += 1

                end_nb_cc, end_biggest_cc_size, end_all_cc_sum_size, end_frag = stats_utils.get_connectivity_stats(
                    world_snapshot, self.human_inflation_radius,
                    [uid for uid, entity in world_snapshot.entities.items() if isinstance(entity, Robot)]
                )

                end_abs_social_cost = stats_utils.get_social_costs_stats(world_snapshot, all_movables_uids)

                total_path_length = transit_path_length + transfer_path_length

                nb_reallocated_obstacles = stats_utils.get_nb_reallocated_obstacles(self.init_ref_world, world_snapshot)

                goal_report = {
                    "goal": goal,
                    "goal_status": goal_status,
                    "number_of_transferred_obstacles": stats_utils.get_nb_transferred_obstacles(
                        actions_results_to_goal
                    ),
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
                    "space_fragmentation_percentage_after_goal": end_frag,
                    "absolute_social_cost_after_goal": end_abs_social_cost,
                    "number_of_connected_components_relative_change": stats_utils.relative_change(
                        self.init_nb_cc, end_nb_cc),
                    "biggest_free_component_size_relative_change": stats_utils.relative_change(
                        self.init_biggest_cc_size, end_biggest_cc_size),
                    "free_space_size_relative_change": stats_utils.relative_change(
                        self.init_all_cc_sum_size, end_all_cc_sum_size),
                    "space_fragmentation_percentage_relative_change": stats_utils.relative_change(
                        self.init_frag_percentage, end_frag, False) * 100.,
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

                if agent_behavior_name == "stilman_2005_behavior":
                    agent_world = copy.deepcopy(self.ref_world)
                    self.rp.cleanup_robot_world()
                    agent_uid_to_behavior[agent_uid] = Stilman2005Behavior(
                        agent_world, agent_uid, agent_navigation_goals, behavior_config, self.abs_path_to_logs_dir)
                else:
                    raise NotImplementedError(
                        "You tried to associate entity '{agent_name}' with a behavior named"
                        "'{b_name}' that is not implemented yet."
                        "Maybe you mispelled something ?".format(
                            agent_name=agent_name, b_name=agent_behavior_name)
                    )
        return agent_uid_to_behavior

    def save_world_snapshot(self, agent_uid, action, trace_polygons, step_count):
        world_snapshot = copy.deepcopy(self.ref_world)
        self.agent_uid_and_goal_to_world_snapshot[agent_uid].append({
            "goal": action.goal,
            "goal_status": str(action),
            "world_snapshot": copy.deepcopy(self.ref_world)
        })
        goal_counter = len(self.agent_uid_and_goal_to_world_snapshot[agent_uid])

        suffix = (
            "at_step_" + str(step_count)
            + "_after_goal_" + str(goal_counter)
            + "_of_" + self.ref_world.entities[agent_uid].name
        )
        json_filepath = self.abs_path_to_logs_dir + "simulation/" + self.simulation_filename + suffix + ".json"
        svg_filepath = utils.append_suffix(self.init_ref_world.init_geometry_filename, suffix)
        svg_data = world_snapshot.to_svg()

        new_group = svg_data.createElement('svg:g')
        new_group.setAttribute('id', "traces"+suffix)
        new_group.setAttribute('inkscape:groupmode', "layer")
        new_group.setAttribute('inkscape:label', "traces"+suffix)
        svg_data.childNodes[0].appendChild(new_group)
        for polygon in trace_polygons:
            conversion.add_shapely_geometry_to_svg_with_projection(
                polygon,
                self.ref_world.scaling_value,
                self.ref_world.dd.width,
                self.ref_world.dd.height,
                'goal_generated_' + str(goal_counter),
                conversion.OBSTACE_TRACE_STYLE,
                svg_data,
                new_group
            )
        del trace_polygons[:len(trace_polygons)]

        json_data = world_snapshot.to_json(svg_filepath)
        world_snapshot.save_to_files(
            json_data=json_data,
            svg_data=svg_data,
            json_filepath=json_filepath,
            svg_filepath=svg_filepath
        )

    def sense(self, active_agents, step_count):
        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            if agent_uid in active_agents:
                last_action_result = (self.agent_uid_to_action_results[agent_uid][-1]
                                      if self.agent_uid_to_action_results[agent_uid]
                                      else ar.ActionSuccess)
                behavior.sense(self.ref_world, last_action_result, step_count)

    def think(self, active_agents, trace_polygons, step_count):
        agent_uid_to_next_action = {}
        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            if agent_uid in active_agents:
                planning_start_time = time.time()
                agent_next_action = behavior.think()

                # TODO Change goal coordinates for easier reading to goal name in log.
                if isinstance(agent_next_action, ba.GoalsFinished):
                    # If the agent has executed all of its goals, remove it from the active agents
                    active_agents.remove(agent_uid)
                    self.simulation_log.append(
                        utils.BasicLog("{} finished executed all its goals.".format(agent_uid), step_count)
                    )
                elif isinstance(agent_next_action, ba.GoalFailed):
                    self.save_world_snapshot(agent_uid, agent_next_action, trace_polygons, step_count)
                    self.simulation_log.append(
                        utils.BasicLog(
                            "{} failed executing goal {}.".format(
                                self.ref_world.entities[agent_uid].name, str(agent_next_action.goal)
                            ),
                            step_count
                        )
                    )
                    simulation_report = {"temp_goals": self.saved_goals, "simulation_log": self.simulation_log}
                    simulation_report["agents_logs"] = {}
                    for uid, behavior in self.agent_uid_to_behavior.items():
                        simulation_report["agents_logs"][
                            self.ref_world.entities[uid].name] = behavior.simulation_log
                    with open(self.log_filepath, 'w+') as f:
                        json.dump(simulation_report, f, default=lambda o: o.__dict__, indent=4, sort_keys=True)
                    if agent_next_action.goal not in self.agent_uid_and_goal_to_action_results[agent_uid]:
                        self.agent_uid_and_goal_to_action_results[agent_uid][agent_next_action.goal] = []
                elif isinstance(agent_next_action, ba.GoalSuccess):
                    # If the agent reached its current goal
                    self.save_world_snapshot(agent_uid, agent_next_action, trace_polygons, step_count)
                    self.simulation_log.append(
                        utils.BasicLog(
                            "{} successfully executed goal {}.".format(
                                self.ref_world.entities[agent_uid].name, str(agent_next_action.goal)
                            ),
                            step_count
                        )
                    )
                    simulation_report = {"temp_goals": self.saved_goals, "simulation_log": self.simulation_log}
                    simulation_report["agents_logs"] = {}
                    for uid, behavior in self.agent_uid_to_behavior.items():
                        simulation_report["agents_logs"][
                            self.ref_world.entities[uid].name] = behavior.simulation_log
                    with open(self.log_filepath, 'w+') as f:
                        json.dump(simulation_report, f, default=lambda o: o.__dict__, indent=4, sort_keys=True)
                else:
                    # If the agent could think of a plan and its step
                    agent_uid_to_next_action[agent_uid] = agent_next_action
                self.agent_uid_to_think_time[agent_uid] += time.time() - planning_start_time
        return agent_uid_to_next_action

    def act(self, agent_uid_to_next_action, entity_to_agent, trace_polygons, step_count):
        # Only Grab and Release actions require further checks, and Wait actions are necessarily valid
        to_check = {
            uid: a for uid, a in agent_uid_to_next_action.items()
            if isinstance(a, (ba.Translation, ba.Rotation)) and not isinstance(a, (ba.Grab, ba.Release))
        }
        failed = {}
        succeeded = {
            uid: ar.ActionSuccess(a, self.ref_world.entities[uid].pose)
            for uid, a in agent_uid_to_next_action.items() if isinstance(a, ba.Wait)
        }

        # Check if released entity is already grabbed by the right agent
        for agent_uid, action in agent_uid_to_next_action.items():
            if isinstance(action, ba.Release):
                entity_uid = action.entity_uid
                if agent_uid not in entity_to_agent.inverse or entity_uid not in entity_to_agent:
                    failed[agent_uid] = ar.NotGrabbedFailure(action)
                else:
                    other_agent_uid = entity_to_agent[entity_uid]
                    if other_agent_uid != agent_uid:
                        failed[agent_uid] = ar.GrabbedByOtherFailure(action, other_agent_uid)
                    else:
                        to_check[agent_uid] = action

        # Check if grabbed entity not already grabbed by another, and if about to be released by another
        entity_to_grab_agents = {}
        for agent_uid, action in agent_uid_to_next_action.items():
            if isinstance(action, ba.Grab):
                entity_uid = action.entity_uid
                if entity_uid in entity_to_grab_agents:
                    entity_to_grab_agents[entity_uid].add(entity_uid)
                else:
                    entity_to_grab_agents[entity_uid] = {entity_uid}
        for agent_uid, action in agent_uid_to_next_action.items():
            if isinstance(action, ba.Grab):
                entity_uid = action.entity_uid
                if len(entity_to_grab_agents[entity_uid]) > 1:
                    failed[agent_uid] = ar.SimultaneousGrabFailure(action, entity_to_grab_agents[entity_uid])
                    continue
                if agent_uid in entity_to_agent.inverse:
                    failed[agent_uid] = ar.GrabMoreThanOneFailure(action)
                    continue
                if entity_uid in entity_to_agent:
                    other_agent_uid = entity_to_agent[entity_uid]
                    other_releases = other_agent_uid in to_check and isinstance(to_check[other_agent_uid], ba.Release)
                    if not other_releases:
                        failed[agent_uid] = ar.AlreadyGrabbedFailure(action, other_agent_uid)
                        continue
                to_check[agent_uid] = action

        # Check actions regarding dynamic collisions and apply the valid ones using Box2D
        collides_with = self.b2_sim.simulate_simple_kinematics([to_check], apply=True)

        # Finish separating succeeded and failed actions, and apply result to world state on success
        for agent_uid, action in to_check.items():
            action_dynamically_collides = (
                    (  # The agent associated with the action collides
                        (
                            agent_uid in collides_with
                            and not isinstance(action, ba.Grab)
                        )
                        or (  # Special case for Grab: ignore collision with grabbed obstacle
                            agent_uid in collides_with
                            and isinstance(action, ba.Grab)
                            and (
                                len(collides_with[agent_uid]) > 1
                                or action.entity_uid not in collides_with[agent_uid]
                            )
                        )
                    )
                    or (  # The obstacle associated with the action collides
                        agent_uid in entity_to_agent.inverse
                        and entity_to_agent.inverse[agent_uid] in collides_with
                        and not isinstance(action, ba.Release)
                    )
            )
            if action_dynamically_collides:
                if agent_uid in entity_to_agent.inverse and not isinstance(action, ba.Release):
                    colliding = collides_with[agent_uid].union(collides_with[entity_to_agent.inverse[agent_uid]])
                else:
                    colliding = collides_with[agent_uid]
                failed[agent_uid] = ar.DynamicCollisionFailure(action, colliding)
            else:
                # SUCCESS
                # If Grab or Release, first update entity_to_agent
                if isinstance(action, ba.Grab):
                    entity_to_agent[action.entity_uid] = agent_uid
                if isinstance(action, ba.Release):
                    del entity_to_agent[action.entity_uid]

                # Then apply to world
                agent = self.ref_world.entities[agent_uid]
                agent_new_pose = self.b2_sim.get_entity_pose(agent_uid)
                agent_new_polygon = utils.set_polygon_pose(agent.polygon, agent.pose, agent_new_pose)
                agent.pose, agent.polygon = agent_new_pose, agent_new_polygon
                if agent_uid in entity_to_agent.inverse:
                    entity_uid = entity_to_agent.inverse[agent_uid]
                    entity = self.ref_world.entities[entity_uid]
                    entity_new_pose = self.b2_sim.get_entity_pose(entity_uid)
                    entity_new_polygon = utils.set_polygon_pose(entity.polygon, entity.pose, entity_new_pose)
                    entity.pose, entity.polygon = entity_new_pose, entity_new_polygon

                succeeded[agent_uid] = ar.ActionSuccess(action, agent_new_pose)

        # Save Action Result in action result history
        for agent_uid, action_result in succeeded.items():
            self.agent_uid_to_action_results[agent_uid].append(action_result)

            agent_current_goal = self.agent_uid_to_behavior[agent_uid].get_current_goal()
            if agent_current_goal in self.agent_uid_and_goal_to_action_results[agent_uid]:
                self.agent_uid_and_goal_to_action_results[agent_uid][agent_current_goal].append(action_result)
            else:
                self.agent_uid_and_goal_to_action_results[agent_uid][agent_current_goal] = [action_result]
        for agent_uid, action_result in failed.items():
            self.agent_uid_to_action_results[agent_uid].append(action_result)

            agent_current_goal = self.agent_uid_to_behavior[agent_uid].get_current_goal()
            if agent_current_goal in self.agent_uid_and_goal_to_action_results[agent_uid]:
                self.agent_uid_and_goal_to_action_results[agent_uid][agent_current_goal].append(action_result)
            else:
                self.agent_uid_and_goal_to_action_results[agent_uid][agent_current_goal] = [action_result]
