import time
import copy
import json
import jsonpickle
import os
import traceback
import signal
from contextlib import contextmanager
import numpy as np

import snamosim.behaviors.stilman_2005_behavior as stilman_2005_behavior
from snamosim.behaviors.stilman_2005_behavior import Stilman2005Behavior

import snamosim.behaviors.plan.basic_actions as ba
import snamosim.behaviors.plan.action_result as ar

from snamosim.display.ros_publisher import RosPublisher

from snamosim.worldreps.entity_based.world import World
from snamosim.worldreps.entity_based.robot import Robot
from snamosim.worldreps.entity_based.obstacle import Obstacle

from snamosim.utils import stats_utils, utils, conversion, b2_collision, collision


class SimulationStepResult:
    def __init__(self, sense_durations, think_durations, act_duration, action_results, step_index):
        self.sense_durations = sense_durations
        self.think_durations = think_durations
        self.act_duration = act_duration
        self.action_results = action_results
        self.step_index = step_index


class AgentStepStats:
    def __init__(self, transit_path_length=0., transfer_path_length=0., path_length=0., nb_transfers=0,
                 nb_successful_goals=0, nb_failed_goals=0, nb_goals=0, nb_conflicts=0, nb_robot_robot_conflicts=0,
                 nb_robot_obstacle_conflicts=0, nb_stolen_movable_conflicts=0, nb_concurrent_grab_conflicts=0,
                 nb_wait_steps=0, nb_transit_steps=0, nb_transfer_steps=0, nb_of_postponements=0,
                 nb_of_unpostponements=0, nb_of_plan_computations=0, sense_time=0., think_time=0.):
        self.transit_path_length = transit_path_length
        self.transfer_path_length = transfer_path_length
        self.path_length = path_length
        self.nb_transfers = nb_transfers
        self.nb_successful_goals = nb_successful_goals
        self.nb_failed_goals = nb_failed_goals
        self.nb_goals = nb_goals
        self.nb_conflicts = nb_conflicts
        self.nb_robot_robot_conflicts = nb_robot_robot_conflicts
        self.nb_robot_obstacle_conflicts = nb_robot_obstacle_conflicts
        self.nb_stolen_movable_conflicts = nb_stolen_movable_conflicts
        self.nb_concurrent_grab_conflicts = nb_concurrent_grab_conflicts
        self.nb_wait_steps = nb_wait_steps
        self.nb_transit_steps = nb_transit_steps
        self.nb_transfer_steps = nb_transfer_steps
        self.nb_of_postponements = nb_of_postponements
        self.nb_of_unpostponements = nb_of_unpostponements
        self.nb_of_plan_computations = nb_of_plan_computations
        self.sense_time = sense_time
        self.think_time = think_time


class WorldStepStats:
    def __init__(self, nb_components=0, biggest_component_size=0, free_space_size=0,
                 fragmentation=0., absolute_social_cost=0.):
        self.nb_components = nb_components
        self.biggest_component_size = biggest_component_size
        self.free_space_size = free_space_size
        self.fragmentation = fragmentation
        self.absolute_social_cost = absolute_social_cost


class StepStats:
    def __init__(self, world_stats=None, agents_stats=None, act_time=0.):
        self.world_stats = world_stats or WorldStepStats()
        self.agents_stats = agents_stats or AgentStepStats()
        self.act_time = act_time


class TimeoutError(Exception):
    def __init(self):
        pass


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError


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

        self.history = []

        # Time stats
        self.agent_uid_and_goal_to_world_snapshot = {agent_uid: [] for agent_uid in self.agent_uid_to_behavior.keys()}

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
        exception = None

        step_count = 0

        while run_active:

            active_agents = set(self.agent_uid_to_behavior.keys())

            self.rp.publish_sim_world(self.ref_world)

            trace_polygons = []

            step_count = 0

            self.simulation_log.append(utils.BasicLog("Starting run.", step_count))

            while active_agents:
                try:
                    # Increment simulation step count
                    step_count += 1

                    # Sense loop: update each agent's knowledge of the world
                    sense_durations = {}
                    self.sense(active_agents, step_count, sense_durations)

                    # Think loop: get each agent to think about their next step
                    think_durations = {}
                    with timeout(10*60):
                        actions = self.think(active_agents, trace_polygons, step_count, think_durations)

                    # Act loops: Verify that each action is doable individually and together, if so, execute them
                    act_start = time.time()
                    action_results = self.act(actions, step_count)
                    act_duration = time.time() - act_start

                    self.history.append(
                        SimulationStepResult(sense_durations, think_durations, act_duration, action_results, step_count)
                    )

                    # Once the simulation reference world has been modified, display the modification
                    if not self.display_sim_knowledge_only_once:
                        self.rp.publish_sim_world(self.ref_world)
                except Exception as e:
                    if self.catch_exceptions:
                        tb = traceback.format_exc()
                        run_exceptions_traces.append(tb)
                        self.simulation_log.append(utils.BasicLog(tb, step_count))
                    else:
                        self.simulation_log.append(utils.BasicLog("MET A RUNTIME EXCEPTION, EXITING !", step_count))
                        run_active = False
                        tb = traceback.format_exc()
                        run_exceptions_traces.append(tb)
                        exception = e
                        break

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
        self.simulation_log.append(utils.BasicLog("Saved simulation final state.", step_count))

        simulation_report = self.create_simulation_report()
        if run_exceptions_traces:
            simulation_report['exceptions'] = run_exceptions_traces

        simulation_report["agents_logs"] = {}
        for uid, behavior in self.agent_uid_to_behavior.items():
            simulation_report["agents_logs"][self.ref_world.entities[uid].name] = behavior.simulation_log

        # TODO Remove this temporary measure for a better separation between scenario generation and execution
        simulation_report["temp_goals"] = self.saved_goals
        self.simulation_log.append(
            utils.BasicLog("Simulation report saved at: {}".format(self.log_filepath), step_count)
        )
        simulation_report["simulation_log"] = self.simulation_log

        p = jsonpickle.Pickler()
        simulation_report["simulation_history"] = p.flatten(self.history)
        simulation_report["agent_plans_history"] = {
            agent_uid: p.flatten(dict(behavior.goal_to_plans)) for agent_uid, behavior in self.agent_uid_to_behavior.items()
        }

        simulation_report_json = json.dumps(simulation_report, default=lambda o: o.__dict__, indent=4, sort_keys=True)
        with open(self.log_filepath, 'w+') as f:
            f.write(simulation_report_json)

        if exception:
            for exception_trace in run_exceptions_traces:
                print(exception_trace)
            raise exception

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

        all_movables_uids = {
            entity_uid for entity_uid, entity in self.init_ref_world.entities.items()
            if isinstance(entity, Obstacle) and entity.type in all_movable_types}

        init_nb_cc, init_biggest_cc_size, init_all_cc_sum_size, init_frag_percentage = \
            stats_utils.get_connectivity_stats(
                self.init_ref_world, self.human_inflation_radius,
                [uid for uid, entity in self.init_ref_world.entities.items() if isinstance(entity, Robot)]
            )
        init_abs_social_cost = stats_utils.get_social_costs_stats(self.init_ref_world, tuple(all_movables_uids))

        replay_world = copy.deepcopy(self.init_ref_world)
        stats = [
            StepStats(
                world_stats=WorldStepStats(
                    init_nb_cc, init_biggest_cc_size, init_all_cc_sum_size, init_frag_percentage, init_abs_social_cost
                ),
                agents_stats={replay_world.entities[uid].name: AgentStepStats() for uid in self.agent_uid_to_behavior.keys()},
                act_time=0.
            )
        ]
        prev_agent_poses = {uid: replay_world.entities[uid].pose for uid in self.agent_uid_to_behavior.keys()}
        for sim_step_result in self.history:
            # Only repeat successful actions when replaying the simulation
            successful_actions = {
                uid: action_result.action for uid, action_result in sim_step_result.action_results.items()
                if (
                    isinstance(action_result, ar.ActionSuccess)
                    and isinstance(action_result.action, (ba.Rotation, ba.Translation, ba.Grab, ba.Release))
                )
            }

            collision.csv_simulate_simple_kinematics(replay_world, successful_actions, apply=True, ignore_collisions=True)
            for agent_uid, action in successful_actions.items():
                if isinstance(action, ba.Grab):
                    replay_world.entity_to_agent[action.entity_uid] = agent_uid
                if isinstance(action, ba.Release):
                    del replay_world.entity_to_agent[action.entity_uid]

            # Compute world state stats ignoring all dynamic obstacles (robots and grabbed obstacles, typically)
            # Only when a release action happens, otherwise preserve previous stats
            if any([isinstance(action, ba.Release) for action in successful_actions.values()]):
                end_nb_cc, end_biggest_cc_size, end_all_cc_sum_size, end_frag = stats_utils.get_connectivity_stats(
                    replay_world, self.human_inflation_radius,
                    [
                        uid for uid, entity in replay_world.entities.items()
                        if isinstance(entity, Robot) or uid in replay_world.entity_to_agent.keys()
                    ]
                )
                end_abs_social_cost = stats_utils.get_social_costs_stats(
                    replay_world, all_movables_uids.difference(set(replay_world.entity_to_agent.keys()))
                )
                world_stats = WorldStepStats(
                    end_nb_cc, end_biggest_cc_size, end_all_cc_sum_size, end_frag, end_abs_social_cost
                )
            else:
                world_stats = stats[-1].world_stats

            # Compute agents stats
            prev_agents_stats = stats[-1].agents_stats
            agents_stats = copy.deepcopy(prev_agents_stats)
            for name, agent_stats in agents_stats.items():
                uid = replay_world.get_entity_uid_from_name(name)

                if uid not in sim_step_result.action_results:
                    continue

                step_distance = utils.euclidean_distance(prev_agent_poses[uid], replay_world.entities[uid].pose)
                if uid in replay_world.entity_to_agent.inverse:
                    agent_stats.transfer_path_length += step_distance
                    agent_stats.nb_transfer_steps += 1
                else:
                    agent_stats.transit_path_length += step_distance
                    agent_stats.nb_transit_steps += 1
                agent_stats.path_length = step_distance

                robot_action_result = sim_step_result.action_results[uid]
                robot_action = robot_action_result.action

                if isinstance(robot_action_result, ar.ActionSuccess):
                    if isinstance(robot_action, ba.Grab):
                        agent_stats.nb_transfers += 1
                    elif isinstance(robot_action, ba.Wait):
                        agent_stats.nb_wait_steps += 1
                    elif isinstance(robot_action, ba.GoalSuccess):
                        agent_stats.nb_goals = + 1
                        agent_stats.nb_successful_goals += 1
                    elif isinstance(robot_action, ba.GoalFailed):
                        agent_stats.nb_goals =+ 1
                        agent_stats.nb_failed_goals += 1

                # TODO Find a way to ditch the self.saved_goals variable
                if not isinstance(robot_action, (ba.GoalResult, ba.GoalsFinished)):
                    current_goal = self.saved_goals[replay_world.entities[uid].name][agent_stats.nb_goals]
                    current_dynamic_plan = self.agent_uid_to_behavior[uid].goal_to_plans[current_goal]

                    step_index = sim_step_result.step_index
                    if step_index in current_dynamic_plan.conflicts_history:
                        conflicts = current_dynamic_plan.conflicts_history[step_index]

                        # Filter redundant conflicts
                        filtered_conflicts = []
                        robot_robot_uids, robot_obstacle_uids, stolen_uids, concurrent_uids = set(), set(), set(), set()
                        for conflict in conflicts:
                            if isinstance(conflict, stilman_2005_behavior.RobotRobotConflict):
                                if conflict.obstacle_uid not in robot_robot_uids:
                                    filtered_conflicts.append(conflict)
                                robot_robot_uids.add(conflict.obstacle_uid)
                            elif isinstance(conflict, stilman_2005_behavior.RobotObstacleConflict):
                                if conflict.obstacle_uid not in robot_obstacle_uids:
                                    filtered_conflicts.append(conflict)
                                robot_obstacle_uids.add(conflict.obstacle_uid)
                            elif isinstance(conflict, stilman_2005_behavior.StolenMovableConflict):
                                if conflict.obstacle_uid not in stolen_uids:
                                    filtered_conflicts.append(conflict)
                                stolen_uids.add(conflict.obstacle_uid)
                            elif isinstance(conflict, stilman_2005_behavior.ConcurrentGrabConflict):
                                if conflict.obstacle_uid not in concurrent_uids:
                                    filtered_conflicts.append(conflict)
                                concurrent_uids.add(conflict.obstacle_uid)

                        agent_stats.nb_conflicts += len(filtered_conflicts)
                        for conflict in filtered_conflicts:
                            if isinstance(conflict, stilman_2005_behavior.RobotRobotConflict):
                                agent_stats.nb_robot_robot_conflicts += 1
                            elif isinstance(conflict, stilman_2005_behavior.RobotObstacleConflict):
                                agent_stats.nb_robot_obstacle_conflicts += 1
                            elif isinstance(conflict, stilman_2005_behavior.StolenMovableConflict):
                                agent_stats.nb_stolen_movable_conflicts += 1
                            elif isinstance(conflict, stilman_2005_behavior.ConcurrentGrabConflict):
                                agent_stats.nb_concurrent_grab_conflicts += 1

                    if step_index in current_dynamic_plan.postponements_history:
                        agent_stats.nb_of_postponements += 1

                    if step_index in current_dynamic_plan.unpostponements_history:
                        agent_stats.nb_of_unpostponements += 1

                    if step_index in current_dynamic_plan.plan_history:
                        plans = current_dynamic_plan.plan_history[step_index]
                        if isinstance(plans, list):
                            agent_stats.nb_of_plan_computations += len(plans)
                        else:
                            agent_stats.nb_of_plan_computations += 1

                agent_stats.sense_time += sim_step_result.sense_durations[uid]
                agent_stats.think_time += sim_step_result.think_durations[uid]

            # Update act_time
            act_time = stats[-1].act_time + sim_step_result.act_duration

            stats.append(StepStats(world_stats, agents_stats, act_time))

            prev_agent_poses = {uid: replay_world.entities[uid].pose for uid in self.agent_uid_to_behavior.keys()}

        p = jsonpickle.Pickler(unpicklable=False)
        report = {"stats": p.flatten(stats)}

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

    def sense(self, active_agents, step_count, sense_durations):
        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            if agent_uid in active_agents:
                sense_start = time.time()
                last_action_result = (
                    self.history[-1].action_results[agent_uid] if (self.history and agent_uid in self.history[-1].action_results) else ar.ActionSuccess
                )
                behavior.sense(self.ref_world, last_action_result, step_count)
                sense_durations[agent_uid] = time.time() - sense_start

    def think(self, active_agents, trace_polygons, step_count, think_durations):
        agent_uid_to_next_action = {}
        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            if agent_uid in active_agents:
                think_start = time.time()
                agent_next_action = behavior.think()
                think_durations[agent_uid] = time.time() - think_start

                # TODO Change goal coordinates for easier reading to goal name in log.
                if isinstance(agent_next_action, ba.GoalsFinished):
                    # If the agent has executed all of its goals, remove it from the active agents
                    active_agents.remove(agent_uid)
                    self.simulation_log.append(
                        utils.BasicLog("Agent {} finished executing all its goals.".format(self.ref_world.entities[agent_uid].name), step_count)
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
                elif isinstance(agent_next_action, ba.GoalSuccess):
                    # If the agent reached its current goal
                    self.save_world_snapshot(agent_uid, agent_next_action, trace_polygons, step_count)
                    self.simulation_log.append(
                        utils.BasicLog(
                            "Agent {} successfully executed goal {}.".format(
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
                agent_uid_to_next_action[agent_uid] = agent_next_action
        return agent_uid_to_next_action

    def act(self, agent_uid_to_next_action, step_count, use_b2=False, ignore_collisions=True):
        # Only Grab and Release actions require further checks, and Wait actions are necessarily valid
        to_check = {
            uid: a for uid, a in agent_uid_to_next_action.items()
            if isinstance(a, (ba.Translation, ba.Rotation)) and not isinstance(a, (ba.Grab, ba.Release))
        }
        action_results = {
            uid: ar.ActionSuccess(a, self.ref_world.entities[uid].pose)
            for uid, a in agent_uid_to_next_action.items() if isinstance(a, (ba.Wait, ba.GoalSuccess, ba.GoalFailed, ba.GoalsFinished))
        }

        # Check if released entity is already grabbed by the right agent
        for agent_uid, action in agent_uid_to_next_action.items():
            if isinstance(action, ba.Release):
                entity_uid = action.entity_uid
                if agent_uid not in self.ref_world.entity_to_agent.inverse or entity_uid not in self.ref_world.entity_to_agent:
                    action_results[agent_uid] = ar.NotGrabbedFailure(action)
                else:
                    other_agent_uid = self.ref_world.entity_to_agent[entity_uid]
                    if other_agent_uid != agent_uid:
                        action_results[agent_uid] = ar.GrabbedByOtherFailure(action, other_agent_uid)
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
                    action_results[agent_uid] = ar.SimultaneousGrabFailure(action, entity_to_grab_agents[entity_uid])
                    continue
                if agent_uid in self.ref_world.entity_to_agent.inverse:
                    action_results[agent_uid] = ar.GrabMoreThanOneFailure(action)
                    continue
                if entity_uid in self.ref_world.entity_to_agent:
                    other_agent_uid = self.ref_world.entity_to_agent[entity_uid]
                    other_releases = other_agent_uid in to_check and isinstance(to_check[other_agent_uid], ba.Release)
                    if not other_releases:
                        action_results[agent_uid] = ar.AlreadyGrabbedFailure(action, other_agent_uid)
                        continue
                to_check[agent_uid] = action

        # Check actions regarding dynamic collisions and apply the valid ones using Box2D
        if use_b2:
            collides_with = self.b2_sim.simulate_simple_kinematics([to_check], apply=True)
        else:
            collides_with = collision.csv_simulate_simple_kinematics(self.ref_world, to_check, apply=True, ignore_collisions=ignore_collisions)

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
                    agent_uid in self.ref_world.entity_to_agent.inverse
                    and self.ref_world.entity_to_agent.inverse[agent_uid] in collides_with
                    and not isinstance(action, ba.Release)
                )
            )
            if action_dynamically_collides and not ignore_collisions:
                action_results[agent_uid] = ar.DynamicCollisionFailure(action, collides_with)
            else:
                if action_dynamically_collides and ignore_collisions:
                    self.simulation_log.append(utils.BasicLog(
                        'Dynamic collision ignored, entities: {}'.format({
                            self.ref_world.entities[uid].name: {self.ref_world.entities[uid2].name for uid2 in uids}
                            for uid, uids in collides_with.items()
                        }), step_count
                    ))

                # SUCCESS
                # If Grab or Release, first update self.ref_world.entity_to_agent
                if isinstance(action, ba.Grab):
                    self.ref_world.entity_to_agent[action.entity_uid] = agent_uid
                if isinstance(action, ba.Release):
                    del self.ref_world.entity_to_agent[action.entity_uid]

                # Then apply to world
                if use_b2:
                    agent = self.ref_world.entities[agent_uid]
                    agent_new_pose = self.b2_sim.get_entity_pose(agent_uid)
                    agent_new_polygon = utils.set_polygon_pose(agent.polygon, agent.pose, agent_new_pose)
                    agent.pose, agent.polygon = agent_new_pose, agent_new_polygon
                    if agent_uid in self.ref_world.entity_to_agent.inverse:
                        entity_uid = self.ref_world.entity_to_agent.inverse[agent_uid]
                        entity = self.ref_world.entities[entity_uid]
                        entity_new_pose = self.b2_sim.get_entity_pose(entity_uid)
                        entity_new_polygon = utils.set_polygon_pose(entity.polygon, entity.pose, entity_new_pose)
                        entity.pose, entity.polygon = entity_new_pose, entity_new_polygon

                action_results[agent_uid] = ar.ActionSuccess(action, self.ref_world.entities[agent_uid].pose)

        return action_results
