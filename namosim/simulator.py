import copy
import io
import json
import os
import pickle
import random
import time
import tkinter as tk
import traceback
import typing as t
from queue import Queue

import cairosvg
import jsonpickle
from PIL import Image, ImageTk
from shapely.geometry import Polygon

import namosim.config as config
import namosim.display.ros2_publisher as ros2
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
from namosim.behaviors.baseline_behavior import BaselineBehavior
from namosim.behaviors.stilman_2005_behavior import DynamicPlan, Stilman2005Behavior
from namosim.exceptions import timeout
from namosim.models import PoseModel, SimulationModel
from namosim.navigation.conflict import (
    ConcurrentGrabConflict,
    RobotObstacleConflict,
    RobotRobotConflict,
    SimultaneousSpaceAccess,
    StealingMovableConflict,
    StolenMovableConflict,
)
from namosim.utils import collision, conversion, stats_utils, utils
from namosim.world.obstacle import Obstacle
from namosim.world.robot import Robot
from namosim.world.world import World


class SimulationStepResult:
    def __init__(
        self,
        sense_durations: t.Dict[int, float],
        think_durations: t.Dict[int, float],
        act_duration: float,
        action_results: t.Dict[int, ar.ActionResult],
        step_index: int,
    ):
        self.sense_durations = sense_durations
        self.think_durations = think_durations
        self.act_duration = act_duration
        self.action_results = action_results
        self.step_index = step_index


class AgentStepStats:
    def __init__(
        self,
        transit_path_length: float = 0.0,
        transfer_path_length: float = 0.0,
        path_length: float = 0.0,
        nb_transfers: int = 0,
        nb_successful_goals: int = 0,
        nb_failed_goals: int = 0,
        nb_goals: int = 0,
        nb_conflicts: int = 0,
        nb_robot_robot_conflicts: int = 0,
        nb_robot_obstacle_conflicts: int = 0,
        nb_stolen_movable_conflicts: int = 0,
        nb_stealing_movable_conflicts: int = 0,
        nb_concurrent_grab_conflicts: int = 0,
        nb_simultaneous_space_access_conflicts: int = 0,
        nb_wait_steps: int = 0,
        nb_transit_steps: int = 0,
        nb_transfer_steps: int = 0,
        nb_steps: int = 0,
        nb_of_postponements: int = 0,
        nb_of_unpostponements: int = 0,
        nb_of_plan_computations: int = 0,
        sense_time: float = 0.0,
        think_time: float = 0.0,
    ):
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
        self.nb_stealing_movable_conflicts = nb_stealing_movable_conflicts
        self.nb_concurrent_grab_conflicts = nb_concurrent_grab_conflicts
        self.nb_simultaneous_space_access_conflicts = (
            nb_simultaneous_space_access_conflicts
        )
        self.nb_wait_steps = nb_wait_steps
        self.nb_transit_steps = nb_transit_steps
        self.nb_transfer_steps = nb_transfer_steps
        self.nb_steps = nb_steps
        self.nb_of_postponements = nb_of_postponements
        self.nb_of_unpostponements = nb_of_unpostponements
        self.nb_of_plan_computations = nb_of_plan_computations
        self.sense_time = sense_time
        self.think_time = think_time


class WorldStepStats:
    def __init__(
        self,
        nb_components: int = 0,
        biggest_component_size: int = 0,
        free_space_size: int = 0,
        fragmentation: float = 0.0,
        absolute_social_cost: float = 0.0,
    ):
        self.nb_components = nb_components
        self.biggest_component_size = biggest_component_size
        self.free_space_size = free_space_size
        self.fragmentation = fragmentation
        self.absolute_social_cost = absolute_social_cost


class StepStats:
    def __init__(
        self,
        world_stats: t.Optional[WorldStepStats] = None,
        agents_stats: t.Optional[t.Dict[str, AgentStepStats]] = None,
        act_time: float = 0.0,
    ):
        self.world_stats = world_stats or WorldStepStats()
        self.agents_stats = agents_stats or {}
        self.act_time = act_time


class Simulator:
    """The main simulator class manages all aspects of the simulation. It initializes
    the world and agents and executes a **sense** -> **think** -> **act** loop until all agents have
    either completed or failed their navigation goals."""

    def __init__(
        self,
        simulation_file_path: str,
        simulation_log_stub: str = "",
        goals: t.Optional[t.Dict[str, t.List[PoseModel]]] = None,
        timestring: t.Optional[str] = None,
    ):
        self.window: tk.Tk | None = None
        self.background: tk.Label | None = None

        if config.DISPLAY_WINDOW:
            self.window = tk.Tk()
            self.window.title("NAMOSIM")
            self.window.resizable(True, True)
            self.background = tk.Label(self.window)
            self.background.pack()

        # Load simulation file and initialize logs
        if timestring:
            self.sim_start_timestring = timestring
        else:
            self.sim_start_timestring = utils.timestamp_string()
        simulation_file_abs_path = os.path.abspath(simulation_file_path)

        with open(simulation_file_abs_path) as f:
            config_json = json.load(f)
        self.config = SimulationModel.model_validate(config_json)

        sim_file_parent_dirname = os.path.basename(
            os.path.normpath(
                os.path.abspath(os.path.join(simulation_file_abs_path, ".."))
            )
        )
        self.simulation_filename = os.path.splitext(
            os.path.basename(simulation_file_abs_path)
        )[0]

        main_logs_dir = os.path.join(
            os.path.dirname(__file__),
            "../logs/",
            simulation_log_stub,
            sim_file_parent_dirname,
            self.simulation_filename,
        )
        self.logs_dir = os.path.join(main_logs_dir, self.sim_start_timestring + "/")

        os.makedirs(self.logs_dir)
        os.makedirs(self.logs_dir + "simulation/")

        self.simulation_log = utils.CustomLogger()

        self.simulation_log.append(
            utils.BasicLog("Simulation file successfully loaded", 0)
        )

        # Save general simulation parameters
        self.random_seed = self.config.random_seed or 10
        random.seed(self.random_seed)
        self.provide_walls = self.config.provide_walls
        self.human_inflation_radius = 0.55 / 2.0  # [m]

        self.simulation_log.append(
            utils.BasicLog("Created log folders at:{}".format(str(self.logs_dir)), 0)
        )

        self.save_init_world_state = True
        self.save_intermediate_world_states = False
        self.save_end_world_state = True
        self.save_stats = True
        self.save_history = False
        self.save_logs = True
        self.pickle_saved_data = True

        if self.pickle_saved_data:

            def pickle_save(obj: t.Any, filepath: str):
                filepath += ".pickle"
                with open(filepath, "wb") as f:
                    pickle.dump(obj, f)

            self.save = pickle_save
        else:

            def json_save(obj: t.Any, filepath: str):
                filepath += ".json"
                p = jsonpickle.Pickler(unpicklable=False)
                flattened_obj = p.flatten(obj)
                with open(filepath, "w+") as f:
                    json.dump(
                        flattened_obj,
                        f,
                        default=lambda o: o.__dict__,
                        indent=4,
                        sort_keys=True,
                    )

            self.save = json_save

        # Reinitialize rviz display
        self.ros_publisher = ros2.RosPublisher(
            node_name=self.simulation_filename, sim_config=self.config
        )
        self.ros_publisher.cleanup_all()

        self.simulation_log.append(utils.BasicLog("Display backend initialized.", 0))

        # Create world from world description json file
        world_file_path = self.config.files.world_file
        world_abs_path = os.path.join(
            os.path.dirname(simulation_file_abs_path), world_file_path
        )
        self.init_ref_world = World.load_from_json(world_abs_path)

        self.simulation_log.append(utils.BasicLog("World file successfully loaded.", 0))

        if self.save_init_world_state:
            self.init_ref_world.save_to_files(
                json_filepath=self.logs_dir
                + "simulation/"
                + self.simulation_filename
                + ".json",
                svg_filepath=self.init_ref_world.init_geometry_filename,
            )
        self.ref_world: World = copy.deepcopy(self.init_ref_world)

        # Associate autonomous agents with goals and behaviors
        self.goal_poses = {
            goal.name: goal.pose for goal in self.init_ref_world.goals.values()
        }

        self.agent_uid_to_goals: t.Dict[int, t.List[PoseModel]]
        """
        Maps an agent uid to a list of goal poses
        """

        self.saved_goals: t.Dict[str, t.List[PoseModel]]
        """
        Maps an agent name to a list of goal poses
        """

        self.agent_uid_to_behavior: t.Dict[int, BaselineBehavior]
        """
        Maps an agent uid to an instance of `BaselineBehavior`
        """

        if goals:
            self.saved_goals = goals
            self.agent_uid_to_goals = {
                self.ref_world.get_entity_uid_from_name(agent_name): gls
                for agent_name, gls in goals.items()
            }
        else:
            self.agent_uid_to_goals = self.initialize_agents_goals(self.goal_poses)
            self.saved_goals = {
                self.ref_world.entities[id].name: copy.deepcopy(goals)
                for id, goals in self.agent_uid_to_goals.items()
            }

        self.agent_uid_to_behavior = self.initialize_agents_behaviors(
            self.agent_uid_to_goals
        )

        self.history: t.List[SimulationStepResult] = []
        """
        A list of simulation step results
        """

        # Time stats
        self.agent_uid_and_goal_to_world_snapshot = {
            agent_uid: [] for agent_uid in self.agent_uid_to_behavior.keys()
        }

        self.catch_exceptions = False

        self.simulation_log.append(utils.BasicLog("Simulation successfully loaded.", 0))
        self.run_exceptions_traces: t.List[t.Any] = []
        self.exception: t.Union[Exception, None] = None

    def step(
        self, active_agents: set[int], trace_polygons: t.List[Polygon], step_count: int
    ) -> t.Tuple[set[int], t.List[Polygon], int]:
        if len(active_agents) == 0:
            self.end_simulation(step_count=step_count)
            return (active_agents, trace_polygons, step_count)

        try:
            # Increment simulation step count
            step_count += 1
            self.ros_publisher.publish_message(
                "Sim steps: {}".format(step_count),
                pose=(
                    0.0,
                    self.ref_world.discretization_data.grid_pose[1]
                    + self.ref_world.discretization_data.height
                    + 0.25,
                    0.0,
                ),
                font_size=0.5,
            )

            # Sense loop: update each agent's knowledge of the world
            sense_durations = {}
            self.sense(active_agents, step_count, sense_durations)

            # Think loop: get each agent to think about their next step
            think_durations = {}
            with timeout(10 * 60):
                actions = self.think(
                    active_agents=active_agents,
                    trace_polygons=trace_polygons,
                    step_count=step_count,
                    think_durations=think_durations,
                )

            # Act loops: Verify that each action is doable individually and together, if so, execute them
            act_start = time.time()
            action_results = self.act(actions, step_count)
            act_duration = time.time() - act_start

            self.history.append(
                SimulationStepResult(
                    sense_durations,
                    think_durations,
                    act_duration,
                    action_results,
                    step_count,
                )
            )

            # Once the simulation reference world has been modified, display the modification
            self.ros_publisher.publish_sim_world(self.ref_world)
        except Exception as e:
            self.end_simulation(step_count=step_count, err=e)

        return (active_agents, trace_polygons, step_count)

    def end_simulation(self, step_count: int, err: Exception | None = None):
        self.run_active = False
        if self.window:
            self.window.quit()
        if self.background:
            self.background.quit()

        if err is not None:
            if self.catch_exceptions:
                tb = traceback.format_exc()
                self.run_exceptions_traces.append(tb)
                self.simulation_log.append(utils.BasicLog(tb, step_count))
            else:
                self.simulation_log.append(
                    utils.BasicLog("MET A RUNTIME EXCEPTION, EXITING !", step_count)
                )
                tb = traceback.format_exc()
                self.run_exceptions_traces.append(tb)
                self.exception = err
                return

    def render_window(self):
        if not self.window:
            raise Exception("No window")
        if not self.background:
            raise Exception("No background")

        svg = self.ref_world.to_svg().toprettyxml()
        image_data = cairosvg.svg2png(svg, dpi=200, output_width=600)
        if not image_data:
            raise Exception("Failed to convert world to image")

        image = Image.open(io.BytesIO(image_data))
        tk_image = ImageTk.PhotoImage(image)
        self.window.geometry(f"{tk_image.width()}x{tk_image.height()}")

        self.background.configure(image=tk_image)

        # store tk_image on background.image to prevent garbage collection
        self.background.image = tk_image  # type: ignore

    def run(self) -> t.List[SimulationStepResult]:
        self.run_active = True
        self.run_exceptions_traces = []
        self.exception = None
        step_count = 0

        while self.run_active:
            active_agents: set[int] = set(self.agent_uid_to_behavior.keys())
            self.ros_publisher.publish_sim_world(self.ref_world)
            trace_polygons: t.List[Polygon] = []
            step_count = 0
            self.simulation_log.append(utils.BasicLog("Starting run.", step_count))
            self.ros_publisher.publish_message(
                "Sim steps: {}".format(step_count),
                pose=(
                    0.0,
                    self.ref_world.discretization_data.grid_pose[1]
                    + self.ref_world.discretization_data.height
                    + 0.25,
                    0.0,
                ),
                font_size=0.5,
            )

            print("")

            if self.window is not None:
                self._run_window_loop(
                    active_agents=active_agents,
                    trace_polygons=trace_polygons,
                    step_count=step_count,
                )
            else:
                while len(active_agents) > 0:
                    (active_agents, trace_polygons, step_count) = self.step(
                        active_agents=active_agents,
                        trace_polygons=trace_polygons,
                        step_count=step_count,
                    )
                self.end_simulation(step_count=step_count)

        self._save_results(step_count=step_count)

        return self.history

    def _save_results(self, step_count: int):
        # Save simulation results
        # - Save exception traces
        if self.run_exceptions_traces:
            exceptions = {"exceptions": self.run_exceptions_traces}
            exceptions_filepath = os.path.join(
                os.path.dirname(self.logs_dir), "exceptions"
            )
            self.save(exceptions, exceptions_filepath)
            self.simulation_log.append(
                utils.BasicLog(
                    "Saved exceptions at: {}".format(exceptions_filepath), step_count
                )
            )

        # - Save world end state as SVG+JSON
        if self.save_end_world_state:
            self.ref_world.save_to_files(
                json_filepath=self.logs_dir
                + "simulation/"
                + self.simulation_filename
                + "_end"
                + ".json",
                svg_filepath=utils.append_suffix(
                    self.init_ref_world.init_geometry_filename, "_end"
                ),
            )
            self.simulation_log.append(
                utils.BasicLog("Saved simulation final state.", step_count)
            )

        # - Save stats
        if self.save_stats:
            stats = self.create_simulation_report()
            stats_filepath = os.path.join(os.path.dirname(self.logs_dir), "stats")
            self.save(stats, stats_filepath)
            self.simulation_log.append(
                utils.BasicLog("Saved stats at: {}".format(stats_filepath), step_count)
            )

        # - Save simulation history
        # TODO Remove this temporary measure for a better separation between scenario generation and execution
        if self.save_history:
            history = {}
            history["temp_goals"] = self.saved_goals
            history["random_seed"] = self.random_seed
            history["simulation_history"] = self.history
            history["agent_plans_history"] = {
                agent_uid: dict(behavior.goal_to_plans)
                for agent_uid, behavior in self.agent_uid_to_behavior.items()
            }
            history_filepath = os.path.join(os.path.dirname(self.logs_dir), "history")
            self.save(history, history_filepath)
            self.simulation_log.append(
                utils.BasicLog(
                    "Saved history at: {}".format(history_filepath), step_count
                )
            )

        # - Save simulation and agents logs
        if self.save_logs:
            logs = {}
            logs["simulation_log"] = self.simulation_log
            logs["agents_logs"] = {}
            for uid, behavior in self.agent_uid_to_behavior.items():
                logs["agents_logs"][
                    self.ref_world.entities[uid].name
                ] = behavior.simulation_log
            logs_filepath = os.path.join(os.path.dirname(self.logs_dir), "logs")
            self.save(logs, logs_filepath)

        if self.exception is not None:
            for exception_trace in self.run_exceptions_traces:
                print(exception_trace)
            raise self.exception

    def _run_window_loop(
        self, active_agents: set[int], trace_polygons: t.List[Polygon], step_count: int
    ):
        if self.window is None:
            raise Exception("No window")
        self._window_step(
            active_agents=active_agents,
            trace_polygons=trace_polygons,
            step_count=step_count,
        )
        self.window.mainloop()

    def _window_step(
        self, active_agents: set[int], trace_polygons: t.List[Polygon], step_count: int
    ):
        if not self.window:
            raise Exception("No window")
        (active_agents, trace_polygons, step_count) = self.step(
            active_agents=active_agents,
            trace_polygons=trace_polygons,
            step_count=step_count,
        )
        self.render_window()
        self.window.after(
            1, self._window_step, active_agents, trace_polygons, step_count
        )

    def _create_robot_world_from_sim_world(self):
        entities = dict()
        for entity_uid, entity in self.ref_world.entities.items():
            if isinstance(entity, Robot) or (
                (isinstance(entity, Obstacle) and entity.type_ == "wall")
                if self.provide_walls
                else True
            ):
                entities[entity_uid] = copy.deepcopy(entity)

        return World(
            entities=entities,
            taboo_zones=copy.deepcopy(self.ref_world.taboo_zones),
            discretization_data=copy.deepcopy(self.ref_world.discretization_data),
        )

    def create_simulation_report(self):
        all_movable_types = set()
        for entity in self.init_ref_world.entities.values():
            if isinstance(entity, Robot):
                all_movable_types.update(set(entity.movable_whitelist))

        all_movables_uids = {
            entity_uid
            for entity_uid, entity in self.init_ref_world.entities.items()
            if isinstance(entity, Obstacle) and entity.type_ in all_movable_types
        }

        (
            init_nb_cc,
            init_biggest_cc_size,
            init_all_cc_sum_size,
            init_frag_percentage,
        ) = stats_utils.get_connectivity_stats(
            self.init_ref_world,
            self.human_inflation_radius,
            [
                uid
                for uid, entity in self.init_ref_world.entities.items()
                if isinstance(entity, Robot)
            ],
            ros_publisher=self.ros_publisher,
        )
        init_abs_social_cost = stats_utils.get_social_costs_stats(
            self.init_ref_world,
            tuple(all_movables_uids),
            ros_publisher=self.ros_publisher,
        )

        replay_world = copy.deepcopy(self.init_ref_world)
        stats = [
            StepStats(
                world_stats=WorldStepStats(
                    init_nb_cc,
                    init_biggest_cc_size,
                    init_all_cc_sum_size,
                    init_frag_percentage,
                    init_abs_social_cost,
                ),
                agents_stats={
                    replay_world.entities[uid].name: AgentStepStats()
                    for uid in self.agent_uid_to_behavior.keys()
                },
                act_time=0.0,
            )
        ]
        prev_agent_poses = {
            uid: replay_world.entities[uid].pose
            for uid in self.agent_uid_to_behavior.keys()
        }
        for sim_step_result in self.history:
            # Only repeat successful actions when replaying the simulation
            successful_actions = {
                uid: action_result.action
                for uid, action_result in sim_step_result.action_results.items()
                if (
                    isinstance(action_result, ar.ActionSuccess)
                    and isinstance(
                        action_result.action,
                        (ba.Rotation, ba.Translation, ba.Grab, ba.Release),
                    )
                )
            }

            collision.csv_simulate_simple_kinematics(
                replay_world, successful_actions, apply=True, ignore_collisions=True
            )
            for agent_uid, action in successful_actions.items():
                if isinstance(action, ba.Grab):
                    replay_world.entity_to_agent[action.entity_uid] = agent_uid
                if isinstance(action, ba.Release):
                    del replay_world.entity_to_agent[action.entity_uid]

            # Compute world state stats ignoring all dynamic obstacles (robots and grabbed obstacles, typically)
            # Only when a release action happens, otherwise preserve previous stats
            if any(
                [
                    isinstance(action, ba.Release)
                    for action in successful_actions.values()
                ]
            ):
                (
                    end_nb_cc,
                    end_biggest_cc_size,
                    end_all_cc_sum_size,
                    end_frag,
                ) = stats_utils.get_connectivity_stats(
                    replay_world,
                    self.human_inflation_radius,
                    [
                        uid
                        for uid, entity in replay_world.entities.items()
                        if isinstance(entity, Robot)
                        or uid in replay_world.entity_to_agent.keys()
                    ],
                    ros_publisher=self.ros_publisher,
                )
                end_abs_social_cost = stats_utils.get_social_costs_stats(
                    replay_world,
                    all_movables_uids.difference(
                        set(replay_world.entity_to_agent.keys())
                    ),
                    ros_publisher=self.ros_publisher,
                )
                world_stats = WorldStepStats(
                    end_nb_cc,
                    end_biggest_cc_size,
                    end_all_cc_sum_size,
                    end_frag,
                    end_abs_social_cost,
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

                step_distance = utils.euclidean_distance(
                    prev_agent_poses[uid], replay_world.entities[uid].pose
                )
                if uid in replay_world.entity_to_agent.inverse:
                    agent_stats.transfer_path_length += step_distance
                else:
                    agent_stats.transit_path_length += step_distance
                agent_stats.path_length += step_distance

                robot_action_result = sim_step_result.action_results[uid]
                robot_action = robot_action_result.action

                if isinstance(robot_action_result, ar.ActionSuccess):
                    if isinstance(robot_action, ba.Grab):
                        agent_stats.nb_transfers += 1
                        agent_stats.nb_transfer_steps += 1
                        agent_stats.nb_steps += 1
                    elif isinstance(robot_action, ba.Wait):
                        agent_stats.nb_wait_steps += 1
                        agent_stats.nb_steps += 1
                    elif isinstance(robot_action, ba.GoalSuccess):
                        agent_stats.nb_goals += 1
                        agent_stats.nb_successful_goals += 1
                    elif isinstance(robot_action, ba.GoalFailed):
                        agent_stats.nb_goals += 1
                        agent_stats.nb_failed_goals += 1
                    elif isinstance(
                        robot_action, (ba.Translation, ba.Rotation, ba.Release)
                    ):
                        agent_stats.nb_steps += 1
                        if uid in replay_world.entity_to_agent.inverse:
                            agent_stats.nb_transfer_steps += 1
                        else:
                            agent_stats.nb_transit_steps += 1

                # TODO Find a way to ditch the self.saved_goals variable
                if not isinstance(robot_action, (ba.GoalResult, ba.GoalsFinished)):
                    current_goal = self.saved_goals[replay_world.entities[uid].name][
                        agent_stats.nb_goals
                    ]
                    current_plan = self.agent_uid_to_behavior[uid].goal_to_plans[
                        current_goal
                    ]

                    step_index = sim_step_result.step_index

                    if isinstance(current_plan, DynamicPlan):
                        if step_index in current_plan.conflicts_history:
                            conflict = current_plan.conflicts_history[step_index]
                            agent_stats.nb_conflicts += 1
                            if isinstance(conflict, RobotRobotConflict):
                                agent_stats.nb_robot_robot_conflicts += 1
                            elif isinstance(conflict, RobotObstacleConflict):
                                agent_stats.nb_robot_obstacle_conflicts += 1
                            elif isinstance(conflict, StolenMovableConflict):
                                agent_stats.nb_stolen_movable_conflicts += 1
                            elif isinstance(conflict, StealingMovableConflict):
                                agent_stats.nb_stealing_movable_conflicts += 1
                            elif isinstance(conflict, ConcurrentGrabConflict):
                                agent_stats.nb_concurrent_grab_conflicts += 1
                            elif isinstance(conflict, SimultaneousSpaceAccess):
                                agent_stats.nb_simultaneous_space_access_conflicts += 1

                        if step_index in current_plan.postponements_history:
                            agent_stats.nb_of_postponements += 1

                        if step_index in current_plan.unpostponements_history:
                            agent_stats.nb_of_unpostponements += 1

                        if step_index in current_plan.steps_with_replan_call:
                            agent_stats.nb_of_plan_computations += 1

                agent_stats.sense_time += sim_step_result.sense_durations[uid]
                agent_stats.think_time += sim_step_result.think_durations[uid]

            # Update act_time
            act_time = stats[-1].act_time + sim_step_result.act_duration

            stats.append(StepStats(world_stats, agents_stats, act_time))

            prev_agent_poses = {
                uid: replay_world.entities[uid].pose
                for uid in self.agent_uid_to_behavior.keys()
            }

        report = {"stats": stats}

        return report

    def initialize_agents_goals(
        self,
        goals_geometries: t.Dict[str, PoseModel],
        max_nb_goals: float = float("inf"),
    ) -> t.Dict[int, t.List[PoseModel]]:
        """
        Contructs and returns a dictionary that maps an agent uid to a list of nativation goal poses. Each
        agent may multiple navigation goals.
        """
        agent_uid_to_goals = {}
        for agent_behavior in self.config.agents_behaviors:
            agent_uid = self.ref_world.get_entity_uid_from_name(
                agent_behavior.agent_name
            )
            if agent_uid in agent_uid_to_goals:
                raise RuntimeError(
                    "You can only associate a single behavior with entity: {entity_name}.".format(
                        entity_name=agent_behavior.agent_name
                    )
                )
            else:
                agent_navigation_goals: t.List[PoseModel] = []

                if agent_behavior.behavior.navigation_goals is not None:
                    for count, config_goal in enumerate(
                        agent_behavior.behavior.navigation_goals
                    ):
                        if count > max_nb_goals:
                            break
                        if config_goal.name in goals_geometries:
                            agent_navigation_goals.append(
                                goals_geometries[config_goal.name]
                            )

                agent_uid_to_goals[agent_uid] = agent_navigation_goals

        return agent_uid_to_goals

    def initialize_agents_behaviors(
        self, agents_navigation_goals: t.Dict[int, t.List[PoseModel]]
    ) -> t.Dict[int, BaselineBehavior]:
        agent_uid_to_behavior = dict()

        for agent in self.config.agents_behaviors:
            agent_uid = self.ref_world.get_entity_uid_from_name(agent.agent_name)
            agent_navigation_goals = agents_navigation_goals[agent_uid]
            if agent_uid in agent_uid_to_behavior:
                raise RuntimeError(
                    "You can only associate a single behavior with entity: {entity_name}.".format(
                        entity_name=agent.agent_name
                    )
                )
            else:
                behavior_config = agent.behavior

                if behavior_config.name == "stilman_2005_behavior":
                    agent_world = copy.deepcopy(self.ref_world)
                    self.ros_publisher.cleanup_robot_world(ns=agent.agent_name)
                    agent_uid_to_behavior[agent_uid] = Stilman2005Behavior(
                        agent_world,
                        agent_uid,
                        agent_navigation_goals,
                        behavior_config,
                        self.logs_dir,
                        ros_publisher=self.ros_publisher,
                    )
                else:
                    raise NotImplementedError(
                        "You tried to associate entity '{agent_name}' with a behavior named"
                        "'{b_name}' that is not implemented yet."
                        "Maybe you mispelled something ?".format(
                            agent_name=agent.agent_name, b_name=behavior_config.name
                        )
                    )
        return agent_uid_to_behavior

    def save_world_snapshot(
        self,
        agent_uid: int,
        action: ba.BasicAction,
        trace_polygons: t.List[Polygon],
        step_count: int,
    ):
        world_snapshot = copy.deepcopy(self.ref_world)
        self.agent_uid_and_goal_to_world_snapshot[agent_uid].append(
            {
                "goal": action.goal,  # type: ignore
                "goal_status": str(action),
                "world_snapshot": copy.deepcopy(self.ref_world),
            }
        )
        goal_counter = len(self.agent_uid_and_goal_to_world_snapshot[agent_uid])

        suffix = (
            "at_step_"
            + str(step_count)
            + "_after_goal_"
            + str(goal_counter)
            + "_of_"
            + self.ref_world.entities[agent_uid].name
        )
        json_filepath = (
            self.logs_dir + "simulation/" + self.simulation_filename + suffix + ".json"
        )
        svg_filepath = utils.append_suffix(
            self.init_ref_world.init_geometry_filename, suffix
        )
        svg_data = world_snapshot.to_svg()

        new_group = svg_data.createElement("svg:g")
        new_group.setAttribute("id", "traces" + suffix)
        new_group.setAttribute("inkscape:groupmode", "layer")
        new_group.setAttribute("inkscape:label", "traces" + suffix)
        svg_data.childNodes[0].appendChild(new_group)
        for polygon in trace_polygons:
            conversion.add_shapely_geometry_to_svg(
                polygon,
                "goal_generated_" + str(goal_counter),
                conversion.OBSTACE_TRACE_STYLE,
                svg_data,
                new_group,
                self.ref_world.scaling_value,
                self.ref_world.discretization_data.width,
                self.ref_world.discretization_data.height,
            )
        del trace_polygons[: len(trace_polygons)]

        json_data = world_snapshot.to_json(svg_filepath)
        world_snapshot.save_to_files(
            json_data=json_data,
            svg_data=svg_data,
            json_filepath=json_filepath,
            svg_filepath=svg_filepath,
        )

    def sense(
        self,
        active_agents: set[int],
        step_count: int,
        sense_durations: t.Dict[int, float],
    ):
        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            if agent_uid in active_agents:
                sense_start = time.time()
                last_action_result = (
                    self.history[-1].action_results[agent_uid]
                    if (self.history and agent_uid in self.history[-1].action_results)
                    else ar.ActionSuccess()
                )
                behavior.sense(self.ref_world, last_action_result, step_count)
                sense_durations[agent_uid] = time.time() - sense_start

    def _agent_think(
        self,
        agent_uid: int,
        behavior: BaselineBehavior,
        results: Queue[t.Tuple[int, float, ba.BasicAction | None]],
    ):
        think_start = time.time()
        next_action = behavior.think()
        think_duration = time.time() - think_start
        results.put((agent_uid, think_duration, next_action))

    def process_think_results(
        self,
        results: t.Iterable[t.Tuple[int, float, ba.BasicAction | None]],
        think_durations: t.Dict[int, float],
        active_agents: t.Set[int],
        trace_polygons: t.List[Polygon],
        step_count: int,
    ) -> t.Dict[int, ba.BasicAction]:
        """Process the results of each agent's think step. Updates the set of activate agents and the dictionary of think durations."""
        agent_uid_to_next_action: t.Dict[int, ba.BasicAction] = {}
        for agent_uid, think_duration, agent_next_action in results:
            think_durations[agent_uid] = think_duration

            # TODO Change goal coordinates for easier reading to goal name in log.
            if isinstance(agent_next_action, ba.GoalsFinished):
                # If the agent has executed all of its goals, remove it from the active agents
                active_agents.remove(agent_uid)
                self.simulation_log.append(
                    utils.BasicLog(
                        "Agent {} finished executing all its goals.".format(
                            self.ref_world.entities[agent_uid].name
                        ),
                        step_count,
                    )
                )
            elif isinstance(agent_next_action, ba.GoalFailed):
                if self.save_intermediate_world_states:
                    self.save_world_snapshot(
                        agent_uid, agent_next_action, trace_polygons, step_count
                    )
                self.simulation_log.append(
                    utils.BasicLog(
                        "{} failed executing goal {}.".format(
                            self.ref_world.entities[agent_uid].name,
                            str(agent_next_action.goal),
                        ),
                        step_count,
                    )
                )
            elif isinstance(agent_next_action, ba.GoalSuccess):
                # If the agent reached its current goal
                if self.save_intermediate_world_states:
                    self.save_world_snapshot(
                        agent_uid, agent_next_action, trace_polygons, step_count
                    )
                self.simulation_log.append(
                    utils.BasicLog(
                        "Agent {} successfully executed goal {}.".format(
                            self.ref_world.entities[agent_uid].name,
                            str(agent_next_action.goal),
                        ),
                        step_count,
                    )
                )

            if agent_next_action:
                agent_uid_to_next_action[agent_uid] = agent_next_action

        return agent_uid_to_next_action

    def think(
        self,
        active_agents: set[int],
        step_count: int,
        think_durations: t.Dict[int, float],
        trace_polygons: t.List[Polygon],
    ):
        results: t.List[t.Tuple[int, float, ba.BasicAction | None]] = []
        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            if agent_uid in active_agents:
                think_start = time.time()
                agent_next_action = behavior.think()
                think_duration = time.time() - think_start
                results.append((agent_uid, think_duration, agent_next_action))

        return self.process_think_results(
            results=results,
            think_durations=think_durations,
            step_count=step_count,
            active_agents=active_agents,
            trace_polygons=trace_polygons,
        )

    def act(
        self,
        agent_uid_to_next_action: t.Dict[int, ba.BasicAction],
        step_count: int,
        ignore_collisions: bool = True,
    ) -> t.Dict[int, ar.ActionResult]:
        """
        Processes agent actions and produce the actions results
        """
        # Only Grab and Release actions require further checks, and Wait actions are necessarily valid
        to_check = {
            uid: a
            for uid, a in agent_uid_to_next_action.items()
            if isinstance(a, (ba.Translation, ba.Rotation))
            and not isinstance(a, (ba.Grab, ba.Release))
        }
        action_results: t.Dict[int, ar.ActionResult] = {
            uid: ar.ActionSuccess(a, self.ref_world.entities[uid].pose)
            for uid, a in agent_uid_to_next_action.items()
            if isinstance(a, (ba.Wait, ba.GoalSuccess, ba.GoalFailed, ba.GoalsFinished))
        }

        # Check if released entity is already grabbed by the right agent
        for agent_uid, action in agent_uid_to_next_action.items():
            if isinstance(action, ba.Release):
                entity_uid = action.entity_uid
                if (
                    agent_uid not in self.ref_world.entity_to_agent.inverse
                    or entity_uid not in self.ref_world.entity_to_agent
                ):
                    action_results[agent_uid] = ar.NotGrabbedFailure(action)
                else:
                    other_agent_uid = self.ref_world.entity_to_agent[entity_uid]
                    if other_agent_uid != agent_uid:
                        action_results[agent_uid] = ar.GrabbedByOtherFailure(
                            action, other_agent_uid
                        )
                    else:
                        to_check[agent_uid] = action

        # Check if grabbed entity not already grabbed by another, and if about to be released by another
        entity_to_grab_agents = {}
        for agent_uid, action in agent_uid_to_next_action.items():
            if isinstance(action, ba.Grab):
                entity_uid = action.entity_uid
                if entity_uid in entity_to_grab_agents:
                    entity_to_grab_agents[entity_uid].add(agent_uid)
                else:
                    entity_to_grab_agents[entity_uid] = {agent_uid}
        for agent_uid, action in agent_uid_to_next_action.items():
            if isinstance(action, ba.Grab):
                entity_uid = action.entity_uid
                if len(entity_to_grab_agents[entity_uid]) > 1:
                    action_results[agent_uid] = ar.SimultaneousGrabFailure(
                        action, entity_to_grab_agents[entity_uid]
                    )
                    continue
                if agent_uid in self.ref_world.entity_to_agent.inverse:
                    action_results[agent_uid] = ar.GrabMoreThanOneFailure(action)
                    continue
                if entity_uid in self.ref_world.entity_to_agent:
                    other_agent_uid = self.ref_world.entity_to_agent[entity_uid]
                    other_releases = other_agent_uid in to_check and isinstance(
                        to_check[other_agent_uid], ba.Release
                    )
                    if not other_releases:
                        action_results[agent_uid] = ar.AlreadyGrabbedFailure(
                            action, other_agent_uid
                        )
                        continue
                to_check[agent_uid] = action

        # Check actions regarding dynamic collisions and apply the valid ones
        collides_with = collision.csv_simulate_simple_kinematics(
            self.ref_world,
            to_check,
            apply=True,
            ignore_collisions=ignore_collisions,
            extra_transit_check=False,
        )

        # Finish separating succeeded and failed actions, and apply result to world state on success
        for agent_uid, action in to_check.items():
            action_dynamically_collides = (
                (  # The agent associated with the action collides
                    (agent_uid in collides_with and not isinstance(action, ba.Grab))
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
                    and self.ref_world.entity_to_agent.inverse[agent_uid]
                    in collides_with
                    and not isinstance(action, ba.Release)
                )
            )
            if action_dynamically_collides and not ignore_collisions:
                action_results[agent_uid] = ar.DynamicCollisionFailure(
                    action, collides_with
                )
            else:
                if action_dynamically_collides and ignore_collisions:
                    self.simulation_log.append(
                        utils.BasicLog(
                            "Dynamic collision ignored, entities: {}".format(
                                {
                                    self.ref_world.entities[uid].name: {
                                        self.ref_world.entities[uid2].name
                                        for uid2 in uids
                                    }
                                    for uid, uids in collides_with.items()
                                }
                            ),
                            step_count,
                        )
                    )

                # SUCCESS
                # If Grab or Release, first update self.ref_world.entity_to_agent
                if isinstance(action, ba.Grab):
                    self.ref_world.entity_to_agent[action.entity_uid] = agent_uid
                if isinstance(action, ba.Release):
                    del self.ref_world.entity_to_agent[action.entity_uid]

                action_results[agent_uid] = ar.ActionSuccess(
                    action, self.ref_world.entities[agent_uid].pose
                )

        return action_results
