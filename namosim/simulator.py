import atexit
import copy
import json
import os
import random
import sys
import time
import tkinter as tk
import traceback
import typing as t

import jsonpickle
from PIL import ImageTk
from shapely.geometry import Polygon

import namosim.config as config
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
from namosim.agents.agent import Agent, ThinkResult
from namosim.data_models import Pose2D
from namosim.exceptions import CustomTimeoutError, timeout
from namosim.input import Input
from namosim.report import AgentStats, SimulationReport, WorldStepReport
from namosim.utils import stats_utils, utils
from namosim.world.obstacle import Obstacle
from namosim.world.world import World
import namosim.config as cfg

sys.setrecursionlimit(10000)
os.system("xset r off")


class SimulationStepResult:
    def __init__(
        self,
        sense_durations: t.Dict[str, float],
        think_durations: t.Dict[str, float],
        act_duration: float,
        action_results: t.Dict[str, ar.ActionResult],
        think_results: t.Dict[str, ThinkResult],
        step_index: int,
    ):
        self.sense_durations = sense_durations
        self.think_durations = think_durations
        self.act_duration = act_duration
        self.action_results = action_results
        self.think_results = think_results
        self.step_index = step_index


class Simulator:
    """The main simulator class manages all aspects of the simulation. It initializes
    the world and agents and executes a **sense** -> **think** -> **act** loop until all agents have
    either completed or failed their navigation goals."""

    def __init__(self, *, world: World, logger: utils.NamosimLogger, logs_dir: str):
        self.ref_world = world
        self.init_ref_world = self.ref_world.light_copy([])
        self.logger = logger
        self.logs_dir = logs_dir
        self.window: tk.Tk | None = None
        self.background: tk.Label | None = None
        if config.DISPLAY_WINDOW:
            self.window = tk.Tk()
            self.window.title("NAMO Planner")
            self.window.resizable(True, True)
            self.background = tk.Label(self.window)
            self.background.pack()
        self.teleop_input = Input()
        self.logger.append(utils.NamosimLog("Simulation file successfully loaded", 0))

        # Save general simulation parameters
        self.random_seed = self.init_ref_world.random_seed or 10
        random.seed(self.random_seed)
        self.human_inflation_radius = 0.55 / 2.0  # [m]

        self.save_report = self.init_ref_world.generate_report
        self.save_history = False
        self.save_logs = True

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
        self.ros_publisher = None
        if not cfg.DEACTIVATE_RVIZ:
            # conditional import to support usage without Ros2 installed
            import namosim.display.ros2_publisher as ros2_publisher

            self.ros_publisher = ros2_publisher.create_default_ros_publisher(
                node_name="namosim",
                agent_ids=list(self.ref_world.agents.keys()),
            )
            self.ros_publisher.cleanup_all()

        self.logger.append(utils.NamosimLog("Display backend initialized.", 0))

        self.logger.append(utils.NamosimLog("World file successfully loaded.", 0))

        # Associate autonomous agents with goals and behaviors
        self.goal_poses = {
            goal.uid: goal.pose for goal in self.init_ref_world.get_goals().values()
        }

        self.saved_goals: t.Dict[str, t.List[Pose2D]]
        """
        Maps an agent uid to a list of goal poses
        """

        self.history: t.List[SimulationStepResult] = []
        """
        A list of simulation step results
        """

        # Time stats
        self.agent_uid_and_goal_to_world_snapshot = {
            agent_uid: [] for agent_uid in self.ref_world.agents.keys()
        }

        self.catch_exceptions = False

        self.logger.append(utils.NamosimLog("Simulation successfully loaded.", 0))
        self.run_exceptions_traces: t.List[t.Any] = []
        self.exception: t.Union[Exception, None] = None

        self.report = SimulationReport()
        for agent in self.ref_world.agents.values():
            self.report.agent_stats[agent.uid] = AgentStats(
                agent_id=agent.uid,
            )

        # keyboard actions
        self._paused = False
        self._step = False

    def step(
        self, active_agents: set[str], trace_polygons: t.List[Polygon], step_count: int
    ) -> t.Tuple[set[str], t.List[Polygon], int]:
        if self._paused:
            return (active_agents, trace_polygons, step_count)

        if self._step:
            self._paused = True
            self._step = False

        if len(active_agents) == 0:
            self.end_simulation(step_count=step_count)
            return (active_agents, trace_polygons, step_count + 1)

        try:
            # Sense loop: update each agent's knowledge of the world
            sense_durations = {}
            self.sense(active_agents, step_count, sense_durations)

            # for agent in self.ref_world.agents.values():
            #     self.logger.append(
            #         utils.NamosimLog(
            #             f"Robot {agent.uid} is at pose {agent.pose}",
            #             step=step_count,
            #         )
            #     )

            # Think loop: get each agent to think about their next step
            actions, think_results, think_durations = self.think(
                active_agents=active_agents,
                trace_polygons=trace_polygons,
                step_count=step_count,
            )

            # Act loops: Verify that each action is doable individually and together, if so, execute them
            act_start = time.time()
            action_results = self.ref_world.step(actions, step_count)
            act_duration = time.time() - act_start

            sim_step_result = SimulationStepResult(
                sense_durations,
                think_durations,
                act_duration,
                action_results,
                think_results,
                step_count,
            )
            self.history.append(sim_step_result)

            if self.save_report:
                self.update_report(sim_step_result)

            # Once the simulation reference world has been modified, display the modification
            if self.ros_publisher:
                self.ros_publisher.publish_world(self.ref_world)
        except Exception as e:
            self.end_simulation(step_count=step_count, err=e)

        return (active_agents, trace_polygons, step_count + 1)

    def update_report(self, sim_step_result: SimulationStepResult):
        for uid, action_result in sim_step_result.action_results.items():
            agent_id = self.ref_world.dynamic_entities[uid].uid
            think_result = sim_step_result.think_results[uid]
            self.report.update_for_step(
                agent_id=agent_id,
                action_result=action_result,
                think_result=think_result,
                planning_time=sim_step_result.think_durations[uid],
            )

        world_stats = self.get_next_world_step_report(
            self.ref_world,
            sim_step_result,
            prev=(
                self.report.world_steps[-1]
                if len(self.report.world_steps) > 0
                else None
            ),
        )
        self.report.world_steps.append(world_stats)

    def end_simulation(self, step_count: int, err: Exception | None = None):
        self.run_active = False
        self._paused = False
        self._step = False

        if self.window:
            self.window.quit()
        if self.background:
            self.background.quit()

        if err is not None:
            if self.catch_exceptions:
                tb = traceback.format_exc()
                self.run_exceptions_traces.append(tb)
                self.logger.append(utils.NamosimLog(tb, step_count))
            else:
                self.logger.append(
                    utils.NamosimLog("MET A RUNTIME EXCEPTION, EXITING !", step_count)
                )
                print(err)
                tb = traceback.format_exc()
                self.run_exceptions_traces.append(tb)
                self.exception = err
                return

    def render_window(self):
        if not self.window:
            raise Exception("No window")
        if not self.background:
            raise Exception("No background")

        image = self.ref_world.to_image(grayscale=False, width=600)
        tk_image = ImageTk.PhotoImage(image)
        self.window.geometry(f"{tk_image.width()}x{tk_image.height()}")

        self.background.configure(image=tk_image)  # type: ignore

        # store tk_image on background.image to prevent garbage collection
        self.background.image = tk_image  # type: ignore

    def run(self) -> t.List[SimulationStepResult]:
        self.run_active = True
        self.run_exceptions_traces = []
        self.exception = None
        step_count = 0

        while self.run_active:
            active_agents: set[str] = set(self.ref_world.agents.keys())
            if self.ros_publisher:
                self.ros_publisher.publish_world(self.ref_world)
            trace_polygons: t.List[Polygon] = []
            self.logger.append(utils.NamosimLog("Starting run.", step_count))
            print("")

            if self.window is not None:
                self._run_window_loop(
                    active_agents=active_agents,
                    trace_polygons=trace_polygons,
                )
            else:
                step_count = 0
                while len(active_agents) > 0 and self.run_active:
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
            exceptions_filepath = os.path.join(self.logs_dir, "exceptions")
            self.save(exceptions, exceptions_filepath)
            self.logger.append(
                utils.NamosimLog(
                    "Saved exceptions at: {}".format(exceptions_filepath), step_count
                )
            )

        # - Save report
        if self.save_report:
            report_path = os.path.join(self.logs_dir, "report.json")
            self.report.save(report_path)
            self.logger.append(
                utils.NamosimLog(
                    "Saved simulation report at: {}".format(report_path), step_count
                )
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
                for agent_uid, behavior in self.ref_world.agents.items()
            }
            history_filepath = os.path.join(self.logs_dir, "history")
            self.save(history, history_filepath)
            self.logger.append(
                utils.NamosimLog(
                    "Saved history at: {}".format(history_filepath), step_count
                )
            )

        # - Save simulation and agents logs
        if self.save_logs:
            logs = {}
            logs["simulation_log"] = self.logger
            logs["agents_logs"] = {}
            for uid, behavior in self.ref_world.agents.items():
                logs["agents_logs"][
                    self.ref_world.dynamic_entities[uid].uid
                ] = behavior.logger
            logs_filepath = os.path.join(self.logs_dir, "logs")
            self.save(logs, logs_filepath)

        if self.exception is not None:
            for exception_trace in self.run_exceptions_traces:
                print(exception_trace)
            raise self.exception

    def _run_window_loop(
        self, active_agents: set[str], trace_polygons: t.List[Polygon]
    ):
        if self.window is None:
            raise Exception("No window")
        self.window.bind("<KeyPress>", self._on_key_press)
        self.window.bind("<KeyRelease>", self._on_key_release)
        self._window_step(
            active_agents=active_agents,
            trace_polygons=trace_polygons,
            step_count=0,
        )
        self.window.mainloop()

    def _on_key_press(self, event: t.Any):
        # Get the key symbol from the event object
        if event.keysym == "p":
            self._paused = not self._paused
        elif event.keysym == "space":
            self._paused = False
            self._step = True

        if event.keysym:
            self.teleop_input.handle_key_press(event.keysym)

    def _on_key_release(self, event: t.Any):
        if event.keysym == self.teleop_input.key_pressed:
            self.teleop_input.handle_key_release(event.keysym)

    def _window_step(
        self, active_agents: set[str], trace_polygons: t.List[Polygon], step_count: int
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
            15, self._window_step, active_agents, trace_polygons, step_count
        )

    def _create_robot_world_from_sim_world(self):
        entities = dict()
        for entity_uid, entity in self.ref_world.dynamic_entities.items():
            if isinstance(entity, Agent) or (
                isinstance(entity, Obstacle) and entity.type_ == "wall"
            ):
                entities[entity_uid] = copy.deepcopy(entity)

        return World(
            dynamic_entities=copy.deepcopy(self.ref_world.dynamic_entities),
            agents=copy.deepcopy(self.ref_world.agents),
            map=copy.deepcopy(self.ref_world.map),
            collision_margin=self.ref_world.collision_margin,
            logger=self.logger,
            random_seed=self.ref_world.random_seed,
            generate_report=self.ref_world.generate_report,
            svg_config=self.ref_world.svg_config,
        )

    def create_simulation_report(self):
        report = {"report": self.report.to_json_data()}
        return report

    def get_next_world_step_report(
        self,
        world: World,
        sim_step_result: SimulationStepResult,
        prev: WorldStepReport | None,
    ) -> WorldStepReport:
        successful_actions: t.Dict[str, ba.Action] = {
            uid: action_result.action
            for uid, action_result in sim_step_result.action_results.items()
            if (
                isinstance(action_result, ar.ActionSuccess)
                and isinstance(
                    action_result.action,
                    (
                        ba.Rotation,
                        ba.Advance,
                        ba.Translation,
                        ba.Grab,
                        ba.Release,
                    ),
                )
            )
        }

        if (
            any(
                [
                    isinstance(action, ba.Release)
                    for action in successful_actions.values()
                ]
            )
            or not prev
        ):
            return self.compute_world_step_report(world)

        return prev

    def compute_world_step_report(
        self,
        world: World,
    ) -> WorldStepReport:
        all_movables_uids = list(world.get_movable_obstacles().keys())
        (
            nb_components,
            biggest_component_size,
            free_space_size,
            fragmentation,
        ) = stats_utils.get_connectivity_stats(
            world,
            self.human_inflation_radius,
            set(
                [
                    uid
                    for uid, entity in world.dynamic_entities.items()
                    if isinstance(entity, Agent) or uid in world.entity_to_agent.keys()
                ]
            ),
        )
        end_abs_social_cost = stats_utils.get_social_costs_stats(
            world,
            set(all_movables_uids),
        )
        world_stats = WorldStepReport(
            nb_components=nb_components,
            biggest_component_size=biggest_component_size,
            free_space_size=free_space_size,
            fragmentation=fragmentation,
            absolute_social_cost=end_abs_social_cost,
        )
        return world_stats

    def initialize_agents_goals(
        self,
        goals_geometries: t.Dict[str, Pose2D],
        max_nb_goals: float = float("inf"),
    ) -> t.Dict[str, t.List[Pose2D]]:
        """
        Contructs and returns a dictionary that maps an agent uid to a list of nativation goal poses. Each
        agent may multiple navigation goals.
        """
        goals = {}
        for agent in self.ref_world.agents.values():
            if agent.uid in goals:
                raise RuntimeError(
                    "You can only associate a single behavior with entity: {entity_name}.".format(
                        entity_name=agent.uid
                    )
                )
            else:
                agent_navigation_goals: t.List[Pose2D] = []

                for count, config_goal in enumerate(agent._navigation_goals):
                    if count > max_nb_goals:
                        break
                    if config_goal.uid in goals_geometries:
                        agent_navigation_goals.append(goals_geometries[config_goal.uid])

                goals[agent.uid] = agent_navigation_goals

        return goals

    def sense(
        self,
        active_agents: set[str],
        step_count: int,
        sense_durations: t.Dict[str, float],
    ):
        for agent_uid, behavior in self.ref_world.agents.items():
            if agent_uid in active_agents:
                sense_start = time.time()
                last_action_result = (
                    self.history[-1].action_results[agent_uid]
                    if (self.history and agent_uid in self.history[-1].action_results)
                    else ar.ActionSuccess()
                )

                # The robot's behavior senses the reference world
                behavior.sense(self.ref_world, last_action_result, step_count)

                # Publish the robot's perceived/sensed world to RVIZ
                if self.ros_publisher:
                    self.ros_publisher.publish_robot_observed_world(
                        behavior.world, behavior.uid
                    )

                # Record the time it took the robot to sense the world
                sense_durations[agent_uid] = time.time() - sense_start
            else:
                if self.ros_publisher:
                    self.ros_publisher.cleanup_robot_observed_world(
                        agent_id=behavior.uid
                    )

    def process_think_results(
        self,
        results: t.Dict[str, ThinkResult],
        active_agents: t.Set[str],
        trace_polygons: t.List[Polygon],
        step_count: int,
    ) -> t.Dict[str, ba.Action]:
        """Process the results of each agent's think step. Updates the set of activate agents and the dictionary of think durations."""
        agent_uid_to_next_action: t.Dict[str, ba.Action] = {}
        for agent_uid, think_result in results.items():
            if len(think_result.conflicts) == 0:
                if self.ros_publisher:
                    self.ros_publisher.cleanup_conflicts_checks(
                        agent_id=think_result.agent_id
                    )

            # TODO Change goal coordinates for easier reading to goal name in log.
            if isinstance(think_result.next_action, ba.GoalsFinished):
                # If the agent has executed all of its goals, remove it from the active agents
                active_agents.remove(agent_uid)
                self.logger.append(
                    utils.NamosimLog(
                        "Agent {} finished executing all its goals.".format(
                            self.ref_world.dynamic_entities[agent_uid].uid
                        ),
                        step_count,
                    )
                )
            elif isinstance(think_result.next_action, ba.GoalFailed):
                self.logger.append(
                    utils.NamosimLog(
                        "{} failed executing goal {}.".format(
                            self.ref_world.dynamic_entities[agent_uid].uid,
                            str(think_result.next_action.goal),
                        ),
                        step_count,
                    )
                )
            elif isinstance(think_result.next_action, ba.GoalSuccess):
                self.logger.append(
                    utils.NamosimLog(
                        "Agent {} successfully executed goal {}.".format(
                            self.ref_world.dynamic_entities[agent_uid].uid,
                            str(think_result.next_action.goal),
                        ),
                        step_count,
                    )
                )

            if think_result.next_action:
                agent_uid_to_next_action[agent_uid] = think_result.next_action
            elif think_result.plan:
                agent_uid_to_next_action[agent_uid] = (
                    think_result.plan.pop_next_action()
                )

        return agent_uid_to_next_action

    def think(
        self,
        active_agents: t.Set[str],
        step_count: int,
        trace_polygons: t.List[Polygon],
    ):
        think_results: t.Dict[str, ThinkResult] = {}
        think_durations: t.Dict[str, float] = {}

        for agent_uid, agent in self.ref_world.agents.items():
            if agent_uid in active_agents:
                self.publish_robot_goal(agent_uid=agent_uid)
                agent_goal = agent.get_current_or_next_goal()

                think_start = time.time()
                try:
                    with timeout(600):
                        think_result = agent.think(
                            ros_publisher=self.ros_publisher, input=self.teleop_input
                        )

                except CustomTimeoutError as e:
                    assert isinstance(e, CustomTimeoutError)
                    if not agent_goal:
                        raise Exception("Agent think timed out without a goal")

                    self.logger.append(
                        utils.NamosimLog(
                            f"Robot {agent.uid} timed out while planning. Failing goal and reinitializing.",
                            step_count,
                        )
                    )

                    think_result = ThinkResult(
                        plan=None,
                        next_action=ba.GoalFailed(
                            goal=agent_goal.pose, is_timeout=True
                        ),
                        goal_pose=agent_goal.pose,
                        did_replan=False,
                        did_postpone=False,
                        agent_id=agent.uid,
                    )

                    agent.skip_current_goal()
                    # Reinitialize the agent so it is not left in a bad state after timing out
                    agent.init(self.ref_world)

                except Exception as e:
                    raise e

                think_duration = time.time() - think_start

                think_results[agent_uid] = think_result
                think_durations[agent_uid] = think_duration

                self.publish_robot_plan(
                    agent_uid=agent_uid, did_replan=think_result.did_replan
                )

        actions = self.process_think_results(
            results=think_results,
            step_count=step_count,
            active_agents=active_agents,
            trace_polygons=trace_polygons,
        )

        return actions, think_results, think_durations

    def publish_robot_goal(self, agent_uid: str):
        if not self.ros_publisher:
            return
        agent = self.ref_world.agents[agent_uid]
        goal = agent.get_current_or_next_goal()
        if agent and goal:
            self.ros_publisher.publish_goal(
                q_init=agent.pose,
                q_goal=goal.pose,
                entity=agent,
            )

    def publish_robot_plan(self, agent_uid: str, did_replan: bool):
        if not self.ros_publisher:
            return
        agent = self.ref_world.agents[agent_uid]
        if agent and agent.goal_pose:
            if did_replan:
                self.ros_publisher.clear_robot_plan(agent_id=agent_uid)
            plan = agent.get_plan()
            if plan:
                self.ros_publisher.publish_robot_plan(
                    plan=plan,
                    robot=agent,
                    map=self.ref_world.map,
                )


def before_exit():
    os.system("xset r on")


# Register the function to be called before exit
atexit.register(before_exit)


def create_sim_from_file(
    simulation_file_path: str,
    logs_dir: str | None = None,
) -> Simulator:
    simulation_file_abs_path = os.path.abspath(simulation_file_path)
    simulation_filename = os.path.splitext(os.path.basename(simulation_file_abs_path))[
        0
    ]

    if logs_dir is None:
        logs_dir = os.path.join(
            os.path.dirname(__file__),
            "../namo_logs/",
            simulation_filename,
        )
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)
    logger = utils.NamosimLogger()

    # Load world file

    if simulation_file_path.strip().lower().endswith(".svg"):
        world = World.load_from_svg(
            simulation_file_abs_path,
            logs_dir=logs_dir,
            logger=logger,
        )
    else:
        world = World.load_from_yaml(
            simulation_file_abs_path,
            logs_dir=logs_dir,
            logger=logger,
        )

    return Simulator(world=world, logger=logger, logs_dir=logs_dir)
