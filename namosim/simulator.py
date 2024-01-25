import atexit
import copy
import io
import json
import os
import random
import sys
import time
import tkinter as tk
import traceback
import typing as t

import cairosvg
import jsonpickle
from PIL import Image, ImageTk
from shapely.geometry import Polygon

import namosim.config as config
import namosim.display.ros2_publisher as ros2
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
from namosim.agents.agent import Agent, ThinkResult
from namosim.data_models import UID, PoseModel
from namosim.exceptions import CustomTimeoutError, timeout
from namosim.input import Input
from namosim.report import AgentStats, SimulationReport, WorldStepReport
from namosim.utils import collision, conversion, stats_utils, utils
from namosim.world.obstacle import Obstacle
from namosim.world.world import World

sys.setrecursionlimit(10000)
os.system("xset r off")


class SimulationStepResult:
    def __init__(
        self,
        sense_durations: t.Dict[UID, float],
        think_durations: t.Dict[UID, float],
        act_duration: float,
        action_results: t.Dict[UID, ar.ActionResult],
        think_results: t.Dict[UID, ThinkResult],
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

    def __init__(
        self,
        *,
        simulation_file_path: str,
        goals: t.Optional[t.Dict[str, t.List[PoseModel]]] = None,
        logs_dir: str | None = None,
    ):
        self.window: tk.Tk | None = None
        self.background: tk.Label | None = None
        if config.DISPLAY_WINDOW:
            self.window = tk.Tk()
            self.window.title("NAMOSIM")
            self.window.resizable(True, True)
            self.background = tk.Label(self.window)
            self.background.pack()
        self.teleop_input = Input()
        simulation_file_abs_path = os.path.abspath(simulation_file_path)

        self.simulation_filename = os.path.splitext(
            os.path.basename(simulation_file_abs_path)
        )[0]

        # init logs
        if logs_dir:
            self.logs_dir = logs_dir
        else:
            self.logs_dir = os.path.join(
                os.path.dirname(__file__),
                "../namo_logs/",
                self.simulation_filename,
            )
        if not os.path.isdir(self.logs_dir):
            os.makedirs(self.logs_dir)
        self.simulation_log = utils.CustomLogger()

        # Load world file

        self.ref_world = World.load_from_svg(
            simulation_file_abs_path, logs_dir=self.logs_dir, logger=self.simulation_log
        )
        self.init_ref_world = self.ref_world.light_copy([])

        self.config = self.init_ref_world.config

        self.simulation_log.append(
            utils.BasicLog("Simulation file successfully loaded", 0)
        )

        # Save general simulation parameters
        self.random_seed = self.config.random_seed or 10
        random.seed(self.random_seed)
        self.human_inflation_radius = 0.55 / 2.0  # [m]

        self.simulation_log.append(
            utils.BasicLog("Created log folders at:{}".format(str(self.logs_dir)), 0)
        )

        self.save_init_world_state = True
        self.save_intermediate_world_states = False
        self.save_end_world_state = True
        self.save_report = True
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
                    cls=utils.JsonEncoder,
                )

        self.save = json_save

        # Reinitialize rviz display
        self.ros_publisher = ros2.RosPublisher(
            node_name=self.simulation_filename,
            agent_names=[x.agent_id for x in self.config.agents],
        )
        self.ros_publisher.cleanup_all()

        self.simulation_log.append(utils.BasicLog("Display backend initialized.", 0))

        self.simulation_log.append(utils.BasicLog("World file successfully loaded.", 0))

        if self.save_init_world_state:
            self.init_ref_world.save_to_files(
                svg_filepath=os.path.join(
                    self.logs_dir, self.init_ref_world.init_geometry_filename
                )
            )

        # Associate autonomous agents with goals and behaviors
        self.goal_poses = {
            goal.name: goal.pose for goal in self.init_ref_world.goals.values()
        }

        self.agent_uid_to_goals: t.Dict[UID, t.List[PoseModel]]
        """
        Maps an agent uid to a list of goal poses
        """

        self.saved_goals: t.Dict[str, t.List[PoseModel]]
        """
        Maps an agent name to a list of goal poses
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

        self.history: t.List[SimulationStepResult] = []
        """
        A list of simulation step results
        """

        # Time stats
        self.agent_uid_and_goal_to_world_snapshot = {
            agent_uid: [] for agent_uid in self.ref_world.agents.keys()
        }

        self.catch_exceptions = False

        self.simulation_log.append(utils.BasicLog("Simulation successfully loaded.", 0))
        self.run_exceptions_traces: t.List[t.Any] = []
        self.exception: t.Union[Exception, None] = None

        self.report = SimulationReport()
        for agent in self.ref_world.agents.values():
            self.report.agent_stats[agent.name] = AgentStats(
                agent_id=agent.name, n_goals=agent.num_navigation_goals
            )

        # keyboard actions
        self._paused = False
        self._step = False

    def step(
        self, active_agents: set[UID], trace_polygons: t.List[Polygon], step_count: int
    ) -> t.Tuple[set[UID], t.List[Polygon], int]:
        if self._paused:
            return (active_agents, trace_polygons, step_count)

        if self._step:
            self._paused = True
            self._step = False

        if len(active_agents) == 0:
            self.end_simulation(step_count=step_count)
            return (active_agents, trace_polygons, step_count + 1)

        try:
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
            actions, think_results, think_durations = self.think(
                active_agents=active_agents,
                trace_polygons=trace_polygons,
                step_count=step_count,
            )

            # Act loops: Verify that each action is doable individually and together, if so, execute them
            act_start = time.time()
            action_results = self.act(actions, step_count)
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
            self.ros_publisher.publish_sim_world(self.ref_world)
        except Exception as e:
            self.end_simulation(step_count=step_count, err=e)

        return (active_agents, trace_polygons, step_count + 1)

    def update_report(self, sim_step_result: SimulationStepResult):
        for uid, action_result in sim_step_result.action_results.items():
            agent_id = self.ref_world.entities[uid].name
            think_result = sim_step_result.think_results[uid]
            self.report.update_for_step(
                agent_id=agent_id,
                action_result=action_result,
                think_result=think_result,
            )
            self.report.agent_stats[
                agent_id
            ].planning_time += sim_step_result.think_durations[uid]
            if sim_step_result.think_results[uid].did_replan:
                self.report.agent_stats[agent_id].replans += 1
            if sim_step_result.think_results[uid].did_postpone:
                self.report.agent_stats[agent_id].postponements += 1

        world_stats = self.get_next_world_step_report(
            self.ref_world,
            sim_step_result,
            prev=self.report.world_steps[-1]
            if len(self.report.world_steps) > 0
            else None,
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
            active_agents: set[UID] = set(self.ref_world.agents.keys())
            self.ros_publisher.publish_sim_world(self.ref_world)
            trace_polygons: t.List[Polygon] = []
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
            self.simulation_log.append(
                utils.BasicLog(
                    "Saved exceptions at: {}".format(exceptions_filepath), step_count
                )
            )

        # - Save world end state as SVG+JSON
        if self.save_end_world_state:
            self.ref_world.save_to_files(
                svg_filepath=os.path.join(
                    self.logs_dir,
                    utils.append_suffix(
                        self.init_ref_world.init_geometry_filename, "_end"
                    ),
                )
            )
            self.simulation_log.append(
                utils.BasicLog("Saved simulation final state.", step_count)
            )

        # - Save report
        if self.save_report:
            report_path = os.path.join(self.logs_dir, "report.json")
            self.report.save(report_path)
            self.simulation_log.append(
                utils.BasicLog(
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
            for uid, behavior in self.ref_world.agents.items():
                logs["agents_logs"][self.ref_world.entities[uid].name] = behavior.logger
            logs_filepath = os.path.join(self.logs_dir, "logs")
            self.save(logs, logs_filepath)

        if self.exception is not None:
            for exception_trace in self.run_exceptions_traces:
                print(exception_trace)
            raise self.exception

    def _run_window_loop(
        self, active_agents: set[UID], trace_polygons: t.List[Polygon]
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
        self, active_agents: set[UID], trace_polygons: t.List[Polygon], step_count: int
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
        for entity_uid, entity in self.ref_world.entities.items():
            if isinstance(entity, Agent) or (
                isinstance(entity, Obstacle) and entity.type_ == "wall"
            ):
                entities[entity_uid] = copy.deepcopy(entity)

        return World(
            config=self.config,
            entities=entities,
            discretization_data=copy.deepcopy(self.ref_world.discretization_data),
            logger=self.simulation_log,
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
        successful_actions: t.Dict[UID, ba.Action] = {
            uid: action_result.action
            for uid, action_result in sim_step_result.action_results.items()
            if (
                isinstance(action_result, ar.ActionSuccess)
                and isinstance(
                    action_result.action,
                    (
                        ba.Rotation,
                        ba.Advance,
                        ba.AbsoluteTranslation,
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
        all_movables_uids = [x.uid for x in world.get_movable_obstacles()]
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
                    for uid, entity in world.entities.items()
                    if isinstance(entity, Agent) or uid in world.entity_to_agent.keys()
                ]
            ),
            ros_publisher=self.ros_publisher,
        )
        end_abs_social_cost = stats_utils.get_social_costs_stats(
            world,
            set(all_movables_uids),
            ros_publisher=self.ros_publisher,
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
        goals_geometries: t.Dict[str, PoseModel],
        max_nb_goals: float = float("inf"),
    ) -> t.Dict[UID, t.List[PoseModel]]:
        """
        Contructs and returns a dictionary that maps an agent uid to a list of nativation goal poses. Each
        agent may multiple navigation goals.
        """
        agent_uid_to_goals = {}
        for agent_behavior in self.config.agents:
            agent_uid = self.ref_world.get_entity_uid_from_name(agent_behavior.agent_id)
            if agent_uid in agent_uid_to_goals:
                raise RuntimeError(
                    "You can only associate a single behavior with entity: {entity_name}.".format(
                        entity_name=agent_behavior.agent_id
                    )
                )
            else:
                agent_navigation_goals: t.List[PoseModel] = []

                for count, config_goal in enumerate(agent_behavior.goals):
                    if count > max_nb_goals:
                        break
                    if config_goal.goal_id in goals_geometries:
                        agent_navigation_goals.append(
                            goals_geometries[config_goal.goal_id]
                        )

                agent_uid_to_goals[agent_uid] = agent_navigation_goals

        return agent_uid_to_goals

    def save_world_snapshot(
        self,
        agent_uid: UID,
        action: ba.RelativeAction,
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
                shapely_geometry=polygon,
                uname="goal_generated_" + str(goal_counter),
                style=conversion.OBSTACE_TRACE_STYLE,
                svg_data=svg_data,
                svg_group=new_group,
                map_width=self.ref_world.discretization_data.width,
                map_height=self.ref_world.discretization_data.height,
            )
        del trace_polygons[: len(trace_polygons)]

        world_snapshot.save_to_files(
            svg_filepath=os.path.join(self.logs_dir, svg_filepath)
        )

    def sense(
        self,
        active_agents: set[UID],
        step_count: int,
        sense_durations: t.Dict[UID, float],
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
                self.ros_publisher.publish_robot_world(behavior.world, behavior.uid)

                # Record the time it took the robot to sense the world
                sense_durations[agent_uid] = time.time() - sense_start
            else:
                self.ros_publisher.cleanup_robot_world(ns=behavior.name)

    def process_think_results(
        self,
        results: t.Dict[UID, ThinkResult],
        active_agents: t.Set[UID],
        trace_polygons: t.List[Polygon],
        step_count: int,
    ) -> t.Dict[UID, ba.Action]:
        """Process the results of each agent's think step. Updates the set of activate agents and the dictionary of think durations."""
        agent_uid_to_next_action: t.Dict[UID, ba.Action] = {}
        for agent_uid, think_result in results.items():
            if len(think_result.conflicts) == 0:
                self.ros_publisher.cleanup_conflicts_checks(ns=think_result.robot_name)

            # TODO Change goal coordinates for easier reading to goal name in log.
            if isinstance(think_result.next_action, ba.GoalsFinished):
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
            elif isinstance(think_result.next_action, ba.GoalFailed):
                if self.save_intermediate_world_states:
                    self.save_world_snapshot(
                        agent_uid, think_result.next_action, trace_polygons, step_count
                    )
                self.simulation_log.append(
                    utils.BasicLog(
                        "{} failed executing goal {}.".format(
                            self.ref_world.entities[agent_uid].name,
                            str(think_result.next_action.goal),
                        ),
                        step_count,
                    )
                )
            elif isinstance(think_result.next_action, ba.GoalSuccess):
                # If the agent reached its current goal
                if self.save_intermediate_world_states:
                    self.save_world_snapshot(
                        agent_uid, think_result.next_action, trace_polygons, step_count
                    )
                self.simulation_log.append(
                    utils.BasicLog(
                        "Agent {} successfully executed goal {}.".format(
                            self.ref_world.entities[agent_uid].name,
                            str(think_result.next_action.goal),
                        ),
                        step_count,
                    )
                )

            if think_result.next_action:
                agent_uid_to_next_action[agent_uid] = think_result.next_action

        return agent_uid_to_next_action

    def think(
        self,
        active_agents: t.Set[UID],
        step_count: int,
        trace_polygons: t.List[Polygon],
    ):
        think_results: t.Dict[UID, ThinkResult] = {}
        think_durations: t.Dict[UID, float] = {}

        for agent_uid, agent in self.ref_world.agents.items():
            if agent_uid in active_agents:
                self.publish_robot_goal(agent_uid=agent_uid)
                agent_goal = agent.get_current_or_next_goal()

                think_start = time.time()
                try:
                    with timeout(60):
                        think_result = agent.think(
                            ros_publisher=self.ros_publisher, input=self.teleop_input
                        )
                except CustomTimeoutError as e:
                    assert isinstance(e, CustomTimeoutError)
                    if not agent_goal:
                        raise Exception("Agent think timed out without a goal")

                    self.simulation_log.append(
                        utils.BasicLog(
                            f"Robot ${agent.name} timed out while planning. Failing goal and reinitializing.",
                            step_count,
                        )
                    )

                    think_result = ThinkResult(
                        next_action=ba.GoalFailed(goal=agent_goal, is_timeout=True),
                        did_replan=False,
                        did_postpone=False,
                        robot_name=agent.name,
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

    def act(
        self,
        agent_uid_to_next_action: t.Dict[UID, ba.Action],
        step_count: int,
        ignore_collisions: bool = True,
    ) -> t.Dict[UID, ar.ActionResult]:
        """
        Processes agent actions and produce the actions results
        """
        # Only Grab and Release actions require further checks, and Wait actions are necessarily valid
        to_check: t.Dict[UID, ba.Action] = {
            uid: a
            for uid, a in agent_uid_to_next_action.items()
            if isinstance(a, (ba.Advance, ba.AbsoluteTranslation, ba.Rotation))
            and not isinstance(a, (ba.Grab, ba.Release))
        }
        action_results: t.Dict[UID, ar.ActionResult] = {
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
            world=self.ref_world,
            agent_actions=to_check,
            apply=True,
            ignore_collisions=ignore_collisions,
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
                    action=action,
                    robot_pose=self.ref_world.entities[agent_uid].pose,
                    is_transfer=agent_uid in self.ref_world.entity_to_agent.inverse,
                    obstacle_uid=self.ref_world.entity_to_agent.inverse.get(
                        agent_uid, None
                    ),
                )

        return action_results

    def publish_robot_goal(self, agent_uid: UID):
        behavior = self.ref_world.agents[agent_uid]
        goal = behavior.get_current_or_next_goal()
        if behavior and goal:
            self.ros_publisher.publish_goal(
                q_init=behavior.pose,
                q_goal=goal,
                entity=behavior,
                ns=behavior.name,
            )

    def publish_robot_plan(self, agent_uid: UID, did_replan: bool):
        behavior = self.ref_world.agents[agent_uid]
        if behavior and behavior.goal_pose:
            if did_replan:
                self.ros_publisher.cleanup_p_opt(ns=behavior.name)
            plan = behavior.get_plan()
            if plan:
                self.ros_publisher.publish_p_opt(
                    plan=plan,
                    robot=behavior,
                    ns=behavior.name,
                )


def before_exit():
    os.system("xset r on")


# Register the function to be called before exit
atexit.register(before_exit)
