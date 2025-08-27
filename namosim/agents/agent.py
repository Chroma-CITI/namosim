import abc
import copy
import typing as t
from collections import OrderedDict

from shapely.geometry import Polygon

from namosim import svg_styles
import namosim.display.ros2_publisher as rp
import namosim.navigation.navigation_plan as navp
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
import namosim.world.world as w
from namosim.algorithms import graph_search
from namosim.data_models import AgentBehaviorXmlConfig, Pose2D
from namosim.input import Input
from namosim.navigation.action_result import ActionResult
from namosim.navigation.basic_actions import Action
from namosim.navigation.conflict import Conflict
from namosim.navigation.navigation_path import TransitPath
from namosim.utils import utils
from namosim.world import goal
from namosim.world.entity import Entity, Movability
from namosim.world.sensors.omniscient_sensor import OmniscientSensor
import numpy as np


class ThinkResult:
    """Represents the result of an agent's thinking process.

    This class encapsulates the outcome of an agent's decision-making cycle,
    including planning information, goal details, and any conflicts detected.
    """

    def __init__(
        self,
        *,
        plan: t.Union["navp.Plan", None],
        goal_pose: Pose2D | None,
        did_replan: bool,
        did_postpone: bool = False,
        agent_id: str,
        conflicts: t.Optional[t.Set[Conflict]] = None,
        next_action: Action | None = None,
    ) -> None:
        self.plan = plan
        self.next_action = next_action
        self.goal_pose = goal_pose
        """The agent's goal at the time this think-result was produced.
        """
        self.did_replan = did_replan
        self.did_postpone = did_postpone
        self.agent_id = agent_id
        self.conflicts: t.Set[Conflict] = conflicts if conflicts else set()


class RLThinkResult(ThinkResult):
    """Represents the result of a reinforcement learning agent's thinking process.

    Extends ThinkResult with additional information specific to RL agents,
    such as log probability and action index.
    """

    def __init__(
        self,
        *,
        next_action: Action | None,
        goal_pose: Pose2D | None,
        did_replan: bool,
        did_postpone: bool = False,
        agent_id: str,
        conflicts: t.Optional[t.Set[Conflict]] = None,
        log_prob: float = 0.0,
        action_idx: int = 0,
    ):
        super().__init__(
            plan=None,
            next_action=next_action,
            goal_pose=goal_pose,
            did_replan=did_replan,
            did_postpone=did_postpone,
            agent_id=agent_id,
            conflicts=conflicts,
        )
        self.log_prob = log_prob
        self.action_idx = action_idx
        self.next_action = next_action
        self.goal_pose = goal_pose
        """The agent's goal at the time this think-result was produced.
        """
        self.did_replan = did_replan
        self.did_postpone = did_postpone
        self.agent_id = agent_id
        self.conflicts = conflicts if conflicts else set()


class Agent(Entity):
    """Represents an autonomous agent in the simulation environment.

    An agent is capable of sensing its environment, planning actions,
    and executing movements towards goals while handling conflicts.
    """

    def __init__(
        self,
        *,
        config: t.Any,
        navigation_goals: t.List[goal.Goal],
        logs_dir: str,
        uid: str,
        polygon: Polygon,
        pose: Pose2D,
        sensors: t.List[OmniscientSensor],
        cell_size: float,
        movability: Movability = Movability.UNKNOWN,
        logger: utils.NamosimLogger,
        style: svg_styles.AgentStyle | None = None,
    ):
        """Initialize the Agent.

        Args:
            config: Configuration object for the agent.
            navigation_goals: List of goals for the agent to navigate to.
            logs_dir: Directory path for logging agent activities.
            uid: Unique identifier for the agent.
            polygon: Shapely Polygon representing the agent's physical shape.
            pose: Current pose of the agent (x, y, theta).
            sensors: List of sensors attached to the agent.
            cell_size: Size of the grid cells for path planning.
            movability: Movability status of the agent.
            logger: Logger instance for recording events.
            style: Optional style configuration for visualization.
        """
        super().__init__(
            polygon=polygon,
            pose=pose,
            movability=movability,
            uid=uid,
            type_="robot",
        )
        self.config = config
        self.sensors = sensors
        for sensor in sensors:
            sensor.parent_uid = self.uid
        self.min_inflation_radius = self.compute_inflation_radius()
        self.logger = logger
        self._world: t.Optional["w.World"] = None
        self._navigation_goals = navigation_goals
        self.agent_style = style if style else svg_styles.AgentStyle()
        self.logs_dir = logs_dir

        self.__last_action_result: ActionResult | None = None

        self._prev_goal: goal.Goal | None = (
            None  # used to check if the goal has changed
        )
        self.__goal: goal.Goal | None = None

        self._prev_plan: t.Optional["navp.Plan"] = (
            None  # used to check if a plan has changed
        )
        self.__p_opt: t.Optional["navp.Plan"] = None

        self._added_uids, self._updated_uids, self._removed_uids = set(), set(), set()

        self.goal_to_plans: OrderedDict[goal.Goal, "navp.Plan"] = OrderedDict()
        self.is_initialized = False
        self.cell_size = cell_size
        """The robot must at this distance from an obstacle to grab it.
        """

    def copy(self) -> "Agent":
        """Create a deep copy of the agent.

        Returns:
            A new Agent instance with copied attributes.
        """
        return Agent(
            config=self.config,
            navigation_goals=copy.deepcopy(self._navigation_goals),
            logs_dir=self.logs_dir,
            uid=self.uid,
            polygon=copy.deepcopy(self.polygon),
            pose=self.pose,
            sensors=copy.deepcopy(self.sensors),
            style=copy.deepcopy(self.agent_style),
            cell_size=copy.deepcopy(self.cell_size),
            movability=self.movability,
            logger=self.logger,
        )

    def set_navigation_goals(self, goals: t.List[goal.Goal]):
        """Set the list of navigation goals for the agent.

        Args:
            goals: List of Goal objects to navigate to.
        """
        self._navigation_goals = goals
        self._goal = None

    def init(self, world: "w.World") -> None:
        """Initialize the agent with the world state.

        Args:
            world: The world object containing the environment.
        """
        self._world = world.light_copy([])
        self.is_initialized = True

    def sense(
        self, ref_world: "w.World", last_action_result: ActionResult, step_count: int
    ):
        """Update the agent's perception of the world using sensors.

        Args:
            ref_world: Reference world state.
            last_action_result: Result of the last action performed.
            step_count: Current simulation step count.
        """
        self._last_action_result = last_action_result
        (
            self._added_uids,
            self._updated_uids,
            self._removed_uids,
        ) = self.update_world_from_sensors(ref_world, self.world)
        self._step_count = step_count

    @abc.abstractmethod
    def think(
        self,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        input: t.Optional[Input] = None,
    ) -> ThinkResult:
        """Perform the agent's thinking process to decide on the next action.

        Args:
            ros_publisher: Optional ROS publisher for visualization.
            input: Optional input data for decision making.

        Returns:
            ThinkResult containing the agent's decision.
        """
        raise NotImplementedError

    def skip_current_goal(self):
        """Resets the agent's current goal to None. This will cause the agent to move on to the next goal on the next think step."""
        self._goal = None
        self._p_opt = None

    def get_current_goal(self):
        """Get the agent's current navigation goal.

        Returns:
            The current Goal object or None if no goal is set.
        """
        return self._goal

    @property
    def _goal(self):
        """Get the current goal of the agent."""
        return self.__goal

    @_goal.setter
    def _goal(self, _goal: goal.Goal | None):
        """Set the current goal of the agent and update the previous goal."""
        self._prev_goal = self.__goal
        self.__goal = _goal

    @property
    def _p_opt(self):
        """Get the current optimal plan."""
        return self.__p_opt

    @_p_opt.setter
    def _p_opt(self, p_opt: t.Optional["navp.Plan"]):
        """Set the current optimal plan and update the previous plan."""
        self._prev_plan = self.__p_opt
        self.__p_opt = p_opt

    @property
    def _last_action_result(self):
        """Get the result of the last action performed."""
        return self.__last_action_result

    @_last_action_result.setter
    def _last_action_result(self, last_action_result: ActionResult):
        """Set the result of the last action performed."""
        self.__last_action_result = last_action_result

    @property
    def world(self):
        """Get the agent's current world model.

        Returns:
            The World object representing the agent's perception.

        Raises:
            Exception: If the agent has not been initialized.
        """
        if not self._world:
            raise Exception("Not initialized")

        return self._world

    @property
    def goal_pose(self):
        """Get the pose of the agent's current goal.

        Returns:
            Pose2D of the goal or None if no goal is set.
        """
        if self._goal is None:
            return None
        return self._goal.pose

    def get_current_or_next_goal(self) -> goal.Goal | None:
        """Get the current goal or the next goal in the list.

        Returns:
            The current Goal if set, otherwise the first goal in the navigation goals list.
        """
        if self._goal:
            return self._goal
        if len(self._navigation_goals) > 0:
            return self._navigation_goals[0]

    def get_plan(self):
        """Get the current optimal plan.

        Returns:
            The current Plan object or None.
        """
        return self.__p_opt

    def has_goal_changed(self):
        """Check if the agent's goal has changed since the last update.

        Returns:
            True if the goal has changed, False otherwise.
        """
        return self._prev_goal != self.__goal

    def is_goal_reached(
        self,
        goal_pose: Pose2D,
        robot_pose: Pose2D,
        pos_tol: float = 0.05,
        ang_tol: float = 0.1,
    ):
        """Check if the robot has reached the goal pose within tolerances.

        Args:
            goal_pose: Target pose (x, y, theta).
            robot_pose: Current robot pose (x, y, theta).
            pos_tol: Position tolerance for x and y coordinates.
            ang_tol: Angular tolerance for theta.

        Returns:
            True if the goal is reached, False otherwise.
        """
        return all(
            [
                utils.is_close(goal_pose[0], robot_pose[0], abs_tol=pos_tol),
                utils.is_close(goal_pose[1], robot_pose[1], abs_tol=pos_tol),
                utils.angle_is_close(goal_pose[2], robot_pose[2], abs_tol=ang_tol),
            ]
        )

    def find_path(
        self,
        robot_pose: Pose2D,
        goal_pose: Pose2D,
        robot_inflated_grid: BinaryOccupancyGrid,
        robot_polygon: Polygon,
    ) -> TransitPath | None:
        """Find a path from robot pose to goal pose using A* search.

        Args:
            robot_pose: Current pose of the robot.
            goal_pose: Target pose to reach.
            robot_inflated_grid: Inflated occupancy grid for collision avoidance.
            robot_polygon: Polygon representing the robot's shape.

        Returns:
            TransitPath object if a path is found, None otherwise.
        """
        real_path = graph_search.real_to_grid_search_a_star(
            robot_pose, goal_pose, robot_inflated_grid
        )
        if real_path:

            def g(a: Pose2D, b: Pose2D):
                translation_cost = utils.euclidean_distance(a, b)
                rotation_cost = abs(a[2] - b[2])
                return translation_cost + rotation_cost

            phys_cost = 0.0
            for a, b in zip(real_path, real_path[1:]):
                phys_cost += g(a, b)

            return TransitPath.from_poses(
                real_path, robot_polygon, robot_pose, phys_cost
            )

        return None

    def update_world_from_sensors(
        self, reference_world: "w.World", target_world: "w.World"
    ):
        """Update the target world model using sensor data from the reference world.

        Args:
            reference_world: The true world state.
            target_world: The world model to update.

        Returns:
            Tuple of sets containing added, updated, and removed entity UIDs.
        """
        added_entities: set[str] = set()
        updated_entities: set[str] = set()
        removed_entities: set[str] = set()

        for sensor in self.sensors:
            s_uids_to_add, s_uids_to_update, s_uids_to_remove = sensor.update_from_fov(
                reference_world, target_world
            )  # type: ignore

            # Might need a better update policy if sensors disagree about what happened, but irrelevant for now
            added_entities.update(s_uids_to_add)
            updated_entities.update(s_uids_to_update)
            removed_entities.update(s_uids_to_remove)

        return added_entities, updated_entities, removed_entities

    def compute_inflation_radius(self) -> float:
        """Compute the inflation radius based on the agent's polygon.

        Returns:
            The circumscribed radius of the agent's polygon.
        """
        return utils.get_circumscribed_radius(self.polygon)

    def add_nav_goal(self, g: goal.Goal):
        """Add a navigation goal to the agent's goal list.

        Args:
            g: Goal object to add.
        """
        self._navigation_goals.append(g)

    def get_orientation_polygon(self):
        """Generate a polygon representing the agent's orientation.

        Returns:
            A Polygon object visualizing the agent's facing direction.
        """
        radius = utils.get_inscribed_radius(self.polygon)
        point_a = np.array([self.pose[0], self.pose[1]])
        direction = np.array(utils.direction_from_yaw(self.pose[2]))
        point_b = point_a + direction * radius
        orientation_polygon = utils.path_to_polygon(
            points=[point_a, point_b], line_width=radius / 4
        )
        return orientation_polygon
