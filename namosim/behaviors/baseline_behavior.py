import abc
import copy
import typing as t
from decimal import Decimal

from shapely import Polygon

from namosim.algorithms import graph_search
from namosim.display.ros2_publisher import RosPublisher
from namosim.models import (
    NavigationOnlyBehaviorConfigModel,
    PoseModel,
    StilmanBehaviorConfigModel,
    WuLevihnBehaviorConfigModel,
)
from namosim.navigation.action_result import ActionResult
from namosim.navigation.basic_actions import BasicAction
from namosim.navigation.navigation_path import TransitPath
from namosim.navigation.navigation_plan import Plan
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryInflatedOccupancyGrid
from namosim.world.robot import Robot
from namosim.world.world import World


class ThinkResult:
    def __init__(
        self,
        next_action: BasicAction | None,
        did_replan: bool,
        robot_name: str,
        has_conflicts: bool,
    ) -> None:
        self.next_action = next_action
        self.did_replan = did_replan
        self.robot_name = robot_name
        self.has_conflicts = has_conflicts


class BaselineBehavior(object):
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        initial_world: World,
        robot_uid: int,
        navigation_goals: t.List[PoseModel],
        behavior_config: StilmanBehaviorConfigModel
        | WuLevihnBehaviorConfigModel
        | NavigationOnlyBehaviorConfigModel,
        logs_dir: str,
    ):
        self.simulation_log = utils.CustomLogger()

        self._initial_world = initial_world
        self._robot_uid = robot_uid
        self._robot_name = initial_world.entities[robot_uid].name
        self._navigation_goals = navigation_goals
        self._behavior_config = behavior_config
        self.logs_dir = logs_dir

        decimal_res = Decimal(initial_world.discretization_data.res).as_tuple()
        precision_exponent = (
            -len(decimal_res.digits) - t.cast(int, decimal_res.exponent) + 2
        )

        self.rounder = 1.0 * (10**precision_exponent)
        self.r_tol = 1.0 * (10**-precision_exponent)

        self.__world: World = copy.deepcopy(self._initial_world)
        self._robot: Robot = t.cast(Robot, self.world.entities[self._robot_uid])
        self.__last_action_result: ActionResult | None = None

        self._prev_goal: PoseModel | None = (
            None  # used to check if the goal has changed
        )
        self.__q_goal: PoseModel | None = None

        self._prev_plan: Plan | None = None  # used to check if a plan has changed
        self.__p_opt: Plan | None = None

        self._added_uids, self._updated_uids, self._removed_uids = set(), set(), set()

        self.goal_to_plans: t.Dict[PoseModel, Plan]

    def sense(
        self, ref_world: World, last_action_result: ActionResult, step_count: int
    ):
        self._last_action_result = last_action_result
        (
            self._added_uids,
            self._updated_uids,
            self._removed_uids,
        ) = self._robot.update_world_from_sensors(ref_world, self.world)
        self._step_count = step_count

    @abc.abstractmethod
    def think(self, ros_publisher: RosPublisher) -> ThinkResult:
        raise NotImplementedError

    @property
    def _q_goal(self):
        return self.__q_goal

    @_q_goal.setter
    def _q_goal(self, _q_goal: PoseModel | None):
        self._prev_goal = self.__q_goal
        self.__q_goal = _q_goal

    @property
    def _p_opt(self):
        return self.__p_opt

    @_p_opt.setter
    def _p_opt(self, p_opt: Plan | None):
        self._prev_plan = self.__p_opt
        self.__p_opt = p_opt

    @property
    def _last_action_result(self):
        return self.__last_action_result

    @_last_action_result.setter
    def _last_action_result(self, last_action_result: ActionResult):
        self.__last_action_result = last_action_result

    @property
    def world(self):
        return self.__world

    def set_world(self, world: World):
        self.__world = world
        self._robot = t.cast(Robot, self.__world.entities[self._robot_uid])

    @property
    def robot_uid(self):
        return self._robot_uid

    @property
    def robot(self):
        return self._robot

    @property
    def name(self):
        return self._behavior_config.name

    @property
    def goal_pose(self):
        return self._q_goal

    def get_plan(self):
        return self.__p_opt

    def has_goal_changed(self):
        return self._prev_goal != self.__q_goal

    def is_goal_reached(
        self,
        q_t: PoseModel,
        q_f: PoseModel,
        pos_tol: float = 0.05,
        ang_tol: float = 0.1,
    ):
        return all(
            [
                utils.is_close(q_t[0], q_f[0], rel_tol=pos_tol),
                utils.is_close(q_t[1], q_f[1], rel_tol=pos_tol),
                utils.angle_is_close(q_t[2], q_f[2], rel_tol=ang_tol),
            ]
        )

    def find_path(
        self,
        robot_pose: PoseModel,
        goal_pose: PoseModel,
        robot_inflated_grid: BinaryInflatedOccupancyGrid,
        robot_polygon: Polygon,
    ):
        real_path = graph_search.real_to_grid_search_a_star(
            robot_pose, goal_pose, robot_inflated_grid
        )
        if real_path:

            def g(a: PoseModel, b: PoseModel):
                translation_cost = utils.euclidean_distance(a, b)
                rotation_cost = abs(a[2] - b[2])
                return translation_cost + rotation_cost

            phys_cost = 0.0
            for a, b in zip(real_path, real_path[1:]):
                phys_cost += g(a, b)
            return TransitPath.from_poses(
                real_path, robot_polygon, robot_pose, phys_cost
            )
        else:
            return None
