import abc
import copy
import typing as t
from decimal import Decimal

from namosim.display.ros2_publisher import RosPublisher
from namosim.models import (
    NavigationOnlyBehaviorConfigModel,
    PoseModel,
    StilmanBehaviorConfigModel,
    WuLevihnBehaviorConfigModel,
)
from namosim.navigation.action_result import ActionResult
from namosim.navigation.basic_actions import BasicAction
from namosim.navigation.navigation_plan import Plan
from namosim.utils import utils
from namosim.worldreps.entity_based.robot import Robot
from namosim.worldreps.entity_based.world import World


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
        ros_publisher: RosPublisher,
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
        self._robot: Robot = t.cast(Robot, self._world.entities[self._robot_uid])
        self.__last_action_result: ActionResult | None = None
        self.__q_goal: PoseModel | None = None
        self.__p_opt: Plan | None = None

        self._added_uids, self._updated_uids, self._removed_uids = set(), set(), set()

        self._rp = ros_publisher
        self.goal_to_plans: t.Dict[PoseModel, Plan]

    def sense(
        self, ref_world: World, last_action_result: ActionResult, step_count: int
    ):
        self._last_action_result = last_action_result
        (
            self._added_uids,
            self._updated_uids,
            self._removed_uids,
        ) = self._robot.update_world_from_sensors(ref_world, self._world)
        self._rp.publish_robot_world(self._world, self._robot_uid)
        self._step_count = step_count

    @abc.abstractmethod
    def think(self) -> BasicAction | None:
        raise NotImplementedError

    @property
    def _q_goal(self):
        return self.__q_goal

    @_q_goal.setter
    def _q_goal(self, _q_goal: PoseModel | None):
        self.__q_goal = _q_goal
        if _q_goal is not None:
            self._rp.publish_goal(
                self._robot.pose, _q_goal, self._robot, ns=self._robot_name
            )

    @property
    def _p_opt(self):
        return self.__p_opt

    @_p_opt.setter
    def _p_opt(self, p_opt: Plan | None):
        self.__p_opt = p_opt
        self._rp.cleanup_p_opt(ns=self._robot_name)
        if self.__p_opt:
            self._rp.publish_p_opt(self.__p_opt, self._robot, ns=self._robot_name)

    @property
    def _last_action_result(self):
        return self.__last_action_result

    @_last_action_result.setter
    def _last_action_result(self, last_action_result: ActionResult):
        self.__last_action_result = last_action_result

    @property
    def _world(self):
        return self.__world

    @_world.setter
    def _world(self, world: World):
        self.__world = world
        self._robot = t.cast(Robot, self.__world.entities[self._robot_uid])

    @property
    def name(self):
        return self._behavior_config.name
