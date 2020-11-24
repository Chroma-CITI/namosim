import abc
import copy
from decimal import Decimal

from snamosim.display.ros_publisher import RosPublisher


class BaselineBehavior(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, initial_world, robot_uid, navigation_goals, behavior_config, abs_path_to_logs_dir):
        self._initial_world = initial_world
        self._robot_uid = robot_uid
        self._robot_name = initial_world.entities[robot_uid].name
        self._navigation_goals = navigation_goals
        self._behavior_config = behavior_config
        self.abs_path_to_logs_dir = abs_path_to_logs_dir

        decimal_res = Decimal(initial_world.dd.res).as_tuple()
        precision_exponent = -len(decimal_res.digits) - decimal_res.exponent + 2

        self.rounder = 1. * (10 ** precision_exponent)
        self.r_tol = 1. * (10 ** -precision_exponent)

        self.__world = copy.deepcopy(self._initial_world)
        self._robot = self._world.entities[self._robot_uid]
        self.__last_action_result = None
        self.__q_goal = None
        self.__p_opt = None

        self._rp = RosPublisher()

    def sense(self, ref_world, last_action_result):
        self._last_action_result = last_action_result
        self._robot.update_world_from_sensors(ref_world, self._world)
        self._rp.publish_robot_world(self._world, self._robot_uid, ns=self._robot_name)

    @abc.abstractmethod
    def think(self):
        raise NotImplementedError

    @property
    def _q_goal(self):
        return self.__q_goal

    @_q_goal.setter
    def _q_goal(self, _q_goal):
        self.__q_goal = _q_goal
        if _q_goal is not None:
            self._rp.publish_goal(self._robot.pose, self.__q_goal, self._robot.polygon, ns=self._robot_name)

    @property
    def _p_opt(self):
        return self.__p_opt

    @_p_opt.setter
    def _p_opt(self, p_opt):
        self.__p_opt = p_opt
        self._rp.cleanup_p_opt(ns=self._robot_name)
        self._rp.publish_p_opt(self.__p_opt, ns=self._robot_name)

    @property
    def _last_action_result(self):
        return self.__last_action_result

    @_last_action_result.setter
    def _last_action_result(self, last_action_result):
        self.__last_action_result = last_action_result

    @property
    def _world(self):
        return self.__world

    @_world.setter
    def _world(self, world):
        self.__world = world
        self._robot = self.__world.entities[self._robot_uid]

    @property
    def name(self):
        return self._behavior_config["name"]
