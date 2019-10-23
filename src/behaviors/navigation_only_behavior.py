from src.behaviors.algorithms.a_star import a_star_real_path
from src.behaviors.plan.path import Path
from src.behaviors.plan.plan import Plan
import numpy as np
from plan.basic_actions import ActionGoalFailure, ActionGoalsFinished, ActionGoalSuccess
from src.display.ros_publisher import RosPublisher
from src.simulation_report import SimulationReport


class NavigationOnlyBehavior(object):
    def __init__(self, simulator, initial_world, robot_uid, navigation_goals, behavior_config):
        self._simulator = simulator
        self._world = initial_world
        self._robot_uid = robot_uid
        self._robot = self._world.entities[self._robot_uid]
        self._navigation_goals = navigation_goals
        self._behavior_config = behavior_config

        self._last_action_result = True
        self.__q_goal = None
        self.__p_opt = None

        self._rp = RosPublisher()

        self._report = SimulationReport()

    def sense(self, ref_world, last_action_result):
        self._last_action_result = last_action_result
        self._robot.update_world_from_sensors(ref_world, self._world)
        self._rp.publish_robot_world(self._world, self._robot_uid)

    def think(self):
        if self._navigation_goals or self.__q_goal is not None:
            if self._q_goal is None:
                self._q_goal = self._navigation_goals.pop(0)
                self._p_opt = Plan([Path([])])

            q_r = self._robot.pose

            # TODO Extract abs_tol constant and make it a parameter for each goal
            is_close_enough_to_goal = all(np.isclose(q_r, self._q_goal, rtol=1e-5))
            if is_close_enough_to_goal:
                print("SUCCESS: Agent '{name}' has successfully reached pose {nav_goal}.".format(
                    name=self._robot.name, nav_goal=str(self._q_goal)))
                self._q_goal = None
                return ActionGoalSuccess()

            if not self._p_opt.is_valid(self._world, self._robot_uid):
                grid = self._world.get_binary_inflated_occupancy_grid((self._robot_uid,))
                self._p_opt = Plan([Path(a_star_real_path(grid, q_r, self._q_goal, self._world.dd))])

            if not self._p_opt.is_empty():
                next_step = self._p_opt.pop_next_step()
                return next_step
            elif self._p_opt.has_infinite_cost():
                print("FAILURE: Agent '{name}' has failed to reach pose {nav_goal}.".format(
                    name=self._robot.name, nav_goal=str(self._q_goal)))
                self._q_goal = None
                return ActionGoalFailure()

        else:
            print("FINISH: Agent '{name}' has finished trying to reach its goals !".format(name=self._robot.name))
            return ActionGoalsFinished(self._report)

    @property
    def _q_goal(self):
        return self.__q_goal
    
    @_q_goal.setter
    def _q_goal(self, _q_goal):
        self.__q_goal = _q_goal
        self._rp.publish_goal(self._robot.pose, self.__q_goal, self._robot.polygon)
        self._report.plans_for_goals[_q_goal] = []
    
    @property
    def _p_opt(self):
        return self.__p_opt

    @_p_opt.setter
    def _p_opt(self, p_opt):
        self.__p_opt = p_opt
        self._rp.publish_p_opt(self.__p_opt)
        self._report.plans_for_goals[self._q_goal].append(p_opt)
