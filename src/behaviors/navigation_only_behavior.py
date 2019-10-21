from src.behaviors.algorithms.a_star import a_star_real_path
from src.behaviors.plan.path import Path
from src.behaviors.plan.plan import Plan
import numpy as np
from plan.basic_actions import ActionGoalFailure, ActionGoalsFinished, ActionGoalSuccess


class NavigationOnlyBehavior:
    def __init__(self, simulator, initial_world, robot_uid, navigation_goals):
        self.simulator = simulator
        self.world = initial_world
        self.robot_uid = robot_uid
        self.robot = self.world.entities[self.robot_uid]
        self.navigation_goals = navigation_goals

        self.last_action_result = True
        self.q_goal = None
        self.p_opt = None

    def sense(self, ref_world, last_action_result):
        self.last_action_result = last_action_result
        self.robot.update_world_from_sensors(ref_world, self.world)

    def think(self):
        if self.navigation_goals or self.q_goal is not None:
            if self.q_goal is None:
                self.q_goal = self.navigation_goals.pop(0)
                self.p_opt = Plan([Path([])])

            q_r = self.robot.pose

            # TODO Extract abs_tol constant and make it a parameter for each goal
            is_close_enough_to_goal = all(np.isclose(q_r, self.q_goal, atol=1e-3))
            if is_close_enough_to_goal:
                print("SUCCESS: Agent '{name}' has successfully reached pose {nav_goal}.".format(
                    name=self.robot.name, nav_goal=str(self.q_goal)))
                self.q_goal, self.p_opt = None, None
                return ActionGoalSuccess()

            if not self.p_opt.is_valid(self.world, self.robot_uid):
                grid = self.world.get_binary_inflated_occupancy_grid((self.robot_uid,))
                self.p_opt = Plan([Path(a_star_real_path(grid, q_r, self.q_goal, self.world.dd))])

            if not self.p_opt.is_empty():
                next_step = self.p_opt.pop_next_step()
                return next_step
            elif self.p_opt.has_infinite_cost():
                print("FAILURE: Agent '{name}' has failed to reach pose {nav_goal}.".format(
                    name=self.robot.name, nav_goal=str(self.q_goal)))
                self.q_goal, self.p_opt = None, None
                return ActionGoalFailure()

        else:
            print("FINISH: Agent '{name}' has finished trying to reach its goals !".format(name=self.robot.name))
            return ActionGoalsFinished()
