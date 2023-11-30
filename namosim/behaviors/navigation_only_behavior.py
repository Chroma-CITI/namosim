import typing as t

import numpy as np
from navigation.basic_actions import GoalFailed, GoalsFinished, GoalSuccess

from namosim.algorithms.graph_search import real_to_grid_search_a_star
from namosim.behaviors.baseline_behavior import BaselineBehavior
from namosim.models import NavigationOnlyBehaviorConfigModel, PoseModel
from namosim.navigation.navigation_path import Path
from namosim.navigation.navigation_plan import Plan
from namosim.world.world import World


class NavigationOnlyBehavior(BaselineBehavior):
    def __init__(
        self,
        initial_world: World,
        robot_uid: int,
        navigation_goals: t.List[PoseModel],
        behavior_config: NavigationOnlyBehaviorConfigModel,
        logs_dir: str,
    ):
        BaselineBehavior.__init__(
            self,
            initial_world,
            robot_uid,
            navigation_goals,
            behavior_config,
            logs_dir,
        )

    def think(self):
        if self._navigation_goals or self._q_goal is not None:
            if self._q_goal is None:
                self._q_goal = self._navigation_goals.pop(0)
                self._p_opt = Plan([], self._q_goal)

            q_r = self._robot.pose

            # TODO Extract abs_tol constant and make it a parameter for each goal
            is_close_enough_to_goal = all(np.isclose(q_r, self._q_goal, rtol=1e-5))
            if is_close_enough_to_goal:
                print(
                    "SUCCESS: Agent '{name}' has successfully reached pose {nav_goal}.".format(
                        name=self._robot.name, nav_goal=str(self._q_goal)
                    )
                )
                action = GoalSuccess(self._q_goal)
                self._q_goal = None
                return action

            if self._p_opt is None:
                raise Exception("No plan")

            if not self._p_opt.is_valid(self.world, self._robot_uid):
                grid = self.world.get_binary_inflated_occupancy_grid(
                    (self._robot_uid,)
                ).get_grid()
                self._p_opt = Plan(
                    [Path(real_to_grid_search_a_star(q_r, self._q_goal, grid))],
                    self._q_goal,
                )

            if not self._p_opt.is_empty():
                next_step = self._p_opt.pop_next_action()
                return next_step
            elif self._p_opt.has_infinite_cost():
                print(
                    "FAILURE: Agent '{name}' has failed to reach pose {nav_goal}.".format(
                        name=self._robot.name, nav_goal=str(self._q_goal)
                    )
                )
                action = GoalFailed(self._q_goal)
                self._q_goal = None
                return action

        else:
            print(
                "FINISH: Agent '{name}' has finished trying to reach its goals !".format(
                    name=self._robot.name
                )
            )
            return GoalsFinished()
