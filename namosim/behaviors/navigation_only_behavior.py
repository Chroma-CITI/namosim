import numpy as np
from plan.basic_actions import GoalFailed, GoalsFinished, GoalSuccess

from namosim import utils
from namosim.behaviors.algorithms.graph_search import real_to_grid_search_a_star
from namosim.behaviors.baseline_behavior import BaselineBehavior


class Path:
    def __init__(
        self, poses, polygons, cells=None, csv_polygons=None, bb_vertices=None
    ):
        self.poses = poses
        self.polygons = polygons
        self.cells = cells
        self.csv_polygons = csv_polygons
        self.bb_vertices = bb_vertices

    # TODO Have these trans and rot precision values be passed from calling functions !
    def is_start_pose(self, pose, trans_mult=100.0, rot_mult=1.0):
        fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
            pose, trans_mult, rot_mult
        )
        fixed_precision_self_pose = utils.real_pose_to_fixed_precision_pose(
            self.poses[0], trans_mult, rot_mult
        )
        return fixed_precision_pose == fixed_precision_self_pose


class Plan:
    def __init__(self, path_components=[], goal=None, robot_uid=None, plan_error=None):
        self.path_components = path_components
        self.goal = goal
        self.robot_uid = robot_uid
        self.phys_cost = 0.0
        self.social_cost = 0.0
        self.total_cost = 0.0
        self.plan_error = plan_error

        self.component_index = 0

        if path_components:
            for path in path_components:
                self.phys_cost += path.phys_cost
                self.social_cost += path.social_cost
                self.total_cost += path.total_cost
        else:
            self.phys_cost = float("inf")
            self.social_cost = float("inf")
            self.total_cost = float("inf")

    def append(self, future_plan):
        self.path_components += future_plan.path_components
        self.phys_cost += future_plan.phys_cost
        self.social_cost += future_plan.social_cost
        self.total_cost += future_plan.total_cost
        return self

    def has_infinite_cost(self):
        return True if self.total_cost == float("inf") else False

    def exists(self):
        return bool(self.path_components)

    def pop_next_step(self):
        """
        Get the next plan step to execute
        :return: the action object to be executed if there is one, None if the plan is empty
        :rtype: action or None
        :except: if pop_next_step is called when the plan is fully executed
        :exception: IndexError
        """
        current_component = self.path_components[self.component_index]
        if current_component.is_fully_executed():
            if self.component_index < len(self.path_components) - 1:
                self.component_index += 1
            current_component = self.path_components[self.component_index]
        return current_component.pop_next_step()


class NavigationOnlyBehavior(BaselineBehavior):
    def __init__(
        self,
        initial_world,
        robot_uid,
        navigation_goals,
        behavior_config,
        abs_path_to_logs_dir,
    ):
        BaselineBehavior.__init__(
            self,
            initial_world,
            robot_uid,
            navigation_goals,
            behavior_config,
            abs_path_to_logs_dir,
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

            if not self._p_opt.is_valid(self._world, self._robot_uid):
                grid = self._world.get_binary_inflated_occupancy_grid(
                    (self._robot_uid,)
                ).get_grid()
                self._p_opt = Plan(
                    [Path(real_to_grid_search_a_star(q_r, self._q_goal, grid))],
                    self._q_goal,
                )

            if not self._p_opt.is_empty():
                next_step = self._p_opt.pop_next_step()
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
