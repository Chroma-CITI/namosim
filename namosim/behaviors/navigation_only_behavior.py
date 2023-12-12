import typing as t

import namosim.navigation.basic_actions as ba
from namosim.behaviors.baseline_behavior import BaselineBehavior, ThinkResult
from namosim.data_models import PoseModel
from namosim.display.ros2_publisher import RosPublisher
from namosim.navigation.navigation_plan import Plan
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryInflatedOccupancyGrid
from namosim.world.obstacle import Obstacle
from namosim.world.world import World


class NavigationOnlyBehavior(BaselineBehavior):
    def __init__(
        self,
        initial_world: World,
        robot_uid: int,
        navigation_goals: t.List[PoseModel],
        logs_dir: str,
    ):
        BaselineBehavior.__init__(
            self,
            initial_world=initial_world,
            robot_uid=robot_uid,
            navigation_goals=navigation_goals,
            name="navigation_only_behavior",
            logs_dir=logs_dir,
        )

        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD
        self.robot_max_inflation_radius = utils.get_circumscribed_radius(
            self._robot.polygon
        )
        all_entities_polygons = {
            uid: e.polygon for uid, e in self.world.entities.items()
        }
        static_obs_polygons = {
            uid: entity.polygon
            for uid, entity in self.world.entities.items()
            if (
                isinstance(entity, Obstacle)
                and entity.movability == "unmovable"
                or entity.movability == "static"
            )
        }
        self.static_obs_inf_grid = BinaryInflatedOccupancyGrid(
            polygons=static_obs_polygons,
            res=self.world.discretization_data.res,
            inflation_radius=self.robot_max_inflation_radius,
            neighborhood=self.neighborhood,
        )
        self.inflated_grid_by_robot = BinaryInflatedOccupancyGrid(
            polygons=all_entities_polygons,
            res=self.world.discretization_data.res,
            inflation_radius=self.robot_max_inflation_radius,
            neighborhood=self.neighborhood,
            params=self.static_obs_inf_grid.params,
        )

    def think(self, ros_publisher: RosPublisher):
        if self._q_goal is None:
            if self._navigation_goals:
                self._q_goal = self._navigation_goals.pop(0)
                self._p_opt = Plan(
                    robot_uid=self.robot.uid, path_components=[], goal=self._q_goal
                )
            else:
                return ThinkResult(
                    next_action=ba.GoalsFinished(),
                    did_replan=False,
                    robot_name=self._robot_name,
                    has_conflicts=False,
                )

        if self._p_opt is None:
            raise Exception("No plan")

        # If current robot pose is close enough to goal, return Success
        if self.is_goal_reached(
            self.world.entities[self._robot_uid].pose, self._q_goal
        ):
            result = ThinkResult(
                next_action=ba.GoalSuccess(goal=self._q_goal),
                did_replan=False,
                robot_name=self._robot_name,
                has_conflicts=False,
            )
            self._q_goal = None
            return result

        if not self._p_opt.is_empty():
            return ThinkResult(
                next_action=self._p_opt.pop_next_action(),
                did_replan=False,
                robot_name=self._robot_name,
                has_conflicts=False,
            )

        path = self.find_path(
            robot_pose=self.world.entities[self._robot_uid].pose,
            goal_pose=self._q_goal,
            robot_inflated_grid=self.static_obs_inf_grid,
            robot_polygon=self.world.entities[self._robot_uid].polygon,
        )

        if path is None:
            return ThinkResult(
                next_action=ba.GoalFailed(self._q_goal),
                did_replan=False,
                robot_name=self._robot_name,
                has_conflicts=False,
            )

        self._p_opt = Plan(
            path_components=[path], goal=self._q_goal, robot_uid=self._robot_uid
        )
        self.goal_to_plans[self._q_goal] = self._p_opt

        return ThinkResult(
            next_action=self._p_opt.pop_next_action(),
            did_replan=True,
            robot_name=self._robot_name,
            has_conflicts=False,
        )
