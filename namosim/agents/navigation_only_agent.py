import copy
import typing as t

from shapely import Polygon
from typing_extensions import Self

import namosim.display.ros2_publisher as rp
import namosim.navigation.basic_actions as ba
import namosim.navigation.navigation_plan as nav_plan
import namosim.world.world as w
from namosim.agents.agent import Agent, ThinkResult
from namosim.data_models import UID, PoseModel
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryInflatedOccupancyGrid
from namosim.world.entity import Style
from namosim.world.obstacle import Obstacle
from namosim.world.sensors.omniscient_sensor import OmniscientSensor


class NavigationOnlyAgent(Agent):
    def __init__(
        self,
        *,
        navigation_goals: t.List[PoseModel],
        logs_dir: str,
        name: str,
        full_geometry_acquired: bool,
        polygon: Polygon,
        pose: PoseModel,
        sensors: t.List[OmniscientSensor],
        push_only_list: t.List[str],
        force_pushes_only: bool,
        movable_whitelist: t.List[str],
        style: Style,
        logger: utils.CustomLogger,
        uid: UID = 0,
    ):
        Agent.__init__(
            self,
            name=name,
            navigation_goals=navigation_goals,
            behavior_type="navigation_only_behavior",
            logs_dir=logs_dir,
            full_geometry_acquired=full_geometry_acquired,
            polygon=polygon,
            pose=pose,
            sensors=sensors,  # type: ignore
            push_only_list=push_only_list,
            force_pushes_only=force_pushes_only,
            movable_whitelist=movable_whitelist,
            style=style,
            logger=logger,
            uid=uid,
        )
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD
        self.robot_max_inflation_radius = utils.get_circumscribed_radius(self.polygon)

    def init(self, world: "w.World"):
        super().init(world)

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

    def think(self, ros_publisher: "rp.RosPublisher"):
        if self._q_goal is None:
            if self._navigation_goals:
                self._q_goal = self._navigation_goals.pop(0)
                self._p_opt = nav_plan.Plan(
                    robot_uid=self.uid, path_components=[], goal=self._q_goal
                )
            else:
                return ThinkResult(
                    next_action=ba.GoalsFinished(),
                    did_replan=False,
                    robot_name=self.name,
                    has_conflicts=False,
                )

        if self._p_opt is None:
            raise Exception("No plan")

        # If current robot pose is close enough to goal, return Success
        if self.is_goal_reached(self.world.entities[self.uid].pose, self._q_goal):
            result = ThinkResult(
                next_action=ba.GoalSuccess(goal=self._q_goal),
                did_replan=False,
                robot_name=self.name,
                has_conflicts=False,
            )
            self._q_goal = None
            return result

        if not self._p_opt.is_empty():
            return ThinkResult(
                next_action=self._p_opt.pop_next_action(),
                did_replan=False,
                robot_name=self.name,
                has_conflicts=False,
            )

        path = self.find_path(
            robot_pose=self.world.entities[self.uid].pose,
            goal_pose=self._q_goal,
            robot_inflated_grid=self.static_obs_inf_grid,
            robot_polygon=self.world.entities[self.uid].polygon,
        )

        if path is None:
            return ThinkResult(
                next_action=ba.GoalFailed(self._q_goal),
                did_replan=False,
                robot_name=self.name,
                has_conflicts=False,
            )

        self._p_opt = nav_plan.Plan(
            path_components=[path], goal=self._q_goal, robot_uid=self.uid
        )
        self.goal_to_plans[self._q_goal] = self._p_opt

        return ThinkResult(
            next_action=self._p_opt.pop_next_action(),
            did_replan=True,
            robot_name=self.name,
            has_conflicts=False,
        )

    def light_copy(self) -> Self:
        return NavigationOnlyAgent(
            uid=self.uid,
            navigation_goals=copy.deepcopy(self._navigation_goals),
            logs_dir=self.logs_dir,
            full_geometry_acquired=self.full_geometry_acquired,
            name=self.name,
            polygon=copy.deepcopy(self.polygon),
            style=copy.deepcopy(self.style),
            pose=copy.deepcopy(self.pose),
            sensors=copy.deepcopy(self.sensors),  # type: ignore
            push_only_list=[],
            force_pushes_only=False,
            movable_whitelist=["box"],
            logger=self.logger,
        )
