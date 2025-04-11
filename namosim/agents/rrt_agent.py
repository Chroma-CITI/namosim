import copy
import typing as t

from shapely.geometry import Polygon
from typing_extensions import Self

from namosim.algorithms.rrt import DiffDriveRRT
import namosim.display.ros2_publisher as rp
import namosim.navigation.basic_actions as ba
from namosim.navigation.navigation_path import TransitPath
import namosim.navigation.navigation_plan as nav_plan
from namosim.svg_styles import AgentStyle
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
import namosim.world.world as w
from namosim.agents.agent import Agent, ThinkResult
from namosim.data_models import RRTAgentConfigModel, PoseModel
from namosim.input import Input
from namosim.utils import utils
from namosim.world.goal import Goal
from namosim.world.sensors.omniscient_sensor import OmniscientSensor


class RRTAgent(Agent):
    def __init__(
        self,
        *,
        navigation_goals: t.List[Goal],
        config: RRTAgentConfigModel,
        logs_dir: str,
        uid: str,
        full_geometry_acquired: bool,
        polygon: Polygon,
        pose: PoseModel,
        sensors: t.List[OmniscientSensor],
        style: AgentStyle,
        logger: utils.NamosimLogger,
        cell_size: float,
    ):
        Agent.__init__(
            self,
            uid=uid,
            navigation_goals=navigation_goals,
            config=config,
            logs_dir=logs_dir,
            full_geometry_acquired=full_geometry_acquired,
            polygon=polygon,
            pose=pose,
            sensors=sensors,  # type: ignore
            style=style,
            logger=logger,
            cell_size=cell_size,
        )
        self.config = config
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD
        self.robot_max_inflation_radius = utils.get_circumscribed_radius(self.polygon)

    def init(self, world: "w.World"):
        super().init(world)
        self.robot_inflated_grid = copy.deepcopy(world.map).inflate_map_destructive(
            self.circumscribed_radius + world.map.cell_size
        )

        # TODO Make sure static and generalist grid share same width and height (occurs naturally if map borders are static, but not otherwise)
        self.robot_inflated_grid.deactivate_entities({self.uid})

    def think(
        self,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        input: t.Optional[Input] = None,
    ) -> ThinkResult:
        if self._goal is None:
            if self._navigation_goals:
                self._goal = self._navigation_goals.pop(0)
                self._p_opt = nav_plan.Plan(agent_id=self.uid, goal=self._goal.pose)
            else:
                return ThinkResult(
                    plan=None,
                    next_action=ba.GoalsFinished(),
                    goal_pose=None,
                    did_replan=False,
                    agent_id=self.uid,
                )

        if self._p_opt is None:
            raise Exception("No plan")

        # If current robot pose is close enough to goal, return Success
        if (
            utils.distance_between_poses(
                a=self.world.dynamic_entities[self.uid].pose,
                b=self._goal.pose,
            )
            <= 0.1
        ):
            result = ThinkResult(
                plan=None,
                next_action=ba.GoalSuccess(goal=self._goal.pose),
                goal_pose=self._goal.pose,
                did_replan=False,
                agent_id=self.uid,
            )
            self._goal = None
            return result

        if not self._p_opt.is_empty():
            return ThinkResult(
                plan=self._p_opt,
                goal_pose=self._goal.pose,
                did_replan=False,
                agent_id=self.uid,
            )

        path = self.find_path_rrt(
            robot_pose=self.world.dynamic_entities[self.uid].pose,
            goal_pose=self._goal.pose,
            robot_inflated_grid=self.robot_inflated_grid,
            robot_polygon=self.world.dynamic_entities[self.uid].polygon,
        )

        if path is None:
            return ThinkResult(
                plan=None,
                next_action=ba.GoalFailed(self._goal.pose),
                goal_pose=self._goal.pose,
                did_replan=False,
                agent_id=self.uid,
            )

        self._p_opt = nav_plan.Plan(
            paths=[path], goal=self._goal.pose, agent_id=self.uid
        )
        self.goal_to_plans[self._goal] = self._p_opt

        return ThinkResult(
            plan=self._p_opt,
            next_action=None,
            goal_pose=self._goal.pose,
            did_replan=True,
            agent_id=self.uid,
        )

    def copy(self) -> Self:
        return RRTAgent(
            navigation_goals=copy.deepcopy(self._navigation_goals),
            config=self.config,
            logs_dir=self.logs_dir,
            full_geometry_acquired=self.full_geometry_acquired,
            uid=self.uid,
            polygon=copy.deepcopy(self.polygon),
            style=copy.deepcopy(self.agent_style),
            pose=copy.deepcopy(self.pose),
            sensors=copy.deepcopy(self.sensors),  # type: ignore
            cell_size=self.cell_size,
            logger=self.logger,
        )

    def find_path_rrt(
        self,
        robot_pose: PoseModel,
        goal_pose: PoseModel,
        robot_inflated_grid: BinaryOccupancyGrid,
        robot_polygon: Polygon,
    ) -> TransitPath | None:
        rrt = DiffDriveRRT(
            polygon=robot_polygon,
            start=robot_pose,
            goal=goal_pose,
            map=robot_inflated_grid,
        )

        plan = rrt.plan()
        # rrt.plot(plan)
        if not plan:
            return None
        poses = [x.pose for x in plan]

        return TransitPath.from_poses(poses, robot_polygon, robot_pose)
