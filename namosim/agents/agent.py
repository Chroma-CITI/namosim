import abc
import copy
import typing as t
from collections import OrderedDict

from shapely import Polygon
from typing_extensions import Self

import namosim.display.ros2_publisher as rp
import namosim.navigation.navigation_plan as navp
import namosim.world.world as w
from namosim.algorithms import graph_search
from namosim.data_models import UID, PoseModel
from namosim.input import Input
from namosim.navigation.action_result import ActionResult
from namosim.navigation.basic_actions import BasicAction
from namosim.navigation.navigation_path import TransitPath
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryInflatedOccupancyGrid
from namosim.world.entity import Entity, Movability, Style
from namosim.world.sensors.g_fov_sensor import GFOVSensor
from namosim.world.sensors.omniscient_sensor import OmniscientSensor
from namosim.world.sensors.s_fov_sensor import SFOVSensor


class ThinkResult:
    def __init__(
        self,
        *,
        next_action: BasicAction | None,
        did_replan: bool,
        did_postpone: bool = False,
        robot_name: str,
        has_conflicts: bool,
    ) -> None:
        self.next_action = next_action
        self.did_replan = did_replan
        self.did_postpone = did_postpone
        self.robot_name = robot_name
        self.has_conflicts = has_conflicts


class Agent(Entity):
    def __init__(
        self,
        *,
        behavior_type: str,
        navigation_goals: t.List[PoseModel],
        logs_dir: str,
        name: str,
        full_geometry_acquired: bool,
        polygon: Polygon,
        pose: PoseModel,
        sensors: t.List[OmniscientSensor | GFOVSensor | SFOVSensor],
        push_only_list: t.List[str],
        force_pushes_only: bool,
        movable_whitelist: t.List[str],
        style: Style,
        cell_size: float,
        movability: Movability = Movability.UNKNOWN,
        logger: utils.CustomLogger,
        uid: UID = 0,
    ):
        super().__init__(
            name=name,
            polygon=polygon,
            pose=pose,
            full_geometry_acquired=full_geometry_acquired,
            movability=movability,
            uid=uid,
            style=style,
            type_="robot",
        )

        self.behavior_type = behavior_type

        self.sensors = sensors
        for sensor in sensors:
            sensor.parent_uid = self.uid

        self.push_only_list = push_only_list
        self.force_pushes_only = force_pushes_only
        self.movable_whitelist = movable_whitelist
        self.min_inflation_radius = self.compute_inflation_radius()
        self.logger = logger
        self.__world: t.Optional["w.World"] = None
        self._navigation_goals = navigation_goals
        self.logs_dir = logs_dir

        self.__last_action_result: ActionResult | None = None

        self._prev_goal: PoseModel | None = (
            None  # used to check if the goal has changed
        )
        self.__q_goal: PoseModel | None = None

        self._prev_plan: t.Optional[
            "navp.Plan"
        ] = None  # used to check if a plan has changed
        self.__p_opt: t.Optional["navp.Plan"] = None

        self._added_uids, self._updated_uids, self._removed_uids = set(), set(), set()

        self.goal_to_plans: t.Dict[PoseModel, "navp.Plan"] = OrderedDict()
        self.is_initialized = False
        self.cell_size = cell_size
        self.grab_and_release_distance = max(
            utils.SQRT_OF_2 * cell_size + 1e-6,
            0.5 * self.circumscribed_radius,
        )
        """The robot will move backwards by this amount when it releases an object. The robot
        must be within this distance from a movable obstacle to grab it.
        This distance must be larger than the cell size otherwise the robot
        may still be colliding when it releases an obstacle.
        """

    def init(self, world: "w.World") -> None:
        self.__world = copy.deepcopy(world)
        self.is_initialized = True

    def sense(
        self, ref_world: "w.World", last_action_result: ActionResult, step_count: int
    ):
        self._last_action_result = last_action_result
        (
            self._added_uids,
            self._updated_uids,
            self._removed_uids,
        ) = self.update_world_from_sensors(ref_world, self.world)
        self._step_count = step_count

    @abc.abstractmethod
    def think(
        self, ros_publisher: "rp.RosPublisher", input: t.Optional[Input] = None
    ) -> ThinkResult:
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
    def _p_opt(self, p_opt: t.Optional["navp.Plan"]):
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
        if not self.__world:
            raise Exception("Not initialized")

        return self.__world

    @property
    def goal_pose(self):
        return self._q_goal

    def get_current_or_next_goal(self):
        if self._q_goal:
            return self._q_goal
        if len(self._navigation_goals) > 0:
            return self._navigation_goals[0]

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

    def update_world_from_sensors(
        self, reference_world: "w.World", target_world: "w.World"
    ):
        added_uids: set[UID] = set()
        updated_uids: set[UID] = set()
        removed_uids: set[UID] = set()

        for sensor in self.sensors:
            s_uids_to_add, s_uids_to_update, s_uids_to_remove = sensor.update_from_fov(
                reference_world, target_world
            )  # type: ignore

            # Might need a better update policy if sensors disagree about what happened, but irrelevant for now
            added_uids.update(s_uids_to_add)
            updated_uids.update(s_uids_to_update)
            removed_uids.update(s_uids_to_remove)

        return added_uids, updated_uids, removed_uids

    def deduce_movability(self, obstacle_type: str):
        if obstacle_type == "unknown" or obstacle_type == "robot":
            return Movability.UNKNOWN
        if obstacle_type == "movable":
            return Movability.MOVABLE
        elif obstacle_type in self.movable_whitelist:
            return Movability.MOVABLE
        else:
            return Movability.STATIC

    def deduce_push_only(self, obstacle_type: str):
        if self.force_pushes_only or obstacle_type in self.push_only_list:
            return True
        else:
            return False

    def compute_inflation_radius(self) -> float:
        return utils.get_circumscribed_radius(self.polygon)

    def to_json(self) -> t.Dict[str, t.Any]:
        json_data = Entity.to_json(self)
        json_data["geometry"]["orientation_id"] = self.name + "_dir"
        json_data["movable_whitelist"] = self.movable_whitelist
        json_data["push_only_list"] = self.push_only_list
        json_data["force_pushes_only"] = self.force_pushes_only
        json_data["sensors"] = []
        for sensor in self.sensors:
            json_data["sensors"].append(sensor.to_json())
        return json_data

    def light_copy(self) -> Self:
        raise NotImplementedError()
