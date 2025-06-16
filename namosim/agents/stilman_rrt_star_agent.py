import copy
import heapq
import time
import typing as t
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon
from shapely.geometry import Point
import shapely.ops
from typing_extensions import Self
from shapely import affinity
import shapely
import namosim.display.ros2_publisher as rp
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
import namosim.navigation.navigation_plan as nav_plan
from namosim.svg_styles import AgentStyle
import namosim.utils.collision as collision
import namosim.utils.connectivity as connectivity
import namosim.world.social_topological_occupation_cost_grid as stocg
import namosim.world.world as w
from namosim.agents.agent import Agent, ThinkResult
from namosim.agents.stilman_configurations import (
    RCHConfiguration,
    RobotConfiguration,
    RobotObstacleConfiguration,
)
from namosim.algorithms import graph_search
from namosim.algorithms.new_local_opening_check import check_new_local_opening
from namosim.data_models import (
    GridCellModel,
    Pose2D,
    StilmanRRTStarBehaviorConfigModel,
)
from namosim.input import Input
from namosim.navigation.conflict import (
    ConcurrentGrabConflict,
    Conflict,
    RobotObstacleConflict,
    RobotRobotConflict,
    StolenMovableConflict,
)
from namosim.navigation.navigation_path import (
    EvasionTransitPath,
    RawPath,
    TransferPath,
    TransitPath,
)
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from namosim.world.entity import Movability
from namosim.world.goal import Goal
from namosim.world.sensors.omniscient_sensor import OmniscientSensor
from namosim.algorithms.rrt_star import DiffDriveRRTStar, RRTNode
from shapely.geometry import JOIN_STYLE


class StilmanRRTStarAgent(Agent):
    def __init__(
        self,
        *,
        navigation_goals: t.List[Goal],
        config: StilmanRRTStarBehaviorConfigModel,
        logs_dir: str,
        uid: str,
        polygon: Polygon,
        pose: Pose2D,
        sensors: t.List[OmniscientSensor],
        style: AgentStyle,
        logger: utils.NamosimLogger,
        cell_size: float,
        collision_margin: float,
    ):
        super().__init__(
            uid=uid,
            navigation_goals=navigation_goals,
            config=config,
            logs_dir=logs_dir,
            polygon=polygon,
            pose=pose,
            sensors=sensors,  # type: ignore
            style=style,
            logger=logger,
            cell_size=cell_size,
        )
        self.config = config
        self.deadlock_strategy: t.Literal["SOCIAL", "DISTANCE"] = (
            "SOCIAL" if config.parameters.use_social_cost else "DISTANCE"
        )
        if config.parameters.deadlock_strategy == "SOCIAL":
            if not config.parameters.use_social_cost:
                raise Exception(
                    "SOCIAL deadlock strategy requires use_social_cost = TRUE"
                )
            self.deadlock_strategy = config.parameters.deadlock_strategy
        elif config.parameters.deadlock_strategy == "DISTANCE":
            self.deadlock_strategy = config.parameters.deadlock_strategy

        self._p_opt: "nav_plan.Plan"

        # - Original Stilman method configuration parameters
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD  # default if bad parameter
        self.translation_unit_cost = 1.0
        self.rotation_unit_cost = 2.0
        self.transfer_coefficient = 2.0  # Note: MUST ALWAYS BE > 1 !
        # - Robot action space parameters
        self.rotation_unit_angle = config.parameters.robot_rotation_unit_angle
        self.translation_unit_length = cell_size
        self.push_only = config.parameters.push_only
        self.translation_factor = (
            self.translation_unit_cost / self.translation_unit_length
        )
        self.rotation_factor = self.rotation_unit_cost / self.rotation_unit_angle
        self.max_evasion_cells_to_visit = 1000

        # - S-NAMO parameters
        self.use_social_cost = config.parameters.use_social_cost
        self.bound_percentage = config.parameters.manip_search_bound_percentage

        if self.use_social_cost:
            self.manip_search_procedure = self.focused_manip_search
        else:
            self.manip_search_procedure = self.manip_search

        self.w_social, self.w_dist, self.w_goal = 20.0, 10.0, 2.0
        self.w_sum = self.w_social + self.w_dist + self.w_goal
        self.TRANSLATION_DISCRETIZATION_FACTOR = (
            self.cell_size
        )  # during grid searches, round pose position to nearest cell
        self.ROTATION_DISCRETIZATION_FACTOR = (
            self.rotation_unit_angle
        )  # during grid searches, round pose angle to nearest rotation-unit-angle

        # - Extra performance parameters
        self.check_new_local_opening_before_global = (
            config.parameters.check_new_local_opening_before_global
        )
        self.activate_grids_logging = config.parameters.activate_grids_logging
        self._social_costmap: npt.NDArray[np.float32] | None = None
        self.conflict_horizon = config.parameters.conflict_horizon
        self.min_nb_steps_to_wait = 5
        self.max_nb_steps_to_wait = 20
        self.replan_count = 20

        if self.push_only:
            self._rot_angles = np.array([])
        else:
            self._rot_angles = np.array(
                [self.rotation_unit_angle, -self.rotation_unit_angle]
            )
        self._all_rot_angles = self.rotation_unit_angle * np.array(
            range(360 // int(self.rotation_unit_angle))
        )

        if (
            config.parameters.drive_type == "differential"
        ):  # pyright: ignore[reportUnnecessaryComparison]
            if self.push_only:
                self._transfer_movement_actions: t.List[ba.Action] = [
                    ba.Advance(distance=self.translation_unit_length),
                ]
            else:
                self._transfer_movement_actions: t.List[ba.Action] = [
                    ba.Advance(distance=self.translation_unit_length),
                    ba.Advance(distance=-self.translation_unit_length),
                ]
        elif config.parameters.drive_type == "holonomic":
            self._transfer_movement_actions: t.List[ba.Action] = [
                ba.Translation((self.translation_unit_length, 0.0)),
                ba.Translation((-self.translation_unit_length, 0.0)),
                ba.Translation((0.0, self.translation_unit_length)),
                ba.Translation((0.0, -self.translation_unit_length)),
            ]
        for rot_angle in self._rot_angles:
            self._transfer_movement_actions.append(ba.Rotation(rot_angle))

        self.goal_position_tolerance = 0.05  # meters
        self.goal_angle_tolerance = 1  # degrees

        # grab_start_distance
        if config.parameters.grab_start_distance is None:
            self.grab_start_distance = 4 * self.cell_size
        else:
            self.grab_start_distance = config.parameters.grab_start_distance

        # grab_end_distance
        if config.parameters.grab_end_distance is None:
            self.grab_end_distance = self.cell_size
        else:
            self.grab_end_distance = config.parameters.grab_end_distance

        self.collision_margin = collision_margin
        self.conflict_radius = (
            self.grab_start_distance
            + self.collision_margin
            + utils.SQRT_OF_2 * self.cell_size
        )

    def init(self, world: "w.World"):
        super().init(world)

        # Initialize movability status of obstacles
        for entity in self.world.dynamic_entities.values():
            if entity.movability != Movability.STATIC:
                entity.movability = self.deduce_movability(entity.type_)

        self.action_space_reduction = (
            "none"  # ['none', 'only_r_acc', 'only_r_acc_then_c_1_x']
        )

        # Initialize static obstacles occupation grid, since it is not supposed to change
        self.robot_inflated_static_map = copy.deepcopy(
            world.map
        ).inflate_map_destructive(self.circumscribed_radius + self.collision_margin)

        # check that goals are valid (i.e., not in static obstacles)
        for goal in self._navigation_goals:
            goal_cell = utils.real_to_grid(
                goal.pose[0],
                goal.pose[1],
                self.robot_inflated_static_map.cell_size,
                self.robot_inflated_static_map.grid_pose,
            )
            if self.robot_inflated_static_map.grid[goal_cell[0]][goal_cell[1]] != 0:
                raise Exception(
                    "Goal cell collides with static obstacle cell. This means the scenario file is invalid."
                )
        self.static_obs_grid = world.map
        movable_polygons = {
            uid: e.polygon for uid, e in self.world.dynamic_entities.items()
        }
        self.robot_inflated_grid = copy.deepcopy(world.map).inflate_map_destructive(
            self.circumscribed_radius + self.collision_margin
        )
        self.robot_inflated_grid.update_polygons(movable_polygons)

        # TODO Make sure static and generalist grid share same width and height (occurs naturally if map borders are static, but not otherwise)
        self.robot_inflated_grid.deactivate_entities({self.uid})

        # Initialize social costmap as None for computation in first think
        self._social_costmap = None

    def init_social_costmap(self, ros_publisher: t.Optional["rp.RosPublisher"] = None):
        # Initialize social occupation costmap
        if self.use_social_cost and self._social_costmap is None:
            self._social_costmap = stocg.compute_social_costmap(
                binary_occ_grid=self.static_obs_grid.grid,
                cell_size=self.world.map.cell_size,
                agent_id=self.uid,
                ros_publisher=ros_publisher,
                log_costmaps=self.activate_grids_logging,
                logs_dir=self.logs_dir,
            )

            if ros_publisher:
                ros_publisher.publish_social_costmap(
                    self._social_costmap,
                    self.world.map.cell_size,
                    agent_id=self.uid,
                )

    def are_all_goals_finished(self):
        return not self._navigation_goals and self._goal is None

    def get_current_goal(self):
        return self._goal

    def potential_deadlocks(
        self,
        current_conflicts: t.Set[Conflict],
        plan: "nav_plan.Plan",
        current_step: int,
    ) -> t.Set[Conflict]:
        robot_robot_conflicts = [
            conflict
            for conflict in current_conflicts
            if isinstance(conflict, RobotRobotConflict)
        ]

        result: t.Set[Conflict] = set()

        for past_step, past_conflicts_at_step in plan.conflicts_history.items():
            for conflict in robot_robot_conflicts:
                if conflict in past_conflicts_at_step:
                    # Check if a replan occurred after this conflict was first detected. If so, we have a potential deadlock.
                    for replan_step in plan.steps_with_replan_call:
                        if replan_step >= past_step:
                            result.add(conflict)
                            break
        return result

    def sense(
        self, ref_world: "w.World", last_action_result: ar.ActionResult, step_count: int
    ):
        # Update baseline world representation (polygons)
        Agent.sense(self, ref_world, last_action_result, step_count)

        # Update grid(s)
        self.robot_inflated_grid.update_polygons(
            new_or_updated_polygons={
                uid: self.world.dynamic_entities[uid].polygon
                for uid in self._added_uids.union(self._updated_uids)
                if uid != self.uid
            },
            removed_polygons=self._removed_uids,
        )
        self.robot_inflated_grid.update_polygons(
            new_or_updated_polygons={
                uid: self.world.dynamic_entities[uid].polygon
                for uid in self._added_uids.union(self._updated_uids)
                if uid != self.uid
            },
            removed_polygons=self._removed_uids,
        )

    def think(
        self,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        input: t.Optional[Input] = None,
    ) -> ThinkResult:
        if not self.is_initialized:
            raise Exception("Not initialized")

        # Initialize the social costmap
        if self._social_costmap is None:
            self.init_social_costmap(ros_publisher=ros_publisher)

        # Initialize the goal if one is not already set
        if self._goal is None:
            if self._navigation_goals:
                self._goal = self._navigation_goals.pop(
                    0
                )  # TODO Stop popping goals, use an index
                self._p_opt = nav_plan.Plan(
                    agent_id=self.uid
                )  # pyright: ignore[reportIncompatibleMethodOverride]
                self.goal_to_plans[self._goal] = self._p_opt
            else:
                return ThinkResult(
                    plan=self._p_opt,
                    next_action=ba.GoalsFinished(),
                    goal_pose=None,
                    did_replan=False,
                    agent_id=self.uid,
                )

        next_step = self.full_coordination_strategy(
            w_t=self.world,
            robot_inflated_static_map=self.robot_inflated_static_map,
            robot_inflated_grid=self.robot_inflated_grid,
            agent_id=self.uid,
            goal=self._goal.pose,
            plan=self._p_opt,
            conflict_horizon=self.conflict_horizon,
            try_max=self.replan_count,
            neighborhood=self.neighborhood,
            step_count=self._step_count,
            action_space_reduction=self.action_space_reduction,
            ros_publisher=ros_publisher,
        )

        self._p_opt.save_conflicts(self._step_count, next_step.conflicts)

        if isinstance(next_step.next_action, (ba.GoalSuccess, ba.GoalFailed)):
            self._goal = None

        return next_step

    def must_replan_now(self, conflicts: t.Set[Conflict]):
        for conflict in conflicts:
            if isinstance(conflict, (StolenMovableConflict, RobotObstacleConflict)):
                return True
        return False

    def full_coordination_strategy(
        self,
        *,
        w_t: "w.World",
        robot_inflated_static_map: BinaryOccupancyGrid,
        robot_inflated_grid: BinaryOccupancyGrid,
        agent_id: str,
        goal: Pose2D,
        plan: "nav_plan.Plan",
        conflict_horizon: int,
        try_max: int,
        neighborhood: t.Sequence[GridCellModel],
        step_count: int,
        action_space_reduction: str,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
    ) -> ThinkResult:
        assert agent_id not in robot_inflated_grid.cell_sets

        # If current robot pose is close enough to goal, return Success
        if self.is_goal_reached(robot_pose=self.pose, goal_pose=goal):
            return ThinkResult(
                plan=plan,
                next_action=ba.GoalSuccess(goal),
                goal_pose=goal,
                did_replan=False,
                agent_id=self.uid,
            )

        if plan.is_empty():
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: Absence of plan requires immediate replanning.".format(
                        self.uid
                    ),
                    step_count,
                )
            )
            return self.replan(
                w_t,
                robot_inflated_static_map,
                robot_inflated_grid,
                agent_id,
                goal,
                plan,
                conflict_horizon,
                try_max,
                neighborhood,
                step_count,
                action_space_reduction,
                ros_publisher=ros_publisher,
            )
        if plan.is_evasion_over():
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: Finished evasion sequence, replanning.".format(self.uid),
                    step_count,
                )
            )
            return self.replan(
                w_t,
                robot_inflated_static_map,
                robot_inflated_grid,
                agent_id,
                goal,
                plan,
                conflict_horizon,
                try_max,
                neighborhood,
                step_count,
                action_space_reduction,
                ros_publisher=ros_publisher,
            )

        conflicts = plan.get_conflicts(
            world=w_t,
            robot_inflated_grid=robot_inflated_grid,
            horizon=conflict_horizon,
            exit_early=True,
            conflict_radius=self.grab_start_distance,
        )
        if len(conflicts) > 0:
            self.logger.append(
                utils.NamosimLog(
                    f"Agent {self.uid}: Conflicts detected: {conflicts}",
                    step_count,
                )
            )
            if self.config.parameters.resolve_conflicts is False:
                self.logger.append(
                    utils.NamosimLog(
                        "Agent {}: Failing goal because conflicts where detected and resolve-conflicts is disabled.".format(
                            self.uid
                        ),
                        step_count,
                    )
                )
                return ThinkResult(
                    plan=plan,
                    next_action=ba.GoalFailed(goal),
                    goal_pose=goal,
                    did_replan=False,
                    agent_id=self.uid,
                    conflicts=conflicts,
                )
            return self.handle_conflicts(
                conflicts=conflicts,
                w_t=w_t,
                step_count=step_count,
                robot_inflated_static_map=robot_inflated_static_map,
                robot_inflated_grid=robot_inflated_grid,
                agent_id=agent_id,
                goal=goal,
                plan=plan,
                ros_publisher=ros_publisher,
            )

        if plan.postpone.is_running():
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: No more conflicts, unpostponing current plan.".format(
                        self.uid
                    ),
                    step_count,
                )
            )
            plan.postpone.clear()

        return ThinkResult(
            plan=plan,
            goal_pose=goal,
            did_replan=False,
            agent_id=self.uid,
        )

    def handle_conflicts(
        self,
        conflicts: t.Set[Conflict],
        w_t: "w.World",
        step_count: int,
        robot_inflated_static_map: BinaryOccupancyGrid,
        robot_inflated_grid: BinaryOccupancyGrid,
        agent_id: str,
        goal: Pose2D,
        plan: "nav_plan.Plan",
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
    ) -> ThinkResult:
        potential_deadlocks = self.potential_deadlocks(conflicts, plan, step_count)

        if potential_deadlocks:
            self.logger.append(
                utils.NamosimLog(
                    f"Agent {self.uid}: Potential deadlocks detected", step_count
                )
            )
            if self.config.parameters.resolve_deadlocks:
                if not plan.has_tries_remaining(self.replan_count):
                    self.logger.append(
                        utils.NamosimLog(
                            "Agent {}: Failing goal, no tries remaining to plan an evasion.".format(
                                self.uid
                            ),
                            step_count,
                        )
                    )
                    return ThinkResult(
                        plan=plan,
                        next_action=ba.GoalFailed(goal),
                        goal_pose=goal,
                        did_replan=False,
                        agent_id=self.uid,
                        conflicts=conflicts,
                    )

                if self.deadlock_strategy == "SOCIAL":
                    return self.resolve_deadlocks_social(
                        agent_id=agent_id,
                        w_t=w_t,
                        robot_inflated_grid=robot_inflated_grid,
                        plan=plan,
                        goal=goal,
                        step_count=step_count,
                        potential_deadlocks=potential_deadlocks,
                        conflicts=conflicts,
                        ros_publisher=ros_publisher,
                    )

                return self.resolve_deadlocks_naive(
                    agent_id=agent_id,
                    w_t=w_t,
                    robot_inflated_grid=robot_inflated_grid,
                    plan=plan,
                    goal=goal,
                    step_count=step_count,
                    potential_deadlocks=potential_deadlocks,
                    conflicts=conflicts,
                    ros_publisher=ros_publisher,
                )

            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: Failing goal because deadlocks where detected and resolve-deadlocks is disabled or unavailable.".format(
                        self.uid
                    ),
                    step_count,
                )
            )
            return ThinkResult(
                plan=plan,
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=False,
                agent_id=self.uid,
                conflicts=conflicts,
            )

        if self.must_replan_now(conflicts):
            self.logger.append(
                utils.NamosimLog(
                    f"Agent {self.uid}: Detected conflicts require immediate replanning. Conflicts: {conflicts}",
                    step_count,
                )
            )
            return self.replan(
                w_t=w_t,
                robot_inflated_static_map=robot_inflated_static_map,
                robot_inflated_grid=robot_inflated_grid,
                agent_id=agent_id,
                goal=goal,
                plan=plan,
                conflict_horizon=self.conflict_horizon,
                max_tries=self.replan_count,
                neighborhood=self.neighborhood,
                step_count=step_count,
                action_space_reduction=self.action_space_reduction,
                ros_publisher=ros_publisher,
            )

        if plan.is_postpone_over():
            plan.postpone.clear()
            self.logger.append(
                utils.NamosimLog(
                    f"Agent {self.uid}: Conflicts remain at end of postpone. Replanning.",
                    step_count,
                )
            )
            return self.replan(
                w_t=w_t,
                robot_inflated_static_map=robot_inflated_static_map,
                robot_inflated_grid=robot_inflated_grid,
                agent_id=agent_id,
                goal=goal,
                plan=plan,
                conflict_horizon=self.conflict_horizon,
                max_tries=self.replan_count,
                neighborhood=self.neighborhood,
                step_count=step_count,
                action_space_reduction=self.action_space_reduction,
                ros_publisher=ros_publisher,
            )

        plan.new_postpone(
            t_max=self.max_nb_steps_to_wait,
            t_min=self.min_nb_steps_to_wait,
            step_count=step_count,
            simulation_log=self.logger,
            agent_id=self.uid,
        )

        return ThinkResult(
            plan=plan,
            goal_pose=goal,
            did_postpone=True,
            did_replan=False,
            agent_id=self.uid,
            conflicts=conflicts,
        )

    def resolve_deadlocks_social(
        self,
        *,
        agent_id: str,
        w_t: "w.World",
        plan: "nav_plan.Plan",
        step_count: int,
        goal: Pose2D,
        robot_inflated_grid: BinaryOccupancyGrid,
        potential_deadlocks: t.Set[Conflict],
        conflicts: t.Set[Conflict],
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
    ):
        robot_cells = robot_inflated_grid.rasterize_polygon(
            w_t.dynamic_entities[agent_id].polygon,
        )
        for conflict in potential_deadlocks:
            if isinstance(conflict, RobotRobotConflict):
                robot_cells.update(
                    robot_inflated_grid.rasterize_polygon(
                        w_t.dynamic_entities[conflict.other_agent_id].polygon
                    )
                )

        assert agent_id not in robot_inflated_grid.cell_sets

        evasion_path = self.compute_evasion(
            robot_inflated_grid=robot_inflated_grid,
            w_t=w_t,
            main_agent_id=agent_id,
            potential_deadlocks=potential_deadlocks,
            forbidden_evasion_cells=set(robot_cells),
            ros_publisher=ros_publisher,
            always_evade=plan.is_postpone_over(),
        )

        assert agent_id not in robot_inflated_grid.cell_sets

        if evasion_path:
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: Executing evasion path.".format(self.uid),
                    step_count,
                )
            )
            plan.set_plan(
                nav_plan.Plan(
                    agent_id=self.uid,
                    paths=[evasion_path],
                    goal=goal,
                ),
                step_count,
            )
            return ThinkResult(
                plan=plan,
                goal_pose=goal,
                did_replan=True,
                agent_id=self.uid,
                conflicts=conflicts,
            )
        self.logger.append(
            utils.NamosimLog(
                "Agent {}: I can not or should not evade, postponing...".format(
                    self.uid,
                ),
                step_count,
            )
        )

        plan.new_postpone(
            t_min=self.min_nb_steps_to_wait,
            t_max=self.max_nb_steps_to_wait,
            step_count=step_count,
            simulation_log=self.logger,
            agent_id=self.uid,
        )

        return ThinkResult(
            plan=plan,
            goal_pose=goal,
            did_postpone=True,
            did_replan=False,
            agent_id=self.uid,
            conflicts=conflicts,
        )

    def resolve_deadlocks_naive(
        self,
        *,
        agent_id: str,
        w_t: "w.World",
        plan: "nav_plan.Plan",
        step_count: int,
        goal: Pose2D,
        robot_inflated_grid: BinaryOccupancyGrid,
        potential_deadlocks: t.Set[Conflict],
        conflicts: t.Set[Conflict],
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
    ):
        assert agent_id not in robot_inflated_grid.cell_sets

        evasion_path = self.compute_evasion_nonsocial(
            robot_inflated_grid=robot_inflated_grid,
            w_t=w_t,
            main_agent_id=agent_id,
            potential_deadlocks=potential_deadlocks,
            always_evade=plan.is_postpone_over(),
        )

        assert agent_id not in robot_inflated_grid.cell_sets

        if evasion_path:
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: Executing evasion path.".format(self.uid),
                    step_count,
                )
            )
            plan.set_plan(
                nav_plan.Plan(
                    agent_id=self.uid,
                    paths=[evasion_path],
                    goal=goal,
                ),
                step_count,
            )
            return ThinkResult(
                plan=plan,
                goal_pose=goal,
                did_replan=True,
                agent_id=self.uid,
                conflicts=conflicts,
            )
        self.logger.append(
            utils.NamosimLog(
                "Agent {}: I can not or should not evade, postponing...".format(
                    self.uid,
                ),
                step_count,
            )
        )

        plan.new_postpone(
            t_min=self.min_nb_steps_to_wait,
            t_max=self.max_nb_steps_to_wait,
            step_count=step_count,
            simulation_log=self.logger,
            agent_id=self.uid,
        )

        return ThinkResult(
            plan=plan,
            goal_pose=goal,
            did_postpone=True,
            did_replan=False,
            agent_id=self.uid,
            conflicts=conflicts,
        )

    def replan(
        self,
        w_t: "w.World",
        robot_inflated_static_map: BinaryOccupancyGrid,
        robot_inflated_grid: BinaryOccupancyGrid,
        agent_id: str,
        goal: Pose2D,
        plan: "nav_plan.Plan",
        conflict_horizon: int,
        max_tries: int,
        neighborhood: t.Sequence[GridCellModel],
        step_count: int,
        action_space_reduction: str,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
    ) -> ThinkResult:
        if not plan.has_tries_remaining(max_tries):
            self.logger.append(
                utils.NamosimLog(
                    f"Agent {self.uid}: Failing goal, no tries remaining to plan",
                    step_count,
                )
            )
            return ThinkResult(
                plan=plan,
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=True,
                agent_id=self.uid,
            )

        plan.steps_with_replan_call.add(step_count)

        # I - Compute plan (ignoring dynamic obstacles) and set it to current plan
        dynamic_entities = {
            uid
            for uid, entity in w_t.dynamic_entities.items()
            if (
                (isinstance(entity, Agent) and uid != agent_id)
                or (uid in w_t.entity_to_agent and w_t.entity_to_agent[uid] != agent_id)
            )
        }
        w_t_no_dyn = w_t.light_copy(ignored_entities=dynamic_entities)
        robot_inflated_grid.deactivate_entities(dynamic_entities)
        p = self.select_connect(
            w_t=w_t_no_dyn,
            robot_inflated_static_map=robot_inflated_static_map,
            robot_inflated_grid=robot_inflated_grid,
            r_f=goal,
            neighborhood=neighborhood,
            action_space_reduction=action_space_reduction,
            ros_publisher=ros_publisher,
            prev_list=set(),
        )
        robot_inflated_grid.activate_entities(dynamic_entities)
        plan.set_plan(p, step_count)

        if plan.is_empty():
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: Failing goal, no plan could be found when ignoring dynamic obstacles.".format(
                        self.uid
                    ),
                    step_count,
                )
            )

            return ThinkResult(
                plan=plan,
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=True,
                agent_id=self.uid,
            )

        conflicts = plan.get_conflicts(
            world=w_t,
            robot_inflated_grid=robot_inflated_grid,
            horizon=conflict_horizon,
            conflict_radius=self.conflict_radius,
        )
        if not conflicts:
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: Found a pure NAMO plan without conflicts with dynamic obstacles, "
                    "executing its first step...".format(self.uid),
                    step_count,
                )
            )
            return ThinkResult(
                plan=plan,
                goal_pose=goal,
                did_replan=True,
                agent_id=self.uid,
            )

        if self.config.parameters.resolve_conflicts is False:
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: Failing goal because conflicts where detected and resolve-conflicts is disabled.".format(
                        self.uid
                    ),
                    step_count,
                )
            )
            return ThinkResult(
                plan=plan,
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=True,
                agent_id=self.uid,
                conflicts=conflicts,
            )

        self.logger.append(
            utils.NamosimLog(
                "Agent {}: A new plan has been computed ignoring dynamic "
                "obstacles but has conflicts with them: {}".format(self.uid, conflicts),
                step_count,
            )
        )

        if not (plan.has_tries_remaining(max_tries) and plan.can_even_be_found()):
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: Failing goal, no tries remaining to plan after conflicts "
                    "were found with the plan ignoring dynamic obstacles.".format(
                        self.uid,
                    ),
                    step_count,
                )
            )
            return ThinkResult(
                plan=plan,
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=True,
                agent_id=self.uid,
                conflicts=conflicts,
            )

        # II - Compute plan with conflicting robots set to static obstacles while ignoring non-conflicting robots

        conflicting_entities = set()
        new_static_polygons: t.Dict[str, Polygon] = {}
        for conflict in conflicts:
            if isinstance(conflict, RobotRobotConflict):
                conflicting_entities.add(conflict.other_agent_id)
                conflicting_robot_obstacle = w_t.get_agent_held_obstacle(
                    conflict.other_agent_id
                )
                if conflicting_robot_obstacle is not None:
                    conflicting_entities.add(conflicting_robot_obstacle.uid)

                polygon_id = f"{conflict.other_agent_id}_static"
                new_static_polygons[polygon_id] = (
                    w_t.get_combined_agent_obstacle_polygon(
                        conflict.other_agent_id
                    ).buffer(self.conflict_radius)
                )
            if isinstance(conflict, ConcurrentGrabConflict):
                conflicting_entities.add(conflict.obstacle_uid)
                conflicting_entities.add(conflict.other_agent_id)
                robot_polygon_id = f"{conflict.other_agent_id}_static"
                new_static_polygons[robot_polygon_id] = w_t.dynamic_entities[
                    conflict.other_agent_id
                ].polygon.buffer(self.conflict_radius)
                obstacle_polygon_id = f"{conflict.obstacle_uid}_static"
                new_static_polygons[obstacle_polygon_id] = w_t.dynamic_entities[
                    conflict.obstacle_uid
                ].polygon.buffer(self.conflict_radius)

        # Make a world copy with conflicting robots (and their obstacles!) set to static obstacles
        non_conflicting_entities = dynamic_entities.difference(conflicting_entities)
        new_world = w_t.light_copy(ignored_entities=non_conflicting_entities)

        robot_inflated_grid.deactivate_entities(non_conflicting_entities)
        robot_inflated_static_map.update_polygons(new_static_polygons)

        # Plan using this modified version of the world
        p = self.select_connect(
            w_t=new_world,
            robot_inflated_static_map=robot_inflated_static_map,
            robot_inflated_grid=robot_inflated_grid,
            r_f=goal,
            neighborhood=neighborhood,
            action_space_reduction=action_space_reduction,
            ros_publisher=ros_publisher,
            prev_list=set(),
        )

        # Reset the inflated grid's state
        robot_inflated_static_map.update_polygons(
            removed_polygons=set(new_static_polygons.keys())
        )
        robot_inflated_grid.activate_entities(non_conflicting_entities)

        if p.is_empty():
            self.logger.append(
                utils.NamosimLog(
                    f"Agent {self.uid}: Postponing. Could not find a plan avoiding the conflicting dynamic obstacles of the pure NAMO plan.",
                    step_count,
                )
            )

            plan.new_postpone(
                t_max=self.max_nb_steps_to_wait,
                t_min=self.min_nb_steps_to_wait,
                step_count=step_count,
                simulation_log=self.logger,
                agent_id=self.uid,
            )
            return ThinkResult(
                plan=plan,
                goal_pose=goal,
                did_postpone=True,
                did_replan=True,
                agent_id=self.uid,
                conflicts=conflicts,
            )

        plan.set_plan(p, step_count)
        new_conflicts = set(
            plan.get_conflicts(
                world=w_t,
                robot_inflated_grid=robot_inflated_grid,
                horizon=conflict_horizon,
                conflict_radius=self.conflict_radius,
            )
        )
        for conflict in conflicts:
            if conflict in new_conflicts:
                new_conflicts.remove(conflict)

        if new_conflicts:
            self.logger.append(
                utils.NamosimLog(
                    f"Agent {self.uid}: Postponing because a new plan has been computed avoiding the conflicting dynamic obstacles of the pure NAMO plan, but has other conflicts.",
                    step_count,
                )
            )

            plan.new_postpone(
                t_max=self.max_nb_steps_to_wait,
                t_min=self.min_nb_steps_to_wait,
                step_count=step_count,
                simulation_log=self.logger,
                agent_id=self.uid,
            )
            return ThinkResult(
                plan=plan,
                goal_pose=goal,
                did_replan=True,
                did_postpone=True,
                agent_id=self.uid,
                conflicts=conflicts,
            )

        self.logger.append(
            utils.NamosimLog(
                "Agent {}: Found a new plan that does not have conflicts with the dynamic obstacles "
                "conflicting with the pure NAMO plan, executing its first step...".format(
                    self.uid
                ),
                step_count,
            )
        )

        return ThinkResult(
            plan=plan,
            goal_pose=goal,
            did_replan=True,
            agent_id=self.uid,
        )

    def select_connect(
        self,
        *,
        w_t: "w.World",
        robot_inflated_static_map: BinaryOccupancyGrid,
        robot_inflated_grid: BinaryOccupancyGrid,
        r_f: Pose2D,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        prev_list: t.Set[str],
        ccs_data: connectivity.CCSData | None = None,
        neighborhood: t.Sequence[GridCellModel] = utils.CHESSBOARD_NEIGHBORHOOD,
        action_space_reduction: str = "none",
        depth=0,
    ):
        """
        High Level Planner _select_connect (SC).
        It makes use of _rch and _manip_search in a greedy heuristic search with backtracking.
        It backtracks locally when the object selected by _rch cannot be moved to merge the selected c_1 in c_free.
        It backtracks globally when all the paths identified by _rch from c_1 are unsuccessful.
        SC calls _find_path to determine a transit path from r_t to a contact point, r_t_plus_1 . The existence of the
        path is guaranteed by the choice of contacts in Manip-Search.
        # :param w_t: state of the world at time t
        # :param r_f: goal robot configuration [x, y, theta] in {m, m, theta}
        # :return: None to backtrack, current partial plan otherwise.
        """
        robot = w_t.dynamic_entities[self.uid]
        r_t = robot.pose

        avoid_list: t.Set[t.Tuple[str, str]] = set()

        robot_cell = utils.real_to_grid(
            r_t[0],
            r_t[1],
            robot_inflated_static_map.cell_size,
            robot_inflated_static_map.grid_pose,
        )
        goal_cell = utils.real_to_grid(
            r_f[0],
            r_f[1],
            robot_inflated_static_map.cell_size,
            robot_inflated_static_map.grid_pose,
        )

        simple_path_to_goal = self.find_path(
            robot_pose=r_t,
            goal_pose=r_f,
            robot_inflated_grid=robot_inflated_grid,
            robot_polygon=robot.polygon,
        )
        if simple_path_to_goal:
            # If the goal is in the same free space component as the robot in simulated w_t
            # Orig. condition in pseudo-code is : x^f in C^acc_R(W)
            # TODO FIX COST COMPUTATION TO FIT SAME MODEL AS MANIP SEARCH !

            if ros_publisher:
                ros_publisher.cleanup_robot_observed_world(agent_id=self.uid)
            return nav_plan.Plan(
                paths=[simple_path_to_goal],
                goal=r_f,
                agent_id=self.uid,
            )

        if ccs_data is None:
            ccs_data = connectivity.init_ccs_for_grid(
                robot_inflated_grid.grid,
                robot_inflated_grid.d_width,
                robot_inflated_grid.d_height,
                neighborhood,
            )
        connected_components_grid = ccs_data.grid

        if ros_publisher:
            ros_publisher.publish_connected_components(
                connected_components_grid, w_t.map.cell_size, agent_id=robot.uid
            )

        c_0 = ccs_data.grid[robot_cell[0]][robot_cell[1]]
        prev_list = prev_list if c_0 == 0 else prev_list.union({c_0})
        r_acc_cells = (
            set()
            if robot_inflated_grid.grid[robot_cell[0]][robot_cell[1]] > 0
            else connectivity.bfs_init(
                robot_inflated_grid.grid,
                robot_inflated_grid.d_width,
                robot_inflated_grid.d_height,
                robot_cell,
                neighborhood,
            ).visited
        )

        if len(robot_inflated_grid.cell_to_dynamic_entity_ids(robot_cell)) > 1:
            n_movable = 0
            for uid in robot_inflated_grid.cell_to_dynamic_entity_ids(robot_cell):
                if w_t.dynamic_entities[uid].movability == Movability.MOVABLE:
                    n_movable += 1
            if n_movable > 1:
                return nav_plan.Plan(
                    plan_error="start_cell_in_several_movable_obstacles_error",
                    agent_id=self.uid,
                )

        goal_cell_obstacles = robot_inflated_grid.cell_to_dynamic_entity_ids(goal_cell)

        if len(goal_cell_obstacles) > 1:
            return nav_plan.Plan(
                plan_error="goal_cell_in_several_movable_obstacles_error",
                agent_id=self.uid,
            )

        if len(goal_cell_obstacles) == 1:
            obs_id = list(goal_cell_obstacles)[0]
            if (
                obs_id != self.uid
                and w_t.dynamic_entities[obs_id].movability != Movability.MOVABLE
            ):
                return nav_plan.Plan(
                    plan_error="goal_cell_occupied_by_unmovable_obstacle",
                    agent_id=self.uid,
                )

        if robot_inflated_static_map.grid[goal_cell[0]][goal_cell[1]] > 0:
            return nav_plan.Plan(
                plan_error="goal_cell_in_static_obstacle_error",
                agent_id=self.uid,
            )

        if robot_inflated_static_map.grid[robot_cell[0]][robot_cell[1]] > 0:
            raise Exception(
                "Robot start position is in collision with a static obstacle. This should never happen."
            )

        forbidden_obstacles = {  # Dynamic obstacles are forbidden !
            uid
            for uid, entity in w_t.dynamic_entities.items()
            if (
                (isinstance(entity, Agent) and uid != self.uid)
                or (uid in w_t.entity_to_agent and w_t.entity_to_agent[uid] != self.uid)
            )
        }
        o_1, c_1 = self.rch(
            start_cell=robot_cell,
            goal_cell=goal_cell,
            robot_inflated_static_map=robot_inflated_static_map,
            connected_components_grid=connected_components_grid,
            inflated_robot_grid=robot_inflated_grid,
            avoid_list=avoid_list,
            prev_list=prev_list,
            forbidden_obstacles=forbidden_obstacles,
            ros_publisher=ros_publisher,
            neighborhood=neighborhood,
        )

        while o_1 != "":
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: select_connect(depth={}, avoid_list={}): selected entity {} for manipulation search to reach component {}.".format(
                        robot.uid, depth, avoid_list, w_t.dynamic_entities[o_1].uid, c_1
                    ),
                    self._step_count,
                )
            )
            if action_space_reduction == "none":
                w_t_next, transfer_path = self.manip_search_procedure(
                    w_t=w_t,
                    o_1=o_1,
                    c_1=c_1,
                    ccs_data=ccs_data,
                    r_acc_cells=r_acc_cells,
                    r_f=r_f,
                    robot_inflated_grid=robot_inflated_grid,
                    ros_publisher=ros_publisher,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=True,
                )
            elif action_space_reduction == "only_r_acc":
                w_t_next, transfer_path = self.manip_search_procedure(
                    w_t=w_t,
                    o_1=o_1,
                    c_1=c_1,
                    ccs_data=ccs_data,
                    r_acc_cells=r_acc_cells,
                    r_f=r_f,
                    robot_inflated_grid=robot_inflated_grid,
                    ros_publisher=ros_publisher,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=False,
                )
            elif action_space_reduction == "only_r_acc_then_c_1_x":
                w_t_next, transfer_path = self.manip_search_procedure(
                    w_t=w_t,
                    o_1=o_1,
                    c_1=c_1,
                    ccs_data=ccs_data,
                    r_acc_cells=r_acc_cells,
                    r_f=r_f,
                    robot_inflated_grid=robot_inflated_grid,
                    ros_publisher=ros_publisher,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=False,
                )
                if transfer_path is None:
                    w_t_next, transfer_path = self.manip_search_procedure(
                        w_t=w_t,
                        o_1=o_1,
                        c_1=c_1,
                        ccs_data=ccs_data,
                        r_acc_cells=r_acc_cells,
                        r_f=r_f,
                        robot_inflated_grid=robot_inflated_grid,
                        ros_publisher=ros_publisher,
                        obstacle_can_intrude_r_acc=False,
                        obstacle_can_intrude_c_1_x=True,
                    )
            else:
                raise ValueError(
                    "action_space_reduction variable value is {}, but it should be one of {}".format(
                        action_space_reduction,
                        ["none", "only_r_acc", "only_r_acc_then_c_1_x"],
                    )
                )

            if ros_publisher:
                ros_publisher.cleanup_manip_search(self.uid)

            if transfer_path is not None:
                self.logger.append(
                    utils.NamosimLog(
                        "Agent {}: select_connect: found partial plan manipulating entity {} to reach component {}.".format(
                            robot.uid, w_t.dynamic_entities[o_1].uid, c_1
                        ),
                        self._step_count,
                    )
                )
                prev_cell_sets = robot_inflated_grid.update_polygons(
                    {o_1: w_t_next.dynamic_entities[o_1].polygon}
                )
                future_plan = self.select_connect(
                    w_t=w_t_next,
                    robot_inflated_static_map=robot_inflated_static_map,
                    robot_inflated_grid=robot_inflated_grid,
                    r_f=r_f,
                    ros_publisher=ros_publisher,
                    ccs_data=ccs_data,
                    prev_list=(prev_list if c_1 == "" else prev_list.union({c_1})),
                    neighborhood=neighborhood,
                    action_space_reduction=action_space_reduction,
                    depth=depth + 1,
                )
                robot_inflated_grid.cell_sets_update(prev_cell_sets)
                if not future_plan.plan_error:
                    transit_path = self.find_path(
                        robot_pose=r_t,
                        goal_pose=transfer_path.robot_path.poses[0],
                        robot_inflated_grid=robot_inflated_grid,
                        robot_polygon=robot.polygon,
                    )
                    if not transit_path:
                        raise ValueError(
                            "Failed to find transit path to start of transfer path"
                        )
                    plan_components: t.List[TransitPath | TransferPath] = (
                        [transit_path, transfer_path]
                        if transit_path.actions
                        else [transfer_path]
                    )
                    return nav_plan.Plan(
                        paths=plan_components,
                        goal=r_f,
                        agent_id=self.uid,
                    ).append(future_plan)

            # Extra check for when the goal is in a movable obstacle that we could not find how to move
            if c_1 == "":
                self.logger.append(
                    utils.NamosimLog(
                        "Agent {}: select_connect: did not find a reachable component if manipulating {}.".format(
                            robot.uid, w_t.dynamic_entities[o_1].uid
                        ),
                        self._step_count,
                    )
                )
                break

            avoid_list.add((o_1, c_1))

            o_1, c_1 = self.rch(
                start_cell=robot_cell,
                goal_cell=goal_cell,
                robot_inflated_static_map=robot_inflated_static_map,
                connected_components_grid=connected_components_grid,
                inflated_robot_grid=robot_inflated_grid,
                avoid_list=avoid_list,
                prev_list=prev_list,
                forbidden_obstacles=forbidden_obstacles,
                ros_publisher=ros_publisher,
                neighborhood=neighborhood,
            )

        if ros_publisher:
            ros_publisher.cleanup_robot_observed_world(agent_id=self.uid)
        return nav_plan.Plan(
            plan_error="no_plan_found_error",
            agent_id=self.uid,
        )

    def rch_get_neighbors(
        self,
        current: RCHConfiguration,
        gscore: t.Dict[RCHConfiguration, float],
        close_set: t.Set[RCHConfiguration],
        open_queue: graph_search.PriorityQueue,
        came_from: t.Dict[RCHConfiguration, RCHConfiguration],
        static_obs_grid: BinaryOccupancyGrid,
        connected_components_grid: npt.NDArray[np.int_],
        inflated_robot_grid: BinaryOccupancyGrid,
        avoid_list: t.Set[t.Tuple[str, str]],
        prev_list: t.Set[str],
        g_function: t.Callable[[RCHConfiguration, RCHConfiguration, bool], float],
        forbidden_obstacles: t.Set[str],
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        neighborhood: t.Sequence[GridCellModel] = utils.TAXI_NEIGHBORHOOD,
    ) -> t.Tuple[t.List[RCHConfiguration], t.List[float]]:
        """
        Combined formulation from Stilman's thesis and his article.
        """
        neighbors, tentative_gscores = [], []
        current_gscore = gscore[current]
        path_has_traversed_first_disconnected_comp = current.first_component_uid != ""
        path_has_traversed_first_obstacle = current.first_obstacle_uid != ""

        if (current.first_obstacle_uid, current.first_component_uid) in avoid_list:
            return [], []

        # Filter out cells that are not in the map, and in static obstacles
        candidate_neighbor_cells = utils.get_neighbors_no_coll(
            current.cell,
            static_obs_grid.grid,
            static_obs_grid.d_width,
            static_obs_grid.d_height,
            neighborhood,
        )

        for neighbor_cell in candidate_neighbor_cells:
            neighbor = None
            if path_has_traversed_first_disconnected_comp:
                # Note: This validation was added according to the description in the article about not allowing
                # transitions between two different obstacles or to a cell with several obstacles, though it was not
                # explicit in the pseudocode formulation in Stilman's thesis.
                cur_cell_obs = inflated_robot_grid.cell_to_dynamic_entity_ids(
                    current.cell
                )
                neighbor_cell_obs = inflated_robot_grid.cell_to_dynamic_entity_ids(
                    neighbor_cell
                )

                cur_and_neighbor_not_in_mult_obs = (
                    len(cur_cell_obs) <= 1 and len(neighbor_cell_obs) <= 1
                )
                current_or_neighbor_in_free_space = (
                    len(cur_cell_obs) == 0 or len(neighbor_cell_obs) == 0
                )
                transition_is_valid = (
                    cur_and_neighbor_not_in_mult_obs
                    and (
                        current_or_neighbor_in_free_space
                        or cur_cell_obs == neighbor_cell_obs
                    )
                    and current.first_obstacle_uid not in neighbor_cell_obs
                )
                if transition_is_valid:
                    neighbor = RCHConfiguration(
                        neighbor_cell,
                        current.first_obstacle_uid,
                        current.first_component_uid,
                    )
            else:
                neighbor_cell_component_uid: str = str(
                    connected_components_grid[neighbor_cell[0]][neighbor_cell[1]]
                )

                neighbor_cell_in_free_space = (
                    inflated_robot_grid.grid[neighbor_cell[0]][neighbor_cell[1]] == 0
                )

                if path_has_traversed_first_obstacle:
                    if neighbor_cell_in_free_space:
                        neighbor_cell_not_in_prev_component_nor_avoid_list_nor_in_init_obstacle = (
                            neighbor_cell_component_uid not in prev_list
                            and (
                                current.first_obstacle_uid,
                                neighbor_cell_component_uid,
                            )
                            not in avoid_list
                            and neighbor_cell_component_uid != ""
                        )
                        if neighbor_cell_not_in_prev_component_nor_avoid_list_nor_in_init_obstacle:
                            neighbor = RCHConfiguration(
                                cell=neighbor_cell,
                                first_obstacle_uid=current.first_obstacle_uid,
                                first_component_uid=neighbor_cell_component_uid,
                            )
                        else:
                            # Either the neighbor tries to go back to robot acc. space, or in a (obs., comp.)
                            # combination that has already been explored and for which no manip. could be found
                            pass

                    else:
                        neighbor_cell_obs = (
                            inflated_robot_grid.cell_to_dynamic_entity_ids(
                                neighbor_cell
                            )
                        )
                        if current.first_obstacle_uid in neighbor_cell_obs:
                            neighbor = RCHConfiguration(
                                neighbor_cell, current.first_obstacle_uid, ""
                            )
                        else:
                            # Either the neighbor is in another obstacle, or in multiple, which is forbidden
                            pass
                else:
                    if neighbor_cell_in_free_space:
                        # If no obstacle has been traversed, we are still in the robot acc. space
                        neighbor = RCHConfiguration(neighbor_cell, "", "")
                    else:
                        neighbor_cell_obstacles = (
                            inflated_robot_grid.cell_to_dynamic_entity_ids(
                                neighbor_cell
                            )
                        )
                        if len(neighbor_cell_obstacles) > 0:
                            neighbor_obs_uid = list(neighbor_cell_obstacles)[0]
                            neighbor = RCHConfiguration(
                                neighbor_cell, neighbor_obs_uid, ""
                            )
                        else:
                            # The neighbor is in multiple obstacles, which is forbidden
                            pass
            if (
                neighbor is not None
                and neighbor not in close_set
                and neighbor.first_obstacle_uid not in forbidden_obstacles
                and (neighbor.first_obstacle_uid, neighbor.first_component_uid)
                not in avoid_list
            ):
                neighbors.append(neighbor)
                tentative_gscores.append(
                    current_gscore
                    + g_function(
                        current,
                        neighbor,
                        inflated_robot_grid.grid[neighbor.cell[0]][neighbor.cell[1]]
                        > 0,
                    )
                )

        return neighbors, tentative_gscores

    def rch(
        self,
        start_cell: GridCellModel,
        goal_cell: GridCellModel,
        robot_inflated_static_map: BinaryOccupancyGrid,
        connected_components_grid: npt.NDArray[np.int_],
        inflated_robot_grid: BinaryOccupancyGrid,
        avoid_list: t.Set[t.Tuple[str, str]],
        prev_list: t.Set[str],
        forbidden_obstacles: t.Set[str],
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        neighborhood: t.Sequence[GridCellModel] = utils.TAXI_NEIGHBORHOOD,
    ) -> t.Tuple[str, str]:
        """Performs an A* search from the start cell to the goal cell, allowing
        only certain types of transitions between cells, as decscribed in Benoit
        Renault's papers and thesis. The search returns the IDs of the first obstacle
        and component encountered on the path to the goal.
        """
        if robot_inflated_static_map.grid[start_cell[0]][start_cell[1]] > 0:
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: rch: The robot start cell {} is occupied by a static obstacle.".format(
                        self.uid, start_cell
                    ),
                    self._step_count,
                )
            )
            return "", ""

        if robot_inflated_static_map.grid[goal_cell[0]][goal_cell[1]] > 0:
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: rch: The robot goal cell {} is occupied by a static obstacle.".format(
                        self.uid, goal_cell
                    ),
                    self._step_count,
                )
            )
            return "", ""

        start_obstacles = inflated_robot_grid.cell_to_dynamic_entity_ids(start_cell)
        if (
            len(start_obstacles) > 1
            or len(start_obstacles.intersection(forbidden_obstacles)) > 0
        ):
            assert self.uid not in start_obstacles
            obstacle_ids = {
                uid for uid in inflated_robot_grid.obstacles_uids_in_cell(start_cell)
            }
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: rch: The robot start cell {} in a rch call must always be at most in one obstacle and not a forbidden one, here: {}.".format(
                        self.uid, start_cell, obstacle_ids
                    ),
                    self._step_count,
                )
            )
            return "", ""

        if inflated_robot_grid.grid[goal_cell[0]][goal_cell[1]] > 1:
            obstacle_ids = {
                self.world.dynamic_entities[uid].uid
                for uid in inflated_robot_grid.obstacles_uids_in_cell(goal_cell)
            }
            self.logger.append(
                utils.NamosimLog(
                    "Agent {}: rch: The robot goal cell {} in a rch call must be at most within one movable obstacle, here: {}.".format(
                        self.uid, goal_cell, obstacle_ids
                    ),
                    self._step_count,
                )
            )
            return "", ""

        # TODO Create custom exceptions for above

        sqrt_of_2_times_res = utils.SQRT_OF_2 * inflated_robot_grid.cell_size
        goal_real = utils.grid_to_real(
            goal_cell[0],
            goal_cell[1],
            inflated_robot_grid.cell_size,
            inflated_robot_grid.grid_pose,
        )

        def g_function(
            current: RCHConfiguration,
            neighbor: RCHConfiguration,
            is_transfer: bool = False,
        ) -> float:
            dist = (
                sqrt_of_2_times_res
                if neighbor.cell
                in [
                    (current.cell[0] + i, current.cell[1] + j)
                    for i, j in utils.CHESSBOARD_NEIGHBORHOOD_EXTRAS
                ]
                else inflated_robot_grid.cell_size
            )
            translation_cost = self.translation_factor * dist
            return translation_cost * (
                1.0 if not is_transfer else self.transfer_coefficient
            )

        def h_function(_c: RCHConfiguration, _g: RCHConfiguration) -> float:
            translation_cost = self.translation_factor * utils.euclidean_distance(
                utils.grid_to_real(
                    _c.cell[0],
                    _c.cell[1],
                    inflated_robot_grid.cell_size,
                    inflated_robot_grid.grid_pose,
                ),
                goal_real,
            )
            return translation_cost

        def rch_get_neighbors_instance(
            current: RCHConfiguration,
            gscore: t.Dict[RCHConfiguration, float],
            close_set: t.Set[RCHConfiguration],
            open_queue: graph_search.PriorityQueue,
            came_from: t.Dict[RCHConfiguration, RCHConfiguration],
        ):
            return self.rch_get_neighbors(
                current,
                gscore,
                close_set,
                open_queue,
                came_from,
                robot_inflated_static_map,
                connected_components_grid,
                inflated_robot_grid,
                avoid_list,
                prev_list,
                g_function,
                forbidden_obstacles,
                ros_publisher,
                neighborhood,
            )

        def exit_condition(_current: RCHConfiguration, _goal: RCHConfiguration) -> bool:
            return _current.cell == _goal.cell

        start_obs_id = list(start_obstacles)[0] if len(start_obstacles) > 0 else ""
        start = RCHConfiguration(start_cell, start_obs_id, "")
        goal = RCHConfiguration(
            goal_cell, "", ""
        )  # Note the empty strings are never used, this line is just for coherence

        end_config: RCHConfiguration
        path_found, end_config, _, _, _, _ = graph_search.new_generic_a_star(
            start, goal, exit_condition, rch_get_neighbors_instance, h_function
        )  # type: ignore

        if path_found:
            if end_config.first_obstacle_uid == "":
                raise ValueError(
                    f"Agent {self.uid} Rch found a path where no obstacle needed to be traversed."
                )
            return end_config.first_obstacle_uid, end_config.first_component_uid

        return "", ""

    def manip_search(
        self,
        w_t: "w.World",
        o_1: str,
        c_1: str,
        ccs_data: connectivity.CCSData,
        r_acc_cells: t.Set[GridCellModel],
        r_f: Pose2D,
        robot_inflated_grid: BinaryOccupancyGrid,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        check_new_local_opening_before_global: bool = True,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ):
        if ros_publisher:
            ros_publisher.cleanup_manip_search(self.uid)

        # Initialize manip search simulation world and some shortcut variables
        w_t_next = w_t.light_copy([])

        # if ros_publisher:
        #     ros_publisher.publish_robot_sim_world(w_t_next, self.uid)

        c_1_cells_set = set() if c_1 == "" else ccs_data.ccs[int(c_1)].visited

        res = w_t_next.map.cell_size

        other_entities = [
            entity
            for entity in w_t_next.dynamic_entities.values()
            if entity.uid != self.uid and entity.uid != o_1
        ]
        other_entities_polygons = {
            entity.uid: entity.polygon for entity in other_entities
        }

        robot = w_t_next.dynamic_entities[self.uid]
        agent_id, robot_pose, robot_polygon, agent_id = (
            robot.uid,
            robot.pose,
            robot.polygon,
            robot.uid,
        )

        obstacle = w_t_next.dynamic_entities[o_1]
        obstacle_uid, obstacle_pose, obstacle_polygon = (
            obstacle.uid,
            obstacle.pose,
            obstacle.polygon,
        )
        obstacle_inflation_radius = (
            utils.get_circumscribed_radius(obstacle_polygon) + self.collision_margin
        )

        goal_pose, goal_cell = (
            r_f,
            utils.real_to_grid(r_f[0], r_f[1], res, robot_inflated_grid.grid_pose),
        )

        # Get accessible sampled navigation points around obstacle
        transfer_start_configs = self.get_transfer_start_configs(
            robot_polygon=robot_polygon,
            robot_pose=robot_pose,
            agent_id=agent_id,
            obstacle_uid=obstacle_uid,
            other_entities_polygons=other_entities_polygons,
            robot_inflated_grid=robot_inflated_grid,
            r_acc_cells=r_acc_cells,
            obstacle_pose=obstacle_pose,
            obstacle_polygon=obstacle_polygon,
            ros_publisher=ros_publisher,
        )

        if not transfer_start_configs:
            # If there are no attainable manipulation configurations, exit early
            if ros_publisher:
                ros_publisher.clear_obstacle_grab_poses(agent_id=self.uid)
            self.logger.append(
                utils.NamosimLog(
                    f"Agent {self.uid}: Failed to find manip points around obstacle.",
                    step=0,
                )
            )
            return w_t_next, None

        inflated_grid_by_obstacle = copy.deepcopy(
            self.world.map
        ).inflate_map_destructive(obstacle_inflation_radius)
        inflated_grid_by_obstacle.update_polygons(other_entities_polygons)
        # Only deactivate obstacle cells once transit end and transfer start are computed (grab action)
        robot_inflated_grid.deactivate_entities([obstacle_uid])

        # Use Dijkstra algorithm to compute a transfer path that allows for an opening to be created
        transfer_path = self.rrt_for_manip_search_no_goal(
            grab_configs=transfer_start_configs,
            agent_id=agent_id,
            c1_cells=c_1_cells_set,
            check_for_local_opening=check_new_local_opening_before_global,
            obstacle_uid=obstacle_uid,
            other_entities_polygons=other_entities_polygons,
            map=w_t.map,
            robot_inflated_grid=robot_inflated_grid,
            sorted_cell_to_combined_cost=OrderedDict(),
            ros_publisher=ros_publisher,
            goal_pose=goal_pose,
            goal_cell=goal_cell,
        )

        # Don't forget to update w_t_next with transfer end state
        if transfer_path:
            robot.pose, robot.polygon = (
                transfer_path.robot_path.poses[-1],
                transfer_path.robot_path.polygons[-1],
            )
            obstacle.pose, obstacle.polygon = (
                transfer_path.obstacle_path.poses[-1],
                transfer_path.obstacle_path.polygons[-1],
            )

        if ros_publisher:
            ros_publisher.publish_robot_observed_world(w_t_next, self.uid)
            ros_publisher.clear_obstacle_grab_poses(agent_id=self.uid)

        robot_inflated_grid.activate_entities([obstacle_uid])

        return w_t_next, transfer_path

    def focused_manip_search(
        self,
        w_t: "w.World",
        o_1: str,
        c_1: str,
        ccs_data: connectivity.CCSData,
        r_acc_cells: t.Set[GridCellModel],
        r_f: Pose2D,
        robot_inflated_grid: BinaryOccupancyGrid,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        check_new_local_opening_before_global: bool = True,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ):
        if ros_publisher:
            ros_publisher.cleanup_manip_search(self.uid)

        # Initialize manip search simulation world and some shortcut variables
        w_t_next = w_t.light_copy([])

        # if ros_publisher:
        #     ros_publisher.publish_robot_sim_world(w_t_next, self.uid)

        c_1_cells_set = set() if c_1 == "" else ccs_data.ccs[int(c_1)].visited

        res = w_t_next.map.cell_size

        other_entities = [
            entity
            for entity in w_t_next.dynamic_entities.values()
            if entity.uid != self.uid and entity.uid != o_1
        ]
        other_entities_polygons = {
            entity.uid: entity.polygon for entity in other_entities
        }

        robot = w_t_next.dynamic_entities[self.uid]
        agent_id, robot_pose, agent_id = robot.uid, robot.pose, robot.uid
        robot_polygon = robot.polygon

        obstacle = w_t_next.dynamic_entities[o_1]
        obstacle_uid, obstacle_pose = obstacle.uid, obstacle.pose
        obstacle_polygon = obstacle.polygon
        obstacle_inflation_radius = (
            utils.get_circumscribed_radius(obstacle_polygon) + self.collision_margin
        )

        goal_pose, goal_cell = (
            r_f,
            utils.real_to_grid(r_f[0], r_f[1], res, robot_inflated_grid.grid_pose),
        )

        # Get accessible sampled navigation points around obstacle
        grab_configs = self.get_transfer_start_configs(
            robot_polygon=robot_polygon,
            robot_pose=robot_pose,
            agent_id=agent_id,
            obstacle_uid=obstacle_uid,
            other_entities_polygons=other_entities_polygons,
            robot_inflated_grid=robot_inflated_grid,
            r_acc_cells=r_acc_cells,
            obstacle_pose=obstacle_pose,
            obstacle_polygon=obstacle_polygon,
            ros_publisher=ros_publisher,
        )

        if not grab_configs:
            # If there are no attainable manipulation configurations, exit early
            if ros_publisher:
                ros_publisher.clear_obstacle_grab_poses(agent_id=self.uid)
            self.logger.append(
                utils.NamosimLog(
                    f"Agent {self.uid}: Failed to find manip points around obstacle.",
                    step=0,
                )
            )
            return w_t_next, None

        inflated_grid_by_obstacle = copy.deepcopy(
            self.world.map
        ).inflate_map_destructive(obstacle_inflation_radius)
        inflated_grid_by_obstacle.update_polygons(other_entities_polygons)

        try:
            robot_inflated_grid.deactivate_entities([obstacle_uid])
            map_copy = copy.deepcopy(w_t.map)
            map_copy.update_polygons(other_entities_polygons)
            map_copy.deactivate_entities([obstacle_uid])

            # Get potentially accessible cells for obstacle ordered by associated combined costs
            (
                cells_sorted_by_combined_cost,
                sorted_cell_to_combined_cost,
            ) = self.new_sorted_cells_by_combined_cost(
                inflated_grid_by_obstacle,
                robot_polygon,
                robot_pose,
                obstacle_pose,
                goal_pose,
                ros_publisher=ros_publisher,
            )
            bound_quantile_index = (
                int(
                    round(
                        len(cells_sorted_by_combined_cost)
                        * (1.0 - self.bound_percentage)
                    )
                )
                - 1
            )
            bound_quantile_index = (
                0 if bound_quantile_index < 0 else bound_quantile_index
            )
            bound_quantile = sorted_cell_to_combined_cost[
                cells_sorted_by_combined_cost[bound_quantile_index]
            ]
            transfer_path = self.rrt_for_manip_search_no_goal(
                grab_configs=grab_configs,
                agent_id=agent_id,
                c1_cells=c_1_cells_set,
                check_for_local_opening=check_new_local_opening_before_global,
                obstacle_uid=obstacle_uid,
                other_entities_polygons=other_entities_polygons,
                map=map_copy,
                sorted_cell_to_combined_cost=sorted_cell_to_combined_cost,
                ros_publisher=ros_publisher,
                robot_inflated_grid=robot_inflated_grid,
                goal_cell=goal_cell,
                goal_pose=goal_pose,
            )

            # Don't forget to update w_t_next with transfer end state
            if transfer_path:
                robot.pose, robot.polygon = (
                    transfer_path.robot_path.poses[-1],
                    transfer_path.robot_path.polygons[-1],
                )
                obstacle.pose, obstacle.polygon = (
                    transfer_path.obstacle_path.poses[-1],
                    transfer_path.obstacle_path.polygons[-1],
                )

            if ros_publisher:
                ros_publisher.publish_robot_observed_world(w_t_next, self.uid)
                ros_publisher.clear_obstacle_grab_poses(agent_id=self.uid)

            return w_t_next, transfer_path
        finally:
            robot_inflated_grid.activate_entities([obstacle_uid])

    def dijkstra_for_manip_search(
        self,
        *,
        start: t.List[RobotObstacleConfiguration],
        agent_id: str,
        obstacle_uid: str,
        obstacle_polygon: Polygon,
        other_entities_polygons: t.Dict[str, Polygon],
        robot_inflated_grid: BinaryOccupancyGrid,
        inflated_grid_by_obstacle: BinaryOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        c_1_cells_set: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        check_new_local_opening_before_global: bool,
        overall_goal_pose: Pose2D,
        overall_goal_cell: GridCellModel,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ) -> t.Any:
        def get_neighbors(
            _current: RobotObstacleConfiguration,
            _gscore: t.Dict[RobotObstacleConfiguration, float],
            _close_set: t.Set[RobotObstacleConfiguration],
            _open_queue: t.List[RobotObstacleConfiguration],
            _came_from: t.Dict[
                RobotObstacleConfiguration, RobotObstacleConfiguration | None
            ],
        ):
            return self.get_manip_search_neighbors(
                _current,
                _gscore,
                _close_set,
                _open_queue,
                _came_from,
                start,
                robot_inflated_grid,
                inflated_grid_by_obstacle,
                r_acc_cells,
                ccs_data,
                agent_id,
                obstacle_uid,
                other_entities_polygons,
                ros_publisher,
                obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
            )

        def exit_condition(_current: RobotObstacleConfiguration):
            robot_config_after_release = self.get_robot_config_after_release(
                robot_inflated_grid,
                _current.robot.floating_point_pose,
                _current.robot.polygon,
                agent_id,
                obstacle_uid,
                other_entities_polygons,
            )
            if robot_config_after_release:
                #   3. ... and creates a global opening to c1
                has_new_global_opening = self.is_there_opening_to_c1(
                    check_for_local_opening=check_new_local_opening_before_global,
                    agent_id=agent_id,
                    robot_cell=robot_config_after_release.cell_in_grid,
                    obstacle_uid=obstacle_uid,
                    old_obstacle_polygon=obstacle_polygon,
                    new_obstacle_polygon=_current.obstacle.polygon,
                    other_entities_polygons=other_entities_polygons,
                    robot_inflated_grid=robot_inflated_grid,
                    c1_cells=c_1_cells_set,
                    goal_pose=overall_goal_pose,
                    goal_cell=overall_goal_cell,
                    ros_publisher=ros_publisher,
                    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                )
                if has_new_global_opening:
                    return True
            return False

        return graph_search.new_generic_dijkstra(
            start, exit_condition=exit_condition, get_neighbors=get_neighbors
        )

    def rrt_for_manip_search_no_goal(
        self,
        grab_configs: t.List[RobotObstacleConfiguration],
        agent_id: str,
        obstacle_uid: str,
        map: BinaryOccupancyGrid,
        robot_inflated_grid: BinaryOccupancyGrid,
        other_entities_polygons: t.Dict[str, Polygon],
        c1_cells: t.Set[GridCellModel],
        check_for_local_opening: bool,
        sorted_cell_to_combined_cost: OrderedDict[GridCellModel, float],
        goal_pose: Pose2D,
        goal_cell: GridCellModel,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
    ) -> TransferPath | None:
        c1_cells = copy.deepcopy(c1_cells)
        map = copy.deepcopy(map)
        map.update_polygons(other_entities_polygons)

        for grab_config in grab_configs:
            robot_pose_before_grab = grab_config.prev_robot_pose
            robot_polygon_before_grab = grab_config.prev_robot_polygon
            robot_pose_after_grab = grab_config.robot.floating_point_pose
            robot_polygon_after_grab = grab_config.robot.polygon
            obstacle_pose = grab_config.obstacle.floating_point_pose
            obstacle_polygon = grab_config.obstacle.polygon

            combined_polygon = shapely.ops.unary_union(
                [robot_polygon_after_grab, obstacle_polygon]
            )
            robot_obstacle_polygon: Polygon = t.cast(
                Polygon,
                combined_polygon.convex_hull.buffer(
                    map.cell_size * 2, join_style=JOIN_STYLE.mitre
                ),
            )

            robot_collision_rrt = DiffDriveRRTStar(
                polygon=robot_obstacle_polygon,
                start=robot_pose_after_grab,
                goal=None,
                map=map,
            )

            self.found_opening = False
            self.has_local_openings: t.List[RRTNode] = []
            release_action = ba.Release(
                entity_uid=obstacle_uid,
                distance=-(self.grab_start_distance - self.grab_end_distance),
            )

            def early_exit_condition(
                tree: t.List[RRTNode], node: RRTNode, iteration: int
            ) -> bool:
                robot_pose_after_release = release_action.predict_pose(
                    node.pose, node.pose
                )
                can_release = robot_collision_rrt.collision_free(
                    RRTNode(
                        robot_pose_after_release,
                        None,
                        node.cost,
                    )
                )
                new_obstacle_polygon = rrt.predict_polygon_for_node(
                    node, obstacle_polygon
                )
                robot_cell = map.pose_to_cell(
                    robot_pose_after_release[0], robot_pose_after_release[1]
                )
                has_opening = can_release and self.is_there_opening_to_c1(
                    check_for_local_opening=check_for_local_opening,
                    agent_id=agent_id,
                    robot_cell=robot_cell,
                    obstacle_uid=obstacle_uid,
                    old_obstacle_polygon=obstacle_polygon,
                    new_obstacle_polygon=new_obstacle_polygon,  # TODO : make sure this is correct
                    other_entities_polygons=other_entities_polygons,
                    robot_inflated_grid=robot_inflated_grid,
                    c1_cells=c1_cells,
                    goal_pose=goal_pose,
                    goal_cell=goal_cell,
                    ros_publisher=ros_publisher,
                    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                )
                self.found_opening = self.found_opening or has_opening
                if has_opening:
                    self.has_local_openings.append(node)
                return self.found_opening and iteration > 1000

            rrt = DiffDriveRRTStar(
                polygon=robot_obstacle_polygon,
                start=robot_pose_after_grab,
                goal=None,
                early_exit_condition=early_exit_condition,
                map=map,
            )

            tree = rrt.plan()

            # rrt.plot()
            if tree is not None and len(self.has_local_openings) > 0:
                # Compute best compromise cost among poses with local openings
                best_node = self.has_local_openings[0]
                best_obs_pose = rrt.predict_pose_for_node(best_node, obstacle_pose)
                best_obs_cell = map.pose_to_cell(best_obs_pose[0], best_obs_pose[1])
                # 2 obstacle test only works with no node distance cost added, minimal test only works with + best_compromise.cost there
                best_node_cost = sorted_cell_to_combined_cost.get(
                    best_obs_cell, float("inf")
                )
                for node in self.has_local_openings:
                    obs_pose = rrt.predict_pose_for_node(node, obstacle_pose)
                    obs_cell = map.pose_to_cell(obs_pose[0], obs_pose[1])
                    # and + best_compromise.cost here
                    cost = sorted_cell_to_combined_cost.get(obs_cell, float("inf"))
                    if cost < best_node_cost:
                        best_node_cost = cost
                        best_node = node

                path_nodes = rrt._get_path(best_node)
                # rrt.debug_plan(path_nodes)
                poses = [x.pose for x in path_nodes]
                path = TransitPath.from_poses(
                    poses, robot_polygon_after_grab, robot_pose_after_grab
                )

                robot_poses = [robot_pose_before_grab]
                robot_polygons = [robot_polygon_before_grab]
                obstacle_poses = [obstacle_pose]
                obstacle_polygons = [obstacle_polygon]

                grab_action = grab_config.action
                assert isinstance(grab_action, ba.Grab)

                release_action = ba.Release(
                    entity_uid=obstacle_uid,
                    distance=-(self.grab_start_distance - self.grab_end_distance),
                )
                actions = [grab_action] + path.actions + [release_action]

                for action in actions:
                    next_robot_pose = action.predict_pose(
                        robot_poses[-1], robot_poses[-1]
                    )
                    next_obstacle_pose = action.predict_pose(
                        robot_poses[-1], obstacle_poses[-1]
                    )
                    next_robot_polygon = action.predict_polygon(
                        robot_poses[-1], robot_polygons[-1]
                    )
                    next_obstacle_polygon = action.predict_polygon(
                        robot_poses[-1], obstacle_polygons[-1]
                    )

                    robot_poses.append(next_robot_pose)
                    obstacle_poses.append(next_obstacle_pose)
                    robot_polygons.append(next_robot_polygon)
                    obstacle_polygons.append(next_obstacle_polygon)
                robot_path = RawPath(robot_poses, robot_polygons)
                obstacle_path = RawPath(obstacle_poses, obstacle_polygons)

                if ros_publisher:
                    ros_publisher.publish_robot_rrt(agent_id=agent_id, rrt_nodes=tree)

                return TransferPath(
                    robot_path=robot_path,
                    obstacle_path=obstacle_path,
                    actions=actions,
                    grab_action=grab_action,
                    release_action=release_action,
                    obstacle_uid=obstacle_uid,
                    manip_pose_id=grab_config.manip_pose_id,
                )

    def rrt_for_manip_search(
        self,
        grab_configs: t.List[RobotObstacleConfiguration],
        obstacle_uid: str,
        goal: RobotObstacleConfiguration,
        map: BinaryOccupancyGrid,
        other_entities_polygons: t.Dict[str, Polygon],
    ) -> TransferPath | None:
        map = copy.deepcopy(map)
        map.update_polygons(other_entities_polygons)

        for grab_config in grab_configs:
            robot_pose_before_grab = grab_config.prev_robot_pose
            robot_polygon_before_grab = grab_config.prev_robot_polygon
            robot_pose_after_grab = grab_config.robot.floating_point_pose
            robot_polygon_after_grab = grab_config.robot.polygon
            obstacle_pose = grab_config.obstacle.floating_point_pose
            obstacle_polygon = grab_config.obstacle.polygon
            goal_pose = goal.robot.floating_point_pose

            combined_polygon = shapely.ops.unary_union(
                [robot_polygon_after_grab, obstacle_polygon]
            )
            robot_obstacle_polygon: Polygon = t.cast(
                Polygon,
                combined_polygon.convex_hull.buffer(
                    self.collision_margin, join_style=JOIN_STYLE.mitre
                ),
            )

            rrt = DiffDriveRRTStar(
                polygon=robot_obstacle_polygon,
                start=robot_pose_after_grab,
                goal=goal_pose,
                map=map,
            )

            nodes = rrt.plan()

            # rrt.plot()
            if nodes:
                poses = [x.pose for x in nodes]
                path = TransitPath.from_poses(
                    poses, robot_polygon_after_grab, robot_pose_after_grab
                )

                robot_poses = [robot_pose_before_grab]
                robot_polygons = [robot_polygon_before_grab]
                obstacle_poses = [obstacle_pose]
                obstacle_polygons = [obstacle_polygon]

                grab_action = grab_config.action
                assert isinstance(grab_action, ba.Grab)

                release_action = ba.Release(
                    entity_uid=obstacle_uid,
                    distance=-(self.grab_start_distance - self.grab_end_distance),
                )
                actions = [grab_action] + path.actions + [release_action]

                for action in actions:
                    next_robot_pose = action.predict_pose(
                        robot_poses[-1], robot_poses[-1]
                    )
                    next_obstacle_pose = action.predict_pose(
                        robot_poses[-1], obstacle_poses[-1]
                    )
                    next_robot_polygon = action.predict_polygon(
                        robot_poses[-1], robot_polygons[-1]
                    )
                    next_obstacle_polygon = action.predict_polygon(
                        robot_poses[-1], obstacle_polygons[-1]
                    )

                    robot_poses.append(next_robot_pose)
                    obstacle_poses.append(next_obstacle_pose)
                    robot_polygons.append(next_robot_polygon)
                    obstacle_polygons.append(next_obstacle_polygon)

                robot_path = RawPath(robot_poses, robot_polygons)
                obstacle_path = RawPath(obstacle_poses, obstacle_polygons)

                return TransferPath(
                    robot_path=robot_path,
                    obstacle_path=obstacle_path,
                    actions=actions,
                    grab_action=grab_action,
                    release_action=release_action,
                    obstacle_uid=obstacle_uid,
                    manip_pose_id=grab_config.manip_pose_id,
                )

    def a_star_for_manip_search(
        self,
        start: t.List[RobotObstacleConfiguration],
        goal: RobotObstacleConfiguration,
        agent_id: str,
        obstacle_uid: str,
        obstacle_polygon: Polygon,
        other_entities_polygons: t.Dict[str, Polygon],
        robot_inflated_grid: BinaryOccupancyGrid,
        inflated_grid_by_obstacle: BinaryOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        c1_cells: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        sorted_cell_to_combined_cost: OrderedDict[GridCellModel, float],
        bound_quantile: float,
        check_for_local_opening: bool,
        overall_goal_pose: Pose2D,
        overall_goal_cell: GridCellModel,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ) -> t.Any:
        def get_neighbors(
            _current: RobotObstacleConfiguration,
            _gscore: t.Dict[RobotObstacleConfiguration, float],
            _close_set: t.Set[RobotObstacleConfiguration],
            _open_queue: t.List[RobotObstacleConfiguration],
            _came_from: t.Dict[
                RobotObstacleConfiguration, RobotObstacleConfiguration | None
            ],
        ):
            neighbors, tentative_g_scores = self.get_manip_search_neighbors(
                _current,
                _gscore,
                _close_set,
                _open_queue,
                _came_from,
                start,
                robot_inflated_grid,
                inflated_grid_by_obstacle,
                r_acc_cells,
                ccs_data,
                agent_id,
                obstacle_uid,
                other_entities_polygons,
                ros_publisher,
                obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
            )
            return neighbors, tentative_g_scores

        def heuristic(
            _neighbor: RobotObstacleConfiguration, _goal: RobotObstacleConfiguration
        ):
            return self.h(
                _neighbor.robot.floating_point_pose, _goal.robot.floating_point_pose
            )

        def flexible_exit_condition(
            _current: RobotObstacleConfiguration, _goal: RobotObstacleConfiguration
        ):
            if _current == _goal:
                return True

            if _current.obstacle.cell_in_grid not in sorted_cell_to_combined_cost:
                # TODO Remove this TEMPORARY condition caused by sometimes missing cell in sorted_cell_to_combined_cost
                return False

            current_cell_cc_within_bound = (
                sorted_cell_to_combined_cost[_current.obstacle.cell_in_grid]
                <= bound_quantile
            )

            if current_cell_cc_within_bound:
                next_transit_start_configuration = self.get_robot_config_after_release(
                    robot_inflated_grid,
                    _current.robot.floating_point_pose,
                    _current.robot.polygon,
                    agent_id,
                    obstacle_uid,
                    other_entities_polygons,
                )
                if next_transit_start_configuration:
                    #   3. ... and creates a global opening to c1
                    has_new_global_opening = self.is_there_opening_to_c1(
                        check_for_local_opening=check_for_local_opening,
                        agent_id=agent_id,
                        robot_cell=next_transit_start_configuration.cell_in_grid,
                        obstacle_uid=obstacle_uid,
                        old_obstacle_polygon=obstacle_polygon,
                        new_obstacle_polygon=_current.obstacle.polygon,
                        other_entities_polygons=other_entities_polygons,
                        robot_inflated_grid=robot_inflated_grid,
                        c1_cells=c1_cells,
                        goal_pose=overall_goal_pose,
                        goal_cell=overall_goal_cell,
                        ros_publisher=ros_publisher,
                        neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                    )
                    if has_new_global_opening:
                        return True
            return False

        return graph_search.new_generic_a_star(
            start,
            goal,
            exit_condition=flexible_exit_condition,
            get_neighbors=get_neighbors,
            heuristic=heuristic,
        )

    def get_grab_start_poses(
        self,
        obstacle_polygon: Polygon,
        robot_inflated_grid: BinaryOccupancyGrid,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
    ) -> t.List[Pose2D]:
        """
        For the given obstacle polygon, computes the valid transit end poses and
        corresponding valid transfer start poses:
            - Transfer start poses are at a robot inflation radius distance from the sides, and facing their middle.
            - Transit end poses are a one and a half times the grid resolution away from the obstacle's sides, so that
                their corresponding cell is **always** outside of the inflated obstacle's cells set.
                They also have the same orientation as their corresponding transfer start pose, to make the
                initialization step of the transfer path as safe as possible (the robot only has to drive a bit forward
                to touch the obstacle's side).

        TODO Add two other sampling strategies:
            - points sampled along buffered polygon
            - points sampled along lines parallel to sides, s.t. we have at least a half robot width from endpoints
        :param obstacle_polygon:
        :type obstacle_polygon:
        :param robot_inflated_grid:
        :type robot_inflated_grid:
        :return: the lists of valid transit end poses and corresponding valid transfer start poses
        :rtype: tuple(list(tuple(float, float, float)), list(tuple(float, float, float)))
        """
        grab_start_poses = utils.sample_poses_at_middle_of_inflated_sides(
            obstacle_polygon,
            self.circumscribed_radius + self.grab_start_distance,
        )

        if ros_publisher:
            ros_publisher.clear_obstacle_grab_poses(agent_id=self.uid)
            ros_publisher.publish_obstacle_grab_poses(
                grab_start_poses, agent_id=self.uid
            )

        return grab_start_poses

    def get_transfer_start_configs(
        self,
        robot_polygon: Polygon,
        robot_pose: Pose2D,
        agent_id: str,
        obstacle_uid: str,
        other_entities_polygons: t.Dict[str, Polygon],
        robot_inflated_grid: BinaryOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        obstacle_pose: Pose2D,
        obstacle_polygon: Polygon,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
    ) -> t.List[RobotObstacleConfiguration]:
        transfer_start_configs: t.List[RobotObstacleConfiguration] = []

        grab_start_poses = self.get_grab_start_poses(
            obstacle_polygon, robot_inflated_grid, ros_publisher=ros_publisher
        )

        for manip_pose_id, grab_start_pose in enumerate(grab_start_poses):
            grab_start_cell = utils.real_to_grid(
                grab_start_pose[0],
                grab_start_pose[1],
                robot_inflated_grid.cell_size,
                robot_inflated_grid.grid_pose,
            )

            if grab_start_cell not in r_acc_cells:
                continue

            robot_polygon_before_grab = utils.set_polygon_pose(
                robot_polygon, robot_pose, grab_start_pose
            )

            grab_action = ba.Grab(
                entity_uid=obstacle_uid,
                distance=self.grab_start_distance - self.grab_end_distance,
            )
            robot_pose_after_grab, robot_polygon_after_grab = grab_action.apply(
                grab_start_pose, robot_polygon_before_grab
            )

            collides_with, csv_polygon = collision.get_csv_collisions(
                agent_id=agent_id,
                robot_pose=grab_start_pose,
                other_polygons=other_entities_polygons,
                polygon=robot_polygon_before_grab,
                robot_action=grab_action,
            )

            if obstacle_uid in collides_with:
                collides_with.remove(obstacle_uid)

            if not collides_with:
                transfer_start_config = RobotObstacleConfiguration(
                    robot_pose=robot_pose_after_grab,
                    robot_polygon=robot_polygon_after_grab,
                    robot_fixed_precision_pose=self.pose_to_fixed_precision(
                        robot_pose_after_grab
                    ),
                    robot_cell_in_grid=utils.real_to_grid(
                        robot_pose_after_grab[0],
                        robot_pose_after_grab[1],
                        robot_inflated_grid.cell_size,
                        robot_inflated_grid.grid_pose,
                    ),
                    obstacle_floating_point_pose=obstacle_pose,
                    obstacle_polygon=obstacle_polygon,
                    obstacle_fixed_precision_pose=self.pose_to_fixed_precision(
                        obstacle_pose
                    ),
                    obstacle_cell_in_grid=utils.real_to_grid(
                        obstacle_pose[0],
                        obstacle_pose[1],
                        robot_inflated_grid.cell_size,
                        robot_inflated_grid.grid_pose,
                    ),
                    manip_pose_id=manip_pose_id,
                    action=grab_action,
                    robot_csv_polygon=csv_polygon,
                    obstacle_csv_polygon=obstacle_polygon,
                    prev_robot_pose=grab_start_pose,
                    prev_robot_polygon=robot_polygon_before_grab,
                )
                transfer_start_configs.append(transfer_start_config)

        return transfer_start_configs

    def get_robot_config_after_release(
        self,
        grid: BinaryOccupancyGrid,
        robot_pose: Pose2D,
        robot_polygon: Polygon,
        agent_id: str,
        obstacle_uid: str,
        other_entities_polygons: t.Dict[str, Polygon],
    ) -> RobotConfiguration | None:
        release_action = ba.Release(
            entity_uid=obstacle_uid,
            distance=-(self.grab_start_distance - self.grab_end_distance),
        )
        robot_pose = Pose2D(robot_pose[0], robot_pose[1], robot_pose[2])
        new_robot_pose, new_robot_polygon = release_action.apply(
            robot_pose, robot_polygon
        )
        cell = utils.real_to_grid(
            new_robot_pose[0], new_robot_pose[1], grid.cell_size, grid.grid_pose
        )

        if utils.is_in_matrix(cell, grid.d_width, grid.d_height):
            if grid.grid[cell[0]][cell[1]] > 0:
                # If the robot cell after release is in an obstacle in the grid, return False
                return None
        else:
            # If robot cell outside of grid, return False
            return None

        # Check if robot is still within map bounds
        if not new_robot_polygon.within(grid.aabb_polygon):
            return None

        # Finally, we check dynamic collisions (between init configuration and after-action configuration)
        (
            collides_with,
            csv_polygon,
        ) = collision.get_csv_collisions(
            agent_id=agent_id,
            robot_pose=robot_pose,
            robot_action=release_action,
            polygon=robot_polygon,
            other_polygons=other_entities_polygons,
        )

        if not collides_with:
            new_fixed_precision_pose = self.pose_to_fixed_precision(
                new_robot_pose,
            )
            next_transit_start_configuration = RobotConfiguration(
                floating_point_pose=new_robot_pose,
                polygon=new_robot_polygon,
                cell_in_grid=cell,
                fixed_precision_pose=new_fixed_precision_pose,
                action=release_action,
                csv_polygon=csv_polygon,
            )
            return next_transit_start_configuration

        return None

    def is_there_opening_to_c1(
        self,
        *,
        check_for_local_opening: bool,
        agent_id: str,
        robot_cell: GridCellModel,
        obstacle_uid: str,
        old_obstacle_polygon: Polygon,
        new_obstacle_polygon: Polygon,
        other_entities_polygons: t.Dict[str, Polygon],
        robot_inflated_grid: BinaryOccupancyGrid,
        c1_cells: t.Set[GridCellModel],
        goal_pose: Pose2D,
        goal_cell: GridCellModel,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        neighborhood: t.Iterable[t.Iterable[int]] = utils.CHESSBOARD_NEIGHBORHOOD,
    ):
        """
        Checks if there is a path between robot_cell and a random cell in c_1_cells_set that is not covered by an
        obstacle (especially the one considered for manipulation).
        :return: True if a path is found, False otherwise
        TODO: Add proper return of init_blocking_areas and init_entity_inflated_polygon and save them in caller methods
        """
        if check_for_local_opening:
            has_new_local_opening = check_new_local_opening(
                old_osbtacle_polygon=old_obstacle_polygon,
                new_obstacle_polygon=new_obstacle_polygon,
                other_entities_polygons=other_entities_polygons,
                robot_radius=robot_inflated_grid.inflation_radius,
                goal_pose=goal_pose,
                ros_publisher=ros_publisher,
                agent_id=agent_id,
            )
        else:
            has_new_local_opening = True

        if has_new_local_opening:
            obstacle_initially_deactivated = (
                obstacle_uid in robot_inflated_grid.deactivated_entities_cell_sets
            )
            if obstacle_initially_deactivated:
                robot_inflated_grid.activate_entities({obstacle_uid})
            previous_cell_sets = robot_inflated_grid.update_polygons(
                new_or_updated_polygons={obstacle_uid: new_obstacle_polygon}
            )

            target_c1_cell = None

            if goal_cell in c1_cells or len(c1_cells) == 0:
                target_c1_cell = goal_cell
            else:
                for cell in c1_cells:
                    if robot_inflated_grid.grid[cell[0]][cell[1]] == 0:
                        target_c1_cell = cell
                        break

            if target_c1_cell is None:
                robot_inflated_grid.cell_sets_update(
                    new_or_updated_cell_sets=previous_cell_sets
                )
                if obstacle_initially_deactivated:
                    robot_inflated_grid.deactivate_entities({obstacle_uid})
                return False

            # TODO Evaluate the performance change (particularly compared to Dijkstra search) if A* star had an
            #  unadmissible heuristic to hasten path discovery (or write Best-FS based solely on heuristic)
            has_new_global_opening, _, _, _, _, _ = graph_search.grid_search_a_star(
                start=robot_cell,
                goal=target_c1_cell,
                grid=robot_inflated_grid.grid,
                width=robot_inflated_grid.d_width,
                height=robot_inflated_grid.d_height,
                neighborhood=neighborhood,
                check_diag_neighbors=False,
            )

            cell_is_clear = True
            if target_c1_cell == goal_cell:
                # make sure goal cell is clear
                cell_is_clear = (
                    robot_inflated_grid.grid[target_c1_cell[0]][target_c1_cell[1]] == 0
                )

            robot_inflated_grid.cell_sets_update(
                new_or_updated_cell_sets=previous_cell_sets
            )
            if obstacle_initially_deactivated:
                robot_inflated_grid.deactivate_entities({obstacle_uid})

            return has_new_global_opening and cell_is_clear

        return False

    def get_manip_search_neighbors(
        self,
        current_configuration: RobotObstacleConfiguration,
        gscore: t.Dict[RobotObstacleConfiguration, float],
        close_set: t.Set[RobotObstacleConfiguration],
        open_queue: t.List[RobotObstacleConfiguration],
        came_from: t.Dict[
            RobotObstacleConfiguration, RobotObstacleConfiguration | None
        ],
        start: t.List[RobotObstacleConfiguration],
        robot_inflated_grid: BinaryOccupancyGrid,
        inflated_grid_by_obstacle: BinaryOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        agent_id: str,
        obstacle_uid: str,
        other_entities_polygons: t.Dict[str, Polygon],
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ):
        """
        Creates list of neighbors that are not in close set, do not collide dynamically nor statically
        """
        # TODO Add debug display option for intersections, be it on grid(s) or in between polygons
        neighbors: t.List[RobotObstacleConfiguration] = []
        tentative_g_scores: t.List[float] = []

        for action in self._transfer_movement_actions:
            if isinstance(action, ba.Rotation):
                neighbor_action_opposes_prev_action = (
                    isinstance(current_configuration.action, ba.Rotation)
                    and action.angle == -1.0 * current_configuration.action.angle
                )
                if neighbor_action_opposes_prev_action:
                    continue

                new_robot_pose, new_robot_polygon = action.apply(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.robot.polygon,
                )
                new_obstacle_pose = action.predict_pose(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.obstacle.floating_point_pose,
                )
                new_obstacle_polygon = action.predict_polygon(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.obstacle.polygon,
                )
                extra_g_cost = self.rotation_unit_cost
            elif isinstance(action, ba.Advance):
                # neighbor_action_opposes_prev_action = isinstance(
                #     current_configuration.action, ba.Advance
                # ) and np.sign(action.distance) != np.sign(
                #     current_configuration.action.distance
                # )
                # if neighbor_action_opposes_prev_action:
                #     continue

                new_robot_pose, new_robot_polygon = action.apply(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.robot.polygon,
                )
                new_obstacle_pose = action.predict_pose(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.obstacle.floating_point_pose,
                )
                new_obstacle_polygon = action.predict_polygon(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.obstacle.polygon,
                )
                extra_g_cost = self.translation_unit_cost
            elif isinstance(action, ba.Translation):
                neighbor_action_opposes_prev_action = (
                    isinstance(current_configuration.action, ba.Translation)
                    and action.v[0] == -1.0 * current_configuration.action.v[0]
                    and action.v[1] == -1.0 * current_configuration.action.v[1]
                )
                if neighbor_action_opposes_prev_action:
                    continue

                new_robot_pose, new_robot_polygon = action.apply(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.robot.polygon,
                )
                new_obstacle_pose = action.predict_pose(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.obstacle.floating_point_pose,
                )
                new_obstacle_polygon = action.predict_polygon(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.obstacle.polygon,
                )
                extra_g_cost = self.translation_unit_cost
            else:
                raise TypeError(
                    "action must either be of type Rotation, Advance, or AbsoluteTranslation"
                )

            # First, check whether the new configuration is in close set, if it is, ignore it
            robot_fixed_precision_pose = self.pose_to_fixed_precision(new_robot_pose)
            obstacle_fixed_precision_pose = self.pose_to_fixed_precision(
                new_obstacle_pose,
            )

            if (
                t.cast(
                    RobotObstacleConfiguration,
                    (robot_fixed_precision_pose, obstacle_fixed_precision_pose),
                )
                in close_set
            ):
                continue

            # Then check for collisions, starting at a grid level
            robot_cell_in_grid = utils.real_to_grid(
                new_robot_pose[0],
                new_robot_pose[1],
                robot_inflated_grid.cell_size,
                robot_inflated_grid.grid_pose,
            )
            obstacle_cell_in_grid = utils.real_to_grid(
                new_obstacle_pose[0],
                new_obstacle_pose[1],
                inflated_grid_by_obstacle.cell_size,
                inflated_grid_by_obstacle.grid_pose,
            )

            is_no_longer_in_grid = not (
                utils.is_in_matrix(
                    robot_cell_in_grid,
                    robot_inflated_grid.d_width,
                    robot_inflated_grid.d_height,
                )
                and utils.is_in_matrix(
                    obstacle_cell_in_grid,
                    inflated_grid_by_obstacle.d_width,
                    inflated_grid_by_obstacle.d_height,
                )
            )
            if is_no_longer_in_grid:
                continue
            if (
                robot_inflated_grid.grid[robot_cell_in_grid[0]][robot_cell_in_grid[1]]
                != 0
            ):
                continue
            if (
                inflated_grid_by_obstacle.grid[obstacle_cell_in_grid[0]][
                    obstacle_cell_in_grid[1]
                ]
                != 0
            ):
                continue

            # Check if robot is still within map bounds
            if not new_robot_polygon.within(robot_inflated_grid.aabb_polygon):
                continue

            # Check if obstacle is still within map bounds
            if not new_obstacle_polygon.within(inflated_grid_by_obstacle.aabb_polygon):
                continue

            # Finally, we check dynamic collisions (between init configuration and after-action configuration)
            (
                collides_with,
                robot_csv_polygon,
            ) = collision.get_csv_collisions(
                agent_id=agent_id,
                robot_pose=current_configuration.robot.floating_point_pose,
                robot_action=action,
                polygon=current_configuration.robot.polygon,
                other_polygons=other_entities_polygons,
            )

            if collides_with:
                continue

            # TODO Refactor collision.csv_check_collisions to check for any number of attached polygons or make new function
            (
                collides_with,
                obstacle_csv_polygon,
            ) = collision.get_csv_collisions(
                agent_id=obstacle_uid,
                robot_pose=current_configuration.robot.floating_point_pose,
                robot_action=action,
                other_polygons=other_entities_polygons,
                polygon=current_configuration.obstacle.polygon,
            )

            if collides_with:
                continue

            # If option is activated, check that obstacle intruded the appropriate component(s)
            intrudes = self.polygon_intrudes_components(
                new_obstacle_polygon,
                robot_inflated_grid,
                r_acc_cells,
                ccs_data,
                obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x,
            )
            if intrudes:
                continue

            if (
                len(robot_inflated_grid.cell_to_dynamic_entity_ids(robot_cell_in_grid))
                > 0
            ):
                continue

            # If we are here, then this newly computed neighbor configuration is valid and we must save it
            neighbor_configuration = RobotObstacleConfiguration(
                robot_pose=new_robot_pose,
                robot_polygon=new_robot_polygon,
                robot_fixed_precision_pose=robot_fixed_precision_pose,
                robot_cell_in_grid=robot_cell_in_grid,
                obstacle_floating_point_pose=new_obstacle_pose,
                obstacle_polygon=new_obstacle_polygon,
                obstacle_fixed_precision_pose=obstacle_fixed_precision_pose,
                obstacle_cell_in_grid=obstacle_cell_in_grid,
                action=action,
                prev_robot_pose=current_configuration.robot.floating_point_pose,
                prev_robot_polygon=current_configuration.robot.polygon,
                manip_pose_id=current_configuration.manip_pose_id,
                robot_csv_polygon=robot_csv_polygon,
                obstacle_csv_polygon=obstacle_csv_polygon,
            )

            neighbors.append(neighbor_configuration)
            tentative_g_scores.append(gscore[current_configuration] + extra_g_cost)

        if ros_publisher:
            visited_cells = set([x.obstacle.cell_in_grid for x in close_set])
            ros_publisher.publish_manip_search(
                agent_id=self.uid,
                current=current_configuration.obstacle.cell_in_grid,
                visited_cells=visited_cells,
                cell_size=self.cell_size,
                grid_pose=robot_inflated_grid.grid_pose,
                robot_pose=current_configuration.robot.floating_point_pose,
                obstacle_pose=current_configuration.obstacle.floating_point_pose,
                robot_polygon=current_configuration.robot.polygon,
                obstacle_polygon=current_configuration.obstacle.polygon,
            )

        return neighbors, tentative_g_scores

    @staticmethod
    def polygon_intrudes_components(
        new_obstacle_polygon: Polygon,
        robot_inflated_grid: BinaryOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        obstacle_can_intrude_r_acc: bool,
        obstacle_can_intrude_c_1_x: bool,
    ):
        if obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            return False

        if obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
            new_obstacle_polygon_cells = robot_inflated_grid.rasterize_polygon(
                new_obstacle_polygon.buffer(robot_inflated_grid.inflation_radius),
                fill=False,
            )
            for cell in new_obstacle_polygon_cells:
                if ccs_data.grid[cell[0]][cell[1]] > 0 and cell not in r_acc_cells:
                    return True
        elif not obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            new_obstacle_polygon_cells = robot_inflated_grid.rasterize_polygon(
                new_obstacle_polygon.buffer(robot_inflated_grid.inflation_radius),
                fill=False,
            )
            for cell in new_obstacle_polygon_cells:
                if cell in r_acc_cells:
                    return True
        elif not obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
            return True

        return False

    @staticmethod
    def cell_intrudes_components(
        cell: GridCellModel,
        r_acc_cells: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        obstacle_can_intrude_r_acc: bool,
        obstacle_can_intrude_c_1_x: bool,
    ):
        if obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            return False
        if obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
            if ccs_data.grid[cell[0]][cell[1]] > 0 and cell not in r_acc_cells:
                return True
        elif not obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            if cell in r_acc_cells:
                return True
        elif not obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
            return True

        return False

    @staticmethod
    def deduce_robot_goal_pose(
        robot_manip_pose: Pose2D, obs_init_pose: Pose2D, obs_goal_pose: Pose2D
    ) -> Pose2D:
        translation, rotation = utils.get_translation_and_rotation(
            obs_init_pose, obs_goal_pose
        )
        robot_goal_point = list(
            affinity.translate(
                Point((robot_manip_pose[0], robot_manip_pose[1])),
                xoff=translation[0],
                yoff=translation[1],
            ).coords[0]
        )
        orientation = (robot_manip_pose[2] + rotation) % 360.0
        orientation = orientation if orientation >= 0.0 else orientation + 360.0
        return Pose2D(robot_goal_point[0], robot_goal_point[1], orientation)

    @staticmethod
    def dijkstra_cc_and_cost(
        start_cell: GridCellModel,
        grid: npt.NDArray[t.Any],
        res: float,
        neighborhood: t.Iterable[t.Iterable[int]] = utils.CHESSBOARD_NEIGHBORHOOD,
    ):
        straight_dist = res
        diag_dist = res * utils.SQRT_OF_2
        width, height = grid.shape

        frontier = []
        heapq.heappush(frontier, (0.0, start_cell))
        cost_so_far = {start_cell: 0.0}

        while frontier:
            current = heapq.heappop(frontier)[1]
            for neighbor in utils.get_neighbors_no_coll(
                current, grid, width, height, neighborhood
            ):
                extra_cost = (
                    straight_dist
                    if current[0] == neighbor[0] or current[1] == neighbor[1]
                    else diag_dist
                )
                new_cost = cost_so_far[current] + extra_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor))

        return cost_so_far

    def new_sorted_cells_by_combined_cost(
        self,
        inflated_grid_by_obstacle: BinaryOccupancyGrid,
        robot_polygon: Polygon,
        robot_pose: Pose2D,
        obstacle_pose: Pose2D,
        goal_pose: Pose2D,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
    ):
        if self._social_costmap is None:
            raise Exception("Social costmap uninitialized")

        # Initialize some needed variables
        obstacle_cell = utils.real_to_grid(
            obstacle_pose[0],
            obstacle_pose[1],
            inflated_grid_by_obstacle.cell_size,
            inflated_grid_by_obstacle.grid_pose,
        )

        robot_poly_at_goal = utils.set_polygon_pose(
            robot_polygon, robot_pose, goal_pose
        )

        robot_cells_at_goal = inflated_grid_by_obstacle.rasterize_polygon(
            robot_poly_at_goal
        )

        # Compute set of potentially reachable cells for obstacle and a heuristic cost to join them
        cell_to_cost = self.dijkstra_cc_and_cost(
            obstacle_cell,
            inflated_grid_by_obstacle.grid,
            inflated_grid_by_obstacle.cell_size,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
        )
        for cell in robot_cells_at_goal:
            if cell in cell_to_cost:
                del cell_to_cost[cell]

        # Filter cells where social == -1.
        for cell in list(cell_to_cost.keys()):
            if self._social_costmap[cell[0]][cell[1]] == -1.0:
                del cell_to_cost[cell]

        acc_cells_for_obs, distance_cost = (
            list(cell_to_cost.keys()),
            np.array(list(cell_to_cost.values())),
        )

        social_cost = np.array(
            [self._social_costmap[cell[0]][cell[1]] for cell in acc_cells_for_obs]
        )

        distance_to_goal = np.array(
            [
                utils.euclidean_distance(
                    utils.grid_to_real(
                        cell[0],
                        cell[1],
                        inflated_grid_by_obstacle.cell_size,
                        inflated_grid_by_obstacle.grid_pose,
                    ),
                    goal_pose,
                )
                for cell in acc_cells_for_obs
            ]
        )

        normalized_social_cost = (
            social_cost
            if len(social_cost) == 1
            else (social_cost - np.min(social_cost)) / np.ptp(social_cost)
        )
        normalized_distance_cost = (
            distance_cost
            if len(distance_cost) == 1
            else (distance_cost - np.min(distance_cost)) / np.ptp(distance_cost)
        )
        normalized_distance_to_goal = (
            distance_to_goal
            if len(distance_to_goal) == 1
            else (distance_to_goal - np.min(distance_to_goal))
            / np.ptp(distance_to_goal)
        )

        combined_cost = (
            self.w_social * normalized_social_cost
            + self.w_dist * normalized_distance_cost
            + self.w_goal * normalized_distance_to_goal
        ) / self.w_sum

        sorted_cell_to_combined_cost = OrderedDict(
            sorted(
                zip(acc_cells_for_obs, combined_cost), key=lambda t: t[1], reverse=True
            )
        )

        if ros_publisher:
            ros_publisher.publish_combined_costmap(
                sorted_cell_to_combined_cost,
                inflated_grid_by_obstacle,
                agent_id=self.uid,
            )

        cells_sorted_by_combined_cost = list(sorted_cell_to_combined_cost.keys())

        if self.activate_grids_logging:
            self.log_grids(
                inflated_grid_by_obstacle,
                acc_cells_for_obs,
                normalized_social_cost,
                normalized_distance_cost,
                sorted_cell_to_combined_cost,
                normalized_distance_to_goal,
            )

        return cells_sorted_by_combined_cost, sorted_cell_to_combined_cost

    def log_grids(
        self,
        inflated_grid_by_obstacle: BinaryOccupancyGrid,
        acc_cells_for_obs: t.List[GridCellModel],
        normalized_social_cost: npt.NDArray[t.Any],
        normalized_distance_cost: npt.NDArray[t.Any],
        sorted_cell_to_combined_cost: t.Dict[GridCellModel, float],
        normalized_distance_to_goal: npt.NDArray[t.Any] | None = None,
    ):
        if inflated_grid_by_obstacle:
            stocg.display_or_log(
                grid=np.invert(inflated_grid_by_obstacle.grid.astype(bool)),
                suffix="-obs_inf_grid",
                start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False,
                log_costmaps=True,
                logs_dir=self.logs_dir,
            )

        normalized_social_cost_costmap = np.zeros(
            (inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height)
        )
        normalized_distance_from_obs_costmap = np.zeros(
            (inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height)
        )
        normalized_distance_from_goal_costmap = np.zeros(
            (inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height)
        )

        for i in range(len(acc_cells_for_obs)):
            cell = acc_cells_for_obs[i]
            normalized_social_cost_costmap[cell[0]][cell[1]] = normalized_social_cost[i]
            normalized_distance_from_obs_costmap[cell[0]][cell[1]] = (
                normalized_distance_cost[i]
            )
            if normalized_distance_to_goal is not None:
                normalized_distance_from_goal_costmap[cell[0]][cell[1]] = (
                    normalized_distance_to_goal[i]
                )

        stocg.display_or_log(
            grid=normalized_social_cost_costmap,
            suffix="-n_social_costmap",
            start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            debug_display=False,
            log_costmaps=True,
            logs_dir=self.logs_dir,
        )
        stocg.display_or_log(
            grid=normalized_distance_from_obs_costmap,
            suffix="-n_d_to_obs_costmap",
            start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            debug_display=False,
            log_costmaps=True,
            logs_dir=self.logs_dir,
        )
        if normalized_distance_to_goal is not None:
            stocg.display_or_log(
                grid=normalized_distance_from_goal_costmap,
                suffix="-n_d_to_goal_costmap",
                start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False,
                log_costmaps=True,
                logs_dir=self.logs_dir,
            )

        combined_costmap = np.zeros(
            (inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height)
        )
        for cell, combined_cost in sorted_cell_to_combined_cost.items():
            combined_costmap[cell[0]][cell[1]] = combined_cost
        stocg.display_or_log(
            grid=combined_costmap,
            suffix="-combined_costmap",
            start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            debug_display=False,
            log_costmaps=True,
            logs_dir=self.logs_dir,
        )

    def compute_evasion(
        self,
        robot_inflated_grid: BinaryOccupancyGrid,
        w_t: "w.World",
        main_agent_id: str,
        potential_deadlocks: t.Set[Conflict],
        forbidden_evasion_cells: t.Set[GridCellModel],
        always_evade: bool,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        use_combined_cost: bool = True,
    ) -> EvasionTransitPath | None:
        # Compute evasion for main robot
        main_robot = t.cast(Agent, w_t.dynamic_entities[main_agent_id])

        # The main robot uid should be deactivated in the robot-inflated grid
        assert main_agent_id in robot_inflated_grid.deactivated_entities_cell_sets

        (
            main_robot_evasion_cost,
            main_robot_evasion_path,
        ) = self.compute_evasion_for_one(
            w_t=w_t,
            robot_inflated_grid=robot_inflated_grid,
            robot=main_robot,
            forbidden_evasion_cells=forbidden_evasion_cells,
            ros_publisher=ros_publisher,
            use_combined_cost=use_combined_cost,
            potential_deadlocks=potential_deadlocks,
        )

        if not main_robot_evasion_path:
            return None

        if always_evade:
            return main_robot_evasion_path

        # If this robot is able to evade, it must check if it should by comparing its evasion path with the one of
        # other robots.
        other_robots_uids = {
            potential_deadlock.other_agent_id
            for potential_deadlock in potential_deadlocks
            if isinstance(potential_deadlock, RobotRobotConflict)
        }

        assert main_agent_id not in other_robots_uids

        robot_inflated_grid.update_polygons(
            new_or_updated_polygons={main_agent_id: main_robot.polygon}
        )

        other_robot_evasion_path_max_duration = 0

        min_other_pos_vec = float("inf")
        max_other_robots_evasion_cost = float("-inf")

        for agent_id in other_robots_uids:
            # TODO : Add check to see if other robot has same radius as main robot : if so use the already computed
            #  inflated grid, else compute a corresponding inflated grid (and save for later just in case ?)
            other_robot = t.cast(Agent, w_t.dynamic_entities[agent_id])
            min_other_pos_vec = min(
                min_other_pos_vec, np.linalg.norm(other_robot.pose[:2])
            )

            robot_inflated_grid.deactivate_entities({agent_id})
            robot_inflated_grid.activate_entities({main_agent_id})
            (
                other_robot_evaion_cost,
                _other_robot_evasion_path,
            ) = self.compute_evasion_for_one(
                w_t=w_t,
                robot_inflated_grid=robot_inflated_grid,
                robot=other_robot,
                forbidden_evasion_cells=set(),
                use_combined_cost=use_combined_cost,
                ros_publisher=ros_publisher,
                potential_deadlocks=potential_deadlocks,
            )
            robot_inflated_grid.deactivate_entities({main_agent_id})

            other_robot_exchange_real_path = graph_search.real_to_grid_search_a_star(
                other_robot.pose, main_robot.pose, robot_inflated_grid
            )

            robot_inflated_grid.activate_entities({agent_id})

            other_robot_exchange_path = TransitPath.from_poses(
                other_robot_exchange_real_path,
                other_robot.polygon,
                other_robot.pose,
            )

            max_other_robots_evasion_cost = max(
                max_other_robots_evasion_cost, other_robot_evaion_cost
            )

            other_robot_evasion_path_max_duration = max(
                other_robot_evasion_path_max_duration,
                len(main_robot_evasion_path.actions)
                + len(other_robot_exchange_path.actions),
            )

        main_robot_evasion_path.set_wait(other_robot_evasion_path_max_duration)
        if main_robot_evasion_cost < max_other_robots_evasion_cost:
            return main_robot_evasion_path

        if main_robot_evasion_cost == max_other_robots_evasion_cost:
            ## tie breaking
            if np.linalg.norm(self.pose[:2]) >= min_other_pos_vec:
                return main_robot_evasion_path

        return None  # Wait for others to evade

    def compute_evasion_nonsocial(
        self,
        robot_inflated_grid: BinaryOccupancyGrid,
        w_t: "w.World",
        main_agent_id: str,
        potential_deadlocks: t.Set[Conflict],
        always_evade: bool,
    ) -> EvasionTransitPath | None:
        """Computes an evasion path for the main robot without using social cost"""
        # Compute evasion for main robot
        robot = t.cast(Agent, w_t.dynamic_entities[main_agent_id])

        other_robots_uids = {
            potential_deadlock.other_agent_id
            for potential_deadlock in potential_deadlocks
            if isinstance(potential_deadlock, RobotRobotConflict)
        }

        if not always_evade:
            d = np.linalg.norm(robot.pose[:2])
            for other_agent_id in other_robots_uids:
                other_robot = w_t.agents[other_agent_id]
                d_other = np.linalg.norm(other_robot.pose[:2])
                if d_other > d:  # type: ignore
                    return None

        # The main robot uid should be deactivated in the robot-inflated grid
        assert main_agent_id in robot_inflated_grid.deactivated_entities_cell_sets

        # If the robot is currently holding an object, try to release it first to find a valid transit starting configuration
        transit_configuration_after_release = None
        if w_t.is_holding_obstacle(robot.uid):
            obstacle_uid = w_t.entity_to_agent.inverse[robot.uid]
            obstacle = w_t.dynamic_entities[obstacle_uid]
            other_entities_polygons = {
                uid: e.polygon
                for uid, e in w_t.dynamic_entities.items()
                if uid not in (robot.uid, obstacle_uid)
            }
            transit_configuration_after_release = self.get_robot_config_after_release(
                robot_inflated_grid,
                robot.pose,
                robot.polygon,
                robot.uid,
                obstacle_uid,
                other_entities_polygons,
            )
            if not transit_configuration_after_release:
                # Could not release obstacle during manipulation because no valid transit pose could be found.
                return None

        # Run A* search to find path to a suitable evasion cell
        robot_polygon = robot.polygon
        robot_pose = robot.pose
        robot_cell = utils.real_to_grid(
            robot_pose[0],
            robot_pose[1],
            robot_inflated_grid.cell_size,
            robot_inflated_grid.grid_pose,
        )
        if transit_configuration_after_release:
            robot_polygon = transit_configuration_after_release.polygon
            robot_pose = transit_configuration_after_release.floating_point_pose
            robot_cell = transit_configuration_after_release.cell_in_grid

        def get_min_dist_to_others(cell: GridCellModel):
            min_dist_to_other_robot = float("inf")

            for other_agent_id in other_robots_uids:
                other_robot = w_t.agents[other_agent_id]
                other_robot_cell = utils.real_to_grid(
                    other_robot.pose[0],
                    other_robot.pose[1],
                    robot_inflated_grid.cell_size,
                    robot_inflated_grid.grid_pose,
                )
                min_dist_to_other_robot = min(
                    min_dist_to_other_robot,
                    utils.euclidean_distance(cell, other_robot_cell),
                )
            return min_dist_to_other_robot

        def get_neighbors_for_evasion(
            current: GridCellModel,
            gscore: t.Dict[GridCellModel, float],
            close_set: t.Set[GridCellModel],
            open_queue: t.List[GridCellModel],
            came_from: t.Dict[GridCellModel, GridCellModel | None],
        ) -> t.Tuple[t.List[GridCellModel], t.List[float]]:
            if len(close_set) >= self.max_evasion_cells_to_visit:
                return [], []

            grid = robot_inflated_grid.grid
            neighbors, tentative_gscores = [], []

            current_gscore = gscore[current]
            for i, j in utils.TAXI_NEIGHBORHOOD:
                neighbor = current[0] + i, current[1] + j
                neighbor_is_valid = (
                    neighbor not in close_set
                    and utils.is_in_matrix(
                        cell=neighbor,
                        width=robot_inflated_grid.d_width,
                        height=robot_inflated_grid.d_height,
                    )
                    and grid[neighbor[0]][neighbor[1]] == 0
                )
                if neighbor_is_valid:
                    neighbors.append(neighbor)
                    tentative_gscores.append(current_gscore + 1.0)

            return neighbors, tentative_gscores

        def exit_condition(current: GridCellModel):
            return False

        _, _, came_from, _, gscore, _ = graph_search.new_generic_dijkstra(
            start=robot_cell,
            exit_condition=exit_condition,
            get_neighbors=get_neighbors_for_evasion,
        )

        if not came_from:
            return None

        best_evasion_cell: GridCellModel | None = None
        best_evasion_score = float("-inf")
        for cell in gscore.keys():
            score = get_min_dist_to_others(cell)
            if score > best_evasion_score:
                best_evasion_cell = cell
                best_evasion_score = score

        if best_evasion_cell is None:
            return None

        raw_cell_path = graph_search.reconstruct_path(came_from, best_evasion_cell)
        real_path = utils.grid_path_to_real_path(
            raw_cell_path,
            robot_pose,
            None,
            robot_inflated_grid.cell_size,
            robot_inflated_grid.grid_pose,
        )

        if len(real_path) < 2:
            return None

        evasion_transit_path = EvasionTransitPath.from_poses(
            real_path, robot_polygon, robot_pose, conflicts=potential_deadlocks
        )

        # remember to release the obstacle, if needed
        if transit_configuration_after_release:
            evasion_transit_path.set_transit_configuration_after_release(
                transit_configuration_after_release
            )

        return evasion_transit_path

    def compute_evasion_for_one(
        self,
        *,
        w_t: "w.World",
        robot_inflated_grid: BinaryOccupancyGrid,
        robot: Agent,
        forbidden_evasion_cells: t.Set[GridCellModel],
        potential_deadlocks: t.Set[Conflict],
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        use_combined_cost: bool = True,
    ) -> t.Tuple[float, EvasionTransitPath | None]:
        """Computes an evasion path for a given robot"""
        if self._social_costmap is None:
            raise Exception("No social costmap")

        robot_start_cell = utils.real_to_grid(
            robot.pose[0],
            robot.pose[1],
            robot_inflated_grid.cell_size,
            robot_inflated_grid.grid_pose,
        )
        robot_start_social_cost = self._social_costmap[robot_start_cell[0]][
            robot_start_cell[1]
        ]

        # If the robot is currently holding an object, try to release it first to find a valid transit starting configuration
        transit_configuration_after_release = None
        if robot.uid in w_t.entity_to_agent.inverse:
            obstacle_uid = w_t.entity_to_agent.inverse[robot.uid]
            obstacle = w_t.dynamic_entities[obstacle_uid]
            other_entities_polygons = {
                uid: e.polygon
                for uid, e in w_t.dynamic_entities.items()
                if uid not in (robot.uid, obstacle_uid)
            }
            transit_configuration_after_release = self.get_robot_config_after_release(
                robot_inflated_grid,
                robot.pose,
                robot.polygon,
                robot.uid,
                obstacle_uid,
                other_entities_polygons,
            )
            if not transit_configuration_after_release:
                # Could not release obstacle during manipulation because no valid transit pose could be found.
                return robot_start_social_cost, None

        # Compute shortest path to each cell of current component of robot
        robot_polygon = robot.polygon
        robot_pose = robot.pose
        robot_cell = utils.real_to_grid(
            robot_pose[0],
            robot_pose[1],
            robot_inflated_grid.cell_size,
            robot_inflated_grid.grid_pose,
        )
        if transit_configuration_after_release:
            robot_polygon = transit_configuration_after_release.polygon
            robot_pose = transit_configuration_after_release.floating_point_pose
            robot_cell = transit_configuration_after_release.cell_in_grid

        def get_neighbors_for_evasion(
            current: GridCellModel,
            gscore: t.Dict[GridCellModel, float],
            close_set: t.Set[GridCellModel],
            open_queue: t.List[GridCellModel],
            came_from: t.Dict[GridCellModel, GridCellModel | None],
        ) -> t.Tuple[t.List[GridCellModel], t.List[float]]:
            if len(close_set) >= self.max_evasion_cells_to_visit:
                return [], []

            grid = robot_inflated_grid.grid
            neighbors, tentative_gscores = [], []

            current_gscore = gscore[current]
            for i, j in utils.TAXI_NEIGHBORHOOD:
                neighbor = current[0] + i, current[1] + j
                neighbor_is_valid = (
                    neighbor not in close_set
                    and utils.is_in_matrix(
                        cell=neighbor,
                        width=robot_inflated_grid.d_width,
                        height=robot_inflated_grid.d_height,
                    )
                    and grid[neighbor[0]][neighbor[1]] == 0
                )
                if neighbor_is_valid:
                    neighbors.append(neighbor)
                    tentative_gscores.append(current_gscore + 1.0)

            return neighbors, tentative_gscores

        def exit_condition(current: GridCellModel):
            return False

        # _, _, came_from, _, gscore, _ = graph_search.di(
        #     start=robot_cell,
        #     goal=None,
        #     exit_condition=exit_condition,
        #     get_neighbors=get_neighbors_for_evasion,
        #     heuristic=evasion_heuristic,
        # )  # type: ignore

        _, _, came_from, _, gscore, _ = graph_search.new_generic_dijkstra(
            start=robot_cell,
            exit_condition=exit_condition,
            get_neighbors=get_neighbors_for_evasion,
        )
        if not came_from:
            # If the robot was in an obstacle, no evasion is possible
            return robot_start_social_cost, None

        accessible_cells: t.List[GridCellModel] = []
        social_cost: t.List[float] = []
        distance_cost: t.List[float] = []
        for cell, value in gscore.items():
            if cell not in forbidden_evasion_cells:
                accessible_cells.append(cell)
                social_cost.append(self._social_costmap[cell[0]][cell[1]])  # type: ignore
                distance_cost.append(value)
        social_cost = np.array(social_cost)  # type: ignore
        distance_cost = np.array(distance_cost)  # type: ignore

        if len(social_cost) == 0:
            return robot_start_social_cost, None

        min_social_cost_index = np.argmin(social_cost)
        evasion_cell_cost = social_cost[min_social_cost_index]

        if not use_combined_cost:
            evasion_cell = accessible_cells[min_social_cost_index]
            evasion_cell_cost = social_cost[min_social_cost_index]
        else:
            normalized_social_cost = (social_cost - np.min(social_cost)) / np.ptp(
                social_cost
            )
            normalized_distance_cost = (distance_cost - np.min(distance_cost)) / np.ptp(
                distance_cost
            )
            combined_cost = (
                self.w_social * normalized_social_cost
                + self.w_dist * normalized_distance_cost
            ) / (self.w_social + self.w_dist)
            min_combined_cost_index = np.argmin(combined_cost)
            evasion_cell = accessible_cells[min_combined_cost_index]
            # evasion_cell_cost = combined_cost[min_combined_cost_index]

            if self.activate_grids_logging:
                sorted_cell_to_combined_cost = OrderedDict(
                    sorted(
                        zip(accessible_cells, combined_cost),
                        key=lambda t: t[1],
                        reverse=True,
                    )
                )
                self.log_grids(
                    robot_inflated_grid,
                    accessible_cells,
                    normalized_social_cost,
                    normalized_distance_cost,
                    sorted_cell_to_combined_cost,
                )

            # ros_publisher.publish_combined_costmap(
            #     sorted_cell_to_combined_cost,
            #     robot_inflated_grid,
            #     ns=self.uid,
            # )
            # ros_publisher.cleanup_grid_map(ns=self.uid)

        raw_cell_path = graph_search.reconstruct_path(came_from, evasion_cell)
        real_path = utils.grid_path_to_real_path(
            raw_cell_path,
            robot_pose,
            None,
            robot_inflated_grid.cell_size,
            robot_inflated_grid.grid_pose,
        )

        if len(real_path) < 2:
            return robot_start_social_cost, None

        evasion_transit_path = EvasionTransitPath.from_poses(
            real_path, robot_polygon, robot_pose, conflicts=potential_deadlocks
        )

        if transit_configuration_after_release:
            evasion_transit_path.set_transit_configuration_after_release(
                transit_configuration_after_release
            )

        return evasion_cell_cost, evasion_transit_path

    def h(self, r_i: Pose2D, r_j: Pose2D):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        # rotation_cost = self.rotation_factor * (abs(r_j[2] - r_i[2]) % 180.)
        return translation_cost  # + rotation_cost

    def g(self, r_i: Pose2D, r_j: Pose2D, is_transfer: float = False):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        rotation_cost = self.rotation_factor * abs(r_j[2] - r_i[2])
        return (translation_cost + rotation_cost) * (
            1.0 if not is_transfer else self.transfer_coefficient
        )

    def get_transfer_path_from_configs(
        self,
        transfer_configurations: t.List[RobotObstacleConfiguration],
        robot_config_after_release: RobotConfiguration,
        obstacle_uid: str,
        phys_cost: t.Optional[float] = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ) -> TransferPath | None:
        if len(transfer_configurations) == 0:
            return None

        manip_pose_id: int = transfer_configurations[0].manip_pose_id  # type: ignore

        actions = [
            configuration.action
            for configuration in transfer_configurations
            if configuration.action
        ]

        grab_action = actions[0]
        if not isinstance(grab_action, ba.Grab):
            raise Exception("The first action in a transfer should be a grab")

        release_action = robot_config_after_release.action
        if not isinstance(release_action, ba.Release):
            raise Exception("The last action in a transfer should be a release")

        robot_poses = [transfer_configurations[0].prev_robot_pose]
        robot_polygons = [transfer_configurations[0].prev_robot_polygon]
        obtacle_poses = [transfer_configurations[0].obstacle.floating_point_pose]
        obtacle_polygons = [transfer_configurations[0].obstacle.polygon]

        for config in transfer_configurations:
            # pad to make obstacle path same lenght as robot path
            robot_poses.append(config.robot.floating_point_pose)
            robot_polygons.append(config.robot.polygon)

            # pad to make obstacle path same lenght as robot path
            obtacle_poses.append(config.obstacle.floating_point_pose)
            obtacle_polygons.append(config.obstacle.polygon)

        # after release
        actions.append(release_action)
        robot_poses.append(robot_config_after_release.floating_point_pose)
        robot_polygons.append(robot_config_after_release.polygon)

        # pad to make obstacle path same lenghth as robot path
        obtacle_poses.append(obtacle_poses[-1])
        obtacle_polygons.append(obtacle_polygons[-1])

        robot_path = RawPath(
            poses=robot_poses,
            polygons=robot_polygons,
        )

        obstacle_path = RawPath(
            poses=obtacle_poses,
            polygons=obtacle_polygons,
        )

        return TransferPath(
            robot_path=robot_path,
            obstacle_path=obstacle_path,
            actions=actions,
            grab_action=grab_action,
            release_action=release_action,
            obstacle_uid=obstacle_uid,
            manip_pose_id=manip_pose_id,
            phys_cost=phys_cost,
            social_cost=social_cost,
            weight=weight,
        )

    def copy(self) -> Self:
        """Returns an uninitialized copy instance of this agent."""
        return StilmanRRTStarAgent(
            navigation_goals=copy.deepcopy(self._navigation_goals),
            config=copy.deepcopy(self.config),
            logs_dir=self.logs_dir,
            uid=self.uid,
            polygon=copy.deepcopy(self.polygon),
            style=copy.deepcopy(self.agent_style),
            pose=copy.deepcopy(self.pose),
            sensors=copy.deepcopy(self.sensors),  # type: ignore
            cell_size=self.cell_size,
            collision_margin=self.collision_margin,
            logger=self.logger,
        )

    def pose_to_fixed_precision(self, pose: Pose2D) -> t.Tuple[int, int, int]:
        return utils.real_pose_to_fixed_precision_pose(
            pose,
            1 / self.TRANSLATION_DISCRETIZATION_FACTOR,
            1 / self.ROTATION_DISCRETIZATION_FACTOR,
        )
