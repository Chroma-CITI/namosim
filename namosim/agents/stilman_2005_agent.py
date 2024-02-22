import copy
import heapq
import time
import typing as t
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from aabbtree import AABBTree
from shapely import Polygon
from shapely.geometry import Point
from typing_extensions import Self

import namosim.display.ros2_publisher as rp
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
import namosim.navigation.navigation_plan as nav_plan
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
    UID,
    GridCellModel,
    PoseModel,
    StilmanBehaviorParametersModel,
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
    Path,
    TransferPath,
    TransitPath,
)
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import (
    BinaryInflatedOccupancyGrid,
    BinaryOccupancyGrid,
)
from namosim.world.entity import Movability, Style
from namosim.world.obstacle import Obstacle
from namosim.world.sensors.omniscient_sensor import OmniscientSensor


class Stilman2005Agent(Agent):
    def __init__(
        self,
        *,
        navigation_goals: t.List[PoseModel],
        params: StilmanBehaviorParametersModel,
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
        cell_size: float,
        uid: UID = 0,
    ):
        super().__init__(
            name=name,
            navigation_goals=navigation_goals,
            behavior_type="stilman_2005_behavior",
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
            cell_size=cell_size,
            uid=uid,
        )
        self.params = params

        self.deadlock_strategy: t.Literal["SOCIAL", "DISTANCE"] = (
            "SOCIAL" if params.use_social_cost else "DISTANCE"
        )
        if params.deadlock_strategy == "SOCIAL":
            if not params.use_social_cost:
                raise Exception(
                    "SOCIAL deadlock strategy requires use_social_cost = TRUE"
                )
            self.deadlock_strategy = params.deadlock_strategy
        elif params.deadlock_strategy == "DISTANCE":
            self.deadlock_strategy = params.deadlock_strategy

        self._p_opt: "nav_plan.DynamicPlan"

        # - Original Stilman method configuration parameters
        self.alpha = params.alpha_for_obstacle_choice_heur
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD  # default if bad parameter
        # self.heur_w = parameters["heuristic_cost_for_traversing_obstacle_in_choice_heur"]
        # self.basic_trans_force = parameters["basic_translation_force"]
        # self.basic_rot_moment = parameters["basic_rotation_moment"]
        self.translation_unit_cost = 1.0
        self.rotation_unit_cost = 1.0
        self.transfer_coefficient = 2.0  # Note: MUST ALWAYS BE > 1 !
        # - Robot action space parameters
        self.rotation_unit_angle = params.robot_rotation_unit_angle
        self.translation_unit_length = params.robot_translation_unit_length
        self.forbid_rotations = params.forbid_rotations
        self.translation_factor = (
            self.translation_unit_cost / self.translation_unit_length
        )
        self.rotation_factor = self.rotation_unit_cost / self.rotation_unit_angle
        self.robot_base_drive_type: t.Literal["holonomic", "differential"] = "holonomic"
        self.max_evasion_cells_to_visit = 1000

        # - S-NAMO parameters
        self.use_social_cost = params.use_social_cost
        self.bound_percentage = params.solution_interval_bound_percentage
        if params.manipulation_search_procedure == "DFS":
            if self.use_social_cost:
                self.manip_search_procedure = self.focused_manip_search
            else:
                raise ValueError(
                    "Focused manipulation search requires the use_social_cost variable to be True !"
                )
        elif params.manipulation_search_procedure == "BFS":
            self.manip_search_procedure = self.manip_search
        self.w_social, self.w_obs, self.w_goal = 15.0, 10.0, 2.0
        self.w_sum = self.w_social + self.w_obs + self.w_goal
        self.distance_to_obs_cost_is_realistic = True

        # - Extra performance parameters
        self.check_new_local_opening_before_global = (
            params.check_new_local_opening_before_global
        )
        self.activate_grids_logging = params.activate_grids_logging
        self._social_costmap: npt.NDArray[np.float_] | None = None
        self.is_first_transfer_step = False
        self.check_horizon = 10
        self.angular_tolerance = 0.1
        self.min_nb_steps_to_wait = 5
        self.max_nb_steps_to_wait = 20
        self.replan_count = 20

        if self.forbid_rotations:
            self._rot_angles = np.array([])
        else:
            self._rot_angles = np.array(
                [self.rotation_unit_angle, -self.rotation_unit_angle]
            )
        self._all_rot_angles = self.rotation_unit_angle * np.array(
            range(1, 360 // int(self.rotation_unit_angle))
        )

        if self.robot_base_drive_type == "differential":  # pyright: ignore[reportUnnecessaryComparison]
            self._transfer_movement_actions: t.List[ba.Action] = [
                ba.Advance(distance=self.translation_unit_length),
                ba.Advance(distance=-self.translation_unit_length),
            ]
        elif self.robot_base_drive_type == "holonomic":
            self._transfer_movement_actions: t.List[ba.Action] = [
                ba.AbsoluteTranslation((self.translation_unit_length, 0.0)),
                ba.AbsoluteTranslation((-self.translation_unit_length, 0.0)),
                ba.AbsoluteTranslation((0.0, self.translation_unit_length)),
                ba.AbsoluteTranslation((0.0, -self.translation_unit_length)),
            ]
        for rot_angle in self._rot_angles:
            self._transfer_movement_actions.append(ba.Rotation(rot_angle))

    def init(self, world: "w.World"):
        super().init(world)
        self.trans_mult = 100.0
        self.rot_mult = 100.0
        self.position_tolerance = self.world.discretization_data.res / 5.0

        # Initialize movability status of obstacles
        for entity in self.world.entities.values():
            if entity.movability != Movability.STATIC:
                entity.movability = self.deduce_movability(entity.type_)

        self.action_space_reduction = (
            "only_r_acc_then_c_1_x"  # ['none', 'only_r_acc', 'only_r_acc_then_c_1_x']
        )

        # Initialize static obstacles occupation grid, since it is not supposed to change
        static_obs_polygons = {
            uid: entity.polygon
            for uid, entity in self.world.entities.items()
            if (isinstance(entity, Obstacle) and entity.movability == Movability.STATIC)
        }

        self.robot_max_inflation_radius = utils.get_circumscribed_radius(self.polygon)
        self.static_obs_inf_grid = BinaryInflatedOccupancyGrid(
            static_obs_polygons,
            self.world.discretization_data.res,
            self.robot_max_inflation_radius,
            neighborhood=self.neighborhood,
        )

        # check that goals are valid (i.e., not in static obstacles)
        for pose in self._navigation_goals:
            goal_cell = utils.real_to_grid(
                pose[0],
                pose[1],
                self.static_obs_inf_grid.res,
                self.static_obs_inf_grid.grid_pose,
            )
            if self.static_obs_inf_grid.grid[goal_cell[0]][goal_cell[1]] != 0:
                raise Exception(
                    "Goal cell collides with static obstacle cell. This means the scenario file is invalid."
                )
        self.static_obs_grid = BinaryOccupancyGrid(
            static_obs_polygons,
            self.world.discretization_data.res,
            neighborhood=self.neighborhood,
            params=self.static_obs_inf_grid.params,
        )

        all_entities_polygons = {
            uid: e.polygon for uid, e in self.world.entities.items()
        }

        self.inflated_grid_by_robot = BinaryInflatedOccupancyGrid(
            all_entities_polygons,
            self.world.discretization_data.res,
            self.robot_max_inflation_radius,
            neighborhood=self.neighborhood,
            params=self.static_obs_inf_grid.params,
        )

        # TODO Make sure static and generalist grid share same width and height (occurs naturally if map borders are static, but not otherwise)
        self.inflated_grid_by_robot.deactivate_entities({self.uid})

        # Initialize social costmap as None for computation in first think
        self._social_costmap = None

        # Init first goal
        if self._q_goal is None:
            if self._navigation_goals:
                self._q_goal = self._navigation_goals.pop(
                    0
                )  # TODO Stop popping goals, use an index
                self._p_opt = nav_plan.DynamicPlan(robot_uid=self.uid)
                self.goal_to_plans[self._q_goal] = self._p_opt
            else:
                return ba.GoalsFinished()

    def init_social_costmap(self, ros_publisher: "rp.RosPublisher"):
        # Initialize social occupation costmap
        if self.use_social_cost and self._social_costmap is None:
            self._social_costmap = stocg.compute_social_costmap(
                self.static_obs_grid.grid,
                self.world.discretization_data.res,
                ros_publisher=ros_publisher,
                log_costmaps=self.activate_grids_logging,
                logs_dir=self.logs_dir,
                ns=self.name,
            )

            ros_publisher.publish_social_grid_map(
                self._social_costmap,
                self.world.discretization_data.res,
                ns=self.name,
            )

    def are_all_goals_finished(self):
        return not self._navigation_goals and self._q_goal is None

    def is_goal_success(self, q_r: PoseModel):
        if self._q_goal is None:
            raise Exception("No goal has been set")

        return all(
            [
                utils.is_close(
                    q_r[0], self._q_goal[0], rel_tol=self.position_tolerance
                ),
                utils.is_close(
                    q_r[1], self._q_goal[1], rel_tol=self.position_tolerance
                ),
                utils.angle_is_close(
                    q_r[2], self._q_goal[2], rel_tol=self.angular_tolerance
                ),
            ]
        )

    def get_current_goal(self):
        return self._q_goal

    def potential_deadlocks(
        self,
        current_conflicts: t.List[Conflict],
        dynamic_plan: "nav_plan.DynamicPlan",
        current_step: int,
    ) -> t.Set[Conflict]:
        robot_robot_conflicts = [
            conflict
            for conflict in current_conflicts
            if isinstance(conflict, RobotRobotConflict)
        ]

        result: t.Set[Conflict] = set()

        for past_step, past_conflicts_at_step in dynamic_plan.conflicts_history.items():
            for conflict in robot_robot_conflicts:
                if conflict in past_conflicts_at_step:
                    # Check if a replan occurred after this conflict was first detected. If so, we have a potential deadlock.
                    for replan_step in dynamic_plan.steps_with_replan_call:
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
        self.inflated_grid_by_robot.update(
            new_or_updated_polygons={
                uid: self.world.entities[uid].polygon
                for uid in self._added_uids.union(self._updated_uids)
                if uid != self.uid
            },
            removed_polygons=self._removed_uids,
        )

    def think(
        self, ros_publisher: "rp.RosPublisher", input: t.Optional[Input] = None
    ) -> ThinkResult:
        if not self.is_initialized:
            raise Exception("Not initialized")

        # Initialize the social costmap
        if self._social_costmap is None:
            self.init_social_costmap(ros_publisher=ros_publisher)

        # Initialize the goal if one is not already set
        if self._q_goal is None:
            if self._navigation_goals:
                self._q_goal = self._navigation_goals.pop(
                    0
                )  # TODO Stop popping goals, use an index
                self._p_opt = nav_plan.DynamicPlan(robot_uid=self.uid)  # pyright: ignore[reportIncompatibleMethodOverride]
                self.goal_to_plans[self._q_goal] = self._p_opt
            else:
                return ThinkResult(
                    next_action=ba.GoalsFinished(),
                    goal_pose=None,
                    did_replan=False,
                    robot_name=self.name,
                )

        next_step = self.full_coordination_strategy(
            w_t=self.world,
            static_obs_inf_grid=self.static_obs_inf_grid,
            inflated_grid_by_robot=self.inflated_grid_by_robot,
            robot_uid=self.uid,
            goal=self._q_goal,
            plan=self._p_opt,
            fov=self.check_horizon,
            try_max=self.replan_count,
            t_min=self.min_nb_steps_to_wait,
            t_max=self.max_nb_steps_to_wait,
            pos_tol=self.position_tolerance,
            ang_tol=self.angular_tolerance,
            neighborhood=self.neighborhood,
            step_count=self._step_count,
            trans_mult=self.trans_mult,
            rot_mult=self.rot_mult,
            action_space_reduction=self.action_space_reduction,
            ros_publisher=ros_publisher,
        )

        self._p_opt.save_conflicts(self._step_count)

        if isinstance(next_step.next_action, (ba.GoalSuccess, ba.GoalFailed)):
            self._q_goal = None

        return next_step

    def must_replan_now(self, conflicts: t.List[Conflict]):
        for conflict in conflicts:
            if isinstance(conflict, (StolenMovableConflict, RobotObstacleConflict)):
                return True
        return False

    def full_coordination_strategy(
        self,
        *,
        w_t: "w.World",
        static_obs_inf_grid: BinaryInflatedOccupancyGrid,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        robot_uid: UID,
        goal: PoseModel,
        plan: "nav_plan.DynamicPlan",
        fov: int,
        try_max: int,
        t_min: int,
        t_max: int,
        pos_tol: float,
        ang_tol: float,
        neighborhood: t.Sequence[GridCellModel],
        step_count: int,
        trans_mult: float,
        rot_mult: float,
        action_space_reduction: str,
        ros_publisher: "rp.RosPublisher",
    ) -> ThinkResult:
        assert robot_uid not in inflated_grid_by_robot.cells_sets

        # If current robot pose is close enough to goal, return Success
        if self.is_goal_reached(w_t.entities[robot_uid].pose, goal, pos_tol, ang_tol):
            return ThinkResult(
                next_action=ba.GoalSuccess(goal),
                goal_pose=goal,
                did_replan=False,
                robot_name=self.name,
            )

        if plan.is_empty():
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Absence of plan requires immediate replanning.".format(
                        self.name
                    ),
                    step_count,
                )
            )
            return self.replan(
                w_t,
                static_obs_inf_grid,
                inflated_grid_by_robot,
                robot_uid,
                goal,
                plan,
                fov,
                try_max,
                t_min,
                t_max,
                pos_tol,
                ang_tol,
                neighborhood,
                step_count,
                trans_mult,
                rot_mult,
                action_space_reduction,
                ros_publisher=ros_publisher,
            )
        if plan.is_evasion_over():
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Finished evasion sequence, replanning.".format(
                        self.name
                    ),
                    step_count,
                )
            )
            return self.replan(
                w_t,
                static_obs_inf_grid,
                inflated_grid_by_robot,
                robot_uid,
                goal,
                plan,
                fov,
                try_max,
                t_min,
                t_max,
                pos_tol,
                ang_tol,
                neighborhood,
                step_count,
                trans_mult,
                rot_mult,
                action_space_reduction,
                ros_publisher=ros_publisher,
            )

        conflicts = plan.get_conflicts(
            world=w_t,
            inflated_grid_by_robot=inflated_grid_by_robot,
            check_horizon=fov,
            ros_publisher=ros_publisher,
            robot_name=self.name,
            exit_early_for_any_conflict=True,
        )
        if not conflicts:
            if plan.timer.is_running and plan.timer.is_timer_over(step_count):
                self.logger.append(
                    utils.BasicLog(
                        "Agent {}: No more conflicts, unpostponing current plan.".format(
                            self.name
                        ),
                        step_count,
                    )
                )
                plan.timer.is_running = False
                plan.unpostponements_history.append(step_count)
            return ThinkResult(
                next_action=plan.pop_next_action(),
                goal_pose=goal,
                did_replan=False,
                robot_name=self.name,
            )  # Normal case, don't log
        if self.params.resolve_conflicts is False:
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Failing goal because conflicts where detected and resolve-conflicts is disabled.".format(
                        self.name
                    ),
                    step_count,
                )
            )
            return ThinkResult(
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=False,
                robot_name=self.name,
                conflicts=conflicts,
            )

        # Detect and resolve deadlocks
        potential_deadlocks = self.potential_deadlocks(conflicts, plan, step_count)

        if potential_deadlocks:
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Potential deadlocks detected: {}.".format(
                        self.name, potential_deadlocks
                    ),
                    step_count,
                )
            )

            if self.params.resolve_deadlocks:
                if plan.timer.is_running and not plan.timer.is_timer_over(step_count):
                    return ThinkResult(
                        next_action=ba.Wait(),
                        goal_pose=goal,
                        did_replan=False,
                        robot_name=self.name,
                        conflicts=conflicts,
                    )

                if not plan.has_tries_remaining(self.replan_count):
                    self.logger.append(
                        utils.BasicLog(
                            "Agent {}: Failing goal, no tries remaining to plan an evasion.".format(
                                self.name
                            ),
                            step_count,
                        )
                    )
                    return ThinkResult(
                        next_action=ba.GoalFailed(goal),
                        goal_pose=goal,
                        did_replan=False,
                        robot_name=self.name,
                        conflicts=conflicts,
                    )

                if self.deadlock_strategy == "SOCIAL":
                    return self.resolve_deadlocks_social(
                        robot_uid=robot_uid,
                        w_t=w_t,
                        robot_inflated_grid=inflated_grid_by_robot,
                        plan=plan,
                        goal=goal,
                        step_count=step_count,
                        potential_deadlocks=potential_deadlocks,
                        conflicts=conflicts,
                        ros_publisher=ros_publisher,
                    )

                return self.resolve_deadlocks_naive(
                    robot_uid=robot_uid,
                    w_t=w_t,
                    robot_inflated_grid=inflated_grid_by_robot,
                    plan=plan,
                    goal=goal,
                    step_count=step_count,
                    potential_deadlocks=potential_deadlocks,
                    conflicts=conflicts,
                    ros_publisher=ros_publisher,
                )

            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Failing goal because deadlocks where detected and resolve-deadlocks is disabled or unavailable.".format(
                        self.name
                    ),
                    step_count,
                )
            )
            return ThinkResult(
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=False,
                robot_name=self.name,
                conflicts=conflicts,
            )

        if not self.must_replan_now(conflicts):
            return ThinkResult(
                next_action=plan.new_postpone(
                    t_min,
                    t_max,
                    step_count,
                    conflicts,
                    self.logger,
                    self.name,
                ),
                goal_pose=goal,
                did_postpone=True,
                did_replan=False,
                robot_name=self.name,
                conflicts=conflicts,
            )

        self.logger.append(
            utils.BasicLog(
                "Agent {}: Detected conflicts require immediate replanning. Conflicts: {}".format(
                    self.name, conflicts
                ),
                step_count,
            )
        )
        return self.replan(
            w_t,
            static_obs_inf_grid,
            inflated_grid_by_robot,
            robot_uid,
            goal,
            plan,
            fov,
            try_max,
            t_min,
            t_max,
            pos_tol,
            ang_tol,
            neighborhood,
            step_count,
            trans_mult,
            rot_mult,
            action_space_reduction,
            ros_publisher=ros_publisher,
        )

    def resolve_deadlocks_social(
        self,
        *,
        robot_uid: UID,
        w_t: "w.World",
        plan: "nav_plan.DynamicPlan",
        step_count: int,
        goal: PoseModel,
        robot_inflated_grid: BinaryInflatedOccupancyGrid,
        potential_deadlocks: t.Set[Conflict],
        conflicts: t.List[Conflict],
        ros_publisher: "rp.RosPublisher",
    ):
        robot_cells = utils.accurate_rasterize_in_grid(
            w_t.entities[robot_uid].polygon,
            robot_inflated_grid.res,
            robot_inflated_grid.grid_pose,
            robot_inflated_grid.d_width,
            robot_inflated_grid.d_height,
            fill=True,
        )
        plan.forbidden_evasion_cells.update(set(robot_cells))
        plan.update_count += 1

        assert robot_uid not in robot_inflated_grid.cells_sets

        evasion_path = self.compute_evasion(
            inflated_grid_by_robot=robot_inflated_grid,
            w_t=w_t,
            main_robot_uid=robot_uid,
            potential_deadlocks=potential_deadlocks,
            forbidden_evasion_cells=plan.forbidden_evasion_cells,
            ros_publisher=ros_publisher,
        )

        assert robot_uid not in robot_inflated_grid.cells_sets

        if evasion_path:
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Executing evasion path.".format(self.name),
                    step_count,
                )
            )
            plan.update_plan(
                nav_plan.Plan(
                    robot_uid=self.uid,
                    path_components=[evasion_path],
                    goal=goal,
                ),
                step_count,
            )
            next_action = plan.pop_next_action()
            return ThinkResult(
                next_action=next_action,
                goal_pose=goal,
                did_replan=True,
                robot_name=self.name,
                conflicts=conflicts,
            )
        self.logger.append(
            utils.BasicLog(
                "Agent {}: I can not or should not evade, postponing...".format(
                    self.name,
                ),
                step_count,
            )
        )
        return ThinkResult(
            next_action=plan.new_postpone(
                t_min=self.min_nb_steps_to_wait,
                t_max=self.max_nb_steps_to_wait,
                step_count=step_count,
                conflicts=conflicts,
                simulation_log=self.logger,
                robot_name=self.name,
            ),
            goal_pose=goal,
            did_postpone=True,
            did_replan=False,
            robot_name=self.name,
            conflicts=conflicts,
        )

    def resolve_deadlocks_naive(
        self,
        *,
        robot_uid: UID,
        w_t: "w.World",
        plan: "nav_plan.DynamicPlan",
        step_count: int,
        goal: PoseModel,
        robot_inflated_grid: BinaryInflatedOccupancyGrid,
        potential_deadlocks: t.Set[Conflict],
        conflicts: t.List[Conflict],
        ros_publisher: "rp.RosPublisher",
    ):
        plan.update_count += 1

        assert robot_uid not in robot_inflated_grid.cells_sets

        evasion_path = self.compute_evasion_nonsocial(
            inflated_grid_by_robot=robot_inflated_grid,
            w_t=w_t,
            main_robot_uid=robot_uid,
            potential_deadlocks=potential_deadlocks,
        )

        assert robot_uid not in robot_inflated_grid.cells_sets

        if evasion_path:
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Executing evasion path.".format(self.name),
                    step_count,
                )
            )
            plan.update_plan(
                nav_plan.Plan(
                    robot_uid=self.uid,
                    path_components=[evasion_path],
                    goal=goal,
                ),
                step_count,
            )
            next_action = plan.pop_next_action()
            return ThinkResult(
                next_action=next_action,
                goal_pose=goal,
                did_replan=True,
                robot_name=self.name,
                conflicts=conflicts,
            )
        self.logger.append(
            utils.BasicLog(
                "Agent {}: I can not or should not evade, postponing...".format(
                    self.name,
                ),
                step_count,
            )
        )
        return ThinkResult(
            next_action=plan.new_postpone(
                t_min=self.min_nb_steps_to_wait,
                t_max=self.max_nb_steps_to_wait,
                step_count=step_count,
                conflicts=conflicts,
                simulation_log=self.logger,
                robot_name=self.name,
            ),
            goal_pose=goal,
            did_postpone=True,
            did_replan=False,
            robot_name=self.name,
            conflicts=conflicts,
        )

    def replan(
        self,
        w_t: "w.World",
        static_obs_inf_grid: BinaryInflatedOccupancyGrid,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        robot_uid: UID,
        goal: PoseModel,
        plan: "nav_plan.DynamicPlan",
        fov: int,
        max_tries: int,
        t_min: int,
        t_max: int,
        pos_tol: float,
        ang_tol: float,
        neighborhood: t.Sequence[GridCellModel],
        step_count: int,
        trans_mult: float,
        rot_mult: float,
        action_space_reduction: str,
        ros_publisher: "rp.RosPublisher",
    ) -> ThinkResult:
        if not plan.has_tries_remaining(max_tries):
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Failing goal, no tries remaining to plan even while ignoring dynamic obstacles.".format(
                        self.name
                    ),
                    step_count,
                )
            )
            return ThinkResult(
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=True,
                robot_name=self.name,
            )

        plan.steps_with_replan_call.add(step_count)

        # I - Compute plan (ignoring dynamic obstacles) and set it to current plan
        dynamic_entities = {
            uid
            for uid, entity in w_t.entities.items()
            if (
                (isinstance(entity, Agent) and uid != robot_uid)
                or (
                    uid in w_t.entity_to_agent and w_t.entity_to_agent[uid] != robot_uid
                )
            )
        }
        w_t_no_dyn = w_t.light_copy(ignored_entities=dynamic_entities)
        inflated_grid_by_robot.deactivate_entities(dynamic_entities)
        plan.update_count += 1
        p = self.select_connect(
            w_t=w_t_no_dyn,
            static_obs_inf_grid=static_obs_inf_grid,
            inflated_grid_by_robot=inflated_grid_by_robot,
            r_f=goal,
            trans_mult=trans_mult,
            rot_mult=rot_mult,
            neighborhood=neighborhood,
            action_space_reduction=action_space_reduction,
            ros_publisher=ros_publisher,
            prev_list=set(),
        )
        inflated_grid_by_robot.activate_entities(dynamic_entities)
        plan.update_plan(p, step_count)

        if plan.is_empty():
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Failing goal, no plan could be found when ignoring dynamic obstacles.".format(
                        self.name
                    ),
                    step_count,
                )
            )

            return ThinkResult(
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=True,
                robot_name=self.name,
            )

        conflicts = plan.get_conflicts(
            world=w_t,
            inflated_grid_by_robot=inflated_grid_by_robot,
            check_horizon=fov,
            ros_publisher=ros_publisher,
            robot_name=self.name,
        )
        if not conflicts:
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Found a pure NAMO plan without conflicts with dynamic obstacles, "
                    "executing its first step...".format(self.name),
                    step_count,
                )
            )
            return ThinkResult(
                next_action=plan.pop_next_action(),
                goal_pose=goal,
                did_replan=True,
                robot_name=self.name,
            )

        if self.params.resolve_conflicts is False:
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Failing goal because conflicts where detected and resolve-conflicts is disabled.".format(
                        self.name
                    ),
                    step_count,
                )
            )
            return ThinkResult(
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=True,
                robot_name=self.name,
                conflicts=conflicts,
            )

        self.logger.append(
            utils.BasicLog(
                "Agent {}: A new plan has been computed ignoring dynamic "
                "obstacles but has conflicts with them: {}".format(
                    self.name, conflicts
                ),
                step_count,
            )
        )

        if not (plan.has_tries_remaining(max_tries) and plan.can_even_be_found()):
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Failing goal, no tries remaining to plan after conflicts "
                    "were found with the plan ignoring dynamic obstacles.".format(
                        self.name,
                    ),
                    step_count,
                )
            )
            return ThinkResult(
                next_action=ba.GoalFailed(goal),
                goal_pose=goal,
                did_replan=True,
                robot_name=self.name,
                conflicts=conflicts,
            )

        # II - Compute plan (with conflicting dynamic obstacles as static)
        # Get uids of conflicting robots and associated
        conflicting_robots_uids = {
            conflict.other_robot_uid
            for conflict in conflicts
            if isinstance(conflict, RobotRobotConflict)
        }
        conflicting_transfered_obstacles_uids = {
            w_t.entity_to_agent.inverse[uid]
            for uid in conflicting_robots_uids
            if uid in w_t.entity_to_agent.inverse
        }
        # Make a world copy without dynamic entities again, but with the conflicting robots
        new_dynamic_entities = dynamic_entities.difference(
            conflicting_robots_uids
        ).difference(conflicting_transfered_obstacles_uids)
        new_w_t_no_dyn = w_t.light_copy(ignored_entities=new_dynamic_entities)
        for conflict in conflicts:
            if (
                isinstance(conflict, ConcurrentGrabConflict)
                and conflict.obstacle_uid not in new_w_t_no_dyn.entity_to_agent
            ):
                new_w_t_no_dyn.entity_to_agent[
                    conflict.obstacle_uid
                ] = conflict.other_robot_uid
        inflated_grid_by_robot.deactivate_entities(new_dynamic_entities)
        # Iterate over each conflicting robot uid, and change its polygon to an encompassing circle
        # encounting for all likely states at at t+1
        polygons_tmp = {}
        for conflicting_robot_uid in conflicting_robots_uids:
            assert conflicting_robot_uid != self.uid

            conflicting_robot = new_w_t_no_dyn.agents[conflicting_robot_uid]
            conflict_radius = new_w_t_no_dyn.get_robot_conflict_radius(
                conflicting_robot_uid
            )
            center = conflicting_robot.polygon.centroid

            # TODO Get inflation from largest robot
            encompassing_circle = center.buffer(conflict_radius)
            polygons_tmp[conflicting_robot_uid] = conflicting_robot.polygon
            conflicting_robot.polygon = encompassing_circle
            inflated_grid_by_robot.update(
                {conflicting_robot_uid: conflicting_robot.polygon}
            )
        # Plan using this modified version of the world
        plan.update_count += 1
        p = self.select_connect(
            w_t=new_w_t_no_dyn,
            static_obs_inf_grid=static_obs_inf_grid,
            inflated_grid_by_robot=inflated_grid_by_robot,
            r_f=goal,
            trans_mult=trans_mult,
            rot_mult=rot_mult,
            neighborhood=neighborhood,
            action_space_reduction=action_space_reduction,
            ros_publisher=ros_publisher,
            prev_list=set(),
        )

        # Reset the inflated grid's state
        for conflicting_uid, prev_polygon in polygons_tmp.items():
            inflated_grid_by_robot.update({conflicting_uid: prev_polygon})
        inflated_grid_by_robot.activate_entities(new_dynamic_entities)

        if p.is_empty():
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Postponing for {} steps, could not find a plan avoiding the conflicting "
                    "dynamic obstacles of the pure NAMO plan.".format(self.name, t_max),
                    step_count,
                )
            )
            return ThinkResult(
                next_action=plan.new_postpone(
                    t_min,
                    t_max,
                    step_count,
                    conflicts,
                    self.logger,
                    self.name,
                ),
                goal_pose=goal,
                did_postpone=True,
                did_replan=True,
                robot_name=self.name,
                conflicts=conflicts,
            )

        plan.update_plan(p, step_count)
        new_conflicts = set(
            plan.get_conflicts(
                world=w_t,
                inflated_grid_by_robot=inflated_grid_by_robot,
                check_horizon=fov,
                ros_publisher=ros_publisher,
                robot_name=self.name,
            )
        )
        for conflict in conflicts:
            if conflict in new_conflicts:
                new_conflicts.remove(conflict)

        if new_conflicts:
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: Postponing for {} steps, a new plan has been computed avoiding the "
                    "conflicting dynamic obstacles of the pure NAMO plan, but has other conflicts: {}".format(
                        self.name, t_max, conflicts
                    ),
                    step_count,
                )
            )
            return ThinkResult(
                next_action=plan.new_postpone(
                    t_min,
                    t_max,
                    step_count,
                    conflicts,
                    self.logger,
                    self.name,
                ),
                goal_pose=goal,
                did_replan=True,
                did_postpone=True,
                robot_name=self.name,
                conflicts=conflicts,
            )

        self.logger.append(
            utils.BasicLog(
                "Agent {}: Found a new plan that does not have conflicts with the dynamic obstacles "
                "conflicting with the pure NAMO plan, executing its first step...".format(
                    self.name
                ),
                step_count,
            )
        )

        return ThinkResult(
            next_action=plan.pop_next_action(),
            goal_pose=goal,
            did_replan=True,
            robot_name=self.name,
        )

    def select_connect(
        self,
        *,
        w_t: "w.World",
        static_obs_inf_grid: BinaryInflatedOccupancyGrid,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        r_f: PoseModel,
        trans_mult: float,
        rot_mult: float,
        ros_publisher: "rp.RosPublisher",
        prev_list: t.Set[UID],
        ccs_data: connectivity.CCSData | None = None,
        neighborhood: t.Sequence[GridCellModel] = utils.CHESSBOARD_NEIGHBORHOOD,
        action_space_reduction: str = "only_r_acc_then_c_1_x",
    ):
        """
        High Level Planner _select_connect (SC).
        It makes use of _rch and _manip_search in a greedy heuristic search with backtracking.
        It backtracks locally when the object selected by _rch cannot be moved to merge the selected c_1 in c_free.
        It backtracks globally when all the paths identified by _rch from c_1 are unsuccessful.
        SC calls _find_path to determine a transit path from r_t to a contact point, r_t_plus_1 . The existence of the
        path is guaranteed by the choice of contacts in Manip-Search.
        # :param w_t: state of the world at time t
        # :param r_f: goal robot configuration [x, y, theta] in {m, m, degrees}
        # :return: None to backtrack, current partial plan otherwise.
        """
        robot = w_t.entities[self.uid]
        r_t = robot.pose

        avoid_list: t.Set[t.Tuple[UID, UID]] = set()

        robot_cell = utils.real_to_grid(
            r_t[0], r_t[1], static_obs_inf_grid.res, static_obs_inf_grid.grid_pose
        )
        goal_cell = utils.real_to_grid(
            r_f[0], r_f[1], static_obs_inf_grid.res, static_obs_inf_grid.grid_pose
        )

        simple_path_to_goal = self.find_path(
            robot_pose=r_t,
            goal_pose=r_f,
            robot_inflated_grid=inflated_grid_by_robot,
            robot_polygon=robot.polygon,
        )
        if simple_path_to_goal:
            # If the goal is in the same free space component as the robot in simulated w_t
            # Orig. condition in pseudo-code is : x^f in C^acc_R(W)
            # TODO FIX COST COMPUTATION TO FIT SAME MODEL AS MANIP SEARCH !
            ros_publisher.cleanup_robot_sim(ns=self.name)
            return nav_plan.Plan(
                path_components=[simple_path_to_goal],
                goal=r_f,
                robot_uid=self.uid,
            )

        if ccs_data is None:
            ccs_data = connectivity.init_ccs_for_grid(
                inflated_grid_by_robot.grid,
                inflated_grid_by_robot.d_width,
                inflated_grid_by_robot.d_height,
                neighborhood,
            )
        connected_components_grid = ccs_data.grid
        ros_publisher.publish_connected_components_grid(
            connected_components_grid, w_t.discretization_data.res, ns=robot.name
        )

        c_0 = ccs_data.grid[robot_cell[0]][robot_cell[1]]
        prev_list = prev_list if c_0 == 0 else prev_list.union({c_0})
        r_acc_cells = (
            set()
            if inflated_grid_by_robot.grid[robot_cell[0]][robot_cell[1]] > 0
            else connectivity.bfs_init(
                inflated_grid_by_robot.grid,
                inflated_grid_by_robot.d_width,
                inflated_grid_by_robot.d_height,
                robot_cell,
                neighborhood,
            ).visited
        )

        if len(inflated_grid_by_robot.cell_to_obstacle_ids(robot_cell)) > 1:
            n_movable = 0
            for uid in inflated_grid_by_robot.cell_to_obstacle_ids(robot_cell):
                if w_t.entities[uid].movability == Movability.MOVABLE:
                    n_movable += 1
            if n_movable > 1:
                return nav_plan.Plan(
                    plan_error="start_cell_in_several_movable_obstacles_error",
                    robot_uid=self.uid,
                )

        goal_cell_obstacles = inflated_grid_by_robot.cell_to_obstacle_ids(goal_cell)

        if len(goal_cell_obstacles) > 1:
            return nav_plan.Plan(
                plan_error="goal_cell_in_several_movable_obstacles_error",
                robot_uid=self.uid,
            )

        if len(goal_cell_obstacles) == 1:
            obs_id = list(goal_cell_obstacles)[0]
            if (
                obs_id != self.uid
                and w_t.entities[obs_id].movability != Movability.MOVABLE
            ):
                return nav_plan.Plan(
                    plan_error="goal_cell_occupied_by_unmovable_obstacle",
                    robot_uid=self.uid,
                )

        if static_obs_inf_grid.grid[goal_cell[0]][goal_cell[1]] > 0:
            raise Exception(
                "Goal cell collides with a static obstacle cell. This should never happen.",
            )

        if static_obs_inf_grid.grid[robot_cell[0]][robot_cell[1]] > 0:
            static_entities_polygons = {
                entity.uid: entity.polygon
                for entity in w_t.entities.values()
                if entity.movability == Movability.STATIC
            }
            static_entities_aabb_tree = collision.polygons_to_aabb_tree(
                static_entities_polygons
            )
            collisions, _ = collision.check_static_collision(
                main_uid=self.uid,
                polygon=self.polygon,
                other_entities_polygons=static_entities_polygons,
                aabb_tree=static_entities_aabb_tree,
            )

            if collisions:
                raise Exception(
                    "Robot start position is in collision with a static obstacle. This should never happen."
                )

        forbidden_obstacles = {  # Dynamic obstacles are forbidden !
            uid
            for uid, entity in w_t.entities.items()
            if (
                (isinstance(entity, Agent) and uid != self.uid)
                or (uid in w_t.entity_to_agent and w_t.entity_to_agent[uid] != self.uid)
            )
        }
        o_1, c_1 = self.rch(
            start_cell=robot_cell,
            goal_cell=goal_cell,
            static_obs_grid=static_obs_inf_grid,
            connected_components_grid=connected_components_grid,
            inflated_robot_grid=inflated_grid_by_robot,
            avoid_list=avoid_list,
            prev_list=prev_list,
            forbidden_obstacles=forbidden_obstacles,
            ros_publisher=ros_publisher,
            neighborhood=neighborhood,
        )

        while o_1 != 0:
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: select_connect: selected entity {} for manipulation search to reach component {}.".format(
                        robot.name, w_t.entities[o_1].name, c_1
                    ),
                    self._step_count,
                )
            )
            if action_space_reduction == "none":
                w_t_plus_2, tho_m = self.manip_search_procedure(
                    w_t=w_t,
                    o_1=o_1,
                    c_1=c_1,
                    ccs_data=ccs_data,
                    r_acc_cells=r_acc_cells,
                    r_f=r_f,
                    inflated_grid_by_robot=inflated_grid_by_robot,
                    trans_mult=trans_mult,
                    rot_mult=rot_mult,
                    ros_publisher=ros_publisher,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=True,
                )
            elif action_space_reduction == "only_r_acc":
                w_t_plus_2, tho_m = self.manip_search_procedure(
                    w_t=w_t,
                    o_1=o_1,
                    c_1=c_1,
                    ccs_data=ccs_data,
                    r_acc_cells=r_acc_cells,
                    r_f=r_f,
                    inflated_grid_by_robot=inflated_grid_by_robot,
                    trans_mult=trans_mult,
                    rot_mult=rot_mult,
                    ros_publisher=ros_publisher,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=False,
                )
            elif action_space_reduction == "only_r_acc_then_c_1_x":
                w_t_plus_2, tho_m = self.manip_search_procedure(
                    w_t=w_t,
                    o_1=o_1,
                    c_1=c_1,
                    ccs_data=ccs_data,
                    r_acc_cells=r_acc_cells,
                    r_f=r_f,
                    inflated_grid_by_robot=inflated_grid_by_robot,
                    trans_mult=trans_mult,
                    rot_mult=rot_mult,
                    ros_publisher=ros_publisher,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=False,
                )
                if tho_m is None:
                    w_t_plus_2, tho_m = self.manip_search_procedure(
                        w_t=w_t,
                        o_1=o_1,
                        c_1=c_1,
                        ccs_data=ccs_data,
                        r_acc_cells=r_acc_cells,
                        r_f=r_f,
                        inflated_grid_by_robot=inflated_grid_by_robot,
                        trans_mult=trans_mult,
                        rot_mult=rot_mult,
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

            if tho_m is not None:
                self.logger.append(
                    utils.BasicLog(
                        "Agent {}: select_connect: found partial plan manipulating entity {} to reach component {}.".format(
                            robot.name, w_t.entities[o_1].name, c_1
                        ),
                        self._step_count,
                    )
                )
                prev_cells_sets = inflated_grid_by_robot.update(
                    {o_1: w_t_plus_2.entities[o_1].polygon}
                )
                future_plan = self.select_connect(
                    w_t=w_t_plus_2,
                    static_obs_inf_grid=static_obs_inf_grid,
                    inflated_grid_by_robot=inflated_grid_by_robot,
                    r_f=r_f,
                    trans_mult=trans_mult,
                    rot_mult=rot_mult,
                    ros_publisher=ros_publisher,
                    ccs_data=ccs_data,
                    prev_list=(prev_list if c_1 == 0 else prev_list.union({c_1})),
                    neighborhood=neighborhood,
                    action_space_reduction=action_space_reduction,
                )
                inflated_grid_by_robot.cells_sets_update(prev_cells_sets)
                if not future_plan.plan_error:
                    tho_n = self.find_path(
                        robot_pose=r_t,
                        goal_pose=tho_m.robot_path.poses[0],
                        robot_inflated_grid=inflated_grid_by_robot,
                        robot_polygon=robot.polygon,
                    )
                    if not tho_n:
                        raise ValueError(
                            "Failed to find transit path to start of transfer path"
                        )
                    plan_components: t.List[TransitPath | TransferPath] = (
                        [tho_n, tho_m] if tho_n.actions else [tho_m]
                    )
                    return nav_plan.Plan(
                        path_components=plan_components,
                        goal=r_f,
                        robot_uid=self.uid,
                    ).append(future_plan)

            # Extra check for when the goal is in a movable obstacle that we could not find how to move
            if c_1 == 0:
                self.logger.append(
                    utils.BasicLog(
                        "Agent {}: select_connect: did not find a reachable component if manipulating {}.".format(
                            robot.name, w_t.entities[o_1].name
                        ),
                        self._step_count,
                    )
                )
                break

            avoid_list.add((o_1, c_1))

            o_1, c_1 = self.rch(
                start_cell=robot_cell,
                goal_cell=goal_cell,
                static_obs_grid=static_obs_inf_grid,
                connected_components_grid=connected_components_grid,
                inflated_robot_grid=inflated_grid_by_robot,
                avoid_list=avoid_list,
                prev_list=prev_list,
                forbidden_obstacles=forbidden_obstacles,
                ros_publisher=ros_publisher,
                neighborhood=neighborhood,
            )

        ros_publisher.cleanup_robot_sim(ns=self.name)
        return nav_plan.Plan(
            plan_error="no_plan_found_error",
            robot_uid=self.uid,
        )

    def rch_get_neighbors(
        self,
        current: RCHConfiguration,
        gscore: t.Dict[RCHConfiguration, float],
        close_set: t.Set[RCHConfiguration],
        open_queue: graph_search.PriorityQueue,
        came_from: t.Dict[RCHConfiguration, RCHConfiguration],
        static_obs_grid: BinaryInflatedOccupancyGrid,
        connected_components_grid: npt.NDArray[np.int_],
        inflated_robot_grid: BinaryInflatedOccupancyGrid,
        avoid_list: t.Set[t.Tuple[UID, UID]],
        prev_list: t.Set[UID],
        g_function: t.Callable[[RCHConfiguration, RCHConfiguration, bool], float],
        traversed_obstacles_ids: utils.OrderedSet,
        forbidden_obstacles: t.Set[UID],
        ros_publisher: "rp.RosPublisher",
        neighborhood: t.Sequence[GridCellModel] = utils.TAXI_NEIGHBORHOOD,
    ) -> t.Tuple[t.List[RCHConfiguration], t.List[float]]:
        """
        Combined formulation from Stilman's thesis and his article.
        """
        neighbors, tentative_gscores = [], []
        current_gscore = gscore[current]
        path_has_traversed_first_disconnected_comp = current.first_component_uid != 0
        path_has_traversed_first_obstacle = current.first_obstacle_uid != 0

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
                cur_cell_obs = inflated_robot_grid.cell_to_obstacle_ids(current.cell)
                neighbor_cell_obs = inflated_robot_grid.cell_to_obstacle_ids(
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
                neighbor_cell_component_uid: int = t.cast(
                    int, connected_components_grid[neighbor_cell[0]][neighbor_cell[1]]
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
                            and neighbor_cell_component_uid != 0
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
                        neighbor_cell_obs = inflated_robot_grid.cell_to_obstacle_ids(
                            neighbor_cell
                        )
                        if current.first_obstacle_uid in neighbor_cell_obs:
                            neighbor = RCHConfiguration(
                                neighbor_cell, current.first_obstacle_uid, 0
                            )
                        else:
                            # Either the neighbor is in another obstacle, or in multiple, which is forbidden
                            pass
                else:
                    if neighbor_cell_in_free_space:
                        # If no obstacle has been traversed, we are still in the robot acc. space
                        neighbor = RCHConfiguration(neighbor_cell, 0, 0)
                    else:
                        neighbor_cell_obstacles = (
                            inflated_robot_grid.cell_to_obstacle_ids(neighbor_cell)
                        )
                        if len(neighbor_cell_obstacles) > 0:
                            neighbor_obs_uid = list(neighbor_cell_obstacles)[0]
                            neighbor = RCHConfiguration(
                                neighbor_cell, neighbor_obs_uid, 0
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
                traversed_obstacles_ids.add(neighbor.first_obstacle_uid)

        ros_publisher.publish_rch_data(
            current=current,
            came_from=came_from,
            neighbors=neighbors,
            traversed_obstacles_ids=traversed_obstacles_ids,
            res=inflated_robot_grid.res,
            grid_pose=inflated_robot_grid.grid_pose,
            ns=self.name,
        )

        return neighbors, tentative_gscores

    def rch(
        self,
        start_cell: GridCellModel,
        goal_cell: GridCellModel,
        static_obs_grid: BinaryInflatedOccupancyGrid,
        connected_components_grid: npt.NDArray[np.int_],
        inflated_robot_grid: BinaryInflatedOccupancyGrid,
        avoid_list: t.Set[t.Tuple[UID, UID]],
        prev_list: t.Set[UID],
        forbidden_obstacles: t.Set[UID],
        ros_publisher: "rp.RosPublisher",
        neighborhood: t.Sequence[GridCellModel] = utils.TAXI_NEIGHBORHOOD,
    ) -> t.Tuple[UID, UID]:
        """Performs an A* search from the start cell to the goal cell, allowing
        only certain types of transitions between cells, as decscribed in Benoit
        Renault's papers and thesis. The search returns the IDs of the first obstacle
        and component encountered on the path to the goal.
        """
        if static_obs_grid.grid[start_cell[0]][start_cell[1]] > 0:
            obstacle_names = {
                self.world.entities[uid].name
                for uid in static_obs_grid.obstacles_uids_in_cell(start_cell)
            }
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot start cell {} in a rch call must always be outside of static obstacles, here: {}.".format(
                        self.name, start_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        if static_obs_grid.grid[goal_cell[0]][goal_cell[1]] > 0:
            obstacle_names = {
                self.world.entities[uid].name
                for uid in static_obs_grid.obstacles_uids_in_cell(goal_cell)
            }
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot goal cell {} in a rch call must always be outside of static obstacles, here: {}.".format(
                        self.name, goal_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        start_obstacles = inflated_robot_grid.cell_to_obstacle_ids(start_cell)
        if (
            len(start_obstacles) > 1
            or len(start_obstacles.intersection(forbidden_obstacles)) > 0
        ):
            assert self.uid not in start_obstacles
            obstacle_names = {
                self.world.entities[uid].name
                for uid in inflated_robot_grid.obstacles_uids_in_cell(start_cell)
            }
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot start cell {} in a rch call must always be at most in one obstacle and not a forbidden one, here: {}.".format(
                        self.name, start_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        if inflated_robot_grid.grid[goal_cell[0]][goal_cell[1]] > 1:
            obstacle_names = {
                self.world.entities[uid].name
                for uid in inflated_robot_grid.obstacles_uids_in_cell(goal_cell)
            }
            self.logger.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot goal cell {} in a rch call must be at most within one movable obstacle, here: {}.".format(
                        self.name, goal_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        # TODO Create custom exceptions for above

        sqrt_of_2_times_res = utils.SQRT_OF_2 * inflated_robot_grid.res
        goal_real = utils.grid_to_real(
            goal_cell[0],
            goal_cell[1],
            inflated_robot_grid.res,
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
                else inflated_robot_grid.res
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
                    inflated_robot_grid.res,
                    inflated_robot_grid.grid_pose,
                ),
                goal_real,
            )
            return translation_cost

        traversed_obstacles_ids = utils.OrderedSet()

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
                static_obs_grid,
                connected_components_grid,
                inflated_robot_grid,
                avoid_list,
                prev_list,
                g_function,
                traversed_obstacles_ids,
                forbidden_obstacles,
                ros_publisher,
                neighborhood,
            )

        def exit_condition(_current: RCHConfiguration, _goal: RCHConfiguration) -> bool:
            return _current.cell == _goal.cell

        start_obs_id = list(start_obstacles)[0] if len(start_obstacles) > 0 else 0
        start = RCHConfiguration(start_cell, start_obs_id, 0)
        goal = RCHConfiguration(
            goal_cell, 0, 0
        )  # Note the zeroes are never used, this line is just for coherence

        end_config: RCHConfiguration
        path_found, end_config, _, _, _, _ = graph_search.new_generic_a_star(
            start, goal, exit_condition, rch_get_neighbors_instance, h_function
        )  # type: ignore

        if path_found:
            if end_config.first_obstacle_uid == 0:
                raise ValueError(
                    "Rch found a path where no obstacle needed to be traversed."
                )
            return end_config.first_obstacle_uid, end_config.first_component_uid

        return 0, 0

    def manip_search(
        self,
        w_t: "w.World",
        o_1: UID,
        c_1: UID,
        ccs_data: connectivity.CCSData,
        r_acc_cells: t.Set[GridCellModel],
        r_f: PoseModel,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        trans_mult: float,
        rot_mult: float,
        ros_publisher: "rp.RosPublisher",
        check_new_local_opening_before_global: bool = True,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ):
        # Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = w_t.light_copy([])

        ros_publisher.publish_robot_sim_world(w_t_plus_2, self.uid)

        c_1_cells_set = set() if c_1 == 0 else ccs_data.ccs[c_1].visited

        res = w_t_plus_2.discretization_data.res

        other_entities = [
            entity
            for entity in w_t_plus_2.entities.values()
            if entity.uid != self.uid and entity.uid != o_1
        ]
        other_entities_polygons = {
            entity.uid: entity.polygon for entity in other_entities
        }
        other_entities_aabb_tree = collision.polygons_to_aabb_tree(
            other_entities_polygons
        )

        robot = w_t_plus_2.entities[self.uid]
        robot_uid, robot_pose, robot_polygon, robot_name = (
            robot.uid,
            robot.pose,
            robot.polygon,
            robot.name,
        )

        obstacle = w_t_plus_2.entities[o_1]
        obstacle_uid, obstacle_pose, obstacle_polygon = (
            obstacle.uid,
            obstacle.pose,
            obstacle.polygon,
        )
        obstacle_min_inflation_radius = utils.get_inscribed_radius(obstacle_polygon)

        goal_pose, goal_cell = (
            r_f,
            utils.real_to_grid(r_f[0], r_f[1], res, inflated_grid_by_robot.grid_pose),
        )

        # Get accessible sampled navigation points around obstacle
        (
            transfer_start_configs_to_cost,
            transfer_start_to_prev_transit_end,
        ) = self.get_transfer_start_to_transit_end_and_cost(
            robot_polygon=robot_polygon,
            robot_pose=robot_pose,
            robot_uid=robot_uid,
            obstacle_uid=obstacle_uid,
            other_entities_polygons=other_entities_polygons,
            other_entities_aabb_tree=other_entities_aabb_tree,
            inflated_grid_by_robot=inflated_grid_by_robot,
            r_acc_cells=r_acc_cells,
            obstacle_pose=obstacle_pose,
            obstacle_polygon=obstacle_polygon,
            trans_mult=trans_mult,
            rot_mult=rot_mult,
            ros_publisher=ros_publisher,
        )

        if not transfer_start_configs_to_cost:
            # If there are no attainable manipulation configurations, exit early
            ros_publisher.cleanup_q_manips_for_obs(ns=self.name)
            return w_t_plus_2, None

        # CAREFUL : We inflate by inscribed radius MINUS sqrt(2)*res to make sure occupied cells are really where the
        # entity's center should NEVER be to avoid collisions.
        # Poses in free cells of this grid may sometimes be colliding.
        inflated_grid_by_obstacle = BinaryInflatedOccupancyGrid(
            other_entities_polygons,
            res,
            max(obstacle_min_inflation_radius - utils.SQRT_OF_2 * res, 0.0),
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
            params=inflated_grid_by_robot.params,
        )

        # Only deactivate obstacle cells once transit end and transfer start are computed (grab action)
        inflated_grid_by_robot.deactivate_entities([obstacle_uid])

        # Use Dijkstra algorithm to compute a transfer path that allows for an opening to be created
        (
            path_found,
            transfer_end_configuration,
            came_from,
            _close_set,
            gscore,
            _,
        ) = self.dijkstra_for_manip_search(
            start=transfer_start_configs_to_cost,
            robot_uid=robot_uid,
            robot_name=robot_name,
            obstacle_uid=obstacle_uid,
            obstacle_polygon=obstacle_polygon,
            other_entities_polygons=other_entities_polygons,
            other_entities_aabb_tree=other_entities_aabb_tree,
            inflated_grid_by_robot=inflated_grid_by_robot,
            inflated_grid_by_obstacle=inflated_grid_by_obstacle,
            r_acc_cells=r_acc_cells,
            c_1_cells_set=c_1_cells_set,
            ccs_data=ccs_data,
            trans_mult=trans_mult,
            rot_mult=rot_mult,
            check_new_local_opening_before_global=check_new_local_opening_before_global,
            overall_goal_pose=goal_pose,
            overall_goal_cell=goal_cell,
            ros_publisher=ros_publisher,
            obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
            obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
        )

        if path_found:
            # ros_publisher.publish_sim(
            #     transfer_end_configuration.robot.polygon, transfer_end_configuration.obstacle.polygon,
            #     "/target", ns=self.name
            # )
            if transfer_end_configuration is None:
                raise Exception("Manip path found but transfer end config is None")

            raw_path: t.List[
                RobotObstacleConfiguration
            ] = graph_search.reconstruct_path(came_from, transfer_end_configuration)  # type: ignore

            prev_transit_end_configuration: RobotConfiguration | None = (
                transfer_start_to_prev_transit_end[raw_path[0]]
            )
            next_transit_start_configuration = (
                self.get_next_transit_start_configuration(
                    inflated_grid_by_robot,
                    raw_path[-1].robot.floating_point_pose,
                    raw_path[-1].robot.polygon,
                    robot_uid,
                    obstacle_uid,
                    raw_path[-1].obstacle.floating_point_pose,
                    other_entities_polygons,
                    other_entities_aabb_tree,
                    trans_mult,
                    rot_mult,
                )
            )

            if next_transit_start_configuration is None:
                raise Exception(
                    "Manip path found but failed to find next transit start config"
                )

            tho_m_phys_cost = gscore[transfer_end_configuration] + self.g(
                transfer_end_configuration.robot.floating_point_pose,
                next_transit_start_configuration.floating_point_pose,
                is_transfer=True,
            )
            tho_m = self.get_transfer_path_from_config(
                prev_transit_end_configuration,
                next_transit_start_configuration,
                raw_path,
                obstacle_uid,
                tho_m_phys_cost,
            )
        else:
            # If after exhausting all possible configurations, none opens a path to the connected component,
            # return None
            tho_m = None

        # Don't forget to update w_t_plus_2 with transfer end state
        if tho_m:
            robot.pose, robot.polygon = (
                tho_m.robot_path.poses[-1],
                tho_m.robot_path.polygons[-1],
            )
            obstacle.pose, obstacle.polygon = (
                tho_m.obstacle_path.poses[-1],
                tho_m.obstacle_path.polygons[-1],
            )

        ros_publisher.publish_robot_sim_world(w_t_plus_2, self.uid)
        ros_publisher.cleanup_robot_sim(ns=self.name)
        ros_publisher.cleanup_q_manips_for_obs(ns=self.name)

        inflated_grid_by_robot.activate_entities([obstacle_uid])

        return w_t_plus_2, tho_m

    def focused_manip_search(
        self,
        w_t: "w.World",
        o_1: UID,
        c_1: UID,
        ccs_data: connectivity.CCSData,
        r_acc_cells: t.Set[GridCellModel],
        r_f: PoseModel,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        trans_mult: float,
        rot_mult: float,
        ros_publisher: "rp.RosPublisher",
        check_new_local_opening_before_global: bool = True,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ):
        # Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = w_t.light_copy([])
        ros_publisher.publish_robot_sim_world(w_t_plus_2, self.uid)

        c_1_cells_set = set() if c_1 == 0 else ccs_data.ccs[c_1].visited

        res = w_t_plus_2.discretization_data.res

        other_entities = [
            entity
            for entity in w_t_plus_2.entities.values()
            if entity.uid != self.uid and entity.uid != o_1
        ]
        other_entities_polygons = {
            entity.uid: entity.polygon for entity in other_entities
        }
        other_entities_aabb_tree = collision.polygons_to_aabb_tree(
            other_entities_polygons
        )

        robot = w_t_plus_2.entities[self.uid]
        robot_uid, robot_pose, robot_name = robot.uid, robot.pose, robot.name
        robot_polygon = robot.polygon

        obstacle = w_t_plus_2.entities[o_1]
        obstacle_uid, obstacle_pose = obstacle.uid, obstacle.pose
        obstacle_polygon = obstacle.polygon
        obstacle_min_inflation_radius = utils.get_inscribed_radius(obstacle_polygon)

        goal_pose, goal_cell = (
            r_f,
            utils.real_to_grid(r_f[0], r_f[1], res, inflated_grid_by_robot.grid_pose),
        )

        # Get accessible sampled navigation points around obstacle
        (
            transfer_start_configs_to_cost,
            transfer_start_to_prev_transit_end,
        ) = self.get_transfer_start_to_transit_end_and_cost(
            robot_polygon=robot_polygon,
            robot_pose=robot_pose,
            robot_uid=robot_uid,
            obstacle_uid=obstacle_uid,
            other_entities_polygons=other_entities_polygons,
            other_entities_aabb_tree=other_entities_aabb_tree,
            inflated_grid_by_robot=inflated_grid_by_robot,
            r_acc_cells=r_acc_cells,
            obstacle_pose=obstacle_pose,
            obstacle_polygon=obstacle_polygon,
            trans_mult=trans_mult,
            rot_mult=rot_mult,
            ros_publisher=ros_publisher,
        )

        if not transfer_start_configs_to_cost:
            # If there are no attainable manipulation configurations, exit early
            ros_publisher.cleanup_q_manips_for_obs(ns=self.name)
            return w_t_plus_2, None

        # CAREFUL : We inflate by inscribed radius MINUS sqrt(2)*res to make sure occupied cells are really where the
        # entity's center should NEVER be to avoid collisions.
        # Poses in free cells of this grid may sometimes be colliding.
        inflated_grid_by_obstacle = BinaryInflatedOccupancyGrid(
            other_entities_polygons,
            res,
            max(obstacle_min_inflation_radius - utils.SQRT_OF_2 * res, 0.0),
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
            params=inflated_grid_by_robot.params,
        )
        inflated_grid_by_robot.deactivate_entities([obstacle_uid])

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
                    len(cells_sorted_by_combined_cost) * (1.0 - self.bound_percentage)
                )
            )
            - 1
        )
        bound_quantile_index = 0 if bound_quantile_index < 0 else bound_quantile_index
        bound_quantile = sorted_cell_to_combined_cost[
            cells_sorted_by_combined_cost[bound_quantile_index]
        ]

        # 1. Find the best obstacle transfer end configuration, that is, the one with the best compromise cost
        best_transfer_end_configuration = self.find_best_transfer_end_configuration(
            robot_pose=robot_pose,
            robot_polygon=robot_polygon,
            robot_name=robot_name,
            robot_uid=robot_uid,
            obstacle_uid=obstacle_uid,
            obstacle_pose=obstacle_pose,
            obstacle_polygon=obstacle_polygon,
            goal_pose=goal_pose,
            goal_cell=goal_cell,
            other_entities_polygons=other_entities_polygons,
            other_entities_aabb_tree=other_entities_aabb_tree,
            inflated_grid_by_robot=inflated_grid_by_robot,
            ordered_cells_by_cost=cells_sorted_by_combined_cost,
            r_acc_cells=r_acc_cells,
            c_1_cells_set=c_1_cells_set,
            ccs_data=ccs_data,
            init_robot_manip_configs=transfer_start_configs_to_cost.keys(),
            trans_mult=trans_mult,
            rot_mult=rot_mult,
            ros_publisher=ros_publisher,
            gscore=None,
            close_set=None,
            check_new_local_opening_before_global=check_new_local_opening_before_global,
            obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
            obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
        )
        if best_transfer_end_configuration is not None:
            ros_publisher.publish_sim(
                robot_polygon=best_transfer_end_configuration.robot.polygon,
                obs_polygon=best_transfer_end_configuration.obstacle.polygon,
                line_width=robot.circumscribed_radius / 4,
                namespace="/target",
                robot_name=self.name,
            )

            # 2. If a best obstacle transfer end configuration has been found, use A Star to find a path toward it
            transfer_end_configuration: RobotObstacleConfiguration | None

            (
                path_found,
                transfer_end_configuration,
                came_from,
                close_set,
                gscore,
                _,
            ) = self.a_star_for_manip_search(
                start=transfer_start_configs_to_cost,
                goal=best_transfer_end_configuration,
                robot_uid=robot_uid,
                robot_name=robot_name,
                obstacle_uid=obstacle_uid,
                obstacle_polygon=obstacle_polygon,
                other_entities_polygons=other_entities_polygons,
                other_entities_aabb_tree=other_entities_aabb_tree,
                inflated_grid_by_robot=inflated_grid_by_robot,
                inflated_grid_by_obstacle=inflated_grid_by_obstacle,
                r_acc_cells=r_acc_cells,
                c_1_cells_set=c_1_cells_set,
                ccs_data=ccs_data,
                trans_mult=trans_mult,
                rot_mult=rot_mult,
                sorted_cell_to_combined_cost=sorted_cell_to_combined_cost,
                bound_quantile=bound_quantile,
                check_new_local_opening_before_global=check_new_local_opening_before_global,
                overall_goal_pose=goal_pose,
                overall_goal_cell=goal_cell,
                ros_publisher=ros_publisher,
                obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
            )

            if path_found and transfer_end_configuration:
                # 3. If a path is found, return it
                # ros_publisher.publish_sim(
                #     transfer_end_configuration.robot.polygon, transfer_end_configuration.obstacle.polygon,
                #     "/target", ns=self.name
                # )
                raw_path: t.List[
                    RobotObstacleConfiguration
                ] = graph_search.reconstruct_path(came_from, transfer_end_configuration)
                prev_transit_end_configuration = transfer_start_to_prev_transit_end[
                    raw_path[0]
                ]
                next_transit_start_configuration = (
                    self.get_next_transit_start_configuration(
                        inflated_grid_by_robot,
                        raw_path[-1].robot.floating_point_pose,
                        raw_path[-1].robot.polygon,
                        robot_uid,
                        obstacle_uid,
                        raw_path[-1].obstacle.floating_point_pose,
                        other_entities_polygons,
                        other_entities_aabb_tree,
                        trans_mult,
                        rot_mult,
                    )
                )

                if next_transit_start_configuration is None:
                    raise Exception(
                        "Manip path found but failed to find next transit start config"
                    )

                tho_m_phys_cost = gscore[transfer_end_configuration] + self.g(
                    transfer_end_configuration.robot.floating_point_pose,
                    next_transit_start_configuration.floating_point_pose,
                    is_transfer=True,
                )
                tho_m = self.get_transfer_path_from_config(
                    prev_transit_end_configuration,
                    next_transit_start_configuration,
                    raw_path,
                    obstacle_uid,
                    tho_m_phys_cost,
                )
            else:
                # 4. If no path is found on the first, try finding a best configuration that has a path towards it
                #   (because we assume the A Star search to have completed, giving us the paths to ALL reachable
                #   configurations.
                best_transfer_end_configuration = self.find_best_transfer_end_configuration(
                    robot_pose=robot_pose,
                    robot_polygon=robot_polygon,
                    robot_name=robot_name,
                    robot_uid=robot_uid,
                    obstacle_uid=obstacle_uid,
                    obstacle_pose=obstacle_pose,
                    obstacle_polygon=obstacle_polygon,
                    goal_pose=goal_pose,
                    goal_cell=goal_cell,
                    other_entities_polygons=other_entities_polygons,
                    other_entities_aabb_tree=other_entities_aabb_tree,
                    inflated_grid_by_robot=inflated_grid_by_robot,
                    ordered_cells_by_cost=cells_sorted_by_combined_cost,
                    r_acc_cells=r_acc_cells,
                    c_1_cells_set=c_1_cells_set,
                    ccs_data=ccs_data,
                    init_robot_manip_configs=transfer_start_configs_to_cost.keys(),
                    trans_mult=trans_mult,
                    rot_mult=rot_mult,
                    ros_publisher=ros_publisher,
                    gscore=gscore,
                    close_set=close_set,
                    check_new_local_opening_before_global=check_new_local_opening_before_global,
                    obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                    obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
                )
                if best_transfer_end_configuration is not None:
                    # ros_publisher.publish_sim(
                    #     best_transfer_end_configuration.robot.polygon, best_transfer_end_configuration.obstacle.polygon,
                    #     "/target", ns=self.name
                    # )
                    raw_path = graph_search.reconstruct_path(
                        came_from, best_transfer_end_configuration
                    )
                    prev_transit_end_configuration = transfer_start_to_prev_transit_end[
                        raw_path[0]
                    ]
                    next_transit_start_configuration = (
                        self.get_next_transit_start_configuration(
                            inflated_grid_by_robot,
                            raw_path[-1].robot.floating_point_pose,
                            raw_path[-1].robot.polygon,
                            robot_uid,
                            obstacle_uid,
                            raw_path[-1].obstacle.floating_point_pose,
                            other_entities_polygons,
                            other_entities_aabb_tree,
                            trans_mult,
                            rot_mult,
                        )
                    )

                    if next_transit_start_configuration is None:
                        raise Exception(
                            "Manip path found but failed to find next transit start config"
                        )

                    tho_m_phys_cost = gscore[best_transfer_end_configuration] + self.g(
                        best_transfer_end_configuration.robot.floating_point_pose,
                        next_transit_start_configuration.floating_point_pose,
                        is_transfer=True,
                    )
                    tho_m = self.get_transfer_path_from_config(
                        prev_transit_end_configuration,
                        next_transit_start_configuration,
                        raw_path,
                        obstacle_uid,
                        tho_m_phys_cost,
                    )
                else:
                    # If after exhausting all possible configurations, none opens a path to the connected component,
                    # return None
                    tho_m = None
        else:
            # If after exhausting all possible configurations, none opens a path to the connected component,
            # return None
            tho_m = None

        # Don't forget to update w_t_plus_2 with transfer end state
        if tho_m:
            robot.pose, robot.polygon = (
                tho_m.robot_path.poses[-1],
                tho_m.robot_path.polygons[-1],
            )
            obstacle.pose, obstacle.polygon = (
                tho_m.obstacle_path.poses[-1],
                tho_m.obstacle_path.polygons[-1],
            )

        ros_publisher.publish_robot_sim_world(w_t_plus_2, self.uid)
        ros_publisher.cleanup_robot_sim(ns=self.name)
        ros_publisher.cleanup_q_manips_for_obs(ns=self.name)

        inflated_grid_by_robot.activate_entities([obstacle_uid])

        return w_t_plus_2, tho_m

    def dijkstra_for_manip_search(
        self,
        *,
        start: t.Dict[RobotObstacleConfiguration, float],
        robot_uid: UID,
        robot_name: str,
        obstacle_uid: UID,
        obstacle_polygon: Polygon,
        other_entities_polygons: t.Dict[UID, Polygon],
        other_entities_aabb_tree: AABBTree,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        inflated_grid_by_obstacle: BinaryInflatedOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        c_1_cells_set: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        trans_mult: float,
        rot_mult: float,
        check_new_local_opening_before_global: bool,
        overall_goal_pose: PoseModel,
        overall_goal_cell: GridCellModel,
        ros_publisher: "rp.RosPublisher",
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
                inflated_grid_by_robot,
                inflated_grid_by_obstacle,
                r_acc_cells,
                ccs_data,
                robot_uid,
                obstacle_uid,
                trans_mult,
                rot_mult,
                other_entities_polygons,
                other_entities_aabb_tree,
                ros_publisher,
                obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
            )

        def exit_condition(_current: RobotObstacleConfiguration):
            next_transit_start_configuration = (
                self.get_next_transit_start_configuration(
                    inflated_grid_by_robot,
                    _current.robot.floating_point_pose,
                    _current.robot.polygon,
                    robot_uid,
                    obstacle_uid,
                    _current.obstacle.floating_point_pose,
                    other_entities_polygons,
                    other_entities_aabb_tree,
                    trans_mult,
                    rot_mult,
                )
            )
            if next_transit_start_configuration:
                #   3. ... and creates a global opening to c1
                has_new_global_opening, _, _ = self.is_there_opening_to_c_1(
                    check_new_local_opening_before_global=check_new_local_opening_before_global,
                    robot_name=robot_name,
                    robot_cell=next_transit_start_configuration.cell_in_grid,
                    obstacle_uid=obstacle_uid,
                    old_obstacle_polygon=obstacle_polygon,
                    new_obstacle_polygon=_current.obstacle.polygon,
                    other_entities_polygons=other_entities_polygons,
                    other_entities_aabb_tree=other_entities_aabb_tree,
                    inflated_grid_by_robot=inflated_grid_by_robot,
                    c_1_cells_set=c_1_cells_set,
                    goal_pose=overall_goal_pose,
                    goal_cell=overall_goal_cell,
                    ros_publisher=ros_publisher,
                    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                    init_blocking_areas=None,
                    init_entity_inflated_polygon=None,
                )
                if has_new_global_opening:
                    return True
            return False

        return graph_search.new_generic_dijkstra(
            start, exit_condition=exit_condition, get_neighbors=get_neighbors
        )

    def a_star_for_manip_search(
        self,
        start: t.Dict[RobotObstacleConfiguration, float],
        goal: RobotObstacleConfiguration,
        robot_uid: UID,
        robot_name: str,
        obstacle_uid: UID,
        obstacle_polygon: Polygon,
        other_entities_polygons: t.Dict[UID, Polygon],
        other_entities_aabb_tree: AABBTree,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        inflated_grid_by_obstacle: BinaryInflatedOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        c_1_cells_set: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        trans_mult: float,
        rot_mult: float,
        sorted_cell_to_combined_cost: OrderedDict[GridCellModel, float],
        bound_quantile: float,
        check_new_local_opening_before_global: bool,
        overall_goal_pose: PoseModel,
        overall_goal_cell: GridCellModel,
        ros_publisher: "rp.RosPublisher",
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
                inflated_grid_by_robot,
                inflated_grid_by_obstacle,
                r_acc_cells,
                ccs_data,
                robot_uid,
                obstacle_uid,
                trans_mult,
                rot_mult,
                other_entities_polygons,
                other_entities_aabb_tree,
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
                next_transit_start_configuration = (
                    self.get_next_transit_start_configuration(
                        inflated_grid_by_robot,
                        _current.robot.floating_point_pose,
                        _current.robot.polygon,
                        robot_uid,
                        obstacle_uid,
                        _current.obstacle.floating_point_pose,
                        other_entities_polygons,
                        other_entities_aabb_tree,
                        trans_mult,
                        rot_mult,
                    )
                )
                if next_transit_start_configuration:
                    #   3. ... and creates a global opening to c1
                    has_new_global_opening, _, _ = self.is_there_opening_to_c_1(
                        check_new_local_opening_before_global,
                        robot_name,
                        next_transit_start_configuration.cell_in_grid,
                        obstacle_uid,
                        obstacle_polygon,
                        _current.obstacle.polygon,
                        other_entities_polygons,
                        other_entities_aabb_tree,
                        inflated_grid_by_robot,
                        c_1_cells_set,
                        overall_goal_pose,
                        overall_goal_cell,
                        ros_publisher=ros_publisher,
                        neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                        init_blocking_areas=None,
                        init_entity_inflated_polygon=None,
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

    def get_transit_end_and_transfer_start_poses(
        self,
        obstacle_polygon: Polygon,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        ros_publisher: "rp.RosPublisher",
    ) -> t.Tuple[t.List[PoseModel], t.List[PoseModel]]:
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
        :param inflated_grid_by_robot:
        :type inflated_grid_by_robot:
        :return: the lists of valid transit end poses and corresponding valid transfer start poses
        :rtype: tuple(list(tuple(float, float, float)), list(tuple(float, float, float)))
        """
        candidate_transfer_start_poses = utils.sample_poses_at_middle_of_inflated_sides(
            obstacle_polygon, inflated_grid_by_robot.inflation_radius
        )
        candidate_transit_end_poses = utils.sample_poses_at_middle_of_inflated_sides(
            obstacle_polygon, self.circumscribed_radius + self.grab_and_release_distance
        )

        valid_transit_end_poses, valid_transfer_start_poses = [], []
        for transit_end_pose, transfer_start_pose in zip(
            candidate_transit_end_poses, candidate_transfer_start_poses
        ):
            valid_transit_end_poses.append(transit_end_pose)
            valid_transfer_start_poses.append(transfer_start_pose)

        ros_publisher.cleanup_q_manips_for_obs(ns=self.name)
        ros_publisher.publish_q_manips_for_obs(valid_transfer_start_poses, ns=self.name)

        return valid_transit_end_poses, valid_transfer_start_poses

    def get_transfer_start_to_transit_end_and_cost(
        self,
        robot_polygon: Polygon,
        robot_pose: PoseModel,
        robot_uid: UID,
        obstacle_uid: UID,
        other_entities_polygons: t.Dict[UID, Polygon],
        other_entities_aabb_tree: AABBTree,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        obstacle_pose: PoseModel,
        obstacle_polygon: Polygon,
        trans_mult: float,
        rot_mult: float,
        ros_publisher: "rp.RosPublisher",
    ):
        robot_cell = utils.real_to_grid(
            robot_pose[0],
            robot_pose[1],
            inflated_grid_by_robot.res,
            inflated_grid_by_robot.grid_pose,
        )
        robot_cell_in_manip_obs = (
            obstacle_uid in inflated_grid_by_robot.cell_to_obstacle_ids(robot_cell)
        )

        transfer_start_configs_to_cost: t.Dict[RobotObstacleConfiguration, float] = {}
        transfer_start_to_prev_transit_end: t.Dict[
            RobotObstacleConfiguration, RobotConfiguration | None
        ] = {}

        if robot_cell_in_manip_obs:
            # If we are in the case where the robot starts from within the inflation of the manipulated obstacle,
            # exit early with only the start transfer configuration
            transfer_start_configuration = RobotObstacleConfiguration(
                robot_floating_point_pose=robot_pose,
                robot_polygon=robot_polygon,
                robot_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                    robot_pose, trans_mult, rot_mult
                ),
                robot_cell_in_grid=utils.real_to_grid(
                    robot_pose[0],
                    robot_pose[1],
                    inflated_grid_by_robot.res,
                    inflated_grid_by_robot.grid_pose,
                ),
                obstacle_floating_point_pose=obstacle_pose,
                obstacle_polygon=obstacle_polygon,
                obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                    obstacle_pose, trans_mult, rot_mult
                ),
                obstacle_cell_in_grid=utils.real_to_grid(
                    obstacle_pose[0],
                    obstacle_pose[1],
                    inflated_grid_by_robot.res,
                    inflated_grid_by_robot.grid_pose,
                ),
                manip_pose_id=0,
            )

            transfer_start_configs_to_cost = {transfer_start_configuration: 0.0}
            transfer_start_to_prev_transit_end = {transfer_start_configuration: None}

            return transfer_start_configs_to_cost, transfer_start_to_prev_transit_end

        # General case otherwise
        (
            transit_end_robot_poses,
            transfer_start_robot_poses,
        ) = self.get_transit_end_and_transfer_start_poses(
            obstacle_polygon, inflated_grid_by_robot, ros_publisher=ros_publisher
        )

        transfer_start_to_transit_end_robot_pose = {
            manip_pose: nav_pose
            for nav_pose, manip_pose in zip(
                transit_end_robot_poses, transfer_start_robot_poses
            )
        }

        for manip_pose_id, (transfer_start_pose, transit_end_pose) in enumerate(
            transfer_start_to_transit_end_robot_pose.items()
        ):
            transit_end_cell = utils.real_to_grid(
                transit_end_pose[0],
                transit_end_pose[1],
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
            )

            if transit_end_cell not in r_acc_cells:
                continue

            prev_transit_end_robot_polygon = utils.set_polygon_pose(
                robot_polygon, robot_pose, transit_end_pose
            )

            grab_action = ba.Grab(
                distance=utils.euclidean_distance(
                    transfer_start_pose, transit_end_pose
                ),
                entity_uid=obstacle_uid,
            )
            transfer_start_robot_polygon = grab_action.apply(
                prev_transit_end_robot_polygon, transit_end_pose
            )

            (
                _,
                collides_with,
                _,
                csv_polygons,
                _,
                _bb_vertices,
            ) = collision.csv_check_collisions(
                main_uid=robot_uid,
                other_polygons=other_entities_polygons,
                polygon_sequence=[
                    prev_transit_end_robot_polygon,
                    transfer_start_robot_polygon,
                ],
                action_sequence=[grab_action.to_absolute(transit_end_pose)],
                aabb_tree=other_entities_aabb_tree,
            )

            if not collides_with:
                prev_transit_end_configuration = RobotConfiguration(
                    floating_point_pose=transit_end_pose,
                    polygon=prev_transit_end_robot_polygon,
                    cell_in_grid=utils.real_to_grid(
                        transit_end_pose[0],
                        transit_end_pose[1],
                        inflated_grid_by_robot.res,
                        inflated_grid_by_robot.grid_pose,
                    ),
                    fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                        transit_end_pose, trans_mult, rot_mult
                    ),
                    action=None,
                    csv_polygon=prev_transit_end_robot_polygon,
                )
                temp_transfer_start_configuration = RobotObstacleConfiguration(
                    robot_floating_point_pose=transfer_start_pose,
                    robot_polygon=utils.set_polygon_pose(
                        robot_polygon, robot_pose, transfer_start_pose
                    ),
                    robot_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                        transfer_start_pose, trans_mult, rot_mult
                    ),
                    robot_cell_in_grid=utils.real_to_grid(
                        transfer_start_pose[0],
                        transfer_start_pose[1],
                        inflated_grid_by_robot.res,
                        inflated_grid_by_robot.grid_pose,
                    ),
                    obstacle_floating_point_pose=obstacle_pose,
                    obstacle_polygon=obstacle_polygon,
                    obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                        obstacle_pose, trans_mult, rot_mult
                    ),
                    obstacle_cell_in_grid=utils.real_to_grid(
                        obstacle_pose[0],
                        obstacle_pose[1],
                        inflated_grid_by_robot.res,
                        inflated_grid_by_robot.grid_pose,
                    ),
                    manip_pose_id=manip_pose_id,
                    action=grab_action,
                    robot_csv_polygon=csv_polygons[(0,)],
                    obstacle_csv_polygon=obstacle_polygon,
                )
                transfer_start_configs_to_cost[
                    temp_transfer_start_configuration
                ] = self.g(transit_end_pose, transfer_start_pose, is_transfer=True)
                transfer_start_to_prev_transit_end[
                    temp_transfer_start_configuration
                ] = prev_transit_end_configuration

        return transfer_start_configs_to_cost, transfer_start_to_prev_transit_end

    def find_best_transfer_end_configuration(
        self,
        *,
        robot_pose: PoseModel,
        robot_polygon: Polygon,
        robot_name: str,
        robot_uid: UID,
        obstacle_uid: UID,
        obstacle_pose: PoseModel,
        obstacle_polygon: Polygon,
        goal_pose: PoseModel,
        goal_cell: GridCellModel,
        other_entities_polygons: t.Dict[UID, Polygon],
        other_entities_aabb_tree: AABBTree,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        ordered_cells_by_cost: t.List[GridCellModel],
        r_acc_cells: t.Set[GridCellModel],
        c_1_cells_set: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        init_robot_manip_configs: t.Iterable[RobotObstacleConfiguration],
        trans_mult: float,
        rot_mult: float,
        ros_publisher: "rp.RosPublisher",
        gscore: t.Optional[t.Dict[RobotObstacleConfiguration, float]] = None,
        close_set: t.Optional[t.Set[RobotObstacleConfiguration]] = None,
        check_new_local_opening_before_global: bool = True,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ) -> RobotObstacleConfiguration | None:
        if close_set:
            assert gscore is not None

            # If all reachable configurations have been explored, index them by obstacle cell
            obs_cell_to_reachable_configurations: t.Dict[
                GridCellModel, t.List[RobotObstacleConfiguration]
            ] = {}
            for c in close_set:
                if c.obstacle.cell_in_grid in obs_cell_to_reachable_configurations:
                    obs_cell_to_reachable_configurations[
                        c.obstacle.cell_in_grid
                    ].append(c)
                else:
                    obs_cell_to_reachable_configurations[c.obstacle.cell_in_grid] = [c]

            # Then iterate over ordered_cells_by_cost until we find a configuration that:
            while ordered_cells_by_cost:
                current_best_cell = ordered_cells_by_cost.pop()

                intrudes = self.cell_intrudes_components(
                    current_best_cell,
                    r_acc_cells,
                    ccs_data,
                    obstacle_can_intrude_r_acc,
                    obstacle_can_intrude_c_1_x,
                )
                if intrudes:
                    continue

                if current_best_cell in obs_cell_to_reachable_configurations:
                    #   1. Is reachable, ...
                    possible_configurations = sorted(
                        obs_cell_to_reachable_configurations[current_best_cell],
                        key=lambda x: gscore[x],
                    )
                    for configuration in possible_configurations:
                        if not configuration.robot.polygon.within(
                            inflated_grid_by_robot.aabb_polygon
                        ):
                            continue
                        if not configuration.obstacle.polygon.within(
                            inflated_grid_by_robot.aabb_polygon
                        ):
                            continue

                        #   2. ... allows sufficient space for the robot to release the object, ...
                        next_transit_start_configuration = (
                            self.get_next_transit_start_configuration(
                                inflated_grid_by_robot,
                                configuration.robot.floating_point_pose,
                                configuration.robot.polygon,
                                robot_uid,
                                obstacle_uid,
                                configuration.obstacle.floating_point_pose,
                                other_entities_polygons,
                                other_entities_aabb_tree,
                                trans_mult,
                                rot_mult,
                            )
                        )
                        if next_transit_start_configuration:
                            #   2bis. ..., does not intrude forbidden component(s), ...
                            intrudes = self.polygon_intrudes_components(
                                configuration.obstacle.polygon,
                                inflated_grid_by_robot,
                                r_acc_cells,
                                ccs_data,
                                obstacle_can_intrude_r_acc,
                                obstacle_can_intrude_c_1_x,
                            )
                            if not intrudes:
                                #   3. ... and creates a global opening to c1
                                (
                                    has_new_global_opening,
                                    _,
                                    _,
                                ) = self.is_there_opening_to_c_1(
                                    check_new_local_opening_before_global,
                                    robot_name,
                                    next_transit_start_configuration.cell_in_grid,
                                    obstacle_uid,
                                    obstacle_polygon,
                                    configuration.obstacle.polygon,
                                    other_entities_polygons,
                                    other_entities_aabb_tree,
                                    inflated_grid_by_robot,
                                    c_1_cells_set,
                                    goal_pose,
                                    goal_cell,
                                    ros_publisher=ros_publisher,
                                    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                                    init_blocking_areas=None,
                                    init_entity_inflated_polygon=None,
                                )
                                if has_new_global_opening:
                                    return configuration
        else:
            # If we do not already have the set of reachable configurations, close_set...
            while ordered_cells_by_cost:
                # We iterate over the cells ordered by combined cost until we find a valid transfer end configuration
                current_cell = ordered_cells_by_cost[-1]

                intrudes = self.cell_intrudes_components(
                    current_cell,
                    r_acc_cells,
                    ccs_data,
                    obstacle_can_intrude_r_acc,
                    obstacle_can_intrude_c_1_x,
                )
                if intrudes:
                    ordered_cells_by_cost.pop()
                    continue

                # For that, we:
                for rot in [0.0] + self._all_rot_angles:
                    # Iterate over the possible obstacle rotations in this cell
                    obstacle_pose_at_transfer_end = utils.grid_pose_to_real_pose(
                        list(current_cell) + [rot],
                        inflated_grid_by_robot.res,
                        inflated_grid_by_robot.grid_pose,
                    )

                    # If the obstacle collides at this pose, don't consider checking further
                    obstacle_transfer_end_poly = utils.set_polygon_pose(
                        obstacle_polygon, obstacle_pose, obstacle_pose_at_transfer_end
                    )
                    collides_with, _ = collision.check_static_collision(
                        obstacle_uid,
                        obstacle_transfer_end_poly,
                        other_entities_polygons,
                        other_entities_aabb_tree,
                    )

                    if collides_with:
                        continue

                    for init_robot_manip_config in init_robot_manip_configs:
                        # Iterate over the possible robot poses corresponding to each obstacle pose
                        robot_pose_at_transfer_end = self.deduce_robot_goal_pose(
                            init_robot_manip_config.robot.floating_point_pose,
                            obstacle_pose,
                            obstacle_pose_at_transfer_end,
                        )

                        # For this (robot, obstacle) configuration, check if:
                        #   1. there are no static collisions for robot too, ...
                        robot_transfer_end_poly = utils.set_polygon_pose(
                            robot_polygon, robot_pose, robot_pose_at_transfer_end
                        )
                        collides_with, _ = collision.check_static_collision(
                            robot_uid,
                            robot_transfer_end_poly,
                            other_entities_polygons,
                            other_entities_aabb_tree,
                        )

                        if collides_with:
                            continue

                        #   2. ... the configuration allows sufficient space for the robot to release the object, ...
                        if not robot_transfer_end_poly:
                            robot_transfer_end_poly = utils.set_polygon_pose(
                                robot_polygon, robot_pose, robot_pose_at_transfer_end
                            )

                        if not robot_transfer_end_poly.within(
                            inflated_grid_by_robot.aabb_polygon
                        ):
                            continue

                        next_transit_start_configuration = (
                            self.get_next_transit_start_configuration(
                                inflated_grid_by_robot,
                                robot_pose_at_transfer_end,
                                robot_transfer_end_poly,
                                robot_uid,
                                obstacle_uid,
                                obstacle_pose_at_transfer_end,
                                other_entities_polygons,
                                other_entities_aabb_tree,
                                trans_mult,
                                rot_mult,
                            )
                        )
                        if next_transit_start_configuration:
                            if not obstacle_transfer_end_poly:
                                obstacle_transfer_end_poly = utils.set_polygon_pose(
                                    obstacle_polygon,
                                    obstacle_pose,
                                    obstacle_pose_at_transfer_end,
                                )

                            if not obstacle_transfer_end_poly.within(
                                inflated_grid_by_robot.aabb_polygon
                            ):
                                continue

                            #   2bis. ..., does not intrude forbidden component(s), ...
                            intrudes = self.polygon_intrudes_components(
                                obstacle_transfer_end_poly,
                                inflated_grid_by_robot,
                                r_acc_cells,
                                ccs_data,
                                obstacle_can_intrude_r_acc,
                                obstacle_can_intrude_c_1_x,
                            )
                            if not intrudes:
                                #   3. ... and creates a global opening to c1
                                (
                                    has_new_global_opening,
                                    _,
                                    _,
                                ) = self.is_there_opening_to_c_1(
                                    check_new_local_opening_before_global=check_new_local_opening_before_global,
                                    robot_name=robot_name,
                                    robot_cell=next_transit_start_configuration.cell_in_grid,
                                    obstacle_uid=obstacle_uid,
                                    old_obstacle_polygon=obstacle_polygon,
                                    new_obstacle_polygon=obstacle_transfer_end_poly,
                                    other_entities_polygons=other_entities_polygons,
                                    other_entities_aabb_tree=other_entities_aabb_tree,
                                    inflated_grid_by_robot=inflated_grid_by_robot,
                                    c_1_cells_set=c_1_cells_set,
                                    goal_pose=goal_pose,
                                    goal_cell=goal_cell,
                                    ros_publisher=ros_publisher,
                                    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                                    init_blocking_areas=None,
                                    init_entity_inflated_polygon=None,
                                )
                                if has_new_global_opening:
                                    return RobotObstacleConfiguration(
                                        robot_floating_point_pose=robot_pose_at_transfer_end,
                                        robot_polygon=robot_transfer_end_poly,
                                        robot_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                                            robot_pose_at_transfer_end,
                                            trans_mult,
                                            rot_mult,
                                        ),
                                        robot_cell_in_grid=utils.real_to_grid(
                                            robot_pose_at_transfer_end[0],
                                            robot_pose_at_transfer_end[1],
                                            inflated_grid_by_robot.res,
                                            inflated_grid_by_robot.grid_pose,
                                        ),
                                        obstacle_floating_point_pose=obstacle_pose_at_transfer_end,
                                        obstacle_polygon=obstacle_transfer_end_poly,
                                        obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                                            obstacle_pose_at_transfer_end,
                                            trans_mult,
                                            rot_mult,
                                        ),
                                        obstacle_cell_in_grid=utils.real_to_grid(
                                            obstacle_pose_at_transfer_end[0],
                                            obstacle_pose_at_transfer_end[1],
                                            inflated_grid_by_robot.res,
                                            inflated_grid_by_robot.grid_pose,
                                        ),
                                        manip_pose_id=0,
                                    )
                ordered_cells_by_cost.pop()
        return None  # If no valid configuration could be found...

    def get_next_transit_start_configuration(
        self,
        grid: BinaryInflatedOccupancyGrid,
        robot_pose: PoseModel,
        robot_polygon: Polygon,
        robot_uid: UID,
        obstacle_uid: UID,
        obstacle_pose: PoseModel,
        other_entities_polygons: t.Dict[UID, Polygon],
        other_entities_aabb_tree: AABBTree,
        trans_mult: float,
        rot_mult: float,
    ):
        release_action = ba.Release(
            distance=-1.0 * self.grab_and_release_distance,
            entity_uid=obstacle_uid,
        )
        robot_pose = (robot_pose[0], robot_pose[1], robot_pose[2])
        new_robot_pose = release_action.predict_pose(robot_pose, robot_pose[2])
        cell = utils.real_to_grid(
            new_robot_pose[0], new_robot_pose[1], grid.res, grid.grid_pose
        )

        if utils.is_in_matrix(cell, grid.d_width, grid.d_height):
            if grid.grid[cell[0]][cell[1]] > 0:
                # If the robot cell after release is in an obstacle in the grid, return False
                return None
        else:
            # If robot cell outside of grid, return False
            return None

        new_robot_polygon = release_action.apply(robot_polygon, new_robot_pose)

        # Check if robot is still within map bounds
        if not new_robot_polygon.within(grid.aabb_polygon):
            return None

        # Finally, we check dynamic collisions (between init configuration and after-action configuration)
        (
            _,
            collides_with,
            _,
            csv_polygons,
            _intersections,
            _bb_vertices,
        ) = collision.csv_check_collisions(
            main_uid=robot_uid,
            other_polygons=other_entities_polygons,
            polygon_sequence=[robot_polygon, new_robot_polygon],
            action_sequence=[release_action.to_absolute(robot_pose)],
            aabb_tree=other_entities_aabb_tree,
        )

        if not collides_with:
            new_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
                new_robot_pose, trans_mult, rot_mult
            )
            next_transit_start_configuration = RobotConfiguration(
                floating_point_pose=new_robot_pose,
                polygon=new_robot_polygon,
                cell_in_grid=cell,
                fixed_precision_pose=new_fixed_precision_pose,
                action=release_action,
                csv_polygon=csv_polygons[(0,)],
            )
            return next_transit_start_configuration

        return None

    def is_there_opening_to_c_1(
        self,
        check_new_local_opening_before_global: bool,
        robot_name: str,
        robot_cell: GridCellModel,
        obstacle_uid: UID,
        old_obstacle_polygon: Polygon,
        new_obstacle_polygon: Polygon,
        other_entities_polygons: t.Dict[UID, Polygon],
        other_entities_aabb_tree: AABBTree,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        c_1_cells_set: t.Set[GridCellModel],
        goal_pose: PoseModel,
        goal_cell: GridCellModel,
        ros_publisher: "rp.RosPublisher",
        neighborhood: t.Iterable[t.Iterable[int]] = utils.CHESSBOARD_NEIGHBORHOOD,
        init_blocking_areas: t.List[Polygon] | None = None,
        init_entity_inflated_polygon: Polygon | None = None,
    ):
        """
        Checks if there is a path between robot_cell and a random cell in c_1_cells_set that is not covered by an
        obstacle (especially the one considered for manipulation).
        :return: True if a path is found, False otherwise
        TODO: Add proper return of init_blocking_areas and init_entity_inflated_polygon and save them in caller methods
        """
        if check_new_local_opening_before_global:
            (
                has_new_local_opening,
                init_blocking_areas,
                init_entity_inflated_polygon,
            ) = check_new_local_opening(
                init_entity_polygon=old_obstacle_polygon,
                target_entity_polygon=new_obstacle_polygon,
                other_entities_polygons=other_entities_polygons,
                other_entities_aabb_tree=other_entities_aabb_tree,
                inflation_radius=inflated_grid_by_robot.inflation_radius,
                goal_pose=goal_pose,
                ros_publisher=ros_publisher,
                init_blocking_areas=init_blocking_areas,
                init_entity_inflated_polygon=init_entity_inflated_polygon,
                ns=robot_name,
            )
        else:
            has_new_local_opening = True

        if has_new_local_opening:
            obstacle_initially_deactivated = (
                obstacle_uid in inflated_grid_by_robot.deactivated_entities_cells_sets
            )
            if obstacle_initially_deactivated:
                inflated_grid_by_robot.activate_entities({obstacle_uid})
            previous_cells_sets = inflated_grid_by_robot.update(
                new_or_updated_polygons={obstacle_uid: new_obstacle_polygon}
            )

            if not c_1_cells_set or (c_1_cells_set and goal_cell in c_1_cells_set):
                cell_in_c_1 = goal_cell
            else:
                c_1_cells_set_iterator = iter(c_1_cells_set)
                cell_in_c_1 = next(c_1_cells_set_iterator)
                while inflated_grid_by_robot.grid[cell_in_c_1[0]][cell_in_c_1[1]] != 0:
                    # While selected cell not in free space after manipulation, try another cell
                    try:
                        cell_in_c_1 = next(c_1_cells_set_iterator)
                    except StopIteration:
                        # Note: using the the exception detection is the pythonic way it seems (no has_next)
                        # No opening because c_1_cells_set is entirely inaccessible to the robot after manipulation
                        inflated_grid_by_robot.cells_sets_update(
                            new_or_updated_cells_sets=previous_cells_sets
                        )
                        if obstacle_initially_deactivated:
                            inflated_grid_by_robot.deactivate_entities({obstacle_uid})
                        has_new_global_opening, skipped_global_opening_check = (
                            False,
                            False,
                        )
                        return (
                            has_new_global_opening,
                            has_new_local_opening,
                            skipped_global_opening_check,
                        )

            # TODO Evaluate the performance change (particularly compared to Dijkstra search) if A* star had an
            #  unadmissible heuristic to hasten path discovery (or write Best-FS based solely on heuristic)
            has_new_global_opening, _, _, _, _, _ = graph_search.grid_search_a_star(
                robot_cell,
                cell_in_c_1,
                inflated_grid_by_robot.grid,
                inflated_grid_by_robot.d_width,
                inflated_grid_by_robot.d_height,
                neighborhood,
                check_diag_neighbors=False,
            )

            cell_is_clear = True
            if cell_in_c_1 == goal_cell:
                # make sure goal cell is clear
                cell_is_clear = (
                    inflated_grid_by_robot.grid[cell_in_c_1[0]][cell_in_c_1[1]] == 0
                )

            inflated_grid_by_robot.cells_sets_update(
                new_or_updated_cells_sets=previous_cells_sets
            )
            if obstacle_initially_deactivated:
                inflated_grid_by_robot.deactivate_entities({obstacle_uid})
            skipped_global_opening_check = False

            return (
                has_new_global_opening and cell_is_clear,
                has_new_local_opening,
                skipped_global_opening_check,
            )

        has_new_global_opening, skipped_global_opening_check = False, True
        return (
            has_new_global_opening,
            has_new_local_opening,
            skipped_global_opening_check,
        )

    def get_manip_search_neighbors(
        self,
        current_configuration: RobotObstacleConfiguration,
        gscore: t.Dict[RobotObstacleConfiguration, float],
        close_set: t.Set[RobotObstacleConfiguration],
        open_queue: t.List[RobotObstacleConfiguration],
        came_from: t.Dict[
            RobotObstacleConfiguration, RobotObstacleConfiguration | None
        ],
        start: t.Dict[RobotObstacleConfiguration, float],
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        inflated_grid_by_obstacle: BinaryInflatedOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        robot_uid: UID,
        obstacle_uid: UID,
        trans_mult: float,
        rot_mult: float,
        other_entities_polygons: t.Dict[UID, Polygon],
        other_entities_aabb_tree: AABBTree,
        ros_publisher: "rp.RosPublisher",
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

                robot_center = (
                    current_configuration.robot.floating_point_pose[0],
                    current_configuration.robot.floating_point_pose[1],
                )
                new_robot_pose = action.predict_pose(
                    current_configuration.robot.floating_point_pose, robot_center
                )
                new_obstacle_pose = action.predict_pose(
                    current_configuration.obstacle.floating_point_pose, robot_center
                )
                new_robot_polygon = action.apply(
                    current_configuration.robot.polygon,
                    current_configuration.robot.floating_point_pose,
                )
                new_obstacle_polygon = action.apply(
                    current_configuration.obstacle.polygon,
                    current_configuration.robot.floating_point_pose,
                )
                extra_g_cost = self.rotation_unit_cost
            elif isinstance(action, ba.Advance):
                neighbor_action_opposes_prev_action = isinstance(
                    current_configuration.action, ba.Advance
                ) and np.sign(action.distance) != np.sign(
                    current_configuration.action.distance
                )
                if neighbor_action_opposes_prev_action:
                    continue

                new_robot_pose = action.predict_pose(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.robot.floating_point_pose[2],
                )
                new_obstacle_pose = action.predict_pose(
                    current_configuration.obstacle.floating_point_pose,
                    current_configuration.robot.floating_point_pose[2],
                )
                new_robot_polygon = action.apply(
                    current_configuration.robot.polygon,
                    current_configuration.robot.floating_point_pose,
                )
                new_obstacle_polygon = action.apply(
                    current_configuration.obstacle.polygon,
                    current_configuration.robot.floating_point_pose,
                )
                extra_g_cost = self.translation_unit_cost
            elif isinstance(action, ba.AbsoluteTranslation):
                neighbor_action_opposes_prev_action = (
                    isinstance(current_configuration.action, ba.AbsoluteTranslation)
                    and action.v[0] == -1.0 * current_configuration.action.v[0]
                    and action.v[1] == -1.0 * current_configuration.action.v[1]
                )
                if neighbor_action_opposes_prev_action:
                    continue

                new_robot_pose = action.predict_pose(
                    current_configuration.robot.floating_point_pose,
                )
                new_obstacle_pose = action.predict_pose(
                    current_configuration.obstacle.floating_point_pose,
                )
                new_robot_polygon = action.apply(
                    current_configuration.robot.polygon,
                )
                new_obstacle_polygon = action.apply(
                    current_configuration.obstacle.polygon,
                )
                extra_g_cost = self.translation_unit_cost
            else:
                raise TypeError(
                    "action must either be of type Rotation, Advance, or AbsoluteTranslation"
                )

            # First, check whether the new configuration is in close set, if it is, ignore it
            robot_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
                new_robot_pose, trans_mult, rot_mult
            )
            obstacle_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
                new_obstacle_pose, trans_mult, rot_mult
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
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
            )
            obstacle_cell_in_grid = utils.real_to_grid(
                new_obstacle_pose[0],
                new_obstacle_pose[1],
                inflated_grid_by_obstacle.res,
                inflated_grid_by_obstacle.grid_pose,
            )

            is_no_longer_in_grid = not (
                utils.is_in_matrix(
                    robot_cell_in_grid,
                    inflated_grid_by_robot.d_width,
                    inflated_grid_by_robot.d_height,
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
                inflated_grid_by_robot.grid[robot_cell_in_grid[0]][
                    robot_cell_in_grid[1]
                ]
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
            if not new_robot_polygon.within(inflated_grid_by_robot.aabb_polygon):
                continue

            # Check if obstacle is still within map bounds
            if not new_obstacle_polygon.within(inflated_grid_by_obstacle.aabb_polygon):
                continue

            # Finally, we check dynamic collisions (between init configuration and after-action configuration)
            (
                _,
                collides_with,
                _,
                robot_csv_polygons,
                _,
                _robot_bb_vertices,
            ) = collision.csv_check_collisions(
                main_uid=robot_uid,
                other_polygons=other_entities_polygons,
                polygon_sequence=[
                    current_configuration.robot.polygon,
                    new_robot_polygon,
                ],
                action_sequence=[
                    action
                    if isinstance(action, ba.AbsoluteAction)
                    else action.to_absolute(
                        current_configuration.robot.floating_point_pose
                    )
                ],
                aabb_tree=other_entities_aabb_tree,
            )
            if collides_with:
                continue
            # TODO Refactor collision.csv_check_collisions to check for any number of attached polygons or make new function
            (
                _,
                collides_with,
                _,
                obstacle_csv_polygons,
                _,
                _obstacle_bb_vertices,
            ) = collision.csv_check_collisions(
                main_uid=obstacle_uid,
                other_polygons=other_entities_polygons,
                polygon_sequence=[
                    current_configuration.obstacle.polygon,
                    new_obstacle_polygon,
                ],
                action_sequence=[
                    # Actions are always applied relative to the robot pose!
                    action.to_absolute(current_configuration.robot.floating_point_pose)
                ],
                aabb_tree=other_entities_aabb_tree,
            )
            if collides_with:
                continue

            # If option is activated, check that obstacle intruded the appropriate component(s)
            intrudes = self.polygon_intrudes_components(
                new_obstacle_polygon,
                inflated_grid_by_robot,
                r_acc_cells,
                ccs_data,
                obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x,
            )
            if intrudes:
                continue

            if len(inflated_grid_by_robot.cell_to_obstacle_ids(robot_cell_in_grid)) > 0:
                continue

            # If we are here, then this newly computed neighbor configuration is valid and we must save it
            neighbor_configuration = RobotObstacleConfiguration(
                robot_floating_point_pose=new_robot_pose,
                robot_polygon=new_robot_polygon,
                robot_fixed_precision_pose=robot_fixed_precision_pose,
                robot_cell_in_grid=robot_cell_in_grid,
                obstacle_floating_point_pose=new_obstacle_pose,
                obstacle_polygon=new_obstacle_polygon,
                obstacle_fixed_precision_pose=obstacle_fixed_precision_pose,
                obstacle_cell_in_grid=obstacle_cell_in_grid,
                action=action,
                manip_pose_id=current_configuration.manip_pose_id,
                robot_csv_polygon=robot_csv_polygons[(0,)],
                obstacle_csv_polygon=obstacle_csv_polygons[(0,)],
            )

            neighbors.append(neighbor_configuration)
            tentative_g_scores.append(gscore[current_configuration] + extra_g_cost)

        manip_poses_ids = [c.manip_pose_id for c in start.keys()]

        ros_publisher.publish_manip_search_data(
            current_manip_pose_id=current_configuration.manip_pose_id,  # type: ignore
            manip_poses_ids=manip_poses_ids,
            robot_pose=current_configuration.robot.floating_point_pose,
            robot_fixed_precision_pos=current_configuration.robot.fixed_precision_pose,
            robot_polygon=current_configuration.robot.polygon,
            obstacle_polygon=current_configuration.obstacle.polygon,
            obstacle_pose=current_configuration.obstacle.floating_point_pose,
            line_width=self.min_inflation_radius / 4,
            res=inflated_grid_by_robot.res,
            neighbor_poses=[n.robot.floating_point_pose for n in neighbors],
            ns=self.name,
        )

        return neighbors, tentative_g_scores

    @staticmethod
    def polygon_intrudes_components(
        new_obstacle_polygon: Polygon,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        obstacle_can_intrude_r_acc: bool,
        obstacle_can_intrude_c_1_x: bool,
    ):
        if obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            return False

        if obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
            new_obstacle_exterior_cells = utils.accurate_rasterize_in_grid(
                new_obstacle_polygon.buffer(inflated_grid_by_robot.inflation_radius),
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
                inflated_grid_by_robot.d_width,
                inflated_grid_by_robot.d_height,
                fill=False,
            )
            for cell in new_obstacle_exterior_cells:
                if ccs_data.grid[cell[0]][cell[1]] > 0 and cell not in r_acc_cells:
                    return True
        elif not obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            new_obstacle_exterior_cells = utils.accurate_rasterize_in_grid(
                new_obstacle_polygon.buffer(inflated_grid_by_robot.inflation_radius),
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
                inflated_grid_by_robot.d_width,
                inflated_grid_by_robot.d_height,
                fill=False,
            )
            for cell in new_obstacle_exterior_cells:
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
        robot_manip_pose: PoseModel, obs_init_pose: PoseModel, obs_goal_pose: PoseModel
    ) -> PoseModel:
        translation, rotation = utils.get_translation_and_rotation(
            obs_init_pose, obs_goal_pose
        )
        robot_goal_point = list(
            utils.rotate_then_translate_polygon(
                Point((robot_manip_pose[0], robot_manip_pose[1])),
                translation,
                rotation,
                (obs_init_pose[0], obs_init_pose[1]),
            ).coords[0]
        )
        orientation = (robot_manip_pose[2] + rotation) % 360.0
        orientation = orientation if orientation >= 0.0 else orientation + 360.0
        return robot_goal_point[0], robot_goal_point[1], orientation

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
        inflated_grid_by_obstacle: BinaryInflatedOccupancyGrid,
        robot_polygon: Polygon,
        robot_pose: PoseModel,
        obstacle_pose: PoseModel,
        goal_pose: PoseModel,
        ros_publisher: "rp.RosPublisher",
    ):
        if self._social_costmap is None:
            raise Exception("Social costmap uninitialized")

        # Initialize some needed variables
        obstacle_cell = utils.real_to_grid(
            obstacle_pose[0],
            obstacle_pose[1],
            inflated_grid_by_obstacle.res,
            inflated_grid_by_obstacle.grid_pose,
        )

        robot_poly_at_goal = utils.set_polygon_pose(
            robot_polygon, robot_pose, goal_pose
        )

        robot_cells_at_goal = utils.accurate_rasterize_in_grid(
            robot_poly_at_goal,
            inflated_grid_by_obstacle.res,
            inflated_grid_by_obstacle.grid_pose,
            inflated_grid_by_obstacle.d_width,
            inflated_grid_by_obstacle.d_height,
            fill=True,
        )

        # Compute set of potentially reachable cells for obstacle and a heuristic cost to join them
        cell_to_cost = self.dijkstra_cc_and_cost(
            obstacle_cell,
            inflated_grid_by_obstacle.grid,
            inflated_grid_by_obstacle.res,
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

        if not self.distance_to_obs_cost_is_realistic:
            distance_cost = np.array(
                [
                    utils.euclidean_distance(
                        utils.grid_to_real(
                            cell[0],
                            cell[1],
                            inflated_grid_by_obstacle.res,
                            inflated_grid_by_obstacle.grid_pose,
                        ),
                        obstacle_pose,
                    )
                    for cell in acc_cells_for_obs
                ]
            )

        distance_to_goal = np.array(
            [
                utils.euclidean_distance(
                    utils.grid_to_real(
                        cell[0],
                        cell[1],
                        inflated_grid_by_obstacle.res,
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
            + self.w_obs * normalized_distance_cost
            + self.w_goal * normalized_distance_to_goal
        ) / self.w_sum

        sorted_cell_to_combined_cost = OrderedDict(
            sorted(
                zip(acc_cells_for_obs, combined_cost), key=lambda t: t[1], reverse=True
            )
        )

        ros_publisher.publish_combined_costmap(
            sorted_cell_to_combined_cost, inflated_grid_by_obstacle, ns=self.name
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
        inflated_grid_by_obstacle: BinaryInflatedOccupancyGrid,
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
            normalized_distance_from_obs_costmap[cell[0]][
                cell[1]
            ] = normalized_distance_cost[i]
            if normalized_distance_to_goal is not None:
                normalized_distance_from_goal_costmap[cell[0]][
                    cell[1]
                ] = normalized_distance_to_goal[i]

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
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        w_t: "w.World",
        main_robot_uid: UID,
        potential_deadlocks: t.Set[Conflict],
        forbidden_evasion_cells: t.Set[GridCellModel],
        ros_publisher: "rp.RosPublisher",
        use_combined_cost: bool = False,
    ) -> EvasionTransitPath | None:
        # Compute evasion for main robot
        main_robot = t.cast(Agent, w_t.entities[main_robot_uid])

        # The main robot uid should be deactivated in the robot-inflated grid
        assert main_robot_uid in inflated_grid_by_robot.deactivated_entities_cells_sets

        (
            main_robot_evasion_cell_social_cost,
            main_robot_evasion_path,
        ) = self.compute_evasion_for_one(
            w_t=w_t,
            inflated_grid_by_robot=inflated_grid_by_robot,
            robot=main_robot,
            forbidden_evasion_cells=forbidden_evasion_cells,
            ros_publisher=ros_publisher,
            use_combined_cost=use_combined_cost,
        )

        if not main_robot_evasion_path:
            return None

        # If this robot is able to evade, it must check if it should by comparing its evasion path with the one of
        # other robots.
        other_robots_uids = {
            potential_deadlock.other_robot_uid
            for potential_deadlock in potential_deadlocks
            if isinstance(potential_deadlock, RobotRobotConflict)
        }

        assert main_robot_uid not in other_robots_uids

        inflated_grid_by_robot.update(
            new_or_updated_polygons={main_robot_uid: main_robot.polygon}
        )

        other_robots_evasion_costs: t.List[float] = []
        other_robot_evasion_path_max_duration = 0

        max_d = float("-inf")

        for robot_uid in other_robots_uids:
            # TODO : Add check to see if other robot has same radius as main robot : if so use the already computed
            #  inflated grid, else compute a corresponding inflated grid (and save for later just in case ?)
            other_robot = t.cast(Agent, w_t.entities[robot_uid])
            max_d = max(max_d, np.linalg.norm(other_robot.pose[:2]))

            inflated_grid_by_robot.deactivate_entities({robot_uid})
            inflated_grid_by_robot.activate_entities({main_robot_uid})
            (
                other_robot_evaion_cost,
                _other_robot_evasion_path,
            ) = self.compute_evasion_for_one(
                w_t=w_t,
                inflated_grid_by_robot=inflated_grid_by_robot,
                robot=other_robot,
                forbidden_evasion_cells=set(),
                use_combined_cost=use_combined_cost,
                ros_publisher=ros_publisher,
            )
            inflated_grid_by_robot.deactivate_entities({main_robot_uid})

            other_robot_exchange_real_path = graph_search.real_to_grid_search_a_star(
                other_robot.pose, main_robot.pose, inflated_grid_by_robot
            )

            inflated_grid_by_robot.activate_entities({robot_uid})

            other_robot_exchange_path = TransitPath.from_poses(
                other_robot_exchange_real_path,
                other_robot.polygon,
                other_robot.pose,
            )

            other_robots_evasion_costs.append(other_robot_evaion_cost)

            other_robot_evasion_path_max_duration = max(
                other_robot_evasion_path_max_duration,
                len(main_robot_evasion_path.actions)
                + len(other_robot_exchange_path.actions),
            )

        main_robot_evasion_path.set_wait(other_robot_evasion_path_max_duration)
        if main_robot_evasion_cell_social_cost < np.min(other_robots_evasion_costs):
            return main_robot_evasion_path

        if main_robot_evasion_cell_social_cost == np.min(other_robots_evasion_costs):
            ## tie breaking
            if np.linalg.norm(self.pose[:2]) >= max_d:
                return main_robot_evasion_path

        return None  # Wait for others to evade

    def compute_evasion_nonsocial(
        self,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        w_t: "w.World",
        main_robot_uid: UID,
        potential_deadlocks: t.Set[Conflict],
    ) -> EvasionTransitPath | None:
        """Computes an evasion path for the main robot without using social cost"""
        # Compute evasion for main robot
        robot = t.cast(Agent, w_t.entities[main_robot_uid])

        other_robots_uids = {
            potential_deadlock.other_robot_uid
            for potential_deadlock in potential_deadlocks
            if isinstance(potential_deadlock, RobotRobotConflict)
        }

        d = np.linalg.norm(robot.pose[:2])
        for other_robot_uid in other_robots_uids:
            other_robot = w_t.agents[other_robot_uid]
            d_other = np.linalg.norm(other_robot.pose[:2])
            if d_other < d:  # type: ignore
                return None

        # The main robot uid should be deactivated in the robot-inflated grid
        assert main_robot_uid in inflated_grid_by_robot.deactivated_entities_cells_sets

        # If the robot is currently holding an object, try to release it first to find a valid transit starting configuration
        transit_configuration_after_release = None
        if w_t.is_holding_obstacle(robot.uid):
            obstacle_uid = w_t.entity_to_agent.inverse[robot.uid]
            obstacle = w_t.entities[obstacle_uid]
            other_entities_polygons = {
                uid: e.polygon
                for uid, e in w_t.entities.items()
                if uid not in (robot.uid, obstacle_uid)
            }
            other_entities_aabb_tree = collision.polygons_to_aabb_tree(
                other_entities_polygons
            )
            transit_configuration_after_release = (
                self.get_next_transit_start_configuration(
                    inflated_grid_by_robot,
                    robot.pose,
                    robot.polygon,
                    robot.uid,
                    obstacle_uid,
                    obstacle.pose,
                    other_entities_polygons,
                    other_entities_aabb_tree,
                    trans_mult=self.trans_mult,
                    rot_mult=self.rot_mult,
                )
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
            inflated_grid_by_robot.res,
            inflated_grid_by_robot.grid_pose,
        )
        if transit_configuration_after_release:
            robot_polygon = transit_configuration_after_release.polygon
            robot_pose = transit_configuration_after_release.floating_point_pose
            robot_cell = transit_configuration_after_release.cell_in_grid

        def get_min_dist_to_others(cell: GridCellModel):
            min_dist_to_other_robot = float("inf")

            for other_robot_uid in other_robots_uids:
                other_robot = w_t.agents[other_robot_uid]
                other_robot_cell = utils.real_to_grid(
                    other_robot.pose[0],
                    other_robot.pose[1],
                    inflated_grid_by_robot.res,
                    inflated_grid_by_robot.grid_pose,
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

            grid = inflated_grid_by_robot.grid
            neighbors, tentative_gscores = [], []

            current_gscore = gscore[current]
            for i, j in utils.TAXI_NEIGHBORHOOD:
                neighbor = current[0] + i, current[1] + j
                neighbor_is_valid = (
                    neighbor not in close_set
                    and utils.is_in_matrix(
                        cell=neighbor,
                        width=inflated_grid_by_robot.d_width,
                        height=inflated_grid_by_robot.d_height,
                    )
                    and grid[neighbor[0]][neighbor[1]] == 0
                )
                if neighbor_is_valid:
                    neighbors.append(neighbor)
                    tentative_gscores.append(current_gscore + 1.0)

            return neighbors, tentative_gscores

        def evasion_heuristic(
            current: GridCellModel,
            goal: None,
        ) -> float:
            return -get_min_dist_to_others(current)

        def exit_condition(current: GridCellModel, goal: ModuleNotFoundError):
            return False

        _, _, came_from, _, gscore, _ = graph_search.new_generic_a_star(
            start=robot_cell,
            goal=None,
            exit_condition=exit_condition,
            get_neighbors=get_neighbors_for_evasion,
            heuristic=evasion_heuristic,
        )  # type: ignore

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
            inflated_grid_by_robot.res,
            inflated_grid_by_robot.grid_pose,
        )

        if len(real_path) < 2:
            return None

        evasion_transit_path = EvasionTransitPath.from_poses(
            real_path, robot_polygon, robot_pose
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
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        robot: Agent,
        forbidden_evasion_cells: t.Set[GridCellModel],
        ros_publisher: "rp.RosPublisher",
        use_combined_cost: bool = False,
    ) -> t.Tuple[float, EvasionTransitPath | None]:
        """Computes an evasion path for a given robot"""
        if self._social_costmap is None:
            raise Exception("No social costmap")

        robot_start_cell = utils.real_to_grid(
            robot.pose[0],
            robot.pose[1],
            inflated_grid_by_robot.res,
            inflated_grid_by_robot.grid_pose,
        )
        robot_start_social_cost = self._social_costmap[robot_start_cell[0]][
            robot_start_cell[1]
        ]

        # If the robot is currently holding an object, try to release it first to find a valid transit starting configuration
        transit_configuration_after_release = None
        if robot.uid in w_t.entity_to_agent.inverse:
            obstacle_uid = w_t.entity_to_agent.inverse[robot.uid]
            obstacle = w_t.entities[obstacle_uid]
            other_entities_polygons = {
                uid: e.polygon
                for uid, e in w_t.entities.items()
                if uid not in (robot.uid, obstacle_uid)
            }
            other_entities_aabb_tree = collision.polygons_to_aabb_tree(
                other_entities_polygons
            )
            transit_configuration_after_release = (
                self.get_next_transit_start_configuration(
                    inflated_grid_by_robot,
                    robot.pose,
                    robot.polygon,
                    robot.uid,
                    obstacle_uid,
                    obstacle.pose,
                    other_entities_polygons,
                    other_entities_aabb_tree,
                    trans_mult=self.trans_mult,
                    rot_mult=self.rot_mult,
                )
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
            inflated_grid_by_robot.res,
            inflated_grid_by_robot.grid_pose,
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

            grid = inflated_grid_by_robot.grid
            neighbors, tentative_gscores = [], []

            current_gscore = gscore[current]
            for i, j in utils.TAXI_NEIGHBORHOOD:
                neighbor = current[0] + i, current[1] + j
                neighbor_is_valid = (
                    neighbor not in close_set
                    and utils.is_in_matrix(
                        cell=neighbor,
                        width=inflated_grid_by_robot.d_width,
                        height=inflated_grid_by_robot.d_height,
                    )
                    and grid[neighbor[0]][neighbor[1]] == 0
                )
                if neighbor_is_valid:
                    neighbors.append(neighbor)
                    tentative_gscores.append(current_gscore + 1.0)

            return neighbors, tentative_gscores

        def evasion_heuristic(
            current: GridCellModel,
            goal: None,
        ) -> float:
            if self._social_costmap is None:
                raise Exception("No social costmap")
            return self._social_costmap[current[0]][current[1]]

        def exit_condition(current: GridCellModel, goal: ModuleNotFoundError):
            return False

        _, _, came_from, _, gscore, _ = graph_search.new_generic_a_star(
            start=robot_cell,
            goal=None,
            exit_condition=exit_condition,
            get_neighbors=get_neighbors_for_evasion,
            heuristic=evasion_heuristic,
        )  # type: ignore

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

        if not use_combined_cost:
            min_social_cost_index = np.argmin(social_cost)
            evasion_cell = accessible_cells[min_social_cost_index]
        else:
            normalized_social_cost = (social_cost - np.min(social_cost)) / np.ptp(
                social_cost
            )
            normalized_distance_cost = (distance_cost - np.min(distance_cost)) / np.ptp(
                distance_cost
            )
            combined_cost = (
                self.w_social * normalized_social_cost
                + self.w_obs * normalized_distance_cost
            ) / (self.w_social + self.w_obs)
            min_combined_cost_index = np.argmin(combined_cost)
            evasion_cell = accessible_cells[min_combined_cost_index]

            if self.activate_grids_logging:
                sorted_cell_to_combined_cost = OrderedDict(
                    sorted(
                        zip(accessible_cells, combined_cost),
                        key=lambda t: t[1],
                        reverse=True,
                    )
                )
                self.log_grids(
                    inflated_grid_by_robot,
                    accessible_cells,
                    normalized_social_cost,
                    normalized_distance_cost,
                    sorted_cell_to_combined_cost,
                )

            # ros_publisher.publish_combined_costmap(
            #     sorted_cell_to_combined_cost,
            #     inflated_grid_by_robot,
            #     ns=self.name,
            # )
            # ros_publisher.cleanup_grid_map(ns=self.name)

        raw_cell_path = graph_search.reconstruct_path(came_from, evasion_cell)
        real_path = utils.grid_path_to_real_path(
            raw_cell_path,
            robot_pose,
            None,
            inflated_grid_by_robot.res,
            inflated_grid_by_robot.grid_pose,
        )

        if len(real_path) < 2:
            return robot_start_social_cost, None

        evasion_transit_path = EvasionTransitPath.from_poses(
            real_path, robot_polygon, robot_pose
        )

        if transit_configuration_after_release:
            evasion_transit_path.set_transit_configuration_after_release(
                transit_configuration_after_release
            )

        evasion_cell_social_cost = t.cast(
            float, self._social_costmap[evasion_cell[0]][evasion_cell[1]]
        )
        return evasion_cell_social_cost, evasion_transit_path

    def h(self, r_i: PoseModel, r_j: PoseModel):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        # rotation_cost = self.rotation_factor * (abs(r_j[2] - r_i[2]) % 180.)
        return translation_cost  # + rotation_cost

    def g(self, r_i: PoseModel, r_j: PoseModel, is_transfer: float = False):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        rotation_cost = self.rotation_factor * abs(r_j[2] - r_i[2])
        return (translation_cost + rotation_cost) * (
            1.0 if not is_transfer else self.transfer_coefficient
        )

    def get_transfer_path_from_config(
        self,
        prev_transit_end_configuration: RobotConfiguration | None,
        next_transit_start_configuration: RobotConfiguration,
        transfer_configurations: t.List[RobotObstacleConfiguration],
        obstacle_uid: UID,
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

        grab_action: ba.Grab = actions[0] if prev_transit_end_configuration else None  # type: ignore

        if not isinstance(next_transit_start_configuration.action, ba.Release):
            raise Exception(
                "The next transit start configuration after a transfer should start with a release action"
            )
        release_action: ba.Release = next_transit_start_configuration.action
        actions.append(release_action)

        robot_poses = [
            configuration.robot.floating_point_pose
            for configuration in transfer_configurations
        ]
        robot_poses.append(next_transit_start_configuration.floating_point_pose)
        robot_polygons = [
            configuration.robot.polygon for configuration in transfer_configurations
        ]
        robot_polygons.append(next_transit_start_configuration.polygon)
        robot_csv_polygons = {
            (i + 1,): config.robot.csv_polygon
            for i, config in enumerate(transfer_configurations)
        }
        robot_csv_polygons[
            (len(transfer_configurations),)
        ] = next_transit_start_configuration.csv_polygon
        robot_bb_vertices = [
            config.robot.bb_vertices
            for config in transfer_configurations
            if config.robot.bb_vertices
        ]
        robot_bb_vertices.append(next_transit_start_configuration.bb_vertices)
        if prev_transit_end_configuration:
            robot_poses.insert(0, prev_transit_end_configuration.floating_point_pose)
            robot_polygons.insert(0, prev_transit_end_configuration.polygon)
            robot_csv_polygons[(0,)] = prev_transit_end_configuration.csv_polygon
            robot_bb_vertices.insert(0, prev_transit_end_configuration.bb_vertices)

        robot_path = Path(
            poses=robot_poses,
            polygons=robot_polygons,
            csv_polygons=robot_csv_polygons,
            bb_vertices=robot_bb_vertices,
        )

        obstacle_path = Path(
            poses=[
                configuration.obstacle.floating_point_pose
                for configuration in transfer_configurations
            ],
            polygons=[
                configuration.obstacle.polygon
                for configuration in transfer_configurations
            ],
            csv_polygons={
                (i + 1,): config.obstacle.csv_polygon
                for i, config in enumerate(transfer_configurations)
            },
            bb_vertices=[
                config.obstacle.bb_vertices for config in transfer_configurations
            ],
        )
        obstacle_path.poses.append(obstacle_path.poses[-1])
        obstacle_path.polygons.append(obstacle_path.polygons[-1])
        obstacle_path.bb_vertices.append([])
        if prev_transit_end_configuration:
            obstacle_path.poses.insert(0, obstacle_path.poses[0])
            obstacle_path.polygons.insert(0, obstacle_path.polygons[0])
            obstacle_path.bb_vertices.insert(0, [])

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
        return Stilman2005Agent(
            uid=self.uid,
            navigation_goals=copy.deepcopy(self._navigation_goals),
            params=copy.deepcopy(self.params),
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
            cell_size=self.cell_size,
            logger=self.logger,
        )
