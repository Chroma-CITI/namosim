import copy
import heapq
import random
import time
import typing as t
from collections import OrderedDict

import numpy as np
from shapely import Polygon
from shapely.geometry import Point

import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
import namosim.utils.collision as collision
import namosim.utils.connectivity as connectivity
import namosim.worldreps.occupation_based.social_topological_occupation_cost_grid as stocg
from namosim.behaviors.algorithms import graph_search
from namosim.behaviors.algorithms.new_local_opening_check import check_new_local_opening
from namosim.behaviors.baseline_behavior import BaselineBehavior
from namosim.display.ros2_publisher import RosPublisher
from namosim.models import (
    FixedPrecisionPoseModel,
    PoseModel,
    StilmanBehaviorConfigModel,
)
from namosim.navigation.conflict import (
    ConcurrentGrabConflict,
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
from namosim.navigation.navigation_plan import Plan
from namosim.utils import utils
from namosim.worldreps.entity_based.obstacle import Obstacle
from namosim.worldreps.entity_based.robot import Robot
from namosim.worldreps.occupation_based.binary_occupancy_grid import (
    BinaryInflatedOccupancyGrid,
    BinaryOccupancyGrid,
)


class RCHConfiguration(object):
    def __init__(
        self,
        cell: t.Tuple[int, int],
        first_obstacle_uid: str,
        first_component_uid: str,
    ):
        self.cell = cell
        self.first_obstacle_uid = first_obstacle_uid
        self.first_component_uid = first_component_uid

    def __eq__(self, other: object):
        if isinstance(other, tuple):
            return self.cell == other
        elif isinstance(other, RCHConfiguration):
            return (
                self.cell == other.cell
                and self.first_obstacle_uid == other.first_obstacle_uid
                and self.first_component_uid == other.first_component_uid
            )
        else:
            raise Exception("Invalid comparison")

    def __hash__(self):
        return hash((self.cell, self.first_obstacle_uid, self.first_component_uid))


class Configuration:
    def __init__(
        self,
        floating_point_pose: PoseModel,
        polygon: Polygon,
        cell_in_grid: t.Tuple[int, int],
        fixed_precision_pose: FixedPrecisionPoseModel,
        action: ba.BasicAction,
        csv_polygon=None,
        bb_vertices=None,
    ):
        self.floating_point_pose = floating_point_pose
        self.polygon = polygon
        self.cell_in_grid = cell_in_grid
        self.fixed_precision_pose = fixed_precision_pose
        self.action = action
        self.csv_polygon = csv_polygon
        self.bb_vertices = bb_vertices

    def __eq__(self, other):
        if isinstance(other, graph_search.HeapNode):
            return self.fixed_precision_pose == other.element.fixed_precision_pose
        elif isinstance(other, tuple):
            return self.fixed_precision_pose == other
        else:
            return self.fixed_precision_pose == other.fixed_precision_pose

    def __hash__(self):
        return hash(self.fixed_precision_pose)


class RobotObstacleConfiguration(object):
    def __init__(
        self,
        robot_floating_point_pose: PoseModel,
        robot_polygon: Polygon,
        robot_cell_in_grid,
        robot_fixed_precision_pose,
        obstacle_floating_point_pose,
        obstacle_polygon,
        obstacle_cell_in_grid,
        obstacle_fixed_precision_pose,
        action=None,
        manip_pose_id=None,
        robot_csv_polygon=None,
        robot_bb_vertices=None,
        obstacle_csv_polygon=None,
        obstacle_bb_vertices=None,
    ):
        self.robot = Configuration(
            robot_floating_point_pose,
            robot_polygon,
            robot_cell_in_grid,
            robot_fixed_precision_pose,
            action=action,
            csv_polygon=robot_csv_polygon,
            bb_vertices=robot_bb_vertices,
        )
        self.obstacle = Configuration(
            obstacle_floating_point_pose,
            obstacle_polygon,
            obstacle_cell_in_grid,
            obstacle_fixed_precision_pose,
            action=action,
            csv_polygon=obstacle_csv_polygon,
            bb_vertices=obstacle_bb_vertices,
        )
        self.action = action
        self.manip_pose_id = manip_pose_id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, graph_search.HeapNode):
            return (
                self.robot.fixed_precision_pose
                == other.element.robot.fixed_precision_pose
                and self.obstacle.fixed_precision_pose
                == other.element.obstacle.fixed_precision_pose
            )
        elif isinstance(other, tuple):
            return (
                self.robot.fixed_precision_pose == other[0]
                and self.obstacle.fixed_precision_pose == other[1]
            )
        elif isinstance(other, RobotObstacleConfiguration):
            return (
                self.robot.fixed_precision_pose == other.robot.fixed_precision_pose
                and self.obstacle.fixed_precision_pose
                == other.obstacle.fixed_precision_pose
            )
        else:
            raise Exception("Invalid comparison")

    def __hash__(self):
        return hash(
            (self.robot.fixed_precision_pose, self.obstacle.fixed_precision_pose)
        )


# class RobotObstacleConflict(Conflict):
#     def __init__(self, obstacle_uid, robot_uid, robot_pose, colliding_uids,
#                  robot_transfered_obstacle_uid=None, robot_transfered_obstacle_pose=None):
#         self.obstacle_uid = obstacle_uid
#         self.robot_uid = robot_uid
#         self.robot_pose = robot_pose
#         self.colliding_uids = colliding_uids
#         self.robot_transfered_obstacle_uid = robot_transfered_obstacle_uid
#         self.robot_transfered_obstacle_pose = robot_transfered_obstacle_pose
#
#     def __str__(self):
#         s = "Robot-Obstacle conflict between robot uid {} with obstacle uid {}.".format(
#             self.robot_uid, self.obstacle_uid
#         )
#
#         robot_state = (
#             "in transit" if self.robot_transfered_obstacle_uid is None
#             else "transfering obstacle uid {}".format(self.robot_transfered_obstacle_uid)
#         )
#         robot_transfered_obstacle_pose_text = (
#             "" if robot_transfered_obstacle_pose is None
#             else ", robot's transfered obstacle: {}".format(robot_transfered_obstacle_pose)
#         )
#
#         s += " Collision detected between entities {} at configuration: robot: {}.".format(
#             self.colliding_uids, self.robot_pose, robot_transfered_obstacle_pose_text
#         )
#
#         return s


class Timer:
    def __init__(self, start_time=0, duration=0, is_running=False):
        self.start_time = start_time
        self.duration = duration
        self.is_running = is_running

    def start_timer(self, start_time, duration):
        self.start_time = start_time
        self.duration = duration
        self.is_running = True

    def is_timer_over(self, current_time):
        if current_time - self.start_time >= self.duration:
            self.is_running = False
            return True
        else:
            return False


class DynamicPlan(Plan):
    DEBUGGING_WAIT_TIME_GENERATOR = []

    def __init__(self):
        Plan.__init__(self)
        # Core attributes
        self.plan_counter = 0
        self.steps_with_replan_call = set()

        #       # Statistical attributes

        self.current_conflicts = []

        self.plan_history = {}
        self.conflicts_history = {}
        self.postponements_history = {}
        self.unpostponements_history = []

        self.forbidden_evasion_cells = set()

        self.timer = Timer()

    def was_last_step_success(self, w_t, last_action_result):
        # TODO Check if robot state (position and grab) are coherent with next step's preconditions
        return isinstance(last_action_result, ar.ActionSuccess)

    def get_conflicts(
        self,
        world,
        inflated_grid_by_robot,
        step_count,
        check_horizon=None,
        apply_strict_horizon=False,
        exit_early_for_any_conflict=False,
        exit_early_only_for_long_term_conflicts=True,
        rp=None,
        robot_name="",
    ):
        conflicts = Plan.get_conflicts(
            self,
            world,
            inflated_grid_by_robot,
            step_count,
            check_horizon=check_horizon,
            apply_strict_horizon=apply_strict_horizon,
            exit_early_for_any_conflict=exit_early_for_any_conflict,
            exit_early_only_for_long_term_conflicts=exit_early_only_for_long_term_conflicts,
            rp=rp,
            robot_name=robot_name,
        )
        self.current_conflicts += conflicts
        return conflicts

    def save_conflicts(self, step_count):
        if self.current_conflicts:
            if step_count in self.conflicts_history:
                self.conflicts_history[step_count] += self.current_conflicts
            else:
                self.conflicts_history[step_count] = self.current_conflicts
        self.current_conflicts = []

    def has_tries_remaining(self, nb_max_tries):
        return self.plan_counter < nb_max_tries

    def can_even_be_found(self):
        if (
            self.plan_error
            and self.plan_error == "start_or_goal_cell_in_static_obstacle_error"
        ):
            return False
        return True

    # Actions
    def pop_next_action(self):
        return Plan.pop_next_action(self)

    def new_postpone(
        self, t_min, t_max, step_count, conflicts, simulation_log, robot_name
    ):
        if self.timer.is_running:
            if self.timer.is_timer_over(step_count):
                simulation_log.append(
                    utils.BasicLog(
                        "Agent {}: Resetting plan because conflicts still exist after full postponement is over: {}.".format(
                            robot_name, conflicts
                        ),
                        step_count,
                    )
                )
                self.update_plan(Plan([]), step_count)
            else:
                return ba.Wait()
        else:
            duration = random.randint(t_min, t_max)
            simulation_log.append(
                utils.BasicLog(
                    "Agent {}: Starting postponement of current plan for {} steps because conflicts: {}.".format(
                        robot_name, duration, conflicts
                    ),
                    step_count,
                )
            )
            self.timer.start_timer(step_count, duration)
            self.postponements_history[step_count] = duration
            return ba.Wait()

    # def postpone(self, t_min, t_max, step_count):
    #     if self.DEBUGGING_WAIT_TIME_GENERATOR:
    #         self.wait_counter = self.DEBUGGING_WAIT_TIME_GENERATOR.pop(0)
    #     else:
    #         self.wait_counter = random.randint(t_min, t_max)
    #     self.wait_counter = t_max  # TODO - Reconsider the computation of the wait time
    #     self.postponements_history[
    #         step_count] = self.wait_counter

    # def unpostpone(self, step_count):
    #     self.wait_counter = 0
    #     self.unpostponements_history.append(step_count)

    def update_plan(self, plan, step_count):
        if step_count in self.plan_history:
            self.plan_history[step_count].append(plan)
        else:
            self.plan_history[step_count] = [plan]

        self.path_components = plan.path_components
        self.goal = plan.goal
        self.robot_uid = plan.robot_uid
        self.phys_cost = plan.phys_cost
        self.social_cost = plan.social_cost
        self.total_cost = plan.total_cost
        self.plan_error = plan.plan_error
        self.component_index = plan.component_index


class Stilman2005Behavior(BaselineBehavior):
    def __init__(
        self,
        initial_world,
        robot_uid,
        navigation_goals,
        behavior_config: StilmanBehaviorConfigModel,
        abs_path_to_logs_dir,
        ros_publisher: RosPublisher,
    ):
        BaselineBehavior.__init__(
            self,
            initial_world,
            robot_uid,
            navigation_goals,
            behavior_config,
            abs_path_to_logs_dir,
            ros_publisher=ros_publisher,
        )

        # Configuration parameters
        parameters = behavior_config.parameters

        # For each, specify collision model, action space
        # self.transit_search_config =
        # self.transfer_search_config =  # Include new opening detection parameters and social cost parameters
        # self.grab_search_config =
        # self.release_search_config =
        # self.obstacle_selection_config =
        # self.plan_execution_config =

        # - Original Stilman method configuration parameters
        self.alpha = parameters.alpha_for_obstacle_choice_heur
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD  # default if bad parameter
        # self.heur_w = parameters["heuristic_cost_for_traversing_obstacle_in_choice_heur"]
        # self.basic_trans_force = parameters["basic_translation_force"]
        # self.basic_rot_moment = parameters["basic_rotation_moment"]
        self.translation_unit_cost = 1.0
        self.rotation_unit_cost = 1.0
        self.transfer_coefficient = 2.0  # Note: MUST ALWAYS BE > 1 !
        # - Robot action space parameters
        self.angular_res = parameters.collision_check_angular_res
        self.rotation_unit_angle = 60.0  # parameters["robot_rotation_unit_angle"]
        self.translation_unit_length = parameters.robot_translation_unit_length
        self.forbid_rotations = parameters.forbid_rotations
        self.translation_factor = (
            self.translation_unit_cost / self.translation_unit_length
        )
        self.rotation_factor = self.rotation_unit_cost / self.rotation_unit_angle
        self.absolute_translations = True
        self.robot_base_drive_type = "holonomic"
        # self.robot_base_drive_type = "differential"
        self.trans_mult = 1.0 / self._world.discretization_data.res * 10.0
        self.rot_mult = 1.0

        # - S-NAMO parameters
        self.use_social_cost = parameters.use_social_cost
        self.bound_percentage = parameters.solution_interval_bound_percentage
        if parameters.manipulation_search_procedure == "DFS":
            if self.use_social_cost:
                self.manip_search_procedure = self.focused_manip_search
            else:
                raise ValueError(
                    "Focused manipulation search requires the use_social_cost variable to be True !"
                )
        elif parameters.manipulation_search_procedure == "BFS":
            self.manip_search_procedure = self.manip_search
        self.w_social, self.w_obs, self.w_goal = 15.0, 10.0, 2.0
        self.w_sum = self.w_social + self.w_obs + self.w_goal
        self.distance_to_obs_cost_is_realistic = True

        # - Extra performance parameters
        self.check_new_local_opening_before_global = (
            parameters.check_new_local_opening_before_global
        )
        self.activate_grids_logging = True  # not parameters["deactivate_grids_logging"]

        if self.robot_base_drive_type == "differential":
            self._trans_vectors = np.array(
                [
                    (self.translation_unit_length, 0.0),
                    (-self.translation_unit_length, 0.0),
                ]
            )
        elif self.robot_base_drive_type == "holonomic":
            self._trans_vectors = np.array(
                [
                    (self.translation_unit_length, 0.0),
                    (-self.translation_unit_length, 0.0),
                    (0.0, self.translation_unit_length),
                    (0.0, -self.translation_unit_length),
                ]
            )

        if self.forbid_rotations:
            self._rot_angles = np.array([])
        else:
            self._rot_angles = np.array(
                [self.rotation_unit_angle, -self.rotation_unit_angle]
            )
        self._all_rot_angles = self.rotation_unit_angle * np.array(
            range(1, 360 // int(self.rotation_unit_angle))
        )
        self._nb_possible_angles = len(self._all_rot_angles)

        if self.absolute_translations:
            self._new_actions = []
            for trans_vector in self._trans_vectors:
                self._new_actions.append(ba.AbsoluteTranslation(trans_vector))
            for rot_angle in self._rot_angles:
                self._new_actions.append(ba.Rotation(rot_angle))
        else:
            self._new_actions = []
            for trans_vector in self._trans_vectors:
                self._new_actions.append(ba.Translation(trans_vector))
            for rot_angle in self._rot_angles:
                self._new_actions.append(ba.Rotation(rot_angle))

        self._social_costmap = None

        self.is_first_transfer_step = False

        self.check_horizon = 10

        self.angular_tolerance = 0.1
        self.position_tolerance = self._world.discretization_data.res / 2.0

        self.min_nb_steps_to_wait = 5
        self.max_nb_steps_to_wait = 20

        # Initialize movability status of obstacles
        for entity in self._world.entities.values():
            if entity.movability != "static":
                entity.movability = self._robot.deduce_movability(entity.type_)

        self.replan_count = 20
        self.goal_to_plans = OrderedDict()

        self.action_space_reduction = (
            "only_r_acc_then_c_1_x"  # ['none', 'only_r_acc', 'only_r_acc_then_c_1_x']
        )

        # Initialize static obstacles occupation grid, since it is not supposed to change
        static_obs_polygons = {
            uid: entity.polygon
            for uid, entity in self._world.entities.items()
            if (
                isinstance(entity, Obstacle)
                and entity.movability == "unmovable"
                or entity.movability == "static"
            )
        }
        self.robot_max_inflation_radius = utils.get_circumscribed_radius(
            self._robot.polygon
        )
        self.static_obs_inf_grid = BinaryInflatedOccupancyGrid(
            static_obs_polygons,
            self._world.discretization_data.res,
            self.robot_max_inflation_radius,
            neighborhood=self.neighborhood,
        )
        self.static_obs_grid = BinaryOccupancyGrid(
            static_obs_polygons,
            self._world.discretization_data.res,
            neighborhood=self.neighborhood,
            params=self.static_obs_inf_grid.params,
        )
        all_entities_polygons = {
            uid: e.polygon for uid, e in self._world.entities.items()
        }
        self.inflated_grid_by_robot = BinaryInflatedOccupancyGrid(
            all_entities_polygons,
            self._world.discretization_data.res,
            self.robot_max_inflation_radius,
            neighborhood=self.neighborhood,
            params=self.static_obs_inf_grid.params,
        )  # TODO Make sure static and generalist grid share same width and height (occurs naturally if map borders are static, but not otherwise)
        self.inflated_grid_by_robot.deactivate_entities({self._robot.uid})

        # Initialize social costmap as None for computation in first think
        self._social_costmap = None

        # Init first goal
        if self._q_goal is None:
            if self._navigation_goals:
                self._q_goal = self._navigation_goals.pop(
                    0
                )  # TODO Stop popping goals, use an index
                self._p_opt = DynamicPlan()
                self.goal_to_plans[self._q_goal] = self._p_opt
            else:
                return ba.GoalsFinished()

    def init_social_costmap(self):
        # Initialize social occupation costmap
        if self.use_social_cost and self._social_costmap is None:
            self._social_costmap = stocg.compute_social_costmap(
                self.static_obs_grid.grid,
                self._world.discretization_data.res,
                ros_publisher=self._rp,
                log_costmaps=self.activate_grids_logging,
                abs_path_to_logs_dir=self.abs_path_to_logs_dir,
                ns=self._robot_name,
            )
            self._rp.publish_social_grid_map(
                self._social_costmap,
                self._world.discretization_data.res,
                ns=self._robot_name,
            )
            pass

    def are_all_goals_finished(self):
        return not self._navigation_goals and self._q_goal is None

    def is_goal_success(self, q_r):
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

    def potential_deadlocks(self, current_conflicts, dynamic_plan, current_step):
        rr_conflicts = [
            conflict
            for conflict in current_conflicts
            if isinstance(conflict, RobotRobotConflict)
        ]
        return {
            conflict
            for past_step, past_conflicts_at_step in dynamic_plan.conflicts_history.items()
            for conflict in rr_conflicts
            if (
                conflict in past_conflicts_at_step
                and [
                    replan_step
                    for replan_step in dynamic_plan.steps_with_replan_call
                    if replan_step >= past_step
                ]
            )
        }

    def sense(self, ref_world, last_action_result, step_count):
        # Update baseline world representation (polygons)
        BaselineBehavior.sense(self, ref_world, last_action_result, step_count)

        # Update grid(s)
        self.inflated_grid_by_robot.polygon_update(
            new_or_updated_polygons={
                uid: self._world.entities[uid].polygon
                for uid in self._added_uids.union(self._updated_uids)
                if uid != self._robot_uid
            },
            removed_polygons=self._removed_uids,
        )

    def think(self):
        if self._social_costmap is None:
            self.init_social_costmap()

        if self._q_goal is None:
            if self._navigation_goals:
                self._q_goal = self._navigation_goals.pop(
                    0
                )  # TODO Stop popping goals, use an index
                self._p_opt = DynamicPlan()
                self.goal_to_plans[self._q_goal] = self._p_opt
            else:
                return ba.GoalsFinished()

        next_step = self.full_coordination_strategy(
            self._world,
            self.static_obs_inf_grid,
            self.inflated_grid_by_robot,
            self._robot_uid,
            self._q_goal,
            self._p_opt,
            self.check_horizon,
            self.replan_count,
            self.min_nb_steps_to_wait,
            self.max_nb_steps_to_wait,
            self.position_tolerance,
            self.angular_tolerance,
            self.neighborhood,
            self._step_count,
            self.trans_mult,
            self.rot_mult,
            self.action_space_reduction,
        )
        self._p_opt.save_conflicts(self._step_count)

        if isinstance(next_step, (ba.GoalSuccess, ba.GoalFailed)):
            self._rp.cleanup_conflicts_checks(ns=self._robot_name)
            self._q_goal = None

        return next_step

    def is_goal_reached(self, q_t, q_f, pos_tol=0.05, ang_tol=0.1):
        return all(
            [
                utils.is_close(q_t[0], q_f[0], rel_tol=pos_tol),
                utils.is_close(q_t[1], q_f[1], rel_tol=pos_tol),
                utils.angle_is_close(q_t[2], q_f[2], rel_tol=ang_tol),
            ]
        )

    def must_replan_now(self, conflicts):
        for conflict in conflicts:
            if isinstance(conflict, (StolenMovableConflict, RobotObstacleConflict)):
                return True
        return False

    def full_coordination_strategy(
        self,
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
    ):
        # If current robot pose is close enough to goal, return Success
        if self.is_goal_reached(w_t.entities[robot_uid].pose, goal, pos_tol, ang_tol):
            return ba.GoalSuccess(goal)

        if not plan.exists():
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: Absence of plan requires immediate replanning.".format(
                        self._robot_name
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
            )
        else:
            if plan.is_evasion_over():
                self.simulation_log.append(
                    utils.BasicLog(
                        "Agent {}: Finished evasion sequence, replanning.".format(
                            self._robot_name
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
                )

            conflicts = plan.get_conflicts(
                w_t,
                inflated_grid_by_robot,
                step_count,
                fov,
                rp=self._rp,
                robot_name=self._robot_name,
            )
            if not conflicts:
                if plan.timer.is_running and plan.timer.is_timer_over(step_count):
                    self.simulation_log.append(
                        utils.BasicLog(
                            "Agent {}: No more conflicts, unpostponing current plan.".format(
                                self._robot_name
                            ),
                            step_count,
                        )
                    )
                    plan.timer.is_running = False
                    plan.unpostponements_history.append(step_count)
                return plan.pop_next_action()  # Normal case, don't log
            else:
                if self.use_social_cost:
                    potential_deadlocks = self.potential_deadlocks(
                        conflicts, plan, step_count
                    )
                    if potential_deadlocks:
                        if plan.timer.is_running and not plan.timer.is_timer_over(
                            step_count
                        ):
                            return ba.Wait()

                        self.simulation_log.append(
                            utils.BasicLog(
                                "Agent {}: Potential deadlocks detected: {}.".format(
                                    self._robot_name, potential_deadlocks
                                ),
                                step_count,
                            )
                        )

                        if not plan.has_tries_remaining(try_max):
                            self.simulation_log.append(
                                utils.BasicLog(
                                    "Agent {}: Failing goal, no tries remaining to plan an evasion.".format(
                                        self._robot_name
                                    ),
                                    step_count,
                                )
                            )
                            return ba.GoalFailed(goal)

                        robot_cells = utils.accurate_rasterize_in_grid(
                            w_t.entities[robot_uid].polygon,
                            inflated_grid_by_robot.res,
                            inflated_grid_by_robot.grid_pose,
                            inflated_grid_by_robot.d_width,
                            inflated_grid_by_robot.d_height,
                            fill=True,
                        )
                        plan.forbidden_evasion_cells.update(set(robot_cells))
                        plan.plan_counter += 1
                        evasion_path = self.compute_evasion(
                            inflated_grid_by_robot,
                            w_t,
                            robot_uid,
                            potential_deadlocks,
                            plan.forbidden_evasion_cells,
                        )
                        if evasion_path:
                            self.simulation_log.append(
                                utils.BasicLog(
                                    "Agent {}: Executing evasion path.".format(
                                        self._robot_name
                                    ),
                                    step_count,
                                )
                            )
                            plan.update_plan(
                                Plan([evasion_path], goal, self._robot_uid), step_count
                            )
                            self._rp.cleanup_p_opt(ns=self._robot_name)
                            self._rp.publish_p_opt(
                                self._p_opt, self._robot, ns=self._robot_name
                            )
                            return plan.pop_next_action()
                        else:
                            self.simulation_log.append(
                                utils.BasicLog(
                                    "Agent {}: I can not or should not evade, postponing...".format(
                                        self._robot_name,
                                    ),
                                    step_count,
                                )
                            )
                            return plan.new_postpone(
                                t_min,
                                t_max,
                                step_count,
                                conflicts,
                                self.simulation_log,
                                self._robot_name,
                            )
                if not self.must_replan_now(conflicts):
                    return plan.new_postpone(
                        t_min,
                        t_max,
                        step_count,
                        conflicts,
                        self.simulation_log,
                        self._robot_name,
                    )
                else:
                    self.simulation_log.append(
                        utils.BasicLog(
                            "Agent {}: Detected conflicts require immediate replanning".format(
                                self._robot_name
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
                    )

    def replan(
        self,
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
    ):
        if not plan.has_tries_remaining(try_max):
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: Failing goal, no tries remaining to plan even while ignoring dynamic obstacles.".format(
                        self._robot_name
                    ),
                    step_count,
                )
            )
            return ba.GoalFailed(goal)
        else:
            plan.steps_with_replan_call.add(step_count)

            # I - Compute plan (ignoring dynamic obstacles) and set it to current plan
            dynamic_entities = {
                uid
                for uid, entity in w_t.entities.items()
                if (
                    (isinstance(entity, Robot) and uid != robot_uid)
                    or (
                        uid in w_t.entity_to_agent
                        and w_t.entity_to_agent[uid] != robot_uid
                    )
                )
            }
            w_t_no_dyn = w_t.light_copy(ignored_entities=dynamic_entities)
            inflated_grid_by_robot.deactivate_entities(dynamic_entities)
            plan.plan_counter += 1
            p = self.select_connect(
                w_t_no_dyn,
                static_obs_inf_grid,
                inflated_grid_by_robot,
                goal,
                trans_mult,
                rot_mult,
                neighborhood=neighborhood,
                action_space_reduction=action_space_reduction,
            )
            inflated_grid_by_robot.activate_entities(dynamic_entities)
            plan.update_plan(p, step_count)
            self._rp.cleanup_p_opt(ns=self._robot_name)
            self._rp.publish_p_opt(self._p_opt, self._robot, ns=self._robot_name)

            if not plan.exists():
                self.simulation_log.append(
                    utils.BasicLog(
                        "Agent {}: Failing goal, no plan could be found when ignoring dynamic obstacles.".format(
                            self._robot_name
                        ),
                        step_count,
                    )
                )
                return ba.GoalFailed(goal)
            else:
                conflicts = plan.get_conflicts(
                    w_t,
                    inflated_grid_by_robot,
                    step_count,
                    fov,
                    rp=self._rp,
                    robot_name=self._robot_name,
                )
                if not conflicts:
                    self.simulation_log.append(
                        utils.BasicLog(
                            "Agent {}: Found a pure NAMO plan without conflicts with dynamic obstacles, "
                            "executing its first step...".format(self._robot_name),
                            step_count,
                        )
                    )
                    return plan.pop_next_action()
                else:
                    self.simulation_log.append(
                        utils.BasicLog(
                            "Agent {}: A new plan has been computed ignoring dynamic "
                            "obstacles but has conflicts with them: {}".format(
                                self._robot_name, conflicts
                            ),
                            step_count,
                        )
                    )

                    if not (
                        plan.has_tries_remaining(try_max) and plan.can_even_be_found()
                    ):
                        self.simulation_log.append(
                            utils.BasicLog(
                                "Agent {}: Failing goal, no tries remaining to plan after conflicts "
                                "were found with the plan ignoring dynamic obstacles.".format(
                                    self._robot_name,
                                ),
                                step_count,
                            )
                        )
                        return ba.GoalFailed(goal)
                    else:
                        # II - Compute plan (with conflicting dynamic obstacles as static)
                        # Get uids of conflicting robots and associated
                        conflicting_robots_uids = {
                            conflict.other_robot_uid for conflict in conflicts
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
                        new_w_t_no_dyn = w_t.light_copy(
                            ignored_entities=new_dynamic_entities
                        )
                        for conflict in conflicts:
                            if (
                                isinstance(conflict, ConcurrentGrabConflict)
                                and conflict.obstacle_uid
                                not in new_w_t_no_dyn.entity_to_agent
                            ):
                                new_w_t_no_dyn.entity_to_agent[
                                    conflict.obstacle_uid
                                ] = conflict.other_robot_uid
                        inflated_grid_by_robot.deactivate_entities(new_dynamic_entities)
                        # Iterate over each conflicting robot uid, and change its polygon to an encompassing circle
                        # encounting for all likely states at at t+1
                        polygons_tmp = {}
                        for conflicting_robot_uid in conflicting_robots_uids:
                            conflicting_robot = new_w_t_no_dyn.entities[
                                conflicting_robot_uid
                            ]
                            center = conflicting_robot.polygon.centroid
                            robot_radius = (
                                center.hausdorff_distance(conflicting_robot.polygon)
                                + 1.1 * inflated_grid_by_robot.res
                            )
                            radius = robot_radius
                            min_radius_for_release = (
                                robot_radius
                                + inflated_grid_by_robot.inflation_radius
                                + 2.0 * inflated_grid_by_robot.res
                            )
                            # Enlarge radius to account for possible grabs
                            for uid, obstacle in new_w_t_no_dyn.entities.items():
                                if (
                                    isinstance(obstacle, Obstacle)
                                    and uid not in new_w_t_no_dyn.entity_to_agent
                                    and obstacle.movability != "static"
                                ):
                                    if obstacle.polygon.buffer(
                                        2.0 * inflated_grid_by_robot.inflation_radius,
                                        join_style=2,
                                    ).intersects(conflicting_robot.polygon):
                                        radius = min_radius_for_release
                                        break
                            if conflicting_robot.uid in w_t.entity_to_agent.inverse:
                                obstacle = w_t.entities[
                                    w_t.entity_to_agent.inverse[conflicting_robot.uid]
                                ]
                                radius = max(
                                    radius,
                                    center.hausdorff_distance(obstacle.polygon)
                                    + 1.1 * inflated_grid_by_robot.res,
                                )
                                if radius < min_radius_for_release:
                                    # Enlarge radius to account for possible releases
                                    radius = min_radius_for_release
                            # TODO Get inflation from largest robot
                            encompassing_circle = center.buffer(radius)
                            polygons_tmp[
                                conflicting_robot_uid
                            ] = conflicting_robot.polygon
                            conflicting_robot.polygon = encompassing_circle
                            inflated_grid_by_robot.polygon_update(
                                {conflicting_robot_uid: conflicting_robot.polygon}
                            )
                        # Plan using this modified version of the world
                        plan.plan_counter += 1
                        p = self.select_connect(
                            new_w_t_no_dyn,
                            static_obs_inf_grid,
                            inflated_grid_by_robot,
                            goal,
                            trans_mult,
                            rot_mult,
                            neighborhood=neighborhood,
                            action_space_reduction=action_space_reduction,
                        )
                        # Reset the inflated grid's state
                        for conflicting_uid, prev_polygon in polygons_tmp.items():
                            inflated_grid_by_robot.polygon_update(
                                {conflicting_uid: prev_polygon}
                            )
                        inflated_grid_by_robot.activate_entities(new_dynamic_entities)

                        if not p.exists():
                            self.simulation_log.append(
                                utils.BasicLog(
                                    "Agent {}: Postponing for {} steps, could not find a plan avoiding the conflicting "
                                    "dynamic obstacles of the pure NAMO plan.".format(
                                        self._robot_name, t_max
                                    ),
                                    step_count,
                                )
                            )
                            return plan.new_postpone(
                                t_min,
                                t_max,
                                step_count,
                                conflicts,
                                self.simulation_log,
                                self._robot_name,
                            )
                        else:
                            plan.update_plan(p, step_count)
                            self._rp.cleanup_p_opt(ns=self._robot_name)
                            self._rp.publish_p_opt(
                                self._p_opt, self._robot, ns=self._robot_name
                            )

                            conflicts = plan.get_conflicts(
                                w_t,
                                inflated_grid_by_robot,
                                step_count,
                                fov,
                                rp=self._rp,
                                robot_name=self._robot_name,
                            )
                            if conflicts:
                                self.simulation_log.append(
                                    utils.BasicLog(
                                        "Agent {}: Postponing for {} steps, a new plan has been computed avoiding the "
                                        "conflicting dynamic obstacles of the pure NAMO plan, but has other conflicts: {}".format(
                                            self._robot_name, t_max, conflicts
                                        ),
                                        step_count,
                                    )
                                )
                                return plan.new_postpone(
                                    t_min,
                                    t_max,
                                    step_count,
                                    conflicts,
                                    self.simulation_log,
                                    self._robot_name,
                                )
                            else:
                                self.simulation_log.append(
                                    utils.BasicLog(
                                        "Agent {}: Found a new plan that does not have conflicts with the dynamic obstacles "
                                        "conflicting with the pure NAMO plan, executing its first step...".format(
                                            self._robot_name
                                        ),
                                        step_count,
                                    )
                                )

                                return plan.pop_next_action()

    def select_connect(
        self,
        w_t,
        static_obs_inf_grid,
        inflated_grid_by_robot_max,
        r_f,
        trans_mult,
        rot_mult,
        ccs_data=None,
        prev_list=set(),
        neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
        action_space_reduction="only_r_acc_then_c_1_x",
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
        robot = w_t.entities[self._robot_uid]
        r_t = robot.pose

        avoid_list = set()

        robot_cell = utils.real_to_grid(
            r_t[0], r_t[1], static_obs_inf_grid.res, static_obs_inf_grid.grid_pose
        )
        goal_cell = utils.real_to_grid(
            r_f[0], r_f[1], static_obs_inf_grid.res, static_obs_inf_grid.grid_pose
        )

        simple_path_to_goal = self.find_path(
            r_t, r_f, w_t, inflated_grid_by_robot_max, robot.polygon
        )
        if simple_path_to_goal:
            # If the goal is in the same free space component as the robot in simulated w_t
            # Orig. condition in pseudo-code is : x^f in C^acc_R(W)
            # TODO FIX COST COMPUTATION TO FIT SAME MODEL AS MANIP SEARCH !
            self._rp.cleanup_robot_sim(ns=self._robot_name)
            return Plan([simple_path_to_goal], r_f, self._robot_uid)

        if ccs_data is None:
            ccs_data = connectivity.init_ccs_for_grid(
                inflated_grid_by_robot_max.grid,
                inflated_grid_by_robot_max.d_width,
                inflated_grid_by_robot_max.d_height,
                neighborhood,
            )
        connected_components_grid = ccs_data.grid
        self._rp.publish_connected_components_grid(
            connected_components_grid, w_t.discretization_data.res, ns=robot.name
        )

        c_0 = ccs_data.grid[robot_cell[0]][robot_cell[1]]
        prev_list = prev_list if c_0 == 0 else prev_list.union({c_0})
        r_acc_cells = (
            set()
            if inflated_grid_by_robot_max.grid[robot_cell[0]][robot_cell[1]] > 0
            else connectivity.bfs_init(
                inflated_grid_by_robot_max.grid,
                inflated_grid_by_robot_max.d_width,
                inflated_grid_by_robot_max.d_height,
                robot_cell,
                neighborhood,
            ).visited
        )

        if inflated_grid_by_robot_max.only_obstacle_uid_in_cell(robot_cell) == -1:
            return Plan(plan_error="start_cell_in_several_movable_obstacles_error")

        if (
            static_obs_inf_grid.grid[robot_cell[0]][robot_cell[1]] > 0
            or static_obs_inf_grid.grid[goal_cell[0]][goal_cell[1]] > 0
        ):
            return Plan(plan_error="start_or_goal_cell_in_static_obstacle_error")

        # if inflated_grid_by_robot_max.grid[goal_cell[0]][goal_cell[1]] > 1: Should not be necessary thanks to first check
        #     return Plan(plan_error="goal_cell_in_more_than_one_movable_obstacle_error")

        forbidden_obstacles = {  # Dynamic obstacles are forbidden !
            uid
            for uid, entity in w_t.entities.items()
            if (
                (isinstance(entity, Robot) and uid != self._robot.uid)
                or (
                    uid in w_t.entity_to_agent
                    and w_t.entity_to_agent[uid] != self._robot.uid
                )
            )
        }
        o_1, c_1 = self.rch(
            robot_cell,
            goal_cell,
            static_obs_inf_grid,
            connected_components_grid,
            inflated_grid_by_robot_max,
            avoid_list,
            prev_list,
            forbidden_obstacles,
            neighborhood,
        )
        while o_1 != 0:
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: select_connect: selected entity {} for manipulation search to reach component {}.".format(
                        robot.name, w_t.entities[o_1].name, c_1
                    ),
                    self._step_count,
                )
            )
            if action_space_reduction == "none":
                w_t_plus_2, tho_m = self.manip_search_procedure(
                    w_t,
                    o_1,
                    c_1,
                    ccs_data,
                    r_acc_cells,
                    r_f,
                    inflated_grid_by_robot_max,
                    trans_mult,
                    rot_mult,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=True,
                )
            elif action_space_reduction == "only_r_acc":
                w_t_plus_2, tho_m = self.manip_search_procedure(
                    w_t,
                    o_1,
                    c_1,
                    ccs_data,
                    r_acc_cells,
                    r_f,
                    inflated_grid_by_robot_max,
                    trans_mult,
                    rot_mult,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=False,
                )
            elif action_space_reduction == "only_r_acc_then_c_1_x":
                w_t_plus_2, tho_m = self.manip_search_procedure(
                    w_t,
                    o_1,
                    c_1,
                    ccs_data,
                    r_acc_cells,
                    r_f,
                    inflated_grid_by_robot_max,
                    trans_mult,
                    rot_mult,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=False,
                )
                if tho_m is None:
                    w_t_plus_2, tho_m = self.manip_search_procedure(
                        w_t,
                        o_1,
                        c_1,
                        ccs_data,
                        r_acc_cells,
                        r_f,
                        inflated_grid_by_robot_max,
                        trans_mult,
                        rot_mult,
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
                self.simulation_log.append(
                    utils.BasicLog(
                        "Agent {}: select_connect: found partial plan manipulating entity {} to reach component {}.".format(
                            robot.name, w_t.entities[o_1].name, c_1
                        ),
                        self._step_count,
                    )
                )
                prev_cells_sets = inflated_grid_by_robot_max.polygon_update(
                    {o_1: w_t_plus_2.entities[o_1].polygon}
                )
                future_plan = self.select_connect(
                    w_t_plus_2,
                    static_obs_inf_grid,
                    inflated_grid_by_robot_max,
                    r_f,
                    trans_mult,
                    rot_mult,
                    ccs_data=ccs_data,
                    prev_list=(prev_list if c_1 == 0 else prev_list.union({c_1})),
                    neighborhood=neighborhood,
                    action_space_reduction=action_space_reduction,
                )
                inflated_grid_by_robot_max.cells_sets_update(prev_cells_sets)
                if not future_plan.plan_error:
                    tho_n = self.find_path(
                        r_t,
                        tho_m.robot_path.poses[0],
                        w_t,
                        inflated_grid_by_robot_max,
                        robot.polygon,
                    )
                    if not tho_n:
                        raise ValueError(
                            "It should not be possible not to find a transit path when the transfer path is found."
                        )
                    plan_components = [tho_n, tho_m] if tho_n.actions else [tho_m]
                    return Plan(plan_components, r_f, self._robot_uid).append(
                        future_plan
                    )

            # Extra check for when the goal is in a movable obstacle that we could not find how to move
            if c_1 == 0:
                self.simulation_log.append(
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
                robot_cell,
                goal_cell,
                static_obs_inf_grid,
                connected_components_grid,
                inflated_grid_by_robot_max,
                avoid_list,
                prev_list,
                forbidden_obstacles,
                neighborhood,
            )

        self._rp.cleanup_robot_sim(ns=self._robot_name)
        return Plan(plan_error="no_plan_found_error")

    def rch_get_neighbors(
        self,
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
        neighborhood=utils.TAXI_NEIGHBORHOOD,
    ):
        """
        Combined formulation from Stilman's thesis and his article.
        """
        neighbors, tentative_gscores = [], []
        current_gscore = gscore[current]
        path_has_traversed_first_disconnected_comp = current.first_component_uid != 0
        path_has_traversed_first_obstacle = current.first_obstacle_uid != 0

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
                cur_cell_obs_uid = inflated_robot_grid.only_obstacle_uid_in_cell(
                    current.cell
                )
                neighbor_cell_obs_uid = inflated_robot_grid.only_obstacle_uid_in_cell(
                    neighbor_cell
                )

                cur_and_neighbor_not_in_mult_obs = (
                    cur_cell_obs_uid != -1 and neighbor_cell_obs_uid != -1
                )
                current_or_neighbor_in_free_space = (
                    cur_cell_obs_uid == 0 or neighbor_cell_obs_uid == 0
                )
                transition_is_valid = (
                    cur_and_neighbor_not_in_mult_obs
                    and (
                        current_or_neighbor_in_free_space
                        or cur_cell_obs_uid == neighbor_cell_obs_uid
                    )
                    and neighbor_cell_obs_uid != current.first_obstacle_uid
                )
                if transition_is_valid:
                    neighbor = RCHConfiguration(
                        neighbor_cell,
                        current.first_obstacle_uid,
                        current.first_component_uid,
                    )
            else:
                neighbor_cell_component_uid = connected_components_grid[
                    neighbor_cell[0]
                ][neighbor_cell[1]]
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
                                neighbor_cell,
                                current.first_obstacle_uid,
                                neighbor_cell_component_uid,
                            )
                        else:
                            # Either the neighbor tries to go back to robot acc. space, or in a (obs., comp.)
                            # combination that has already been explored and for which no manip. could be found
                            pass

                    else:
                        neighbor_cell_obs_uid = (
                            inflated_robot_grid.only_obstacle_uid_in_cell(neighbor_cell)
                        )
                        if neighbor_cell_obs_uid == current.first_obstacle_uid:
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
                        neighbor_cell_obstacle_uid = (
                            inflated_robot_grid.only_obstacle_uid_in_cell(neighbor_cell)
                        )
                        if neighbor_cell_obstacle_uid > 0:
                            neighbor = RCHConfiguration(
                                neighbor_cell, neighbor_cell_obstacle_uid, 0
                            )
                        else:
                            # The neighbor is in multiple obstacles, which is forbidden
                            pass
            if (
                neighbor is not None
                and neighbor not in close_set
                and neighbor.first_obstacle_uid not in forbidden_obstacles
            ):
                neighbors.append(neighbor)
                tentative_gscores.append(
                    current_gscore
                    + g_function(
                        current,
                        neighbor,
                        is_transfer=inflated_robot_grid.grid[neighbor.cell[0]][
                            neighbor.cell[1]
                        ]
                        > 0,
                    )
                )
                traversed_obstacles_ids.add(neighbor.first_obstacle_uid)

        self._rp.publish_rch_data(
            current,
            gscore,
            close_set,
            open_queue,
            came_from,
            neighbors,
            traversed_obstacles_ids,
            inflated_robot_grid.res,
            inflated_robot_grid.grid_pose,
            ns=self._robot_name,
        )

        return neighbors, tentative_gscores

    def rch(
        self,
        start_cell,
        goal_cell,
        static_obs_grid,
        connected_components_grid,
        inflated_robot_grid,
        avoid_list,
        prev_list,
        forbidden_obstacles,
        neighborhood=utils.TAXI_NEIGHBORHOOD,
    ):
        if static_obs_grid.grid[start_cell[0]][start_cell[1]] > 0:
            obstacle_names = {
                self._world.entities[uid].name
                for uid in static_obs_grid.obstacles_uids_in_cell(start_cell)
            }
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot start cell {} in a rch call must always be outside of static obstacles, here: {}.".format(
                        self._robot_name, start_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        if static_obs_grid.grid[goal_cell[0]][goal_cell[1]] > 0:
            obstacle_names = {
                self._world.entities[uid].name
                for uid in static_obs_grid.obstacles_uids_in_cell(goal_cell)
            }
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot goal cell {} in a rch call must always be outside of static obstacles, here: {}.".format(
                        self._robot_name, goal_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        start_obstacle_uid = inflated_robot_grid.only_obstacle_uid_in_cell(start_cell)
        if start_obstacle_uid == -1 or start_obstacle_uid in forbidden_obstacles:
            obstacle_names = {
                self._world.entities[uid].name
                for uid in inflated_robot_grid.obstacles_uids_in_cell(start_cell)
            }
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot start cell {} in a rch call must always be at most in one obstacle and not a forbidden one, here: {}.".format(
                        self._robot_name, start_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        if inflated_robot_grid.grid[goal_cell[0]][goal_cell[1]] > 1:
            obstacle_names = {
                self._world.entities[uid].name
                for uid in inflated_robot_grid.obstacles_uids_in_cell(goal_cell)
            }
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot goal cell {} in a rch call must be at most within one movable obstacle, here: {}.".format(
                        self._robot_name, goal_cell, obstacle_names
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

        def g_function(current, neighbor, is_transfer=False):
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

        def h_function(_c, _g):
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
            current, gscore, close_set, open_queue, came_from
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
                neighborhood,
            )

        def exit_condition(_current, _goal):
            return _current.cell == _goal.cell

        start = RCHConfiguration(
            start_cell, start_obstacle_uid if start_obstacle_uid > 0 else 0, 0
        )
        goal = RCHConfiguration(
            goal_cell, 0, 0
        )  # Note the zeroes are never used, this line is just for coherence

        path_found, end_config, _, _, _, _ = graph_search.new_generic_a_star(
            start, goal, exit_condition, rch_get_neighbors_instance, h_function
        )
        if path_found:
            if end_config.first_obstacle_uid == 0:
                raise ValueError(
                    "Rch found a path where no obstacle needed to be traversed."
                )
            return end_config.first_obstacle_uid, end_config.first_component_uid
        else:
            return 0, 0

    def manip_search(
        self,
        w_t,
        o_1,
        c_1,
        ccs_data,
        r_acc_cells,
        r_f,
        inflated_grid_by_robot_max,
        trans_mult,
        rot_mult,
        check_new_local_opening_before_global=True,
        obstacle_can_intrude_r_acc=True,
        obstacle_can_intrude_c_1_x=True,
    ):
        # Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = copy.deepcopy(w_t)

        self._rp.publish_robot_sim_world(
            w_t_plus_2, self._robot_uid, ns=self._robot_name
        )

        c_1_cells_set = set() if c_1 == 0 else ccs_data.ccs[c_1].visited

        res = w_t_plus_2.discretization_data.res

        other_entities = [
            entity
            for entity in w_t_plus_2.entities.values()
            if entity.uid != self._robot.uid and entity.uid != o_1
        ]
        other_entities_polygons = {
            entity.uid: entity.polygon for entity in other_entities
        }
        other_entities_aabb_tree = collision.polygons_to_aabb_tree(
            other_entities_polygons
        )

        robot = w_t_plus_2.entities[self._robot.uid]
        robot_uid, robot_pose, robot_polygon, robot_name = (
            robot.uid,
            robot.pose,
            robot.polygon,
            robot.name,
        )
        robot_cell = utils.real_to_grid(
            robot_pose[0],
            robot_pose[1],
            inflated_grid_by_robot_max.res,
            inflated_grid_by_robot_max.grid_pose,
        )
        robot_min_inflation_radius = utils.get_inscribed_radius(robot_polygon)
        robot_max_inflation_radius = utils.get_circumscribed_radius(robot_polygon)

        obstacle = w_t_plus_2.entities[o_1]
        obstacle_uid, obstacle_pose, obstacle_polygon = (
            obstacle.uid,
            obstacle.pose,
            obstacle.polygon,
        )
        obstacle_min_inflation_radius = utils.get_inscribed_radius(obstacle_polygon)

        inf_robot, inf_obstacle = copy.deepcopy(robot), copy.deepcopy(obstacle)
        inf_robot.polygon, inf_obstacle.polygon = (
            robot.polygon.buffer(res, join_style=2),
            obstacle.polygon.buffer(res, join_style=2),
        )

        goal_pose, goal_cell = (
            r_f,
            utils.real_to_grid(
                r_f[0], r_f[1], res, inflated_grid_by_robot_max.grid_pose
            ),
        )

        # Get accessible sampled navigation points around obstacle
        (
            transfer_start_configs_to_cost,
            transfer_start_to_prev_transit_end,
        ) = self.get_transfer_start_to_transit_end_and_cost(
            robot_polygon,
            robot_pose,
            robot_uid,
            obstacle_uid,
            other_entities_polygons,
            other_entities_aabb_tree,
            inflated_grid_by_robot_max,
            ccs_data,
            r_acc_cells,
            obstacle_pose,
            obstacle_polygon,
            trans_mult,
            rot_mult,
        )

        if not transfer_start_configs_to_cost:
            # If there are no attainable manipulation configurations, exit early
            self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)
            return w_t_plus_2, None

        # CAREFUL : We inflate by inscribed radius MINUS sqrt(2)*res to make sure occupied cells are really where the
        # entity's center should NEVER be to avoid collisions.
        # Poses in free cells of this grid may sometimes be colliding.
        inflated_grid_by_robot_min = BinaryInflatedOccupancyGrid(
            other_entities_polygons,
            res,
            max(robot_min_inflation_radius - utils.SQRT_OF_2 * res, 0.0),
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
        )
        inflated_grid_by_obstacle = BinaryInflatedOccupancyGrid(
            other_entities_polygons,
            res,
            obstacle_min_inflation_radius - utils.SQRT_OF_2 * res,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
            params=inflated_grid_by_robot_max.params,
        )
        # Only deactivate obstacle cells once transit end and transfer start are computed (grab action)
        inflated_grid_by_robot_max.deactivate_entities([obstacle_uid])

        # Use Dijkstra algorithm to compute a transfer path that allows for an opening to be created
        (
            path_found,
            transfer_end_configuration,
            came_from,
            close_set,
            gscore,
            _,
        ) = self.dijkstra_for_manip_search(
            transfer_start_configs_to_cost,
            robot_uid,
            robot_name,
            obstacle_uid,
            obstacle_polygon,
            other_entities_polygons,
            other_entities_aabb_tree,
            inflated_grid_by_robot_min,
            inflated_grid_by_robot_max,
            inflated_grid_by_obstacle,
            r_acc_cells,
            c_1_cells_set,
            ccs_data,
            trans_mult,
            rot_mult,
            check_new_local_opening_before_global,
            goal_pose,
            goal_cell,
            obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
            obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
        )
        if path_found:
            # self._rp.publish_sim(
            #     transfer_end_configuration.robot.polygon, transfer_end_configuration.obstacle.polygon,
            #     "/target", ns=self._robot_name
            # )
            raw_path: t.List[
                RobotObstacleConfiguration
            ] = graph_search.reconstruct_path(came_from, transfer_end_configuration)

            prev_transit_end_configuration = transfer_start_to_prev_transit_end[
                raw_path[0]
            ]
            next_transit_start_configuration = (
                self.get_next_transit_start_configuration(
                    inflated_grid_by_robot_max,
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

        self._rp.publish_robot_sim_world(
            w_t_plus_2, self._robot_uid, ns=self._robot_name
        )
        self._rp.cleanup_robot_sim(ns=self._robot_name)
        self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)

        inflated_grid_by_robot_max.activate_entities([obstacle_uid])

        return w_t_plus_2, tho_m

    def focused_manip_search(
        self,
        w_t,
        o_1,
        c_1,
        ccs_data,
        r_acc_cells,
        r_f,
        inflated_grid_by_robot_max,
        trans_mult,
        rot_mult,
        check_new_local_opening_before_global=True,
        obstacle_can_intrude_r_acc=True,
        obstacle_can_intrude_c_1_x=True,
    ):
        # Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = copy.deepcopy(w_t)
        self._rp.publish_robot_sim_world(
            w_t_plus_2, self._robot_uid, ns=self._robot_name
        )

        c_1_cells_set = set() if c_1 == 0 else ccs_data.ccs[c_1].visited

        res = w_t_plus_2.discretization_data.res

        other_entities = [
            entity
            for entity in w_t_plus_2.entities.values()
            if entity.uid != self._robot.uid and entity.uid != o_1
        ]
        other_entities_polygons = {
            entity.uid: entity.polygon for entity in other_entities
        }
        other_entities_aabb_tree = collision.polygons_to_aabb_tree(
            other_entities_polygons
        )

        robot = w_t_plus_2.entities[self._robot.uid]
        robot_uid, robot_pose, robot_name = robot.uid, robot.pose, robot.name
        robot_cell = utils.real_to_grid(
            robot_pose[0],
            robot_pose[1],
            inflated_grid_by_robot_max.res,
            inflated_grid_by_robot_max.grid_pose,
        )
        robot_polygon = robot.polygon
        robot_min_inflation_radius = utils.get_inscribed_radius(robot_polygon)
        robot_max_inflation_radius = utils.get_circumscribed_radius(robot_polygon)

        obstacle = w_t_plus_2.entities[o_1]
        obstacle_uid, obstacle_pose = obstacle.uid, obstacle.pose
        obstacle_polygon = obstacle.polygon
        obstacle_min_inflation_radius = utils.get_inscribed_radius(obstacle_polygon)

        goal_pose, goal_cell = (
            r_f,
            utils.real_to_grid(
                r_f[0], r_f[1], res, inflated_grid_by_robot_max.grid_pose
            ),
        )

        # Get accessible sampled navigation points around obstacle
        (
            transfer_start_configs_to_cost,
            transfer_start_to_prev_transit_end,
        ) = self.get_transfer_start_to_transit_end_and_cost(
            robot_polygon,
            robot_pose,
            robot_uid,
            obstacle_uid,
            other_entities_polygons,
            other_entities_aabb_tree,
            inflated_grid_by_robot_max,
            ccs_data,
            r_acc_cells,
            obstacle_pose,
            obstacle_polygon,
            trans_mult,
            rot_mult,
        )

        if not transfer_start_configs_to_cost:
            # If there are no attainable manipulation configurations, exit early
            self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)
            return w_t_plus_2, None

        # CAREFUL : We inflate by inscribed radius MINUS sqrt(2)*res to make sure occupied cells are really where the
        # entity's center should NEVER be to avoid collisions.
        # Poses in free cells of this grid may sometimes be colliding.
        inflated_grid_by_robot_min = BinaryInflatedOccupancyGrid(
            other_entities_polygons,
            res,
            max(robot_min_inflation_radius - utils.SQRT_OF_2 * res, 0.0),
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
        )
        inflated_grid_by_obstacle = BinaryInflatedOccupancyGrid(
            other_entities_polygons,
            res,
            obstacle_min_inflation_radius - utils.SQRT_OF_2 * res,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
            params=inflated_grid_by_robot_max.params,
        )
        inflated_grid_by_robot_max.deactivate_entities([obstacle_uid])

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
            robot_pose,
            robot_polygon,
            robot_name,
            robot_uid,
            obstacle_uid,
            obstacle_pose,
            obstacle_polygon,
            goal_pose,
            goal_cell,
            other_entities_polygons,
            other_entities_aabb_tree,
            inflated_grid_by_robot_max,
            cells_sorted_by_combined_cost,
            r_acc_cells,
            c_1_cells_set,
            ccs_data,
            transfer_start_configs_to_cost.keys(),
            trans_mult,
            rot_mult,
            gscore=None,
            close_set=None,
            check_new_local_opening_before_global=check_new_local_opening_before_global,
            obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
            obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
        )
        if best_transfer_end_configuration is not None:
            self._rp.publish_sim(
                best_transfer_end_configuration.robot.polygon,
                best_transfer_end_configuration.obstacle.polygon,
                "/target",
                ns=self._robot_name,
            )

            # 2. If a best obstacle transfer end configuration has been found, use A Star to find a path toward it
            (
                path_found,
                transfer_end_configuration,
                came_from,
                close_set,
                gscore,
                _,
            ) = self.a_star_for_manip_search(
                transfer_start_configs_to_cost,
                best_transfer_end_configuration,
                robot_uid,
                robot_name,
                obstacle_uid,
                obstacle_polygon,
                other_entities_polygons,
                other_entities_aabb_tree,
                inflated_grid_by_robot_min,
                inflated_grid_by_robot_max,
                inflated_grid_by_obstacle,
                r_acc_cells,
                c_1_cells_set,
                ccs_data,
                trans_mult,
                rot_mult,
                sorted_cell_to_combined_cost,
                bound_quantile,
                check_new_local_opening_before_global,
                goal_pose,
                goal_cell,
                obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
            )
            if path_found:
                # 3. If a path is found, return it
                # self._rp.publish_sim(
                #     transfer_end_configuration.robot.polygon, transfer_end_configuration.obstacle.polygon,
                #     "/target", ns=self._robot_name
                # )
                raw_path = graph_search.reconstruct_path(
                    came_from, transfer_end_configuration
                )
                prev_transit_end_configuration = transfer_start_to_prev_transit_end[
                    raw_path[0]
                ]
                next_transit_start_configuration = (
                    self.get_next_transit_start_configuration(
                        inflated_grid_by_robot_max,
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
                    robot_pose,
                    robot_polygon,
                    robot_name,
                    robot_uid,
                    obstacle_uid,
                    obstacle_pose,
                    obstacle_polygon,
                    goal_pose,
                    goal_cell,
                    other_entities_polygons,
                    other_entities_aabb_tree,
                    inflated_grid_by_robot_max,
                    cells_sorted_by_combined_cost,
                    r_acc_cells,
                    c_1_cells_set,
                    ccs_data,
                    transfer_start_configs_to_cost.keys(),
                    trans_mult,
                    rot_mult,
                    gscore=gscore,
                    close_set=close_set,
                    check_new_local_opening_before_global=check_new_local_opening_before_global,
                    obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                    obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
                )
                if best_transfer_end_configuration is not None:
                    # self._rp.publish_sim(
                    #     best_transfer_end_configuration.robot.polygon, best_transfer_end_configuration.obstacle.polygon,
                    #     "/target", ns=self._robot_name
                    # )
                    raw_path = graph_search.reconstruct_path(
                        came_from, best_transfer_end_configuration
                    )
                    prev_transit_end_configuration = transfer_start_to_prev_transit_end[
                        raw_path[0]
                    ]
                    next_transit_start_configuration = (
                        self.get_next_transit_start_configuration(
                            inflated_grid_by_robot_max,
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
                    tho_m_phys_cost = gscore[transfer_end_configuration] + self.g(
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

        self._rp.publish_robot_sim_world(
            w_t_plus_2, self._robot_uid, ns=self._robot_name
        )
        self._rp.cleanup_robot_sim(ns=self._robot_name)
        self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)

        inflated_grid_by_robot_max.activate_entities([obstacle_uid])

        return w_t_plus_2, tho_m

    def dijkstra_for_manip_search(
        self,
        start,
        robot_uid,
        robot_name,
        obstacle_uid,
        obstacle_polygon,
        other_entities_polygons,
        other_entities_aabb_tree,
        inflated_grid_by_robot_min,
        inflated_grid_by_robot_max,
        inflated_grid_by_obstacle,
        r_acc_cells,
        c_1_cells_set,
        ccs_data,
        trans_mult,
        rot_mult,
        check_new_local_opening_before_global,
        overall_goal_pose,
        overall_goal_cell,
        obstacle_can_intrude_r_acc=True,
        obstacle_can_intrude_c_1_x=True,
    ):
        def get_neighbors(_current, _gscore, _close_set, _open_queue, _came_from):
            return self.get_neighbors(
                _current,
                _gscore,
                _close_set,
                _open_queue,
                _came_from,
                start,
                inflated_grid_by_robot_min,
                inflated_grid_by_robot_max,
                inflated_grid_by_obstacle,
                r_acc_cells,
                ccs_data,
                robot_uid,
                obstacle_uid,
                trans_mult,
                rot_mult,
                other_entities_polygons,
                other_entities_aabb_tree,
                obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
            )

        def exit_condition(_current):
            next_transit_start_configuration = (
                self.get_next_transit_start_configuration(
                    inflated_grid_by_robot_max,
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
                    inflated_grid_by_robot_max,
                    c_1_cells_set,
                    overall_goal_pose,
                    overall_goal_cell,
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
        start,
        goal,
        robot_uid,
        robot_name,
        obstacle_uid,
        obstacle_polygon,
        other_entities_polygons,
        other_entities_aabb_tree,
        inflated_grid_by_robot_min,
        inflated_grid_by_robot_max,
        inflated_grid_by_obstacle,
        r_acc_cells,
        c_1_cells_set,
        ccs_data,
        trans_mult,
        rot_mult,
        sorted_cell_to_combined_cost,
        bound_quantile,
        check_new_local_opening_before_global,
        overall_goal_pose,
        overall_goal_cell,
        obstacle_can_intrude_r_acc=True,
        obstacle_can_intrude_c_1_x=True,
    ):
        def get_neighbors(_current, _gscore, _close_set, _open_queue, _came_from):
            neighbors, tentative_g_scores = self.get_neighbors(
                _current,
                _gscore,
                _close_set,
                _open_queue,
                _came_from,
                start,
                inflated_grid_by_robot_min,
                inflated_grid_by_robot_max,
                inflated_grid_by_obstacle,
                r_acc_cells,
                ccs_data,
                robot_uid,
                obstacle_uid,
                trans_mult,
                rot_mult,
                other_entities_polygons,
                other_entities_aabb_tree,
                obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
            )
            return neighbors, tentative_g_scores

        def heuristic(_neighbor, _goal):
            return self.h(
                _neighbor.robot.floating_point_pose, _goal.robot.floating_point_pose
            )

        def flexible_exit_condition(_current, _goal):
            if _current == _goal:
                return True

            if _current.obstacle.cell_in_grid not in sorted_cell_to_combined_cost:
                # TODO Remove this TEMPORARY condition caused by sometimes missing cell in sorted_cell_to_combined_cost
                return False

            upper_bound = (1.0 + self.bound_percentage) * sorted_cell_to_combined_cost[
                _goal.obstacle.cell_in_grid
            ]
            current_cell_cc_within_bound = (
                sorted_cell_to_combined_cost[_current.obstacle.cell_in_grid]
                < upper_bound
            )

            if current_cell_cc_within_bound:
                next_transit_start_configuration = (
                    self.get_next_transit_start_configuration(
                        inflated_grid_by_robot_max,
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
                        inflated_grid_by_robot_max,
                        c_1_cells_set,
                        overall_goal_pose,
                        overall_goal_cell,
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
        self, obstacle_polygon, inflated_grid_by_robot_max
    ):
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
        :param inflated_grid_by_robot_max:
        :type inflated_grid_by_robot_max:
        :return: the lists of valid transit end poses and corresponding valid transfer start poses
        :rtype: tuple(list(tuple(float, float, float)), list(tuple(float, float, float)))
        """
        candidate_transfer_start_poses = utils.sample_poses_at_middle_of_inflated_sides(
            obstacle_polygon, inflated_grid_by_robot_max.inflation_radius
        )
        candidate_transit_end_poses = utils.sample_poses_at_middle_of_inflated_sides(
            obstacle_polygon,
            inflated_grid_by_robot_max.inflation_radius
            + 1.5 * inflated_grid_by_robot_max.res,
        )

        valid_transit_end_poses, valid_transfer_start_poses = [], []
        for transit_end_pose, transfer_start_pose in zip(
            candidate_transit_end_poses, candidate_transfer_start_poses
        ):
            valid_transit_end_poses.append(transit_end_pose)
            valid_transfer_start_poses.append(transfer_start_pose)

        self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)
        self._rp.publish_q_manips_for_obs(
            valid_transfer_start_poses, ns=self._robot_name
        )

        return valid_transit_end_poses, valid_transfer_start_poses

    def get_transfer_start_to_transit_end_and_cost(
        self,
        robot_polygon,
        robot_pose,
        robot_uid,
        obstacle_uid,
        other_entities_polygons,
        other_entities_aabb_tree,
        inflated_grid_by_robot_max,
        ccs_data,
        r_acc_cells,
        obstacle_pose,
        obstacle_polygon,
        trans_mult,
        rot_mult,
    ):
        robot_cell = utils.real_to_grid(
            robot_pose[0],
            robot_pose[1],
            inflated_grid_by_robot_max.res,
            inflated_grid_by_robot_max.grid_pose,
        )
        cell_in_manip_obs = (
            inflated_grid_by_robot_max.only_obstacle_uid_in_cell(robot_cell)
            == obstacle_uid
        )

        if cell_in_manip_obs:
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
                    inflated_grid_by_robot_max.res,
                    inflated_grid_by_robot_max.grid_pose,
                ),
                obstacle_floating_point_pose=obstacle_pose,
                obstacle_polygon=obstacle_polygon,
                obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                    obstacle_pose, trans_mult, rot_mult
                ),
                obstacle_cell_in_grid=utils.real_to_grid(
                    obstacle_pose[0],
                    obstacle_pose[1],
                    inflated_grid_by_robot_max.res,
                    inflated_grid_by_robot_max.grid_pose,
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
            obstacle_polygon, inflated_grid_by_robot_max
        )

        transfer_start_to_transit_end_robot_pose = {
            manip_pose: nav_pose
            for nav_pose, manip_pose in zip(
                transit_end_robot_poses, transfer_start_robot_poses
            )
        }

        transfer_start_configs_to_cost = {}
        transfer_start_to_prev_transit_end = {}
        for manip_pose_id, (transfer_start_pose, transit_end_pose) in enumerate(
            transfer_start_to_transit_end_robot_pose.items()
        ):
            transit_end_cell = utils.real_to_grid(
                transit_end_pose[0],
                transit_end_pose[1],
                inflated_grid_by_robot_max.res,
                inflated_grid_by_robot_max.grid_pose,
            )

            if transit_end_cell not in r_acc_cells:
                continue

            prev_transit_end_robot_polygon = utils.set_polygon_pose(
                robot_polygon, robot_pose, transit_end_pose
            )

            grab_action = ba.Grab(
                translation_vector=(
                    utils.euclidean_distance(transfer_start_pose, transit_end_pose),
                    0.0,
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
                bb_vertices,
            ) = collision.csv_check_collisions(
                main_uid=robot_uid,
                other_polygons=other_entities_polygons,
                polygon_sequence=[
                    prev_transit_end_robot_polygon,
                    transfer_start_robot_polygon,
                ],
                action_sequence=[
                    collision.convert_action(grab_action, transit_end_pose)
                ],
                bb_type="minimum_rotated_rectangle",
                aabb_tree=other_entities_aabb_tree,
            )

            if not collides_with:
                prev_transit_end_configuration = Configuration(
                    floating_point_pose=transit_end_pose,
                    polygon=prev_transit_end_robot_polygon,
                    cell_in_grid=utils.real_to_grid(
                        transit_end_pose[0],
                        transit_end_pose[1],
                        inflated_grid_by_robot_max.res,
                        inflated_grid_by_robot_max.grid_pose,
                    ),
                    fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                        transit_end_pose, trans_mult, rot_mult
                    ),
                    action=None,
                    csv_polygon=prev_transit_end_robot_polygon,
                    bb_vertices=list(prev_transit_end_robot_polygon.exterior.coords),
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
                        inflated_grid_by_robot_max.res,
                        inflated_grid_by_robot_max.grid_pose,
                    ),
                    obstacle_floating_point_pose=obstacle_pose,
                    obstacle_polygon=obstacle_polygon,
                    obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                        obstacle_pose, trans_mult, rot_mult
                    ),
                    obstacle_cell_in_grid=utils.real_to_grid(
                        obstacle_pose[0],
                        obstacle_pose[1],
                        inflated_grid_by_robot_max.res,
                        inflated_grid_by_robot_max.grid_pose,
                    ),
                    manip_pose_id=manip_pose_id,
                    action=grab_action,
                    robot_csv_polygon=csv_polygons[(0,)],
                    robot_bb_vertices=bb_vertices[0],
                    obstacle_csv_polygon=obstacle_polygon,
                    obstacle_bb_vertices=list(obstacle_polygon.exterior.coords),
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
        robot_pose,
        robot_polygon,
        robot_name,
        robot_uid,
        obstacle_uid,
        obstacle_pose,
        obstacle_polygon,
        goal_pose,
        goal_cell,
        other_entities_polygons,
        other_entities_aabb_tree,
        inflated_grid_by_robot_max,
        ordered_cells_by_cost,
        r_acc_cells,
        c_1_cells_set,
        ccs_data,
        init_robot_manip_configs,
        trans_mult,
        rot_mult,
        gscore=None,
        close_set=None,
        check_new_local_opening_before_global=True,
        obstacle_can_intrude_r_acc=True,
        obstacle_can_intrude_c_1_x=True,
    ):
        if close_set:
            # If all reachable configurations have been explored, index them by obstacle cell
            obs_cell_to_reachable_configurations = {}
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
                            inflated_grid_by_robot_max.aabb_polygon
                        ):
                            continue
                        if not configuration.obstacle.polygon.within(
                            inflated_grid_by_robot_max.aabb_polygon
                        ):
                            continue

                        #   2. ... allows sufficient space for the robot to release the object, ...
                        next_transit_start_configuration = (
                            self.get_next_transit_start_configuration(
                                inflated_grid_by_robot_max,
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
                                inflated_grid_by_robot_max,
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
                                    inflated_grid_by_robot_max,
                                    c_1_cells_set,
                                    goal_pose,
                                    goal_cell,
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
                        inflated_grid_by_robot_max.res,
                        inflated_grid_by_robot_max.grid_pose,
                    )

                    # If the obstacle collides at this pose, don't consider checking further
                    obstacle_transfer_end_poly = utils.set_polygon_pose(
                        obstacle_polygon, obstacle_pose, obstacle_pose_at_transfer_end
                    )
                    collides_with = collision.check_static_collision(
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
                        collides_with = collision.check_static_collision(
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
                            inflated_grid_by_robot_max.aabb_polygon
                        ):
                            continue

                        next_transit_start_configuration = (
                            self.get_next_transit_start_configuration(
                                inflated_grid_by_robot_max,
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
                                inflated_grid_by_robot_max.aabb_polygon
                            ):
                                continue

                            #   2bis. ..., does not intrude forbidden component(s), ...
                            intrudes = self.polygon_intrudes_components(
                                obstacle_transfer_end_poly,
                                inflated_grid_by_robot_max,
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
                                    obstacle_transfer_end_poly,
                                    other_entities_polygons,
                                    other_entities_aabb_tree,
                                    inflated_grid_by_robot_max,
                                    c_1_cells_set,
                                    goal_pose,
                                    goal_cell,
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
                                            inflated_grid_by_robot_max.res,
                                            inflated_grid_by_robot_max.grid_pose,
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
                                            inflated_grid_by_robot_max.res,
                                            inflated_grid_by_robot_max.grid_pose,
                                        ),
                                    )
                ordered_cells_by_cost.pop()
        return None  # If no valid configuration could be found...

    @staticmethod
    def get_next_transit_start_configuration(
        grid,
        robot_pose,
        robot_polygon,
        robot_uid,
        obstacle_uid,
        obstacle_pose,
        other_entities_polygons,
        other_entities_aabb_tree,
        trans_mult,
        rot_mult,
    ):
        release_action = ba.Release(
            translation_vector=(-1.0 * (grid.inflation_radius + 1.5 * grid.res), 0.0),
            entity_uid=obstacle_uid,
        )
        new_robot_pose = release_action.predict_pose(robot_pose, robot_pose[2])
        old_cell = utils.real_to_grid(
            robot_pose[0], robot_pose[1], grid.res, grid.grid_pose
        )
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

        new_robot_polygon = release_action.apply(robot_polygon, robot_pose)

        # Check if robot is still within map bounds
        if not new_robot_polygon.within(grid.aabb_polygon):
            return None

        # Finally, we check dynamic collisions (between init configuration and after-action configuration)
        (
            _,
            collides_with,
            _,
            csv_polygons,
            intersections,
            bb_vertices,
        ) = collision.csv_check_collisions(
            main_uid=robot_uid,
            other_polygons=other_entities_polygons,
            polygon_sequence=[robot_polygon, new_robot_polygon],
            action_sequence=[collision.convert_action(release_action, robot_pose)],
            bb_type="minimum_rotated_rectangle",
            aabb_tree=other_entities_aabb_tree,
        )

        if not collides_with:
            new_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
                new_robot_pose, trans_mult, rot_mult
            )
            next_transit_start_configuration = Configuration(
                new_robot_pose,
                new_robot_polygon,
                cell,
                new_fixed_precision_pose,
                release_action,
                csv_polygon=csv_polygons[(0,)],
                bb_vertices=bb_vertices[0],
            )
            return next_transit_start_configuration
        else:
            return None

    def is_there_opening_to_c_1(
        self,
        check_new_local_opening_before_global,
        robot_name,
        robot_cell,
        obstacle_uid,
        old_obstacle_polygon,
        new_obstacle_polygon,
        other_entities_polygons,
        other_entities_aabb_tree,
        inflated_grid_by_robot_max,
        c_1_cells_set,
        goal_pose,
        goal_cell,
        neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
        init_blocking_areas=None,
        init_entity_inflated_polygon=None,
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
                old_obstacle_polygon,
                new_obstacle_polygon,
                other_entities_polygons,
                other_entities_aabb_tree,
                inflated_grid_by_robot_max.inflation_radius,
                goal_pose,
                ros_publisher=self._rp,
                init_blocking_areas=init_blocking_areas,
                init_entity_inflated_polygon=init_entity_inflated_polygon,
                ns=robot_name,
            )
        else:
            has_new_local_opening = True

        if has_new_local_opening:
            obstacle_initially_deactivated = (
                obstacle_uid
                in inflated_grid_by_robot_max.deactivated_entities_cells_sets
            )
            if obstacle_initially_deactivated:
                inflated_grid_by_robot_max.activate_entities({obstacle_uid})
            previous_cells_sets = inflated_grid_by_robot_max.polygon_update(
                new_or_updated_polygons={obstacle_uid: new_obstacle_polygon}
            )

            if not c_1_cells_set or (c_1_cells_set and goal_cell in c_1_cells_set):
                cell_in_c_1 = goal_cell
            else:
                c_1_cells_set_iterator = iter(c_1_cells_set)
                cell_in_c_1 = next(c_1_cells_set_iterator)
                while (
                    inflated_grid_by_robot_max.grid[cell_in_c_1[0]][cell_in_c_1[1]] != 0
                ):
                    # While selected cell not in free space after manipulation, try another cell
                    try:
                        cell_in_c_1 = next(c_1_cells_set_iterator)
                    except StopIteration:
                        # Note: using the the exception detection is the pythonic way it seems (no has_next)
                        # No opening because c_1_cells_set is entirely inaccessible to the robot after manipulation
                        inflated_grid_by_robot_max.cells_sets_update(
                            new_or_updated_cells_sets=previous_cells_sets
                        )
                        if obstacle_initially_deactivated:
                            inflated_grid_by_robot_max.deactivate_entities(
                                {obstacle_uid}
                            )
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
                inflated_grid_by_robot_max.grid,
                inflated_grid_by_robot_max.d_width,
                inflated_grid_by_robot_max.d_height,
                neighborhood,
                check_diag_neighbors=False,
            )

            inflated_grid_by_robot_max.cells_sets_update(
                new_or_updated_cells_sets=previous_cells_sets
            )
            if obstacle_initially_deactivated:
                inflated_grid_by_robot_max.deactivate_entities({obstacle_uid})
            skipped_global_opening_check = False

            return (
                has_new_global_opening,
                has_new_local_opening,
                skipped_global_opening_check,
            )
        else:
            has_new_global_opening, skipped_global_opening_check = False, True
            return (
                has_new_global_opening,
                has_new_local_opening,
                skipped_global_opening_check,
            )

    def get_neighbors(
        self,
        current_configuration,
        gscore,
        close_set,
        open_queue,
        came_from,
        start,
        inflated_grid_by_robot_min,
        inflated_grid_by_robot_max,
        inflated_grid_by_obstacle,
        r_acc_cells,
        ccs_data,
        robot_uid,
        obstacle_uid,
        trans_mult,
        rot_mult,
        other_entities_polygons,
        other_entities_aabb_tree,
        obstacle_can_intrude_r_acc=True,
        obstacle_can_intrude_c_1_x=True,
    ):
        """
        Creates list of neighbors that are not in close set, do not collide dynamically nor statically
        """
        # TODO Add debug display option for intersections, be it on grid(s) or in between polygons
        neighbors = []
        tentative_g_scores = []

        for action in self._new_actions:
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
                extra_g_cost = self.rotation_unit_cost
            elif isinstance(action, ba.Translation):
                neighbor_action_opposes_prev_action = (
                    isinstance(current_configuration.action, ba.Translation)
                    and action.translation_vector[0]
                    == -1.0 * current_configuration.action.translation_vector[0]
                    and action.translation_vector[1]
                    == -1.0 * current_configuration.action.translation_vector[1]
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
                extra_g_cost = self.translation_unit_cost
            else:
                raise TypeError(
                    "action must either be of type NewRotation or NewTranslation"
                )

            # First, check whether the new configuration is in close set, if it is, ignore it
            robot_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
                new_robot_pose, trans_mult, rot_mult
            )
            obstacle_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
                new_obstacle_pose, trans_mult, rot_mult
            )

            if (robot_fixed_precision_pose, obstacle_fixed_precision_pose) in close_set:
                continue

            # Then check for collisions, starting at a grid level
            robot_cell_in_grid = utils.real_to_grid(
                new_robot_pose[0],
                new_robot_pose[1],
                inflated_grid_by_robot_min.res,
                inflated_grid_by_robot_min.grid_pose,
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
                    inflated_grid_by_robot_min.d_width,
                    inflated_grid_by_robot_min.d_height,
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
                inflated_grid_by_robot_min.grid[robot_cell_in_grid[0]][
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

            # Continue at static polygon level, check if still in map
            new_robot_polygon = action.apply(
                current_configuration.robot.polygon,
                current_configuration.robot.floating_point_pose,
            )

            # Check if robot is still within map bounds
            if not new_robot_polygon.within(inflated_grid_by_robot_min.aabb_polygon):
                continue

            new_obstacle_polygon = action.apply(
                current_configuration.obstacle.polygon,
                current_configuration.robot.floating_point_pose,
            )

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
                robot_bb_vertices,
            ) = collision.csv_check_collisions(
                main_uid=robot_uid,
                other_polygons=other_entities_polygons,
                polygon_sequence=[
                    current_configuration.robot.polygon,
                    new_robot_polygon,
                ],
                action_sequence=[
                    collision.convert_action(
                        action, current_configuration.robot.floating_point_pose
                    )
                ],
                bb_type="minimum_rotated_rectangle",
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
                obstacle_bb_vertices,
            ) = collision.csv_check_collisions(
                main_uid=obstacle_uid,
                other_polygons=other_entities_polygons,
                polygon_sequence=[
                    current_configuration.obstacle.polygon,
                    new_obstacle_polygon,
                ],
                action_sequence=[
                    collision.convert_action(
                        action, current_configuration.obstacle.floating_point_pose
                    )
                ],
                bb_type="minimum_rotated_rectangle",
                aabb_tree=other_entities_aabb_tree,
            )
            if collides_with:
                continue

            # If option is activated, check that obstacle intruded the appropriate component(s)
            intrudes = self.polygon_intrudes_components(
                new_obstacle_polygon,
                inflated_grid_by_robot_max,
                r_acc_cells,
                ccs_data,
                obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x,
            )
            if intrudes:
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
                robot_bb_vertices=robot_bb_vertices[0],
                obstacle_csv_polygon=obstacle_csv_polygons[(0,)],
                obstacle_bb_vertices=obstacle_bb_vertices[0],
            )

            neighbors.append(neighbor_configuration)
            tentative_g_scores.append(gscore[current_configuration] + extra_g_cost)

        self._rp.publish_manip_search_data(
            current_configuration,
            gscore,
            close_set,
            open_queue,
            came_from,
            neighbors,
            start,
            inflated_grid_by_robot_min.res,
            inflated_grid_by_robot_min.grid_pose,
            ns=self._robot_name,
        )

        return neighbors, tentative_g_scores

    def find_path(
        self, robot_pose, goal_pose, w_t, inflated_grid_by_robot, robot_polygon
    ):
        real_path = graph_search.real_to_grid_search_a_star(
            robot_pose, goal_pose, inflated_grid_by_robot
        )
        if real_path:
            phys_cost = 0.0
            raw_path_iterator = iter(real_path)
            prev_step = next(raw_path_iterator)
            for cur_step in raw_path_iterator:
                phys_cost += self.g(prev_step, cur_step, is_transfer=False)
            return TransitPath.from_poses(
                real_path, robot_polygon, robot_pose, phys_cost
            )
        else:
            return None

    @staticmethod
    def polygon_intrudes_components(
        new_obstacle_polygon,
        inflated_grid_by_robot,
        r_acc_cells,
        ccs_data,
        obstacle_can_intrude_r_acc,
        obstacle_can_intrude_c_1_x,
    ):
        if obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            return False
        elif obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
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
        cell,
        r_acc_cells,
        ccs_data,
        obstacle_can_intrude_r_acc,
        obstacle_can_intrude_c_1_x,
    ):
        if obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            return False
        elif obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
            if ccs_data.grid[cell[0]][cell[1]] > 0 and cell not in r_acc_cells:
                return True
        elif not obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            if cell in r_acc_cells:
                return True
        elif not obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
            return True

        return False

    @staticmethod
    def deduce_robot_goal_pose(robot_manip_pose, obs_init_pose, obs_goal_pose):
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
        start_cell, grid, res, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD
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
        inflated_grid_by_obstacle,
        robot_polygon,
        robot_pose,
        obstacle_pose,
        goal_pose,
    ):
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

        self._rp.publish_combined_costmap(
            sorted_cell_to_combined_cost, inflated_grid_by_obstacle, ns=self._robot_name
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
        inflated_grid_by_obstacle,
        acc_cells_for_obs,
        normalized_social_cost,
        normalized_distance_cost,
        sorted_cell_to_combined_cost,
        normalized_distance_to_goal=None,
    ):
        stocg.display_or_log(
            grid=np.invert(inflated_grid_by_obstacle.grid.astype(bool)),
            suffix="-obs_inf_grid",
            start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            debug_display=False,
            log_costmaps=True,
            abs_path_to_logs_dir=self.abs_path_to_logs_dir,
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
            abs_path_to_logs_dir=self.abs_path_to_logs_dir,
        )
        stocg.display_or_log(
            grid=normalized_distance_from_obs_costmap,
            suffix="-n_d_to_obs_costmap",
            start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            debug_display=False,
            log_costmaps=True,
            abs_path_to_logs_dir=self.abs_path_to_logs_dir,
        )
        if normalized_distance_to_goal is not None:
            stocg.display_or_log(
                grid=normalized_distance_from_goal_costmap,
                suffix="-n_d_to_goal_costmap",
                start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False,
                log_costmaps=True,
                abs_path_to_logs_dir=self.abs_path_to_logs_dir,
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
            abs_path_to_logs_dir=self.abs_path_to_logs_dir,
        )

    def compute_evasion(
        self,
        inflated_grid_by_robot_max,
        w_t,
        main_robot_uid,
        potential_deadlocks,
        forbidden_evasion_cells,
        use_combined_cost=True,
    ):
        # Compute evasion for main robot
        main_robot = w_t.entities[main_robot_uid]

        inflated_grid_by_robot_max.deactivate_entities({main_robot_uid})
        (
            main_robot_evasion_cell_social_cost,
            main_robot_evasion_path,
        ) = self.compute_evasion_for_one(
            w_t,
            inflated_grid_by_robot_max,
            main_robot,
            forbidden_evasion_cells,
            use_combined_cost,
        )
        inflated_grid_by_robot_max.activate_entities({main_robot_uid})

        if not main_robot_evasion_path:
            return None
        else:
            # If this robot is able to evade, it must check if it should by comparing its evasion path with the one of
            # other robots.
            other_robots_uids = {
                potential_deadlock.other_robot_uid
                for potential_deadlock in potential_deadlocks
                if isinstance(potential_deadlock, RobotRobotConflict)
            }
            inflated_grid_by_robot_max.polygon_update(
                new_or_updated_polygons={main_robot_uid: main_robot.polygon}
            )

            max_evasion_cell_social_cost = main_robot_evasion_cell_social_cost
            other_robot_evasion_path_max_duration = 0
            for robot_uid in other_robots_uids:
                # TODO : Add check to see if other robot has same radius as main robot : if so use the already computed
                #  inflated grid, else compute a corresponding inflated grid (and save for later just in case ?)
                other_robot = w_t.entities[robot_uid]

                inflated_grid_by_robot_max.deactivate_entities({robot_uid})
                inflated_grid_by_robot_max.activate_entities({main_robot_uid})
                (
                    other_robot_evasion_cell_social_cost,
                    other_robot_evasion_path,
                ) = self.compute_evasion_for_one(
                    w_t,
                    inflated_grid_by_robot_max,
                    other_robot,
                    set(),
                    use_combined_cost,
                )
                inflated_grid_by_robot_max.deactivate_entities({main_robot_uid})

                other_robot_exchange_real_path = (
                    graph_search.real_to_grid_search_a_star(
                        other_robot.pose, main_robot.pose, inflated_grid_by_robot_max
                    )
                )
                other_robot_exchange_path = TransitPath.from_poses(
                    other_robot_exchange_real_path,
                    other_robot.polygon,
                    other_robot.pose,
                )

                max_evasion_cell_social_cost = max(
                    max_evasion_cell_social_cost, other_robot_evasion_cell_social_cost
                )
                other_robot_evasion_path_max_duration = max(
                    other_robot_evasion_path_max_duration,
                    (
                        0
                        if main_robot_evasion_path is None
                        else len(main_robot_evasion_path.actions)
                    )
                    + (
                        0
                        if other_robot_exchange_path is None
                        else len(other_robot_exchange_path.actions)
                    ),
                )

                inflated_grid_by_robot_max.activate_entities({robot_uid})

            if main_robot_evasion_cell_social_cost < max_evasion_cell_social_cost:
                main_robot_evasion_path.set_wait(other_robot_evasion_path_max_duration)
                # main_robot_evasion_path.set_wait(100)
                return main_robot_evasion_path
            else:
                return None

    def compute_evasion_for_one(
        self,
        w_t,
        inflated_grid_by_robot_max,
        robot,
        forbidden_evasion_cells,
        use_combined_cost=False,
        return_path=True,
    ):
        robot_start_cell = utils.real_to_grid(
            robot.pose[0],
            robot.pose[1],
            inflated_grid_by_robot_max.res,
            inflated_grid_by_robot_max.grid_pose,
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
                    inflated_grid_by_robot_max,
                    robot.pose,
                    robot.polygon,
                    robot.uid,
                    obstacle_uid,
                    obstacle.pose,
                    other_entities_polygons,
                    other_entities_aabb_tree,
                    100.0,
                    1.0,
                )
            )
            if not transit_configuration_after_release:
                # Could not release obstacle during manipulation because no valid transit pose could be found.
                if return_path:
                    return robot_start_social_cost, None
                else:
                    return robot_start_social_cost

        # Compute shortest path to each cell of current component of robot
        robot_polygon = robot.polygon
        robot_pose = robot.pose
        robot_cell = utils.real_to_grid(
            robot_pose[0],
            robot_pose[1],
            inflated_grid_by_robot_max.res,
            inflated_grid_by_robot_max.grid_pose,
        )
        if transit_configuration_after_release:
            robot_polygon = transit_configuration_after_release.polygon
            robot_pose = transit_configuration_after_release.floating_point_pose
            robot_cell = transit_configuration_after_release.cell_in_grid

        _, _, came_from, _, gscore, _ = graph_search.grid_search_dijkstra(
            robot_cell,
            None,
            inflated_grid_by_robot_max.grid,
            inflated_grid_by_robot_max.d_width,
            inflated_grid_by_robot_max.d_height,
        )

        if not came_from:
            # If the robot was in an obstacle, no evasion is possible
            if return_path:
                return robot_start_social_cost, None
            else:
                return robot_start_social_cost
        else:
            accessible_cells = []
            social_cost = []
            distance_cost = []
            for cell, value in gscore.items():
                if cell not in forbidden_evasion_cells:
                    accessible_cells.append(cell)
                    social_cost.append(self._social_costmap[cell[0]][cell[1]])
                    distance_cost.append(value)
            social_cost = np.array(social_cost)
            distance_cost = np.array(distance_cost)

            if not use_combined_cost:
                min_social_cost_index = np.argmin(social_cost)
                evasion_cell = accessible_cells[min_social_cost_index]
            else:
                normalized_social_cost = (social_cost - np.min(social_cost)) / np.ptp(
                    social_cost
                )
                normalized_distance_cost = (
                    distance_cost - np.min(distance_cost)
                ) / np.ptp(distance_cost)
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
                        inflated_grid_by_robot_max,
                        accessible_cells,
                        normalized_social_cost,
                        normalized_distance_cost,
                        sorted_cell_to_combined_cost,
                    )

                # self._rp.publish_combined_costmap(
                #     sorted_cell_to_combined_cost, inflated_grid_by_robot_max, ns=self._robot_name
                # )
                # self._rp.cleanup_grid_map(ns=self._robot_name)

            if not return_path:
                return self._social_costmap[evasion_cell[0]][evasion_cell[1]]
            else:
                raw_cell_path = graph_search.reconstruct_path(came_from, evasion_cell)
                real_path = utils.grid_path_to_real_path(
                    raw_cell_path,
                    robot_pose,
                    None,
                    inflated_grid_by_robot_max.res,
                    inflated_grid_by_robot_max.grid_pose,
                )

                evasion_transit_path = (
                    None
                    if len(real_path) < 2
                    else EvasionTransitPath.from_poses(
                        real_path, robot_polygon, robot_pose
                    )
                )

                if transit_configuration_after_release:
                    evasion_transit_path.set_transit_configuration_after_release(
                        transit_configuration_after_release
                    )

                return self._social_costmap[evasion_cell[0]][
                    evasion_cell[1]
                ], evasion_transit_path

    def h(self, r_i, r_j):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        # rotation_cost = self.rotation_factor * (abs(r_j[2] - r_i[2]) % 180.)
        return translation_cost  # + rotation_cost

    def g(self, r_i, r_j, is_transfer=False):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        rotation_cost = self.rotation_factor * abs(r_j[2] - r_i[2])
        return (translation_cost + rotation_cost) * (
            1.0 if not is_transfer else self.transfer_coefficient
        )

    def get_transfer_path_from_config(
        self,
        prev_transit_end_configuration: Configuration,
        next_transit_start_configuration: Configuration,
        transfer_configurations: t.List[RobotObstacleConfiguration],
        obstacle_uid: int,
        phys_cost: t.Optional[float] = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ) -> TransferPath | None:
        if not transfer_configurations:
            return None

        manip_pose_id = transfer_configurations[0].manip_pose_id

        actions = [
            configuration.action
            for configuration in transfer_configurations
            if configuration.action
        ]
        grab_action = actions[0] if prev_transit_end_configuration else None
        release_action = next_transit_start_configuration.action
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
            config.robot.bb_vertices for config in transfer_configurations
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
