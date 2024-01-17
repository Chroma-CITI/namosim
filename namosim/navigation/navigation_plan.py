import copy
import random
import typing as t

from typing_extensions import Self

import namosim.display.ros2_publisher as ros2
import namosim.display.ros2_publisher as rp
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
import namosim.navigation.navigation_plan as nav_plan
import namosim.utils.collision as collision
import namosim.world.world as w
import namosim.world.world as world
from namosim.data_models import UID, PoseModel
from namosim.navigation.basic_actions import Action
from namosim.navigation.conflict import (
    Conflict,
    RobotObstacleConflict,
    StolenMovableConflict,
)
from namosim.navigation.navigation_path import (
    EvasionTransitPath,
    TransferPath,
    TransitPath,
)
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryInflatedOccupancyGrid
from namosim.world.entity import Movability


class Plan:
    def __init__(
        self,
        *,
        robot_uid: UID,
        path_components: t.List[t.Union[TransitPath, TransferPath]] = [],
        goal: t.Optional[PoseModel] = None,
        plan_error: t.Optional[str] = None,
    ):
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

    def append(self, future_plan: Self):
        self.path_components += future_plan.path_components
        self.phys_cost += future_plan.phys_cost
        self.social_cost += future_plan.social_cost
        self.total_cost += future_plan.total_cost
        return self

    def has_infinite_cost(self):
        return True if self.total_cost == float("inf") else False

    def is_empty(self):
        return len(self.path_components) == 0

    def get_conflicts(
        self,
        *,
        world: "world.World",
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        rp: "ros2.RosPublisher",
        check_horizon: int = 0,
        apply_strict_horizon: bool = False,
        exit_early_for_any_conflict: bool = False,
        exit_early_only_for_long_term_conflicts: bool = True,
        robot_name: str = "",
    ) -> t.List[Conflict]:
        # Check validity of each component
        previously_moved_entities_uids = set()
        remaining_components = self.path_components[self.component_index :]
        conflicts = []

        # Define sets of polygons and associated aabb trees to check for collisions
        other_entities_polygons = {
            uid: e.polygon
            for uid, e in world.entities.items()
            if uid != self.robot_uid and e.movability != Movability.STATIC
        }
        other_entities_aabb_tree = collision.polygons_to_aabb_tree(
            other_entities_polygons
        )

        other_entities_polygons_with_encompassing_circles = copy.copy(
            other_entities_polygons
        )
        other_entities_with_encompassing_circles_aabb_tree = copy.deepcopy(
            other_entities_aabb_tree
        )
        encompassing_circle_uid_to_robot_uid = {}
        max_uid = 10000  # TODO: find a better way compute this temporary uid
        for other_robot in world.agents.values():
            if other_robot.uid == self.robot_uid:
                continue

            # Inflate all other robots and their associated obstacles by the maximum translation at t+1 to prevent
            # SimultaneousSpaceAccess-type Conflicts
            other_robot_center = other_robot.polygon.centroid
            radius = (
                world.get_robot_conflict_radius(other_robot.uid)
                # Enlarge radius so that conflict is detected before the robot enters another robot's conflict radius, after which a dealock may occur
                + utils.SQRT_OF_2 * world.config.cell_size
            )

            # TODO Get inflation from largest robot
            encompassing_circle = other_robot_center.buffer(radius)
            temp_uid = (
                max(
                    max_uid,
                    max(
                        [0]
                        if not encompassing_circle_uid_to_robot_uid
                        else encompassing_circle_uid_to_robot_uid.keys()
                    ),
                )
                + 1
            )
            other_entities_polygons_with_encompassing_circles[
                temp_uid
            ] = encompassing_circle
            other_entities_with_encompassing_circles_aabb_tree.add(
                collision.polygon_to_aabb(encompassing_circle), temp_uid
            )
            encompassing_circle_uid_to_robot_uid[temp_uid] = other_robot.uid
            inflated_grid_by_robot.update(
                new_or_updated_polygons={temp_uid: encompassing_circle}
            )

        for i, path in enumerate(remaining_components):
            if check_horizon > 0 or apply_strict_horizon is False:
                has_first_action = i == 0
                if isinstance(path, TransitPath):
                    conflicts += path.get_conflicts(
                        robot_uid=self.robot_uid,
                        world=world,
                        inflated_grid_by_robot=inflated_grid_by_robot,
                        encompassing_circle_uid_to_robot_uid=encompassing_circle_uid_to_robot_uid,
                        check_horizon=check_horizon,
                        has_first_action=has_first_action,
                        apply_strict_horizon=apply_strict_horizon,
                        exit_early_for_any_conflict=exit_early_for_any_conflict,
                        exit_early_only_for_long_term_conflicts=exit_early_only_for_long_term_conflicts,
                        rp=rp,
                        robot_name=robot_name,
                    )
                else:
                    conflicts += path.get_conflicts(
                        robot_uid=self.robot_uid,
                        world=world,
                        inflated_grid_by_robot=inflated_grid_by_robot,
                        other_entities_polygons=other_entities_polygons,
                        other_entities_aabb_tree=other_entities_aabb_tree,
                        other_entities_polygons_with_encompassing_circles=other_entities_polygons_with_encompassing_circles,
                        other_entities_with_encompassing_circles_aabb_tree=other_entities_with_encompassing_circles_aabb_tree,
                        encompassing_circle_uid_to_robot_uid=encompassing_circle_uid_to_robot_uid,
                        previously_moved_entities_uids=previously_moved_entities_uids,
                        has_first_action=has_first_action,
                        check_horizon=check_horizon,
                        apply_strict_horizon=apply_strict_horizon,
                        exit_early_for_any_conflict=exit_early_for_any_conflict,
                        exit_early_only_for_long_term_conflicts=exit_early_only_for_long_term_conflicts,
                        rp=rp,
                        robot_name=robot_name,
                    )

                    # If the previously checked path components are valid, we assume it leaves any manipulated
                    # obstacles in the right place so we don't check again:
                    # - We simply deactivate collisions with them from the world representation
                    # - or if another path component needs to move them (check_start_pose)
                    previously_moved_entities_uids.add(path.obstacle_uid)

                    inflated_grid_by_robot.deactivate_entities([path.obstacle_uid])

                if exit_early_for_any_conflict and conflicts:
                    break
                if exit_early_only_for_long_term_conflicts and conflicts:
                    is_there_long_term_conflict = any(
                        [
                            (
                                isinstance(conflict, RobotObstacleConflict)
                                or (isinstance(conflict, StolenMovableConflict))
                            )
                            for conflict in conflicts
                        ]
                    )
                    if is_there_long_term_conflict:
                        break

                if check_horizon:
                    check_horizon = max(0, check_horizon - path.get_remaining_length())
            else:
                break

        # Reactivate entities that had been deactivated during checks
        inflated_grid_by_robot.activate_entities(previously_moved_entities_uids)
        inflated_grid_by_robot.update(
            removed_polygons=set(encompassing_circle_uid_to_robot_uid.keys())
        )

        return conflicts

    def pop_next_action(self) -> Action:
        """
        Get the next plan step to execute
        :return: the action object to be executed if there is one, None if the plan is empty
        :rtype: action or None
        :except: if pop_next_action is called when the plan is fully executed
        :exception: IndexError
        """
        current_component = self.path_components[self.component_index]
        if current_component.is_fully_executed():
            if self.component_index < len(self.path_components) - 1:
                self.component_index += 1
            current_component = self.path_components[self.component_index]
        return current_component.pop_next_action()

    def is_evading(self):
        return self.is_empty() is False and isinstance(
            self.path_components[self.component_index], EvasionTransitPath
        )

    def is_evasion_over(self):
        return (
            self.is_evading()
            and self.path_components[self.component_index].is_fully_executed()
        )


class Timer:
    def __init__(
        self, start_time: int = 0, duration: int = 0, is_running: bool = False
    ):
        self.start_time = start_time
        self.duration = duration
        self.is_running = is_running

    def start_timer(self, start_time: int, duration: int):
        self.start_time = start_time
        self.duration = duration
        self.is_running = True

    def is_timer_over(self, current_time: int):
        if current_time - self.start_time >= self.duration:
            self.is_running = False
            return True
        return False


class DynamicPlan(Plan):
    DEBUGGING_WAIT_TIME_GENERATOR = []

    def __init__(self, robot_uid: UID):
        super().__init__(robot_uid=robot_uid)
        self.update_count = 0
        """
        The number of times the plan was updated
        """

        self.steps_with_replan_call = set()
        """
        The steps in which a replan occurred
        """

        self.current_conflicts = []
        self.plan_history = {}
        self.conflicts_history = {}
        self.postponements_history = {}
        self.unpostponements_history = []
        self.forbidden_evasion_cells = set()
        self.timer = Timer()

    def was_last_step_success(
        self, w_t: "w.World", last_action_result: ar.ActionResult
    ):
        # TODO Check if robot state (position and grab) are coherent with next step's preconditions
        return isinstance(last_action_result, ar.ActionSuccess)

    def get_conflicts(
        self,
        world: "w.World",
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        ros_publisher: "rp.RosPublisher",
        check_horizon: int,
        apply_strict_horizon: bool = False,
        exit_early_for_any_conflict: bool = True,
        exit_early_only_for_long_term_conflicts: bool = True,
        robot_name: str = "",
    ):
        conflicts = super().get_conflicts(
            world=world,
            inflated_grid_by_robot=inflated_grid_by_robot,
            check_horizon=check_horizon,
            apply_strict_horizon=apply_strict_horizon,
            exit_early_for_any_conflict=exit_early_for_any_conflict,
            exit_early_only_for_long_term_conflicts=exit_early_only_for_long_term_conflicts,
            rp=ros_publisher,
            robot_name=robot_name,
        )
        self.current_conflicts += conflicts
        return conflicts

    def save_conflicts(self, step_count: int):
        if self.current_conflicts:
            if step_count in self.conflicts_history:
                self.conflicts_history[step_count] += self.current_conflicts
            else:
                self.conflicts_history[step_count] = self.current_conflicts
        self.current_conflicts = []

    def has_tries_remaining(self, max_tries: int):
        return self.update_count < max_tries

    def can_even_be_found(self):
        if (
            self.plan_error
            and self.plan_error == "start_or_goal_cell_in_static_obstacle_error"
        ):
            return False
        return True

    def new_postpone(
        self,
        t_min: int,
        t_max: int,
        step_count: int,
        conflicts: t.List[Conflict],
        simulation_log: t.List[utils.BasicLog],
        robot_name: str,
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
                self.update_plan(
                    nav_plan.Plan(robot_uid=self.robot_uid, path_components=[]),
                    step_count,
                )
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

    def update_plan(self, plan: Plan, step_count: int):
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
