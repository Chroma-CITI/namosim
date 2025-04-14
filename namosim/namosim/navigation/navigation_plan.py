import copy
import random
import typing as t

from typing_extensions import Self

import namosim.display.ros2_publisher as rp
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
import namosim.navigation.navigation_plan as nav_plan
import namosim.utils.collision as collision
import namosim.world.world as w
import namosim.world.world as world
from namosim.data_models import PoseModel
from namosim.navigation.basic_actions import Action
from namosim.navigation.conflict import (
    Conflict,
    RobotObstacleConflict,
    RobotRobotConflict,
    StolenMovableConflict,
)
from namosim.navigation.navigation_path import (
    EvasionTransitPath,
    TransferPath,
    TransitPath,
)
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from namosim.world.entity import Movability


class Postpone:
    def __init__(self):
        self.duration = 0
        self._step_index = 0
        self._is_running = False

    def start(self, duration: int):
        self.duration = duration
        self._step_index = 0
        self._is_running = True

    def go_to_end(self):
        self._step_index = self.duration

    def clear(self):
        self.duration = 0
        self._step_index = 0
        self._is_running = False

    def tick(self):
        self._step_index += 1
        return ba.Wait()

    def is_running(self):
        return self._is_running

    def is_done(self):
        return self._step_index >= self.duration


class Plan:
    def __init__(
        self,
        *,
        agent_id: str,
        paths: t.List[t.Union[TransitPath, TransferPath]] | None = None,
        goal: t.Optional[PoseModel] = None,
        plan_error: t.Optional[str] = None,
    ):
        self.paths = [] if paths is None else paths
        self.goal = goal
        self.agent_id = agent_id
        self.phys_cost = 0.0
        self.social_cost = 0.0
        self.total_cost = 0.0
        self.plan_error = plan_error
        self.component_index = 0
        self.postpone: Postpone = Postpone()
        self.postponements_history: t.Dict[int, int] = {}
        self.conflicts_history: t.Dict[int, t.Set[Conflict]] = {}
        self.steps_with_replan_call: t.Set[int] = set()
        self.update_count = 0

        if paths:
            for path in paths:
                self.phys_cost += path.phys_cost
                self.social_cost += path.social_cost
                self.total_cost += path.total_cost
        else:
            self.phys_cost = float("inf")
            self.social_cost = float("inf")
            self.total_cost = float("inf")

    def reset(self):
        self.component_index = 0
        for path in self.paths:
            path.reset()

    def get_current_path(self):
        return self.paths[self.component_index]

    def get_all_robot_poses(self) -> t.List[PoseModel]:
        poses = []
        for path in self.paths:
            poses += path.robot_path.poses
        return poses

    def get_all_actions(self) -> t.List[Action]:
        actions = []
        for path in self.paths:
            actions += path.actions
        return actions

    def get_current_action_index(self):
        idx = 0
        for path in self.paths[0 : self.component_index + 1]:
            if path.is_fully_executed():
                idx += len(path.actions)
            else:
                idx += path.action_index
        return idx

    def get_current_pose_index(self):
        return self.get_current_action_index()

    def append(self, future_plan: Self):
        self.paths += future_plan.paths
        self.phys_cost += future_plan.phys_cost
        self.social_cost += future_plan.social_cost
        self.total_cost += future_plan.total_cost
        return self

    def is_empty(self):
        return len(self.paths) == 0

    def _get_conflicts(
        self,
        *,
        world: "world.World",
        robot_inflated_grid: BinaryOccupancyGrid,
        grab_start_distance,
        rp: t.Optional["rp.RosPublisher"] = None,
        check_horizon: int = 0,
        apply_strict_horizon: bool = False,
        exit_early_for_any_conflict: bool = False,
        exit_early_only_for_long_term_conflicts: bool = True,
    ) -> t.Set[Conflict]:
        # if self.postpone.is_running():
        #     return []

        # Check validity of each component
        previously_moved_entities_uids = set()
        remaining_components = self.paths[self.component_index :]
        conflicts = set()

        # Define sets of polygons and associated aabb trees to check for collisions
        other_entities_polygons = {
            uid: e.polygon
            for uid, e in world.dynamic_entities.items()
            if uid != self.agent_id and e.movability != Movability.STATIC
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
        encompassing_circle_uid_to_agent_id = {}
        for other_robot in world.agents.values():
            if other_robot.uid == self.agent_id:
                continue

            # Inflate all other robots and their associated obstacles by the maximum translation at t+1 to prevent
            # SimultaneousSpaceAccess-type Conflicts
            other_robot_center = other_robot.polygon.centroid
            radius = (
                world.get_robot_conflict_radius(other_robot.uid, grab_start_distance)
                # Enlarge radius so that conflict is detected before the robot enters another robot's conflict radius, after which a dealock may occur
                + utils.SQRT_OF_2 * world.map.cell_size
            )

            # TODO Get inflation from largest robot
            encompassing_circle = other_robot_center.buffer(radius)
            temp_uid = f"{other_robot.uid}_conflict_circle"
            other_entities_polygons_with_encompassing_circles[temp_uid] = (
                encompassing_circle
            )
            other_entities_with_encompassing_circles_aabb_tree.add(
                collision.polygon_to_aabb(encompassing_circle), temp_uid
            )
            encompassing_circle_uid_to_agent_id[temp_uid] = other_robot.uid
            robot_inflated_grid.update_polygons(
                new_or_updated_polygons={temp_uid: encompassing_circle}
            )

        for i, path in enumerate(remaining_components):
            if check_horizon > 0 or apply_strict_horizon is False:
                has_first_action = i == 0
                if isinstance(path, TransitPath):
                    conflicts.update(
                        path.get_conflicts(
                            agent_id=self.agent_id,
                            world=world,
                            robot_inflated_grid=robot_inflated_grid,
                            encompassing_circle_uid_to_agent_id=encompassing_circle_uid_to_agent_id,
                            check_horizon=check_horizon,
                            has_first_action=has_first_action,
                            apply_strict_horizon=apply_strict_horizon,
                            exit_early_for_any_conflict=exit_early_for_any_conflict,
                            exit_early_only_for_long_term_conflicts=exit_early_only_for_long_term_conflicts,
                            rp=rp,
                        )
                    )
                else:
                    conflicts.update(
                        path.get_conflicts(
                            agent_id=self.agent_id,
                            world=world,
                            grab_start_distance=grab_start_distance,
                            robot_inflated_grid=robot_inflated_grid,
                            other_entities_polygons=other_entities_polygons,
                            other_entities_aabb_tree=other_entities_aabb_tree,
                            other_entities_polygons_with_encompassing_circles=other_entities_polygons_with_encompassing_circles,
                            other_entities_with_encompassing_circles_aabb_tree=other_entities_with_encompassing_circles_aabb_tree,
                            encompassing_circle_uid_to_agent_id=encompassing_circle_uid_to_agent_id,
                            previously_moved_entities_uids=previously_moved_entities_uids,
                            has_first_action=has_first_action,
                            check_horizon=check_horizon,
                            apply_strict_horizon=apply_strict_horizon,
                            exit_early_for_any_conflict=exit_early_for_any_conflict,
                            exit_early_only_for_long_term_conflicts=exit_early_only_for_long_term_conflicts,
                            rp=rp,
                        )
                    )

                    # If the previously checked path components are valid, we assume it leaves any manipulated
                    # obstacles in the right place so we don't check again:
                    # - We simply deactivate collisions with them from the world representation
                    # - or if another path component needs to move them (check_start_pose)
                    previously_moved_entities_uids.add(path.obstacle_uid)

                    robot_inflated_grid.deactivate_entities([path.obstacle_uid])

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
        robot_inflated_grid.activate_entities(previously_moved_entities_uids)
        robot_inflated_grid.update_polygons(
            removed_polygons=set(encompassing_circle_uid_to_agent_id.keys())
        )

        return conflicts

    def get_conflicts(
        self,
        *,
        world: "world.World",
        robot_inflated_grid: BinaryOccupancyGrid,
        grab_start_distance,
        rp: t.Optional["rp.RosPublisher"] = None,
        check_horizon: int = 0,
        apply_strict_horizon: bool = False,
        exit_early_for_any_conflict: bool = False,
        exit_early_only_for_long_term_conflicts: bool = True,
    ) -> t.Set[Conflict]:
        conflicts = set(
            self._get_conflicts(
                world=world,
                robot_inflated_grid=robot_inflated_grid,
                grab_start_distance=grab_start_distance,
                rp=rp,
                check_horizon=check_horizon,
                apply_strict_horizon=apply_strict_horizon,
                exit_early_for_any_conflict=exit_early_for_any_conflict,
                exit_early_only_for_long_term_conflicts=exit_early_only_for_long_term_conflicts,
            )
        )

        conflicts_to_ignore: t.Set[Conflict] = set()
        if self.is_evading():
            evasion_path = t.cast(EvasionTransitPath, self.get_current_path())
            for evasion_conflict in evasion_path.conflicts:

                for conflict in conflicts:
                    if (
                        isinstance(conflict, RobotRobotConflict)
                        and isinstance(evasion_conflict, RobotRobotConflict)
                        and conflict.other_agent_id == evasion_conflict.other_agent_id
                    ):
                        conflicts_to_ignore.add(conflict)

        return conflicts.difference(conflicts_to_ignore)

    def pop_next_action(self) -> Action:
        """
        Get the next plan step to execute
        :return: the action object to be executed if there is one, None if the plan is empty
        :rtype: action or None
        :except: if pop_next_action is called when the plan is fully executed
        :exception: IndexError
        """
        if self.is_empty():
            return ba.Wait()

        if self.postpone.is_running():
            if self.postpone.is_done():
                self.postpone.clear()
            else:
                return self.postpone.tick()

        current_component = self.paths[self.component_index]
        if current_component.is_fully_executed():
            if self.component_index < len(self.paths) - 1:
                self.component_index += 1
            current_component = self.paths[self.component_index]
        return current_component.pop_next_action()

    def is_evading(self):
        return self.is_empty() is False and isinstance(
            self.paths[self.component_index], EvasionTransitPath
        )

    def is_evasion_over(self):
        return (
            self.is_evading() and self.paths[self.component_index].is_fully_executed()
        )

    def is_postpone_over(self):
        return self.postpone.is_running() and self.postpone.is_done()

    def was_last_step_success(
        self, w_t: "w.World", last_action_result: ar.ActionResult
    ):
        # TODO Check if robot state (position and grab) are coherent with next step's preconditions
        return isinstance(last_action_result, ar.ActionSuccess)

    def save_conflicts(self, step_count: int, conflicts: t.Set[Conflict]):
        if len(conflicts) > 0:
            if step_count in self.conflicts_history:
                self.conflicts_history[step_count].update(conflicts)
            else:
                self.conflicts_history[step_count] = conflicts
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
        simulation_log: t.List[utils.NamosimLog],
        agent_id: str,
    ):
        if self.postpone.is_running() and not self.postpone.is_done():
            return

        n_steps = random.randint(t_min, t_max)
        simulation_log.append(
            utils.NamosimLog(
                f"Agent {agent_id}: Postponing for {n_steps} steps.",
                step_count,
            )
        )
        self.postpone.start(duration=n_steps)
        self.postponements_history[step_count] = n_steps
        self.update_count += 1

    def set_plan(
        self, plan: "nav_plan.Plan", step_count: int, postpone: Postpone | None = None
    ):
        self.update_count += 1
        self.paths = plan.paths
        self.goal = plan.goal
        self.agent_id = plan.agent_id
        self.phys_cost = plan.phys_cost
        self.social_cost = plan.social_cost
        self.total_cost = plan.total_cost
        self.plan_error = plan.plan_error
        self.component_index = plan.component_index
        self.postpone = postpone or Postpone()
