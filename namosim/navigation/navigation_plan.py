import random
import typing as t

from shapely.geometry import Polygon
from typing_extensions import Self

import namosim.display.ros2_publisher as rp
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
import namosim.navigation.navigation_plan as nav_plan
import namosim.utils.collision as collision
import namosim.world.world as w
import namosim.world.world as world
from namosim.data_models import Pose2D
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

    def start(self, duration: int, is_for_deadlock: bool = False):
        self.duration = duration
        self._step_index = 0
        self._is_running = True
        self.is_for_deadlock = is_for_deadlock

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
        goal: t.Optional[Pose2D] = None,
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

    def set_current_action_index(self, new_path_idx: int, new_action_idx: int):
        self.component_index = new_path_idx

        for path_idx, path in enumerate(self.paths):
            if path_idx < new_path_idx:
                path.reset(len(path.actions))
            elif path_idx == new_path_idx:
                path.reset(new_action_idx)
            else:
                path.reset(0)

    def get_current_path(self):
        return self.paths[self.component_index]

    def get_all_robot_poses(self) -> t.List[Pose2D]:
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

    def get_last_index_of_current_path(self):
        idx = 0
        for path in self.paths[0 : self.component_index + 1]:
            idx += len(path.actions)
            if not path.is_fully_executed():
                break
        return idx - 1

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
        conflict_radius: float,
        horizon: int,
        rp: t.Optional["rp.RosPublisher"] = None,
        exit_early: bool = False,
    ) -> t.Set[Conflict]:
        # Check validity of each component
        previously_moved_obstacles: t.Set[str] = set()
        remaining_paths = self.paths[self.component_index :]
        conflicts = set()

        orig_polygons = {
            uid: e.polygon
            for uid, e in world.dynamic_entities.items()
            if uid != self.agent_id
        }

        other_entity_polygons = {
            uid: e.polygon
            for uid, e in world.dynamic_entities.items()
            if uid != self.agent_id
        }

        # temporarily inflate robot polygons (including obstacles they are carrying) to avoid conflicts with other robots
        for other_robot in world.agents.values():
            if other_robot.uid == self.agent_id:
                continue
            robot_conflict_polygon = world.get_combined_agent_obstacle_polygon(
                other_robot.uid
            ).buffer(conflict_radius)
            other_entity_polygons[other_robot.uid] = robot_conflict_polygon

        robot_inflated_grid.update_polygons(
            new_or_updated_polygons=other_entity_polygons
        )

        for path in remaining_paths:
            if horizon > 0:
                if isinstance(path, TransitPath):
                    conflicts.update(
                        path.get_conflicts(
                            agent_id=self.agent_id,
                            world=world,
                            robot_inflated_grid=robot_inflated_grid,
                            horizon=horizon,
                            exit_early=exit_early,
                            rp=rp,
                        )
                    )
                else:
                    conflicts.update(
                        path.get_conflicts(
                            agent_id=self.agent_id,
                            world=world,
                            other_entities_polygons=other_entity_polygons,
                            previously_moved_obstacles=previously_moved_obstacles,
                            horizon=horizon,
                            exit_early=exit_early,
                            rp=rp,
                        )
                    )

                    # If the previously checked path components are valid, we assume it leaves any manipulated
                    # obstacles in the right place so we don't check again:
                    # - We simply deactivate collisions with them from the world representation
                    # - or if another path component needs to move them (check_start_pose)
                    previously_moved_obstacles.add(path.obstacle_uid)

                    robot_inflated_grid.deactivate_entities([path.obstacle_uid])

                if exit_early and conflicts:
                    break

                if horizon:
                    horizon = max(0, horizon - path.get_remaining_length())
            else:
                break

        # Reactivate entities that had been deactivated during checks
        robot_inflated_grid.activate_entities(previously_moved_obstacles)
        robot_inflated_grid.update_polygons(new_or_updated_polygons=orig_polygons)

        return conflicts

    def get_conflicts(
        self,
        *,
        world: "world.World",
        robot_inflated_grid: BinaryOccupancyGrid,
        conflict_radius: float,
        horizon: int,
        rp: t.Optional["rp.RosPublisher"] = None,
        exit_early: bool = False,
    ) -> t.Set[Conflict]:
        conflicts = self._get_conflicts(
            world=world,
            robot_inflated_grid=robot_inflated_grid,
            conflict_radius=conflict_radius,
            rp=rp,
            horizon=horizon,
            exit_early=exit_early,
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
        is_for_deadlock: bool = False,
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
        self.postpone.start(duration=n_steps, is_for_deadlock=is_for_deadlock)
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
