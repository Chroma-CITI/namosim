import copy
import typing as t

from typing_extensions import Self

import namosim.display.ros2_publisher as ros2
import namosim.utils.collision as collision
from namosim.data_models_v2 import PoseModel
from namosim.navigation.basic_actions import BasicAction
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
from namosim.world.binary_occupancy_grid import BinaryInflatedOccupancyGrid
from namosim.world.obstacle import Obstacle
from namosim.world.robot import Robot
from namosim.world.world_v2 import WorldV2


class Plan:
    def __init__(
        self,
        path_components: t.List[t.Union[TransitPath, TransferPath]] = [],
        goal: t.Optional[PoseModel] = None,
        robot_uid: t.Optional[int] = None,
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
        world: WorldV2,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        rp: "ros2.RosPublisher",
        check_horizon: t.Optional[int] = None,
        apply_strict_horizon: bool = False,
        exit_early_for_any_conflict: bool = False,
        exit_early_only_for_long_term_conflicts: bool = True,
        robot_name: str = "",
    ) -> t.List[Conflict]:
        # Check validity of each component
        shared_horizon = check_horizon
        previously_moved_entities_uids = set()
        remaining_components = self.path_components[self.component_index :]
        conflicts = []

        # Define sets of polygons and associated aabb trees to check for collisions
        other_entities_polygons = {
            uid: e.polygon
            for uid, e in world.entities.items()
            if uid != self.robot_uid and e.movability != "static"
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
        for other_robot in world.entities.values():
            # Inflate all other robots and their associated obstacles by the maximum translation at t+1 to prevent
            # SimultaneousSpaceAccess-type Conflicts
            if isinstance(other_robot, Robot) and other_robot.uid != self.robot_uid:
                center = other_robot.polygon.centroid
                robot_radius = (
                    center.hausdorff_distance(other_robot.polygon)
                    + 1.1 * inflated_grid_by_robot.res
                )
                radius = robot_radius
                min_radius_for_release = (
                    robot_radius
                    + inflated_grid_by_robot.inflation_radius
                    + 2.0 * inflated_grid_by_robot.res
                )
                # Enlarge radius to account for possible grabs
                for uid, obstacle in world.entities.items():
                    if (
                        isinstance(obstacle, Obstacle)
                        and uid not in world.entity_to_agent
                        and obstacle.movability != "static"
                    ):
                        if obstacle.polygon.buffer(
                            2.0 * inflated_grid_by_robot.inflation_radius,
                            join_style="mitre",
                        ).intersects(other_robot.polygon):
                            radius = min_radius_for_release
                            break
                if other_robot.uid in world.entity_to_agent.inverse:
                    obstacle = world.entities[
                        world.entity_to_agent.inverse[other_robot.uid]
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

        for counter, path in enumerate(remaining_components):
            has_first_action = counter == 0
            if (
                shared_horizon is None
                or not apply_strict_horizon
                or (apply_strict_horizon and shared_horizon > 0)
            ):
                if isinstance(path, TransitPath):
                    conflicts += path.get_conflicts(
                        self.robot_uid,
                        world,
                        inflated_grid_by_robot,
                        encompassing_circle_uid_to_robot_uid,
                        has_first_action,
                        shared_horizon,
                        apply_strict_horizon,
                        exit_early_for_any_conflict,
                        exit_early_only_for_long_term_conflicts,
                        rp=rp,
                        robot_name=robot_name,
                    )
                else:
                    # If the previously checked path components are valid, we assume it leaves any manipulated
                    # obstacles in the right place so we don't check again:
                    # - We simply deactivate collisions with them from the world representation
                    # - or if another path component needs to move them (check_start_pose)
                    previously_moved_entities_uids.add(path.obstacle_uid)
                    conflicts += path.get_conflicts(
                        self.robot_uid,
                        world,
                        inflated_grid_by_robot,
                        other_entities_polygons,
                        other_entities_aabb_tree,
                        other_entities_polygons_with_encompassing_circles,
                        other_entities_with_encompassing_circles_aabb_tree,
                        encompassing_circle_uid_to_robot_uid,
                        previously_moved_entities_uids,
                        has_first_action,
                        shared_horizon,
                        apply_strict_horizon,
                        exit_early_for_any_conflict,
                        exit_early_only_for_long_term_conflicts,
                        rp=rp,
                        robot_name=robot_name,
                    )

                    # inflated_grid_by_robot.deactivate_entities([path.obstacle_uid])
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

                if shared_horizon:
                    shared_horizon -= path.get_length()
            else:
                break

        # Reactivate entities that had been deactivated during checks
        # inflated_grid_by_robot.activate_entities(previously_moved_entities_uids)
        inflated_grid_by_robot.activate_entities(previously_moved_entities_uids)
        inflated_grid_by_robot.update(
            removed_polygons=set(encompassing_circle_uid_to_robot_uid.keys())
        )

        return conflicts

    def pop_next_action(self) -> BasicAction:
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
