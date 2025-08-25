import typing as t

import numpy as np
from shapely.geometry import Polygon

import namosim.agents.agent as agent
import namosim.display.ros2_publisher as ros2
import namosim.world.world as world
from namosim.agents.stilman_configurations import RobotConfiguration
from namosim.data_models import GridCellModel, Pose2D
from namosim.navigation import basic_actions as ba
from namosim.navigation.conflict import (
    ConcurrentGrabConflict,
    Conflict,
    RobotObstacleConflict,
    RobotRobotConflict,
    SimultaneousSpaceAccess,
    StealingMovableConflict,
    StolenMovableConflict,
)
from namosim.navigation.path_type import PathType
from namosim.utils import collision, utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from shapely.geometry import JOIN_STYLE
import shapely.ops


class RawPath:
    """
    Represents a sequence of entity poses and their associated polygons
    """

    def __init__(
        self,
        poses: t.List[Pose2D],
        polygons: t.List[Polygon],
    ):
        if len(poses) != len(polygons):
            raise ValueError(
                "A RawPath requires that its polygon and pose arrays be the same size."
                "Current sizes are: polygon({}), pose({}))".format(
                    len(polygons), len(poses)
                )
            )
        self.poses = poses
        self.polygons = polygons

    # TODO Have these trans and rot precision values be passed from calling functions !
    def is_start_pose(
        self,
        pose: Pose2D,
        cell_size: float,
    ):
        """
        Returns `True` if the given pose is equivalen to the first pose in the path,
        up to a fixed degree of precision, otherwise `False`.
        """

        translation_discretization_factor = 2 / cell_size
        rotation_discretization_factor = 1 / 3
        other_pose = utils.real_pose_to_fixed_precision_pose(
            pose, translation_discretization_factor, rotation_discretization_factor
        )
        start_pose = utils.real_pose_to_fixed_precision_pose(
            self.poses[0],
            translation_discretization_factor,
            rotation_discretization_factor,
        )
        return other_pose == start_pose

    def __len__(self):
        return len(self.poses)


class TransferPath:
    """
    Represents a sequence of configurations in which a robot moves (transfers) a particular obstacle.
    """

    path_type: t.Literal[PathType.TRANSFER] = PathType.TRANSFER

    def __init__(
        self,
        robot_path: RawPath,
        obstacle_path: RawPath,
        actions: t.List[ba.Action],
        grab_action: ba.Grab,
        release_action: ba.Release,
        obstacle_uid: str,
        manip_pose_id: int,
        phys_cost: t.Optional[float] = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ):
        if len(robot_path) != len(obstacle_path) or len(robot_path) != len(actions) + 1:
            raise ValueError(
                "A TransferPath requires its robot and obstacle raw paths have the same length and equal to number of actions + 1"
                "Current sizes are: robot_path({}), obstacle_path({}), actions({})".format(
                    len(robot_path), len(obstacle_path), len(actions)
                )
            )
        self.robot_path = robot_path
        self.obstacle_path = obstacle_path
        self.obstacle_uid = obstacle_uid
        self.manip_pose_id = manip_pose_id
        self.phys_cost = (
            phys_cost
            if phys_cost is not None
            else utils.sum_of_euclidean_distances(self.robot_path.poses) * weight
        )
        self.social_cost = social_cost
        self.total_cost = self.phys_cost + self.social_cost

        # TODO Remove this attribute that is currently kept to avoid circular dependency with ros_conversion.py
        #   Simply move this class and the other ones in another module
        self.is_transfer = True

        self.grab_action = grab_action
        self.release_action = release_action
        self.actions = actions
        self.action_index = 0

    def reset(self, action_index: int = 0):
        self.action_index = action_index

    def is_fully_executed(self):
        return self.action_index >= len(self.actions)

    def get_conflicts(
        self,
        agent_id: str,
        world: "world.World",
        other_entities_polygons: t.Dict[str, Polygon],
        previously_moved_obstacles: t.Set[str],
        horizon: int,
        exit_early: bool = False,
        rp: t.Optional["ros2.RosPublisher"] = None,
    ) -> t.Set[Conflict]:
        assert len(self.actions) + 1 == len(self.robot_path.poses)
        assert agent_id not in other_entities_polygons

        conflicts: t.Set[Conflict] = set()
        if horizon <= 0:
            return conflicts

        # Compute and display horizon convex polygons
        if rp:
            rp.publish_transfer_horizon_polygons(
                robot_polygons=self.robot_path.polygons,
                obstacle_polygons=self.obstacle_path.polygons,
                start_index=self.action_index,
                horizon=horizon,
                agent_id=agent_id,
            )

        # Check conflicts for all actions within horizon
        for look_ahead_index, (
            action,
            robot_pose,
            robot_polygon,
            obstacle_pose,
            obstacle_polygon,
        ) in enumerate(
            zip(
                self.actions[self.action_index :],
                self.robot_path.poses[self.action_index :],
                self.robot_path.polygons[self.action_index :],
                self.obstacle_path.poses[self.action_index :],
                self.obstacle_path.polygons[self.action_index :],
            )
        ):
            if look_ahead_index >= horizon:
                break

            if action is self.grab_action:
                conflicts = conflicts.union(
                    self.get_grab_action_conflicts(
                        agent_id=agent_id,
                        robot_pose=robot_pose,
                        robot_polygon=robot_polygon,
                        obstacle_polygon=obstacle_polygon,
                        world=world,
                        previously_moved_obstacles=previously_moved_obstacles,
                        other_entities_polygons=other_entities_polygons,
                        exit_early=exit_early,
                    )
                )
                if len(conflicts) > 0 and exit_early:
                    return conflicts

            # Get robot conflicts

            (
                collides_with,
                _,
            ) = collision.get_csv_collisions(
                agent_id=agent_id,
                robot_pose=robot_pose,
                robot_action=action,
                other_polygons=other_entities_polygons,
                polygon=robot_polygon,
                ignored_entities=previously_moved_obstacles.union({self.obstacle_uid}),
            )

            for uid in collides_with:
                other_robot = None
                if uid in world.agents:
                    other_robot = world.agents[uid]
                elif uid in world.entity_to_agent:
                    other_robot = world.agents[world.entity_to_agent[uid]]

                if other_robot:
                    if look_ahead_index < horizon:
                        conflicts.add(
                            RobotRobotConflict(
                                agent_id=agent_id,
                                robot_pose=robot_pose,
                                other_agent_id=other_robot.uid,
                                other_robot_pose=other_robot.pose,
                            )
                        )
                        if exit_early:
                            return conflicts
                else:
                    conflicts.add(RobotObstacleConflict(uid))
                    if exit_early:
                        return conflicts

            # Obstacle conflicts
            (
                collides_with,
                _,
            ) = collision.get_csv_collisions(
                agent_id=self.obstacle_uid,
                robot_action=action,
                robot_pose=self.robot_path.poses[self.action_index + look_ahead_index],
                other_polygons=other_entities_polygons,
                polygon=self.obstacle_path.polygons[
                    self.action_index + look_ahead_index
                ],
                ignored_entities=previously_moved_obstacles.union({self.obstacle_uid}),
            )

            for uid in collides_with:
                other_robot = None
                if uid in world.agents:
                    other_robot = world.agents[uid]
                elif uid in world.entity_to_agent:
                    other_robot = world.agents[world.entity_to_agent[uid]]

                if other_robot:
                    if look_ahead_index < horizon:
                        conflicts.add(
                            RobotRobotConflict(
                                agent_id=agent_id,
                                robot_pose=robot_pose,
                                other_agent_id=other_robot.uid,
                                other_robot_pose=other_robot.pose,
                            )
                        )
                        if exit_early:
                            return conflicts
                else:
                    conflicts.add(RobotObstacleConflict(uid))
                    if exit_early:
                        return conflicts
        return conflicts

    def get_grab_action_conflicts(
        self,
        *,
        agent_id: str,
        robot_pose: Pose2D,
        robot_polygon: Polygon,
        obstacle_polygon: Polygon,
        world: "world.World",
        previously_moved_obstacles: t.Set[str],
        other_entities_polygons: t.Dict[str, Polygon],
        exit_early: bool
    ) -> t.Set[Conflict]:
        conflicts: t.Set[Conflict] = set()

        already_grabbed_by_current_robot = (
            world.entity_to_agent.get(self.obstacle_uid) == agent_id
        )
        if already_grabbed_by_current_robot:
            ## This happens when the plan has two consecutive transfer paths back-to-back.
            return conflicts

        # If obstacle is held by another agent
        if self.obstacle_uid in world.entity_to_agent:
            conflicts.add(
                StealingMovableConflict(
                    self.obstacle_uid,
                    world.entity_to_agent[self.obstacle_uid],
                )
            )
            if exit_early:
                return conflicts

        # Check that obstacle is at the expected pose
        current_obstacle_pose = world.dynamic_entities[self.obstacle_uid].pose
        obstacle_at_start_pose = self.obstacle_path.is_start_pose(
            current_obstacle_pose, cell_size=world.map.cell_size
        )

        if not obstacle_at_start_pose:
            conflicts.add(
                StolenMovableConflict(
                    self.obstacle_uid,
                    expected_pose=self.obstacle_path.poses[0],
                    actual_pose=current_obstacle_pose,
                )
            )
            if exit_early:
                return conflicts

        # Check for SimultaneousSpace conflict that might result from the grab, since a grab instantly expands the robot's conflict radius.
        robot_obstacle_polygon = t.cast(
            Polygon, shapely.ops.unary_union([robot_polygon, obstacle_polygon])
        )
        collides_with = collision.get_collisions_for_entity(
            robot_obstacle_polygon,
            other_entities_polygons,
            ignored_entities={self.obstacle_uid},
        )

        for uid in collides_with:
            assert uid != agent_id
            if isinstance(
                world.dynamic_entities[uid],
                agent.Agent,
            ):
                ConcurrentGrabConflict(self.obstacle_uid, uid)
                if exit_early:
                    return conflicts

        (
            collides_with,
            _,
        ) = collision.get_csv_collisions(
            agent_id=agent_id,
            robot_pose=robot_pose,
            robot_action=self.grab_action,
            other_polygons=other_entities_polygons,
            polygon=robot_polygon,
            ignored_entities=previously_moved_obstacles.union({self.obstacle_uid}),
        )

        for uid in collides_with:
            if (
                isinstance(world.dynamic_entities[uid], agent.Agent)
                or uid in world.entity_to_agent
            ):
                if isinstance(world.dynamic_entities[uid], agent.Agent):
                    other_robot = world.agents[uid]
                else:
                    other_robot = world.agents[world.entity_to_agent[uid]]

                conflicts.add(
                    RobotRobotConflict(
                        agent_id=agent_id,
                        robot_pose=robot_pose,
                        other_agent_id=other_robot.uid,
                        other_robot_pose=other_robot.pose,
                    )
                )
                if exit_early:
                    return conflicts
            else:
                conflicts.add(RobotObstacleConflict(uid))
        return conflicts

    def pop_next_action(self):
        action = self.actions[self.action_index]
        self.action_index += 1
        return action

    def get_length(self):
        return len(self.actions)

    def get_remaining_length(self):
        return max(0, len(self.actions) - self.action_index)


class TransitPath:
    path_type: t.Literal[PathType.TRANSIT] = PathType.TRANSIT

    def __init__(
        self,
        robot_path: RawPath,
        actions: t.List[ba.Action],
        phys_cost: float | None = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ):
        if len(robot_path) != len(actions) + 1:
            raise ValueError(
                "A TransitPath requires the length of the robot raw path be equal to the number of actions + 1. "
                "Current sizes are: robot_path({}), actions({})".format(
                    len(robot_path.polygons), len(actions)
                )
            )

        self.robot_path = robot_path

        self.phys_cost = (
            phys_cost
            if phys_cost is not None
            else utils.sum_of_euclidean_distances(self.robot_path.poses) * weight
        )
        self.social_cost = social_cost
        self.total_cost = self.phys_cost + self.social_cost

        # TODO Remove this attribute that is currently kept to avoid circular dependency with ros_conversion.py
        #   Simply move this class and the other ones in another module
        self.is_transfer = False

        self.actions = actions
        self.action_index = 0

    def reset(self, action_index: int = 0):
        self.action_index = action_index

    def __str__(self):
        if len(self.actions) < 5:
            return "{" + ", ".join([str(x) for x in self.actions]) + "}"
        return (
            "{"
            + ", ".join([str(x) for x in self.actions[:2]])
            + ", ..., "
            + ", ".join([str(x) for x in self.actions[-2:]])
            + "}"
        )

    @classmethod
    def from_poses(
        cls,
        poses: t.List[Pose2D],
        robot_polygon: Polygon,
        robot_pose: Pose2D,
        phys_cost: float | None = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ):
        # Separate translation from rotation actions
        if len(poses) == 0:
            return cls(
                robot_path=RawPath([], []),
                actions=[],
                phys_cost=phys_cost,
                social_cost=social_cost,
                weight=weight,
            )

        if robot_pose != poses[0]:
            raise Exception("Robot pose not equal to start pose")

        if len(poses) == 1:
            return cls(
                robot_path=RawPath(poses=poses, polygons=[robot_polygon]),
                actions=[],
                phys_cost=phys_cost,
                social_cost=social_cost,
                weight=weight,
            )

        actions: t.List[ba.Action] = []
        updated_poses = [poses[0]]

        for pose, next_pose in zip(poses, poses[1:]):
            has_translation = not all(
                [
                    utils.is_close(pose[0], next_pose[0], abs_tol=1e-6),
                    utils.is_close(pose[1], next_pose[1], abs_tol=1e-6),
                ]
            )

            current_angle = pose[2]
            turn_towards_angle = 0.0

            if has_translation:
                turn_towards_angle = utils.get_angle_to_turn(pose, next_pose)

                dist = utils.euclidean_distance(pose, next_pose)
                if np.abs(turn_towards_angle) > 90:
                    turn_towards_angle = utils.normalize_angle_degrees(
                        turn_towards_angle + 180
                    )  # turn away
                    current_angle = utils.add_angles(current_angle, turn_towards_angle)
                    actions.append(ba.Rotation(angle=turn_towards_angle))
                    updated_poses.append(Pose2D(pose[0], pose[1], current_angle))
                    dist = -dist
                elif np.abs(turn_towards_angle) > 1e-6:
                    current_angle = utils.add_angles(current_angle, turn_towards_angle)
                    actions.append(ba.Rotation(angle=turn_towards_angle))
                    updated_poses.append(Pose2D(pose[0], pose[1], current_angle))

                actions.append(ba.Advance(dist))
                updated_poses.append(Pose2D(next_pose[0], next_pose[1], current_angle))

            has_rotation = not utils.angle_is_close(
                current_angle, next_pose[2], abs_tol=1e-6
            )

            if has_rotation:
                remaining_angle = utils.subtract_angles(next_pose[2], current_angle)
                actions.append(ba.Rotation(angle=remaining_angle))
                updated_poses.append(next_pose)

        polygons = [
            utils.set_polygon_pose(robot_polygon, robot_pose, pose)
            for pose in updated_poses
        ]
        robot_path = RawPath(updated_poses, polygons)

        return cls(
            robot_path,
            actions,
            phys_cost=phys_cost,
            social_cost=social_cost,
            weight=weight,
        )

    def is_fully_executed(self):
        return self.action_index >= len(self.actions)

    def get_conflicts(
        self,
        agent_id: str,
        world: "world.World",
        robot_inflated_grid: BinaryOccupancyGrid,
        horizon: int,
        exit_early: bool = False,
        rp: t.Optional["ros2.RosPublisher"] = None,
    ) -> t.Set[Conflict]:
        assert agent_id not in robot_inflated_grid.cell_sets
        assert len(self.actions) + 1 == len(self.robot_path.poses)

        conflicts: t.Set[Conflict] = set()
        if horizon <= 0:
            return conflicts

        # Compute and display horizon cells
        if rp:
            rp.publish_transit_horizon_cells(
                poses=self.robot_path.poses,
                start_index=self.action_index,
                horizon=horizon,
                robot_inflated_grid=robot_inflated_grid,
                agent_id=agent_id,
            )

        # Check for RobotRobot conflicts within horizon, and RobotObstacle conflicts even beyond
        conflicting_cells: t.Set[GridCellModel] = set()
        conflicting_entities_cells: t.Set[GridCellModel] = set()
        for look_ahead_index, action in enumerate(
            self.actions[self.action_index : self.action_index + horizon]
        ):
            if isinstance(action, ba.Wait):
                continue

            pose = self.robot_path.poses[self.action_index + look_ahead_index]
            cell = utils.real_to_grid(
                pose[0],
                pose[1],
                robot_inflated_grid.cell_size,
                robot_inflated_grid.grid_pose,
            )

            if robot_inflated_grid.grid[cell[0]][cell[1]] != 0:
                colliding_obstacles = robot_inflated_grid.obstacles_uids_in_cell(cell)

                for uid in colliding_obstacles:
                    if isinstance(world.dynamic_entities[uid], agent.Agent) or (
                        uid in world.entity_to_agent
                        # ignore collisions with the obstacle the robot is currently holding
                        and world.entity_to_agent.get(uid) != agent_id
                    ):
                        other_agent_id = uid
                        if uid in world.entity_to_agent:
                            other_agent_id = world.entity_to_agent[uid]

                        if look_ahead_index < horizon:
                            conflicts.add(
                                RobotRobotConflict(
                                    agent_id=agent_id,
                                    robot_pose=pose,
                                    other_agent_id=other_agent_id,
                                    other_robot_pose=(
                                        world.dynamic_entities[uid].pose
                                        if isinstance(
                                            world.dynamic_entities[uid],
                                            agent.Agent,
                                        )
                                        else world.dynamic_entities[
                                            world.entity_to_agent[uid]
                                        ].pose
                                    ),
                                )
                            )
                            conflicting_cells.add(cell)
                            conflicting_entities_cells.update(
                                robot_inflated_grid.cell_sets[uid]
                            )
                            if exit_early:
                                if rp:
                                    rp.publish_transit_conflicting_cells(
                                        conflicting_cells,
                                        robot_inflated_grid,
                                        agent_id,
                                    )
                                    rp.publish_transit_conflicting_polygons_cells(
                                        conflicting_entities_cells,
                                        robot_inflated_grid,
                                        agent_id,
                                    )
                                return conflicts
                    else:
                        conflicts.add(RobotObstacleConflict(uid))
                        conflicting_cells.add(cell)
                        conflicting_entities_cells.update(
                            robot_inflated_grid.cell_sets[uid]
                        )
                        if exit_early:
                            if rp:
                                rp.publish_transit_conflicting_cells(
                                    conflicting_cells, robot_inflated_grid, agent_id
                                )
                                rp.publish_transit_conflicting_polygons_cells(
                                    conflicting_entities_cells,
                                    robot_inflated_grid,
                                    agent_id,
                                )
                            return conflicts

        if rp:
            rp.publish_transit_conflicting_cells(
                conflicting_cells, robot_inflated_grid, agent_id
            )
            rp.publish_transit_conflicting_polygons_cells(
                conflicting_entities_cells, robot_inflated_grid, agent_id
            )
        return conflicts

    def pop_next_action(self):
        action = self.actions[self.action_index]
        self.action_index += 1
        return action

    def get_length(self):
        return len(self.actions)

    def get_remaining_length(self):
        return max(0, len(self.actions) - self.action_index)


class EvasionTransitPath(TransitPath):
    def __init__(
        self,
        robot_path: RawPath,
        actions: t.List[ba.Action],
        conflicts: t.Set[Conflict],
        phys_cost: float | None = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ):
        TransitPath.__init__(self, robot_path, actions, phys_cost, social_cost, weight)
        self.evasion_goal_pose = (
            None if len(robot_path.poses) == 0 else robot_path.poses[-1]
        )
        self.transit_configuration_after_release = None
        self.release_executed = False
        self.conflicts = conflicts

    def set_wait(self, nb_wait_steps: int):
        for _ in range(nb_wait_steps):
            self.actions.append(ba.Wait())
            self.robot_path.poses.append(self.robot_path.poses[-1])

    def set_transit_configuration_after_release(
        self, transit_configuration_after_release: RobotConfiguration
    ):
        # TODO Fix this hack for better management of this non-mandatory first release action
        self.transit_configuration_after_release = transit_configuration_after_release
        self.release_executed = False

    def pop_next_action(self):
        if self.transit_configuration_after_release and not self.release_executed:
            self.release_executed = True
            return self.transit_configuration_after_release.action

        return TransitPath.pop_next_action(self)

    @classmethod
    def from_poses(
        cls,
        poses: t.List[Pose2D],
        robot_polygon: Polygon,
        robot_pose: Pose2D,
        conflicts: t.Set[Conflict],
    ):
        path = TransitPath.from_poses(
            poses=poses, robot_polygon=robot_polygon, robot_pose=robot_pose
        )
        return EvasionTransitPath(
            robot_path=path.robot_path, actions=path.actions, conflicts=conflicts
        )
