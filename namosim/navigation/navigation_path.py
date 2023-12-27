import math
import typing as t

from aabbtree import AABBTree
from shapely import GeometryCollection, Polygon

import namosim.agents.agent as agent
import namosim.display.ros2_publisher as ros2
import namosim.world.world as world
from namosim.data_models import UID, PoseModel
from namosim.navigation import basic_actions as ba
from namosim.navigation.conflict import (
    ConcurrentGrabConflict,
    RobotObstacleConflict,
    RobotRobotConflict,
    SimultaneousSpaceAccess,
    StealingMovableConflict,
    StolenMovableConflict,
)
from namosim.navigation.path_type import PathType
from namosim.utils import collision, utils
from namosim.world.binary_occupancy_grid import BinaryInflatedOccupancyGrid


class Path:
    """
    Represents a sequence of entity poses and their associated geometries such as covered grid cells,
    convex-swept volumes (CSVs) and bounding-box verticies.
    """

    path_type = PathType.PATH

    def __init__(
        self,
        poses: t.List[PoseModel],
        polygons: t.List[Polygon],
        cells: t.Optional[set[t.Tuple[int, int]]] = None,
        csv_polygons: t.Optional[t.Dict[t.Tuple[int], GeometryCollection]] = None,
        bb_vertices: t.Optional[t.List[t.List[t.Tuple[float, float]]]] = None,
    ):
        self.poses = poses
        self.polygons = polygons
        self.cells = cells
        self.csv_polygons = csv_polygons
        self.bb_vertices = bb_vertices or []
        self.is_transfer = False

    # TODO Have these trans and rot precision values be passed from calling functions !
    def is_start_pose(
        self, pose: PoseModel, trans_mult: float = 100.0, rot_mult: float = 1.0
    ):
        """
        Returns `True` if the given pose is equivalen to the first pose in the path,
        up to a fixed degree of precision, otherwise `False`.
        """
        other_pose = utils.real_pose_to_fixed_precision_pose(pose, trans_mult, rot_mult)
        start_pose = utils.real_pose_to_fixed_precision_pose(
            self.poses[0], trans_mult, rot_mult
        )
        return other_pose == start_pose


class TransferPath:
    """
    Represents a sequence of configurations in which a robot moves (transfers) a particular obstacle.
    """

    path_type: t.Literal[PathType.TRANSFER] = PathType.TRANSFER

    def __init__(
        self,
        robot_path: Path,
        obstacle_path: Path,
        actions: t.List[ba.BasicAction],
        grab_action: ba.Grab,
        release_action: ba.Release,
        obstacle_uid: UID,
        manip_pose_id: int,
        phys_cost: t.Optional[float] = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ):
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

    def has_infinite_cost(self):
        return True if self.total_cost == float("inf") else False

    def is_fully_executed(self):
        return self.action_index >= len(self.actions)

    def get_conflicts(
        self,
        robot_uid: UID,
        world: "world.World",
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        other_entities_polygons: t.Dict[UID, Polygon],
        other_entities_aabb_tree: AABBTree,
        other_entities_polygons_with_encompassing_circles: t.Dict[UID, Polygon],
        other_entities_with_encompassing_circles_aabb_tree: AABBTree,
        encompassing_circle_uid_to_robot_uid: t.Dict[UID, int],
        previously_moved_entities_uids: t.Set[UID],
        has_first_action: bool,
        shared_horizon: int | None = None,
        apply_strict_horizon: bool = False,
        exit_early_for_any_conflict: bool = False,
        exit_early_only_for_long_term_conflicts: bool = True,
        rp: t.Optional["ros2.RosPublisher"] = None,
        robot_name: str = "",
    ):
        # if rp:
        #     rp.cleanup_swept_area(ns=robot_name)

        if shared_horizon is None:
            # If no horizon is given, check all unexecuted actions
            shared_horizon = len(self.actions) - self.action_index
        elif shared_horizon <= 0 and apply_strict_horizon:
            return []

        conflicts = []

        collision_polygons = other_entities_polygons
        collision_aabb_tree = other_entities_aabb_tree

        # Compute and display horizon convex polygons
        if rp:
            rp.publish_transfer_horizon_convex_polygons(
                robot_csv_polygons=self.robot_path.csv_polygons or {},
                obstacle_csv_polygons=self.obstacle_path.csv_polygons or {},
                start_index=self.action_index,
                check_horizon=shared_horizon,
                robot_name=robot_name,
            )

        # Check conflicts for all actions within horizon (Robot-Robot) and beyond (other conflicts)
        for counter, index in enumerate(
            range(
                self.action_index,
                len(self.actions),
            )
        ):
            if apply_strict_horizon and counter > shared_horizon:
                break

            if counter == 0 and has_first_action:
                # If the first action in the path is the first action in the check horizon,
                # we also check for simultaneous conflilcts types at t+1
                collision_polygons = other_entities_polygons_with_encompassing_circles
                collision_aabb_tree = other_entities_with_encompassing_circles_aabb_tree
            else:
                collision_polygons = other_entities_polygons
                collision_aabb_tree = other_entities_aabb_tree

            action = self.actions[index]

            if action is self.grab_action:
                ## Grab actions should only occur at start of transfer path
                assert counter == 0

                # Check that obstacle is at the expected pose (except if it supposed to be moved before that)
                current_obstacle_pose = world.entities[self.obstacle_uid].pose
                obstacle_at_start_pose = self.obstacle_path.is_start_pose(
                    current_obstacle_pose
                )
                if (
                    not obstacle_at_start_pose
                    or self.obstacle_uid in world.entity_to_agent
                ):
                    if self.obstacle_uid in world.entity_to_agent:
                        conflicts.append(
                            StealingMovableConflict(
                                self.obstacle_uid,
                                world.entity_to_agent[self.obstacle_uid],
                            )
                        )
                    else:
                        conflicts.append(StolenMovableConflict(self.obstacle_uid))
                        if exit_early_only_for_long_term_conflicts:
                            return conflicts
                    if exit_early_for_any_conflict:
                        return conflicts

                if counter == 0 and has_first_action:
                    # If the first action in the path is the first action in the check horizon,
                    # and is a grab action, we also check the possibility of SimultaneousGrab conflicts t+1
                    radius = 2.0 * inflated_grid_by_robot.res
                    grab_zone = world.entities[self.obstacle_uid].polygon.buffer(
                        radius, join_style="mitre"
                    )
                    collides_with = collision.check_static_collision(
                        self.obstacle_uid,
                        grab_zone,
                        other_entities_polygons,
                        other_entities_aabb_tree,
                        ignored_uids={self.obstacle_uid},
                        break_at_first=False,
                        save_intersections=False,
                    )
                    if self.obstacle_uid in collides_with:
                        for uid in collides_with[self.obstacle_uid]:
                            if (
                                isinstance(
                                    world.entities[uid],
                                    agent.Agent,
                                )
                                and uid not in world.entity_to_agent.inverse
                            ):
                                conflicts.append(
                                    ConcurrentGrabConflict(self.obstacle_uid, uid)
                                )
                                if exit_early_for_any_conflict:
                                    return conflicts

                (
                    _,
                    collides_with,
                    _,
                    csv_polygons,
                    intersections,
                    bb_vertices,
                ) = collision.csv_check_collisions(
                    main_uid=robot_uid,
                    other_polygons=collision_polygons,
                    polygon_sequence=[
                        self.robot_path.polygons[0],
                        self.robot_path.polygons[1],
                    ],
                    action_sequence=[
                        collision.convert_action(
                            self.grab_action, self.robot_path.poses[0]
                        )
                    ],
                    bb_type="minimum_rotated_rectangle",
                    aabb_tree=collision_aabb_tree,
                    ignored_entities=previously_moved_entities_uids.union(
                        {self.obstacle_uid}
                    ),
                )

                if robot_uid in collides_with:
                    for uid in collides_with[robot_uid]:
                        if uid in encompassing_circle_uid_to_robot_uid:
                            if counter == 0 and has_first_action:
                                other_robot_uid = encompassing_circle_uid_to_robot_uid[
                                    uid
                                ]
                                other_robot_transfered_obstacle = world.entities.get(
                                    world.entity_to_agent.inverse.get(
                                        other_robot_uid, None
                                    ),
                                    None,
                                )
                                conflicts.append(
                                    SimultaneousSpaceAccess(
                                        robot_uid=robot_uid,
                                        robot_pose=self.robot_path.poses[0],
                                        other_robot_uid=other_robot_uid,
                                        other_robot_pose=world.entities[
                                            other_robot_uid
                                        ].pose,
                                        colliding_uids=(robot_uid, other_robot_uid),
                                        robot_transfered_obstacle_uid=self.obstacle_uid,
                                        robot_transfered_obstacle_pose=world.entities[
                                            self.obstacle_uid
                                        ].pose,
                                        other_robot_transfered_obstacle_uid=None
                                        if other_robot_transfered_obstacle is None
                                        else other_robot_transfered_obstacle.uid,
                                        other_robot_transfered_obstacle_pose=None
                                        if other_robot_transfered_obstacle is None
                                        else other_robot_transfered_obstacle.pose,
                                        at_grab=True,
                                    )
                                )
                                if exit_early_for_any_conflict:
                                    return conflicts
                        elif (
                            isinstance(world.entities[uid], agent.Agent)
                            or uid in world.entity_to_agent
                        ):
                            if counter <= shared_horizon:
                                conflicts.append(
                                    RobotRobotConflict(
                                        robot_uid=robot_uid,
                                        robot_pose=self.robot_path.poses[0],
                                        other_robot_uid=uid
                                        if isinstance(
                                            world.entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entity_to_agent[uid],
                                        other_robot_pose=world.entities[uid].pose
                                        if isinstance(
                                            world.entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entities[
                                            world.entity_to_agent[uid]
                                        ].pose,
                                        colliding_uids=(robot_uid, uid),
                                        robot_transfered_obstacle_uid=self.obstacle_uid,
                                        robot_transfered_obstacle_pose=world.entities[
                                            self.obstacle_uid
                                        ].pose,
                                        other_robot_transfered_obstacle_uid=uid
                                        if uid in world.entity_to_agent
                                        else None,
                                        other_robot_transfered_obstacle_pose=world.entities[
                                            uid
                                        ].pose
                                        if uid in world.entity_to_agent
                                        else None,
                                        at_grab=True,
                                    )
                                )
                                if exit_early_for_any_conflict:
                                    return conflicts
                        else:
                            conflicts.append(RobotObstacleConflict(uid))
                            if (
                                exit_early_for_any_conflict
                                or exit_early_only_for_long_term_conflicts
                            ):
                                return conflicts
            elif action is self.release_action:
                robot_before_release_pose = self.robot_path.poses[-2]
                obstacle_before_release_pose = self.obstacle_path.poses[-2]

                (
                    _,
                    collides_with,
                    _,
                    csv_polygons,
                    intersections,
                    bb_vertices,
                ) = collision.csv_check_collisions(
                    main_uid=robot_uid,
                    other_polygons=collision_polygons,
                    polygon_sequence=[
                        self.robot_path.polygons[-2],
                        self.robot_path.polygons[-1],
                    ],
                    action_sequence=[
                        collision.convert_action(
                            self.release_action, robot_before_release_pose
                        )
                    ],
                    bb_type="minimum_rotated_rectangle",
                    aabb_tree=collision_aabb_tree,
                    ignored_entities=previously_moved_entities_uids.union(
                        {self.obstacle_uid}
                    ),
                )
                if robot_uid in collides_with:
                    for uid in collides_with[robot_uid]:
                        if uid in encompassing_circle_uid_to_robot_uid:
                            if counter == 0 and has_first_action:
                                other_robot_uid = encompassing_circle_uid_to_robot_uid[
                                    uid
                                ]
                                other_robot_transfered_obstacle = world.entities.get(
                                    world.entity_to_agent.inverse.get(
                                        other_robot_uid, None
                                    ),
                                    None,
                                )
                                conflicts.append(
                                    SimultaneousSpaceAccess(
                                        robot_uid=robot_uid,
                                        robot_pose=robot_before_release_pose,
                                        other_robot_uid=other_robot_uid,
                                        other_robot_pose=world.entities[
                                            other_robot_uid
                                        ].pose,
                                        colliding_uids=(robot_uid, other_robot_uid),
                                        robot_transfered_obstacle_uid=self.obstacle_uid,
                                        robot_transfered_obstacle_pose=obstacle_before_release_pose,
                                        other_robot_transfered_obstacle_uid=None
                                        if other_robot_transfered_obstacle is None
                                        else other_robot_transfered_obstacle.uid,
                                        other_robot_transfered_obstacle_pose=None
                                        if other_robot_transfered_obstacle is None
                                        else other_robot_transfered_obstacle.pose,
                                        at_release=True,
                                    )
                                )
                                if exit_early_for_any_conflict:
                                    return conflicts
                        elif (
                            isinstance(world.entities[uid], agent.Agent)
                            or uid in world.entity_to_agent
                        ):
                            if counter <= shared_horizon:
                                conflicts.append(
                                    RobotRobotConflict(
                                        robot_uid=robot_uid,
                                        robot_pose=robot_before_release_pose,
                                        other_robot_uid=uid
                                        if isinstance(
                                            world.entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entity_to_agent[uid],
                                        other_robot_pose=world.entities[uid].pose
                                        if isinstance(
                                            world.entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entities[
                                            world.entity_to_agent[uid]
                                        ].pose,
                                        colliding_uids=(robot_uid, uid),
                                        robot_transfered_obstacle_uid=self.obstacle_uid,
                                        robot_transfered_obstacle_pose=obstacle_before_release_pose,
                                        other_robot_transfered_obstacle_uid=uid
                                        if uid in world.entity_to_agent
                                        else None,
                                        other_robot_transfered_obstacle_pose=world.entities[
                                            uid
                                        ].pose
                                        if uid in world.entity_to_agent
                                        else None,
                                        at_release=True,
                                    )
                                )
                                if exit_early_for_any_conflict:
                                    return conflicts
                        else:
                            conflicts.append(RobotObstacleConflict(uid))
                            if (
                                exit_early_for_any_conflict
                                or exit_early_only_for_long_term_conflicts
                            ):
                                return conflicts
            else:
                (
                    _,
                    collides_with,
                    _,
                    csv_polygons,
                    intersections,
                    bb_vertices,
                ) = collision.csv_check_collisions(
                    main_uid=robot_uid,
                    other_polygons=collision_polygons,
                    polygon_sequence=self.robot_path.polygons[index : index + 2],
                    action_sequence=[
                        collision.convert_action(
                            self.actions[index], self.robot_path.poses[index]
                        )
                    ],
                    bb_type="minimum_rotated_rectangle",
                    aabb_tree=collision_aabb_tree,
                    ignored_entities=previously_moved_entities_uids.union(
                        {self.obstacle_uid}
                    ),
                )
                if robot_uid in collides_with:
                    for uid in collides_with[robot_uid]:
                        if uid in encompassing_circle_uid_to_robot_uid:
                            if counter == 0 and has_first_action:
                                other_robot_uid = encompassing_circle_uid_to_robot_uid[
                                    uid
                                ]
                                other_robot_transfered_obstacle = world.entities.get(
                                    world.entity_to_agent.inverse.get(
                                        other_robot_uid, None
                                    ),
                                    None,
                                )
                                conflicts.append(
                                    SimultaneousSpaceAccess(
                                        robot_uid=robot_uid,
                                        robot_pose=self.robot_path.poses[index],
                                        other_robot_uid=other_robot_uid,
                                        other_robot_pose=world.entities[
                                            other_robot_uid
                                        ].pose,
                                        colliding_uids=(robot_uid, other_robot_uid),
                                        robot_transfered_obstacle_uid=self.obstacle_uid,
                                        robot_transfered_obstacle_pose=self.obstacle_path.poses[
                                            index
                                        ],
                                        other_robot_transfered_obstacle_uid=None
                                        if other_robot_transfered_obstacle is None
                                        else other_robot_transfered_obstacle.uid,
                                        other_robot_transfered_obstacle_pose=None
                                        if other_robot_transfered_obstacle is None
                                        else other_robot_transfered_obstacle.pose,
                                    )
                                )
                                if exit_early_for_any_conflict:
                                    return conflicts

                        elif (
                            isinstance(world.entities[uid], agent.Agent)
                            or uid in world.entity_to_agent
                        ):
                            if counter <= shared_horizon:
                                conflicts.append(
                                    RobotRobotConflict(
                                        robot_uid=robot_uid,
                                        robot_pose=self.robot_path.poses[index],
                                        other_robot_uid=uid
                                        if isinstance(
                                            world.entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entity_to_agent[uid],
                                        other_robot_pose=world.entities[uid].pose
                                        if isinstance(
                                            world.entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entities[
                                            world.entity_to_agent[uid]
                                        ].pose,
                                        colliding_uids=(robot_uid, uid),
                                        robot_transfered_obstacle_uid=self.obstacle_uid,
                                        robot_transfered_obstacle_pose=self.obstacle_path.poses[
                                            index
                                        ],
                                        other_robot_transfered_obstacle_uid=uid
                                        if uid in world.entity_to_agent
                                        else None,
                                        other_robot_transfered_obstacle_pose=world.entities[
                                            uid
                                        ].pose
                                        if uid in world.entity_to_agent
                                        else None,
                                    )
                                )
                                if exit_early_for_any_conflict:
                                    return conflicts
                        else:
                            conflicts.append(RobotObstacleConflict(uid))
                            if (
                                exit_early_for_any_conflict
                                or exit_early_only_for_long_term_conflicts
                            ):
                                return conflicts
                (
                    _,
                    collides_with,
                    _,
                    csv_polygons,
                    intersections,
                    bb_vertices,
                ) = collision.csv_check_collisions(
                    main_uid=self.obstacle_uid,
                    other_polygons=collision_polygons,
                    polygon_sequence=self.obstacle_path.polygons[index : index + 2],
                    action_sequence=[
                        collision.convert_action(
                            self.actions[index], self.obstacle_path.poses[index]
                        )
                    ],
                    bb_type="minimum_rotated_rectangle",
                    aabb_tree=collision_aabb_tree,
                    ignored_entities=previously_moved_entities_uids.union(
                        {self.obstacle_uid}
                    ),
                )
                if self.obstacle_uid in collides_with:
                    for uid in collides_with[self.obstacle_uid]:
                        if uid in encompassing_circle_uid_to_robot_uid:
                            if counter == 0 and has_first_action:
                                other_robot_uid = encompassing_circle_uid_to_robot_uid[
                                    uid
                                ]
                                other_robot_transfered_obstacle = world.entities.get(
                                    world.entity_to_agent.inverse.get(
                                        other_robot_uid, None
                                    ),
                                    None,
                                )
                                conflicts.append(
                                    SimultaneousSpaceAccess(
                                        robot_uid=robot_uid,
                                        robot_pose=self.robot_path.poses[index],
                                        other_robot_uid=other_robot_uid,
                                        other_robot_pose=world.entities[
                                            other_robot_uid
                                        ].pose,
                                        colliding_uids=(
                                            self.obstacle_uid,
                                            other_robot_uid,
                                        ),
                                        robot_transfered_obstacle_uid=self.obstacle_uid,
                                        robot_transfered_obstacle_pose=self.obstacle_path.poses[
                                            index
                                        ],
                                        other_robot_transfered_obstacle_uid=None
                                        if other_robot_transfered_obstacle is None
                                        else other_robot_transfered_obstacle.uid,
                                        other_robot_transfered_obstacle_pose=None
                                        if other_robot_transfered_obstacle is None
                                        else other_robot_transfered_obstacle.pose,
                                    )
                                )
                                if exit_early_for_any_conflict:
                                    return conflicts
                        elif (
                            isinstance(world.entities[uid], agent.Agent)
                            or uid in world.entity_to_agent
                        ):
                            if counter <= shared_horizon:
                                conflicts.append(
                                    RobotRobotConflict(
                                        robot_uid=robot_uid,
                                        robot_pose=self.robot_path.poses[index],
                                        other_robot_uid=uid
                                        if isinstance(
                                            world.entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entity_to_agent[uid],
                                        other_robot_pose=world.entities[uid].pose
                                        if isinstance(
                                            world.entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entities[
                                            world.entity_to_agent[uid]
                                        ].pose,
                                        colliding_uids=(self.obstacle_uid, uid),
                                        robot_transfered_obstacle_uid=self.obstacle_uid,
                                        robot_transfered_obstacle_pose=self.obstacle_path.poses[
                                            index
                                        ],
                                        other_robot_transfered_obstacle_uid=uid
                                        if uid in world.entity_to_agent
                                        else None,
                                        other_robot_transfered_obstacle_pose=world.entities[
                                            uid
                                        ].pose
                                        if uid in world.entity_to_agent
                                        else None,
                                    )
                                )
                                if exit_early_for_any_conflict:
                                    return conflicts
                        else:
                            conflicts.append(RobotObstacleConflict(uid))
                            if (
                                exit_early_for_any_conflict
                                or exit_early_only_for_long_term_conflicts
                            ):
                                return conflicts

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
        robot_path: Path,
        actions: t.List[ba.BasicAction],
        phys_cost: float | None = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ):
        if len(robot_path.polygons) != len(robot_path.poses) != len(actions) + 1:
            raise ValueError(
                "A TransitPath requires that its polygon and pose arrays are the same size. "
                "The action array must be of this same size -1."
                "Current sizes are: polygon({}), pose({}), action({})".format(
                    len(robot_path.polygons), len(robot_path.poses), len(actions)
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
        poses: t.List[PoseModel],
        robot_polygon: Polygon,
        robot_pose: PoseModel,
        phys_cost: float | None = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ):
        # Separate translation from rotation actions
        if len(poses) == 0:
            return cls(
                robot_path=Path([], []),
                actions=[],
                phys_cost=phys_cost,
                social_cost=social_cost,
                weight=weight,
            )

        if robot_pose != poses[0]:
            raise Exception("Robot pose not equal to start pose")

        if len(poses) == 1:
            return cls(
                robot_path=Path(poses=poses, polygons=[robot_polygon]),
                actions=[],
                phys_cost=phys_cost,
                social_cost=social_cost,
                weight=weight,
            )

        actions: t.List[ba.BasicAction] = []
        updated_poses = [poses[0]]

        for p in poses:
            for e in p:
                if math.isnan(e):
                    pass

        for pose, next_pose in zip(poses, poses[1:]):
            has_translation = not all(
                [
                    utils.is_close(pose[0], next_pose[0], rel_tol=1e-6),
                    utils.is_close(pose[1], next_pose[1], rel_tol=1e-6),
                ]
            )

            current_angle = pose[2]
            turn_towards_angle = 0.0

            if has_translation:
                turn_towards_angle = utils.get_angle_to_turn(pose, next_pose)
                if turn_towards_angle != 0:
                    current_angle = utils.add_angles(current_angle, turn_towards_angle)
                    actions.append(ba.Rotation(angle=turn_towards_angle))
                    updated_poses.append((pose[0], pose[1], current_angle))

                actions.append(
                    ba.Translation.from_absolute_translation_vector(
                        utils.get_translation(pose, next_pose)
                    )
                )
                updated_poses.append((next_pose[0], next_pose[1], current_angle))

            has_rotation = not utils.angle_is_close(
                current_angle, next_pose[2], rel_tol=1e-6
            )

            if has_rotation:
                remaining_angle = utils.subtract_angles(next_pose[2], current_angle)
                actions.append(ba.Rotation(angle=remaining_angle))
                updated_poses.append(next_pose)

            if not has_rotation and not has_translation:
                updated_poses.append(next_pose)

        polygons = [
            utils.set_polygon_pose(robot_polygon, robot_pose, pose)
            for pose in updated_poses
        ]
        robot_path = Path(updated_poses, polygons)

        return cls(
            robot_path,
            actions,
            phys_cost=phys_cost,
            social_cost=social_cost,
            weight=weight,
        )

    def has_infinite_cost(self):
        return True if self.total_cost == float("inf") else False

    def is_fully_executed(self):
        return self.action_index >= len(self.actions)

    def get_conflicts(
        self,
        robot_uid,
        world,
        inflated_grid_by_robot,
        encompassing_circle_uid_to_robot_uid,
        has_first_action,
        shared_horizon=None,
        apply_strict_horizon=False,
        exit_early_for_any_conflict=False,
        exit_early_only_for_long_term_conflicts=True,
        rp: t.Optional["ros2.RosPublisher"] = None,
        robot_name="",
    ):
        if not self.actions:
            return []

        if shared_horizon is None:
            # If no horizon is given, check all unexecuted actions
            shared_horizon = len(self.actions) + 1 - self.action_index
        elif shared_horizon <= 0 and apply_strict_horizon:
            return []

        conflicts = []

        encompassing_circles_uids = set(encompassing_circle_uid_to_robot_uid.keys())

        # Compute and display horizon cells
        if rp:
            rp.publish_transit_horizon_cells(
                poses=self.robot_path.poses,
                start_index=self.action_index,
                check_horizon=shared_horizon,
                inflated_grid_by_robot=inflated_grid_by_robot,
                robot_name=robot_name,
            )

        # Check for RobotRobot conflicts within horizon, and RobotObstacle conflicts even beyond
        conflicting_cells = set()
        conflicting_entities_cells = set()
        for counter, index in enumerate(
            range(self.action_index, len(self.actions) + 1)
        ):
            if index < len(self.actions) and isinstance(self.actions[index], ba.Wait):
                continue

            if apply_strict_horizon and counter > shared_horizon:
                break

            if counter == 0 and has_first_action:
                # If the first action in the path is the first action in the check horizon,
                # we also check for simultaneous conflilcts types at t+1
                inflated_grid_by_robot.activate_entities(encompassing_circles_uids)
            else:
                inflated_grid_by_robot.deactivate_entities(encompassing_circles_uids)

            pose = self.robot_path.poses[index]
            cell = utils.real_to_grid(
                pose[0],
                pose[1],
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
            )

            if inflated_grid_by_robot.grid[cell[0]][cell[1]] != 0:
                colliding_obstacles = inflated_grid_by_robot.obstacles_uids_in_cell(
                    cell
                )
                for uid in colliding_obstacles:
                    if uid in encompassing_circles_uids:
                        if counter == 0 and has_first_action:
                            other_robot_uid = encompassing_circle_uid_to_robot_uid[uid]
                            other_robot_transfered_obstacle = world.entities.get(
                                world.entity_to_agent.inverse.get(
                                    other_robot_uid, None
                                ),
                                None,
                            )
                            conflicts.append(
                                SimultaneousSpaceAccess(
                                    robot_uid=robot_uid,
                                    robot_pose=self.robot_path.poses[index],
                                    other_robot_uid=other_robot_uid,
                                    other_robot_pose=world.entities[
                                        other_robot_uid
                                    ].pose,
                                    colliding_uids=(robot_uid, other_robot_uid),
                                    robot_transfered_obstacle_uid=None,
                                    robot_transfered_obstacle_pose=None,
                                    other_robot_transfered_obstacle_uid=None
                                    if other_robot_transfered_obstacle is None
                                    else other_robot_transfered_obstacle.uid,
                                    other_robot_transfered_obstacle_pose=None
                                    if other_robot_transfered_obstacle is None
                                    else other_robot_transfered_obstacle.pose,
                                )
                            )
                            conflicting_cells.add(cell)
                            conflicting_entities_cells.update(
                                inflated_grid_by_robot.cells_sets[uid]
                            )
                            if exit_early_for_any_conflict:
                                rp.publish_transit_conflicting_cells(
                                    conflicting_cells,
                                    inflated_grid_by_robot,
                                    robot_name,
                                )
                                rp.publish_transit_conflicting_polygons_cells(
                                    conflicting_entities_cells,
                                    inflated_grid_by_robot,
                                    robot_name,
                                )
                                return conflicts
                    elif (
                        isinstance(world.entities[uid], agent.Agent)
                        or uid in world.entity_to_agent
                    ):
                        if counter <= shared_horizon:
                            conflicts.append(
                                RobotRobotConflict(
                                    robot_uid=robot_uid,
                                    robot_pose=self.robot_path.poses[index],
                                    other_robot_uid=uid
                                    if isinstance(
                                        world.entities[uid],
                                        agent.Agent,
                                    )
                                    else world.entity_to_agent[uid],
                                    other_robot_pose=world.entities[uid].pose
                                    if isinstance(
                                        world.entities[uid],
                                        agent.Agent,
                                    )
                                    else world.entities[
                                        world.entity_to_agent[uid]
                                    ].pose,
                                    colliding_uids=(robot_uid, uid),
                                    robot_transfered_obstacle_uid=None,
                                    robot_transfered_obstacle_pose=None,
                                    other_robot_transfered_obstacle_uid=uid
                                    if uid in world.entity_to_agent
                                    else None,
                                    other_robot_transfered_obstacle_pose=world.entities[
                                        uid
                                    ].pose
                                    if uid in world.entity_to_agent
                                    else None,
                                )
                            )
                            conflicting_cells.add(cell)
                            conflicting_entities_cells.update(
                                inflated_grid_by_robot.cells_sets[uid]
                            )
                            if exit_early_for_any_conflict:
                                rp.publish_transit_conflicting_cells(
                                    conflicting_cells,
                                    inflated_grid_by_robot,
                                    robot_name,
                                )
                                rp.publish_transit_conflicting_polygons_cells(
                                    conflicting_entities_cells,
                                    inflated_grid_by_robot,
                                    robot_name,
                                )
                                return conflicts
                    else:
                        conflicts.append(RobotObstacleConflict(uid))
                        conflicting_cells.add(cell)
                        conflicting_entities_cells.update(
                            inflated_grid_by_robot.cells_sets[uid]
                        )
                        if (
                            exit_early_for_any_conflict
                            or exit_early_only_for_long_term_conflicts
                        ):
                            rp.publish_transit_conflicting_cells(
                                conflicting_cells, inflated_grid_by_robot, robot_name
                            )
                            rp.publish_transit_conflicting_polygons_cells(
                                conflicting_entities_cells,
                                inflated_grid_by_robot,
                                robot_name,
                            )
                            return conflicts

        rp.publish_transit_conflicting_cells(
            conflicting_cells, inflated_grid_by_robot, robot_name
        )
        rp.publish_transit_conflicting_polygons_cells(
            conflicting_entities_cells, inflated_grid_by_robot, robot_name
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
        self, robot_path, actions, phys_cost=None, social_cost=0.0, weight=1.0
    ):
        TransitPath.__init__(self, robot_path, actions, phys_cost, social_cost, weight)
        self.evasion_goal_pose = (
            None if len(robot_path.poses) == 0 else robot_path.poses[-1]
        )
        self.transit_configuration_after_release = None
        self.release_executed = False

    def set_wait(self, nb_wait_steps):
        for i in range(nb_wait_steps):
            self.actions.append(ba.Wait())
            self.robot_path.poses.append(self.robot_path.poses[-1])

    def set_transit_configuration_after_release(
        self, transit_configuration_after_release
    ):
        # TODO Fix this hack for better management of this non-mandatory first release action
        self.transit_configuration_after_release = transit_configuration_after_release
        self.release_executed = False

    def pop_next_action(self):
        if self.transit_configuration_after_release and not self.release_executed:
            self.release_executed = True
            return self.transit_configuration_after_release.action
        else:
            return TransitPath.pop_next_action(self)
            return TransitPath.pop_next_action(self)
