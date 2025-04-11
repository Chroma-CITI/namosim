import typing as t

import numpy as np
from aabbtree import AABBTree
from shapely.geometry import Polygon

import namosim.agents.agent as agent
import namosim.display.ros2_publisher as ros2
import namosim.world.world as world
from namosim.agents.stilman_configurations import RobotConfiguration
from namosim.data_models import GridCellModel, PoseModel
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


class RawPath:
    """
    Represents a sequence of entity poses and their associated polygons
    """

    def __init__(
        self,
        poses: t.List[PoseModel],
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

    def reset(self):
        self.action_index = 0

    def is_fully_executed(self):
        return self.action_index >= len(self.actions)

    def get_conflicts(
        self,
        agent_id: str,
        world: "world.World",
        robot_inflated_grid: BinaryOccupancyGrid,
        other_entities_polygons: t.Dict[str, Polygon],
        other_entities_aabb_tree: AABBTree,
        other_entities_polygons_with_encompassing_circles: t.Dict[str, Polygon],
        other_entities_with_encompassing_circles_aabb_tree: AABBTree,
        encompassing_circle_uid_to_agent_id: t.Dict[str, str],
        previously_moved_entities_uids: t.Set[str],
        check_horizon: int,
        has_first_action: bool,
        grab_start_distance: float,
        apply_strict_horizon: bool = False,
        exit_early_for_any_conflict: bool = False,
        exit_early_only_for_long_term_conflicts: bool = True,
        rp: t.Optional["ros2.RosPublisher"] = None,
    ) -> t.Set[Conflict]:
        conflicts: t.Set[Conflict] = set()

        assert agent_id not in robot_inflated_grid.cell_sets

        if check_horizon <= 0 and apply_strict_horizon:
            return conflicts

        robot = t.cast(agent.Agent, world.dynamic_entities[agent_id])

        collision_polygons = other_entities_polygons
        collision_aabb_tree = other_entities_aabb_tree

        # Compute and display horizon convex polygons
        if rp:
            rp.publish_transfer_horizon_polygons(
                robot_polygons=self.robot_path.polygons,
                obstacle_polygons=self.obstacle_path.polygons,
                start_index=self.action_index,
                check_horizon=check_horizon,
                agent_id=agent_id,
            )

        assert len(self.actions) + 1 == len(self.robot_path.poses)

        # Check conflicts for all actions within horizon (Robot-Robot) and beyond (other conflicts)
        for look_ahead_index, (action, robot_pose_prior_to_action) in enumerate(
            zip(
                self.actions[self.action_index :],
                self.robot_path.poses[self.action_index :],
            )
        ):
            if apply_strict_horizon and look_ahead_index >= check_horizon:
                break

            if look_ahead_index < check_horizon and has_first_action:
                # If the first action in the path is the first action in the check horizon,
                # we also check for simultaneous conflilcts types at t+1
                collision_polygons = other_entities_polygons_with_encompassing_circles
                collision_aabb_tree = other_entities_with_encompassing_circles_aabb_tree
            else:
                collision_polygons = other_entities_polygons
                collision_aabb_tree = other_entities_aabb_tree

            assert agent_id not in collision_polygons

            if action is self.grab_action:
                ## Grab actions should only occur at start of transfer path
                assert self.action_index == 0

                # Check that obstacle is at the expected pose (except if it supposed to be moved before that)
                current_obstacle_pose = world.dynamic_entities[self.obstacle_uid].pose
                obstacle_at_start_pose = self.obstacle_path.is_start_pose(
                    current_obstacle_pose
                )

                already_grabbed_by_current_robot = (
                    world.entity_to_agent.get(self.obstacle_uid) == agent_id
                )

                if already_grabbed_by_current_robot:
                    ## This happens when the plan has two consecutive transfer paths back-to-back.
                    break

                # If held by another agent
                if self.obstacle_uid in world.entity_to_agent:
                    conflicts.add(
                        StealingMovableConflict(
                            self.obstacle_uid,
                            world.entity_to_agent[self.obstacle_uid],
                        )
                    )
                    if (
                        exit_early_only_for_long_term_conflicts
                        or exit_early_for_any_conflict
                    ):
                        return conflicts

                # If the obstacle is no longer where the agent thought it would be and it wasn't previously moved by the robot, we have a stolen object conflict.
                if (
                    not obstacle_at_start_pose
                    and self.obstacle_uid not in previously_moved_entities_uids
                ):
                    conflicts.add(StolenMovableConflict(self.obstacle_uid))
                    if (
                        exit_early_only_for_long_term_conflicts
                        or exit_early_for_any_conflict
                    ):
                        return conflicts

                # Check for SimultaneousSpace conflict that might result from the grab, since a grab instantly expands the robot's conflict radius.
                if look_ahead_index < check_horizon:
                    radius = world.get_robot_conflict_radius(
                        agent_id=agent_id,
                        grab_start_distance=grab_start_distance,
                        obstacle_id=self.obstacle_uid,
                    )
                    grab_zone = robot.polygon.centroid.buffer(
                        radius, join_style=JOIN_STYLE.mitre
                    )
                    collides_with = collision.get_collisions_for_entity(
                        grab_zone,
                        collision_polygons,
                        collision_aabb_tree,
                        ignored_entities={self.obstacle_uid},
                        break_at_first=False,
                    )

                    for uid in collides_with:
                        if uid in encompassing_circle_uid_to_agent_id:
                            uid = encompassing_circle_uid_to_agent_id[uid]
                        assert uid != agent_id
                        if isinstance(
                            world.dynamic_entities[uid],
                            agent.Agent,
                        ):
                            other_robot_obstacle = world.get_agent_held_obstacle(uid)

                            conflicts.add(
                                SimultaneousSpaceAccess(
                                    agent_id=agent_id,
                                    robot_pose=robot_pose_prior_to_action,
                                    other_agent_id=uid,
                                    other_robot_pose=world.dynamic_entities[uid].pose,
                                    colliding_uids=(agent_id, uid),
                                    robot_transfered_obstacle_uid=self.obstacle_uid,
                                    robot_transfered_obstacle_pose=world.dynamic_entities[
                                        self.obstacle_uid
                                    ].pose,
                                    other_robot_transfered_obstacle_uid=(
                                        other_robot_obstacle.uid
                                        if other_robot_obstacle
                                        else None
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        other_robot_obstacle.pose
                                        if other_robot_obstacle
                                        else None
                                    ),
                                    at_grab=True,
                                )
                            )
                            if exit_early_for_any_conflict:
                                return conflicts

                # Check for ConcurrentGrabConflict if the first action in the path is the first action in the check horizon,
                if look_ahead_index < check_horizon and has_first_action:
                    grab_zone = world.dynamic_entities[
                        self.obstacle_uid
                    ].polygon.buffer(grab_start_distance, join_style=JOIN_STYLE.mitre)
                    collides_with, _ = collision.get_collisions_for_entity(
                        grab_zone,
                        collision_polygons,
                        collision_aabb_tree,
                        ignored_entities={self.obstacle_uid},
                        break_at_first=False,
                    )
                    for uid in collides_with:
                        if uid in encompassing_circle_uid_to_agent_id:
                            uid = encompassing_circle_uid_to_agent_id[uid]

                        if (
                            isinstance(
                                world.dynamic_entities[uid],
                                agent.Agent,
                            )
                            and uid not in world.entity_to_agent.inverse
                        ):
                            conflicts.add(
                                ConcurrentGrabConflict(self.obstacle_uid, uid)
                            )
                            if exit_early_for_any_conflict:
                                return conflicts

                (
                    collides_with,
                    _,
                ) = collision.get_csv_collisions(
                    agent_id=agent_id,
                    robot_pose=self.robot_path.poses[0],
                    robot_action=self.grab_action,
                    other_polygons=collision_polygons,
                    polygon=self.robot_path.polygons[0],
                    ignored_entities=previously_moved_entities_uids.union(
                        {self.obstacle_uid}
                    ),
                    others_aabb_tree=collision_aabb_tree,
                )

                for uid in collides_with:
                    if uid in encompassing_circle_uid_to_agent_id:
                        if look_ahead_index < check_horizon and has_first_action:
                            other_agent_id = encompassing_circle_uid_to_agent_id[uid]
                            other_robot_obs = world.get_agent_held_obstacle(
                                other_agent_id
                            )
                            conflicts.add(
                                SimultaneousSpaceAccess(
                                    agent_id=agent_id,
                                    robot_pose=robot_pose_prior_to_action,
                                    other_agent_id=other_agent_id,
                                    other_robot_pose=world.dynamic_entities[
                                        other_agent_id
                                    ].pose,
                                    colliding_uids=(agent_id, other_agent_id),
                                    robot_transfered_obstacle_uid=self.obstacle_uid,
                                    robot_transfered_obstacle_pose=world.dynamic_entities[
                                        self.obstacle_uid
                                    ].pose,
                                    other_robot_transfered_obstacle_uid=(
                                        other_robot_obs.uid if other_robot_obs else None
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        other_robot_obs.pose
                                        if other_robot_obs
                                        else None
                                    ),
                                    at_grab=True,
                                )
                            )
                            if exit_early_for_any_conflict:
                                return conflicts
                    elif (
                        isinstance(world.dynamic_entities[uid], agent.Agent)
                        or uid in world.entity_to_agent
                    ):
                        if look_ahead_index < check_horizon:
                            conflicts.add(
                                RobotRobotConflict(
                                    agent_id=agent_id,
                                    robot_pose=robot_pose_prior_to_action,
                                    other_agent_id=(
                                        uid
                                        if isinstance(
                                            world.dynamic_entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entity_to_agent[uid]
                                    ),
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
                                    colliding_uids=(agent_id, uid),
                                    robot_transfered_obstacle_uid=self.obstacle_uid,
                                    robot_transfered_obstacle_pose=world.dynamic_entities[
                                        self.obstacle_uid
                                    ].pose,
                                    other_robot_transfered_obstacle_uid=(
                                        uid if uid in world.entity_to_agent else None
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        world.dynamic_entities[uid].pose
                                        if uid in world.entity_to_agent
                                        else None
                                    ),
                                    at_grab=True,
                                )
                            )
                            if exit_early_for_any_conflict:
                                return conflicts
                    else:
                        conflicts.add(RobotObstacleConflict(uid))
                        if (
                            exit_early_for_any_conflict
                            or exit_early_only_for_long_term_conflicts
                        ):
                            return conflicts
            elif action is self.release_action:
                robot_before_release_pose = self.robot_path.poses[-2]
                obstacle_before_release_pose = self.obstacle_path.poses[-2]

                (
                    collides_with,
                    _,
                ) = collision.get_csv_collisions(
                    agent_id=agent_id,
                    robot_pose=robot_before_release_pose,
                    robot_action=self.release_action,
                    other_polygons=collision_polygons,
                    polygon=self.robot_path.polygons[-2],
                    ignored_entities=previously_moved_entities_uids.union(
                        {self.obstacle_uid}
                    ),
                    others_aabb_tree=collision_aabb_tree,
                )

                for uid in collides_with:
                    if uid in encompassing_circle_uid_to_agent_id:
                        if look_ahead_index < check_horizon and has_first_action:
                            other_agent_id = encompassing_circle_uid_to_agent_id[uid]
                            other_robot_obs = world.get_agent_held_obstacle(
                                other_agent_id
                            )
                            conflicts.add(
                                SimultaneousSpaceAccess(
                                    agent_id=agent_id,
                                    robot_pose=robot_before_release_pose,
                                    other_agent_id=other_agent_id,
                                    other_robot_pose=world.dynamic_entities[
                                        other_agent_id
                                    ].pose,
                                    colliding_uids=(agent_id, other_agent_id),
                                    robot_transfered_obstacle_uid=self.obstacle_uid,
                                    robot_transfered_obstacle_pose=obstacle_before_release_pose,
                                    other_robot_transfered_obstacle_uid=(
                                        None
                                        if other_robot_obs is None
                                        else other_robot_obs.uid
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        None
                                        if other_robot_obs is None
                                        else other_robot_obs.pose
                                    ),
                                    at_release=True,
                                )
                            )
                            if exit_early_for_any_conflict:
                                return conflicts
                    elif (
                        isinstance(world.dynamic_entities[uid], agent.Agent)
                        or uid in world.entity_to_agent
                    ):
                        if look_ahead_index < check_horizon:
                            conflicts.add(
                                RobotRobotConflict(
                                    agent_id=agent_id,
                                    robot_pose=robot_before_release_pose,
                                    other_agent_id=(
                                        uid
                                        if isinstance(
                                            world.dynamic_entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entity_to_agent[uid]
                                    ),
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
                                    colliding_uids=(agent_id, uid),
                                    robot_transfered_obstacle_uid=self.obstacle_uid,
                                    robot_transfered_obstacle_pose=obstacle_before_release_pose,
                                    other_robot_transfered_obstacle_uid=(
                                        uid if uid in world.entity_to_agent else None
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        world.dynamic_entities[uid].pose
                                        if uid in world.entity_to_agent
                                        else None
                                    ),
                                    at_release=True,
                                )
                            )
                            if exit_early_for_any_conflict:
                                return conflicts
                    else:
                        conflicts.add(RobotObstacleConflict(uid))
                        if (
                            exit_early_for_any_conflict
                            or exit_early_only_for_long_term_conflicts
                        ):
                            return conflicts
            else:
                (
                    collides_with,
                    _,
                ) = collision.get_csv_collisions(
                    agent_id=agent_id,
                    robot_pose=robot_pose_prior_to_action,
                    robot_action=action,
                    other_polygons=collision_polygons,
                    polygon=self.robot_path.polygons[
                        self.action_index + look_ahead_index
                    ],
                    ignored_entities=previously_moved_entities_uids.union(
                        {self.obstacle_uid}
                    ),
                    others_aabb_tree=collision_aabb_tree,
                )

                for uid in collides_with:
                    if uid in encompassing_circle_uid_to_agent_id:
                        if look_ahead_index < check_horizon and has_first_action:
                            other_agent_id = encompassing_circle_uid_to_agent_id[uid]
                            other_robot_obs = world.get_agent_held_obstacle(
                                other_agent_id
                            )
                            conflicts.add(
                                SimultaneousSpaceAccess(
                                    agent_id=agent_id,
                                    robot_pose=robot_pose_prior_to_action,
                                    other_agent_id=other_agent_id,
                                    other_robot_pose=world.dynamic_entities[
                                        other_agent_id
                                    ].pose,
                                    colliding_uids=(agent_id, other_agent_id),
                                    robot_transfered_obstacle_uid=self.obstacle_uid,
                                    robot_transfered_obstacle_pose=self.obstacle_path.poses[
                                        self.action_index + look_ahead_index
                                    ],
                                    other_robot_transfered_obstacle_uid=(
                                        None
                                        if other_robot_obs is None
                                        else other_robot_obs.uid
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        None
                                        if other_robot_obs is None
                                        else other_robot_obs.pose
                                    ),
                                )
                            )
                            if exit_early_for_any_conflict:
                                return conflicts

                    elif (
                        isinstance(world.dynamic_entities[uid], agent.Agent)
                        or uid in world.entity_to_agent
                    ):
                        if look_ahead_index < check_horizon:
                            conflicts.add(
                                RobotRobotConflict(
                                    agent_id=agent_id,
                                    robot_pose=robot_pose_prior_to_action,
                                    other_agent_id=(
                                        uid
                                        if isinstance(
                                            world.dynamic_entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entity_to_agent[uid]
                                    ),
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
                                    colliding_uids=(agent_id, uid),
                                    robot_transfered_obstacle_uid=self.obstacle_uid,
                                    robot_transfered_obstacle_pose=self.obstacle_path.poses[
                                        self.action_index + look_ahead_index
                                    ],
                                    other_robot_transfered_obstacle_uid=(
                                        uid if uid in world.entity_to_agent else None
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        world.dynamic_entities[uid].pose
                                        if uid in world.entity_to_agent
                                        else None
                                    ),
                                )
                            )
                            if exit_early_for_any_conflict:
                                return conflicts
                    else:
                        conflicts.add(RobotObstacleConflict(uid))
                        if (
                            exit_early_for_any_conflict
                            or exit_early_only_for_long_term_conflicts
                        ):
                            return conflicts

                (
                    collides_with,
                    _,
                ) = collision.get_csv_collisions(
                    agent_id=self.obstacle_uid,
                    robot_action=action,
                    robot_pose=self.robot_path.poses[
                        self.action_index + look_ahead_index
                    ],
                    other_polygons=collision_polygons,
                    polygon=self.obstacle_path.polygons[
                        self.action_index + look_ahead_index
                    ],
                    others_aabb_tree=collision_aabb_tree,
                    ignored_entities=previously_moved_entities_uids.union(
                        {self.obstacle_uid}
                    ),
                )

                for uid in collides_with:
                    if uid in encompassing_circle_uid_to_agent_id:
                        if look_ahead_index < check_horizon and has_first_action:
                            other_agent_id = encompassing_circle_uid_to_agent_id[uid]
                            other_robot_obs = world.get_agent_held_obstacle(
                                other_agent_id
                            )
                            conflicts.add(
                                SimultaneousSpaceAccess(
                                    agent_id=agent_id,
                                    robot_pose=self.robot_path.poses[
                                        self.action_index + look_ahead_index
                                    ],
                                    other_agent_id=other_agent_id,
                                    other_robot_pose=world.dynamic_entities[
                                        other_agent_id
                                    ].pose,
                                    colliding_uids=(
                                        self.obstacle_uid,
                                        other_agent_id,
                                    ),
                                    robot_transfered_obstacle_uid=self.obstacle_uid,
                                    robot_transfered_obstacle_pose=self.obstacle_path.poses[
                                        self.action_index + look_ahead_index
                                    ],
                                    other_robot_transfered_obstacle_uid=(
                                        None
                                        if other_robot_obs is None
                                        else other_robot_obs.uid
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        None
                                        if other_robot_obs is None
                                        else other_robot_obs.pose
                                    ),
                                )
                            )
                            if exit_early_for_any_conflict:
                                return conflicts
                    elif (
                        isinstance(world.dynamic_entities[uid], agent.Agent)
                        or uid in world.entity_to_agent
                    ):
                        if look_ahead_index < check_horizon:
                            conflicts.add(
                                RobotRobotConflict(
                                    agent_id=agent_id,
                                    robot_pose=self.robot_path.poses[
                                        self.action_index + look_ahead_index
                                    ],
                                    other_agent_id=(
                                        uid
                                        if isinstance(
                                            world.dynamic_entities[uid],
                                            agent.Agent,
                                        )
                                        else world.entity_to_agent[uid]
                                    ),
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
                                    colliding_uids=(self.obstacle_uid, uid),
                                    robot_transfered_obstacle_uid=self.obstacle_uid,
                                    robot_transfered_obstacle_pose=self.obstacle_path.poses[
                                        self.action_index + look_ahead_index
                                    ],
                                    other_robot_transfered_obstacle_uid=(
                                        uid if uid in world.entity_to_agent else None
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        world.dynamic_entities[uid].pose
                                        if uid in world.entity_to_agent
                                        else None
                                    ),
                                )
                            )
                            if exit_early_for_any_conflict:
                                return conflicts
                    else:
                        conflicts.add(RobotObstacleConflict(uid))
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

    def reset(self):
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
                    updated_poses.append((pose[0], pose[1], current_angle))
                    dist = -dist
                elif np.abs(turn_towards_angle) > 1e-6:
                    current_angle = utils.add_angles(current_angle, turn_towards_angle)
                    actions.append(ba.Rotation(angle=turn_towards_angle))
                    updated_poses.append((pose[0], pose[1], current_angle))

                actions.append(ba.Advance(dist))
                updated_poses.append((next_pose[0], next_pose[1], current_angle))

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
        encompassing_circle_uid_to_agent_id: t.Dict[str, str],
        check_horizon: int,
        has_first_action: bool,
        apply_strict_horizon: bool = False,
        exit_early_for_any_conflict: bool = False,
        exit_early_only_for_long_term_conflicts: bool = True,
        rp: t.Optional["ros2.RosPublisher"] = None,
    ) -> t.Set[Conflict]:
        conflicts: t.Set[Conflict] = set()

        assert agent_id not in robot_inflated_grid.cell_sets
        if not self.actions:
            return conflicts

        if check_horizon <= 0 and apply_strict_horizon:
            return conflicts

        conflicts = conflicts

        encompassing_circles_uids = set(encompassing_circle_uid_to_agent_id.keys())

        # Compute and display horizon cells
        if rp:
            rp.publish_transit_horizon_cells(
                poses=self.robot_path.poses,
                start_index=self.action_index,
                check_horizon=check_horizon,
                robot_inflated_grid=robot_inflated_grid,
                agent_id=agent_id,
            )

        # Check for RobotRobot conflicts within horizon, and RobotObstacle conflicts even beyond
        conflicting_cells: t.Set[GridCellModel] = set()
        conflicting_entities_cells: t.Set[GridCellModel] = set()
        for look_ahead_index, action in enumerate(self.actions[self.action_index :]):
            if isinstance(action, ba.Wait):
                continue

            if apply_strict_horizon and look_ahead_index >= check_horizon:
                break

            if look_ahead_index < check_horizon and has_first_action:
                # If the first action in the path is the first action in the check horizon,
                # we also check for simultaneous conflilcts types at t+1
                robot_inflated_grid.activate_entities(encompassing_circles_uids)
            else:
                robot_inflated_grid.deactivate_entities(encompassing_circles_uids)

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
                    if uid in encompassing_circles_uids:
                        if look_ahead_index < check_horizon and has_first_action:
                            other_agent_id = encompassing_circle_uid_to_agent_id[uid]
                            other_robot_obs = world.get_agent_held_obstacle(
                                other_agent_id
                            )
                            conflicts.add(
                                SimultaneousSpaceAccess(
                                    agent_id=agent_id,
                                    robot_pose=pose,
                                    other_agent_id=other_agent_id,
                                    other_robot_pose=world.dynamic_entities[
                                        other_agent_id
                                    ].pose,
                                    colliding_uids=(agent_id, other_agent_id),
                                    robot_transfered_obstacle_uid=None,
                                    robot_transfered_obstacle_pose=None,
                                    other_robot_transfered_obstacle_uid=(
                                        other_robot_obs.uid if other_robot_obs else None
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        other_robot_obs.pose
                                        if other_robot_obs
                                        else None
                                    ),
                                )
                            )
                            conflicting_cells.add(cell)
                            conflicting_entities_cells.update(
                                robot_inflated_grid.cell_sets[uid]
                            )
                            if exit_early_for_any_conflict:
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
                    elif isinstance(world.dynamic_entities[uid], agent.Agent) or (
                        uid in world.entity_to_agent
                        # ignore collisions with the obstacle the robot is currently holding
                        and world.entity_to_agent.get(uid) != agent_id
                    ):
                        other_agent_id = uid
                        if uid in world.entity_to_agent:
                            other_agent_id = world.entity_to_agent[uid]

                        if look_ahead_index < check_horizon:
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
                                    colliding_uids=(agent_id, uid),
                                    robot_transfered_obstacle_uid=None,
                                    robot_transfered_obstacle_pose=None,
                                    other_robot_transfered_obstacle_uid=(
                                        uid if uid in world.entity_to_agent else None
                                    ),
                                    other_robot_transfered_obstacle_pose=(
                                        world.dynamic_entities[uid].pose
                                        if uid in world.entity_to_agent
                                        else None
                                    ),
                                )
                            )
                            conflicting_cells.add(cell)
                            conflicting_entities_cells.update(
                                robot_inflated_grid.cell_sets[uid]
                            )
                            if exit_early_for_any_conflict:
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
                        # check for polygon-level collisions
                        # collisions = world.get_polygon_collisions(agent_id, {uid})
                        # if len(collisions) == 0:
                        #     continue

                        conflicts.add(RobotObstacleConflict(uid))
                        conflicting_cells.add(cell)
                        conflicting_entities_cells.update(
                            robot_inflated_grid.cell_sets[uid]
                        )
                        if (
                            exit_early_for_any_conflict
                            or exit_early_only_for_long_term_conflicts
                        ):
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
        poses: t.List[PoseModel],
        robot_polygon: Polygon,
        robot_pose: PoseModel,
        conflicts: t.Set[Conflict],
    ):
        path = TransitPath.from_poses(
            poses=poses, robot_polygon=robot_polygon, robot_pose=robot_pose
        )
        return EvasionTransitPath(
            robot_path=path.robot_path, actions=path.actions, conflicts=conflicts
        )
