import math
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from aabbtree import AABB, AABBTree
from shapely import Polygon
from shapely.geometry import MultiPoint, Point

import namosim.navigation.basic_actions as ba
import namosim.world.world as w
from namosim.data_models import UID, PoseModel
from namosim.utils import utils


def bounds(points: t.Iterable[t.Tuple[float, float]]):
    minx, miny, maxx, maxy = float("inf"), float("inf"), -float("inf"), -float("inf")
    for point in points:
        minx, miny, maxx, maxy = (
            min(minx, point[0]),
            min(miny, point[1]),
            max(maxx, point[0]),
            max(maxy, point[1]),
        )
    return minx, miny, maxx, maxy


def rotate_point(
    *, point: t.Tuple[float, float], center: t.Tuple[float, float], angle_degrees: float
):
    """
    Rotate a 2D point around a center point by a specified angle in degrees.

    Parameters:
        point (tuple): (x, y) coordinates of the point to be rotated.
        center (tuple): (x, y) coordinates of the center point.
        angle_degrees (float): Rotation angle in degrees.

    Returns:
        tuple: (x, y) coordinates of the rotated point.
    """
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)

    # Translate the point to the origin
    translated_point = (point[0] - center[0], point[1] - center[1])

    # Rotate the translated point using the rotation matrix
    rotated_x = translated_point[0] * math.cos(angle_radians) - translated_point[
        1
    ] * math.sin(angle_radians)
    rotated_y = translated_point[0] * math.sin(angle_radians) + translated_point[
        1
    ] * math.cos(angle_radians)

    # Translate the rotated point back to the original position
    rotated_point = (rotated_x + center[0], rotated_y + center[1])

    return rotated_point


def arc_bounding_box(
    degrees: float,
    point: t.Tuple[float, float],
    center: t.Tuple[float, float],
) -> t.List[t.Tuple[float, float]]:
    """Computes the verticies a rectangular box bounding the circular arc from point a to point b."""

    # This function first computes the box assuming the arc is centered on the x-axis, then it rotates it to the arc's true position.

    a = point
    b = rotate_point(point=point, angle_degrees=degrees, center=center)
    midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    radius = utils.euclidean_distance(center, a)
    dist_center_to_mid = utils.euclidean_distance(center, midpoint)
    arc_center = rotate_point(point=a, angle_degrees=degrees / 2, center=center)
    dy = arc_center[1] - center[1]
    dx = arc_center[0] - center[0]
    arc_position = np.arctan2(dy, dx) * 180 / math.pi

    ## looking at bounding box from the center of the circle to center of the arc

    assert np.abs(degrees) <= 360

    if np.abs(degrees) < 180:
        box_height = utils.euclidean_distance(a, b)
        box_width = radius - dist_center_to_mid
        box_left = center[0] + dist_center_to_mid

    else:
        box_width = radius + dist_center_to_mid
        box_height = 2 * radius
        box_left = center[0] - dist_center_to_mid

    box_top = center[1] + box_height / 2
    points = [
        (box_left, box_top - box_height),
        (box_left, box_top),
        (box_left + box_width, box_top - box_height),
        (box_left + box_width, box_top),
    ]

    points = [
        rotate_point(point=p, angle_degrees=arc_position, center=center) for p in points
    ]

    return points


def bounding_boxes_vertices(
    action_sequence: t.List[ba.AbsoluteAction],
    polygon_sequence: t.List[Polygon],
    bb_type: t.Literal[
        "minimum_rotated_rectangle", "aabbox"
    ] = "minimum_rotated_rectangle",
) -> t.List[t.List[t.Tuple[float, float]]]:
    """
    Returns for each action the pointclouds of the bounding boxes that cover each polygon's point trajectory
    during the action.
    :param action_sequence:
    :type action_sequence:
    :param polygon_sequence:
    :type polygon_sequence:
    :param bb_type: Type of bounding box, either 'minimum_rotated_rectangle' or 'aabbox', first one is most accurate
    :type bb_type: str
    :return:
    :rtype:
    """
    bb_vertices = []
    for index, action in enumerate(action_sequence):
        init_poly_coords = list(polygon_sequence[index].exterior.coords)
        end_poly_coords = list(polygon_sequence[index + 1].exterior.coords)
        action_bb_vertices = []
        if isinstance(action, ba.AbsoluteTranslation):
            for coord in init_poly_coords:
                action_bb_vertices.append(coord)
            for coord in end_poly_coords:
                action_bb_vertices.append(coord)
        elif isinstance(action, ba.AbsoluteRotation):
            for point_a, _ in zip(init_poly_coords, end_poly_coords):
                bb = arc_bounding_box(
                    point=point_a,
                    degrees=action.angle,
                    center=action.center,
                )
                for coord in bb:
                    action_bb_vertices.append(coord)
        else:
            raise TypeError(
                "Actions must be pure AbsoluteRotation or AbsoluteTranslation."
            )
        bb_vertices.append(action_bb_vertices)
    return bb_vertices


def csv_from_bb_vertices(bb_vertices: t.List[t.List[t.Tuple[float, float]]]) -> Polygon:
    """
    Computes the CSV (Convex Swept Volume) approximation polygon of the provided bounding boxes vertices
    :param bb_vertices: List of Bounding boxes vertices for each action
    :type bb_vertices:
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.Polygon
    """
    all_vertices = [vertex for vertices in bb_vertices for vertex in vertices]
    return MultiPoint(all_vertices).convex_hull


def polygon_to_aabb(polygon: Polygon):
    xmin, ymin, xmax, ymax = polygon.bounds
    return AABB([(xmin, xmax), (ymin, ymax)])


def polygons_to_aabb_tree(polygons: t.Dict[UID, Polygon]):
    aabb_tree = AABBTree()
    for uid, polygon in polygons.items():
        aabb_tree.add(polygon_to_aabb(polygon), uid)
    return aabb_tree


def check_static_collision(
    main_uid: UID,
    polygon: Polygon,
    other_entities_polygons: t.Dict[UID, Polygon],
    aabb_tree: AABBTree,
    ignored_uids: t.Iterable[UID] | None = None,
    break_at_first: bool = True,
    save_intersections: bool = False,
) -> t.Tuple[t.Dict[UID, t.Set[UID]], t.Dict[t.Tuple[UID, UID], Polygon]]:
    aabb = polygon_to_aabb(polygon)
    potential_collision_uids = aabb_tree.overlap_values(aabb)
    if ignored_uids:
        potential_collision_uids = set(potential_collision_uids).difference(
            set(ignored_uids)
        )
    if break_at_first:
        for uid in potential_collision_uids:
            if polygon.intersects(other_entities_polygons[uid]):
                if save_intersections:
                    intersection = polygon.intersection(other_entities_polygons[uid])
                    return {main_uid: {uid}, uid: {main_uid}}, {
                        (main_uid, uid): intersection,
                        (uid, main_uid): intersection,
                    }

                return {main_uid: {uid}, uid: {main_uid}}, {}
        return {}, {}

    collides_with: t.Dict[UID, t.Set[UID]] = {}
    intersections: t.Dict[t.Tuple[UID, UID], Polygon] = {}

    for uid in potential_collision_uids:
        if polygon.intersects(other_entities_polygons[uid]):
            if save_intersections:
                intersection: Polygon = polygon.intersection(
                    other_entities_polygons[uid]
                )
                intersections[(main_uid, uid)] = intersection
                intersections[(uid, main_uid)] = intersection

            if main_uid in collides_with:
                collides_with[main_uid].add(uid)
            else:
                collides_with[main_uid] = {uid}

            if uid in collides_with:
                collides_with[uid].add(main_uid)
            else:
                collides_with[uid] = {main_uid}

    return collides_with, intersections


def merge_collides_with(
    source: t.Dict[UID, t.Set[UID]], other: t.Dict[UID, t.Set[UID]]
):
    for uid, uids in other.items():
        if uid in source:
            source[uid].update(uids)
            for uid_2 in uids:
                if uid_2 in source:
                    source[uid_2].add(uid)
                else:
                    source[uid_2] = {uid}
        else:
            source[uid] = uids
            for uid_2 in uids:
                if uid_2 in source:
                    source[uid_2].add(uid)
                else:
                    source[uid_2] = {uid}
    return source


def csv_check_collisions(
    *,
    main_uid: UID,
    other_polygons: t.Dict[UID, Polygon],
    polygon_sequence: t.List[Polygon],
    action_sequence: t.List[ba.AbsoluteAction],
    id_sequence: t.List[int] | None = None,
    aabb_tree: AABBTree | None = None,
    bb_vertices: t.List[t.List[t.Tuple[float, float]]] | None = None,
    csv_polygons: t.Dict[t.Sequence[int], Polygon] | None = None,
    intersections: t.Dict[t.Tuple[int, int], Polygon] | None = None,
    ignored_entities: t.Set[UID] | None = None,
    display_debug: bool = False,
    break_at_first: bool = True,
    save_intersections: bool = False,
) -> t.Tuple[
    bool,
    t.Dict[UID, t.Set[UID]],
    AABBTree,
    t.Dict[t.Sequence[int], Polygon],
    t.Dict[t.Tuple[int, int], Polygon],
    t.List[t.List[t.Tuple[float, float]]],
]:
    # Initialize at first recursive iteration
    if not aabb_tree:
        aabb_tree = polygons_to_aabb_tree(other_polygons)
    if not bb_vertices:
        bb_vertices = bounding_boxes_vertices(action_sequence, polygon_sequence)
    if not csv_polygons:
        csv_polygons = {}
    if not intersections:
        intersections = {}
    if not id_sequence:
        id_sequence = list(range(len(action_sequence)))

    csv_polygon = csv_from_bb_vertices(bb_vertices)
    csv_polygons[tuple(id_sequence)] = csv_polygon

    # Dichotomy-check for collision between polygon and CSV as long as:
    # - there is no collision
    # - AND the CSV envelops more than one action (two consecutive polygons)
    if save_intersections:
        collides_with, local_intersections = check_static_collision(
            main_uid,
            csv_polygon,
            other_polygons,
            aabb_tree,
            ignored_entities,
            break_at_first,
            save_intersections,
        )
        intersections[tuple(id_sequence)] = local_intersections
    else:
        collides_with, _ = check_static_collision(
            main_uid,
            csv_polygon,
            other_polygons,
            aabb_tree,
            ignored_entities,
            break_at_first,
            save_intersections,
        )

    if collides_with:
        if display_debug:
            fig, ax = plt.subplots()
            for p in polygon_sequence:
                ax.plot(*p.exterior.xy, color="grey")
            # for i in indexes:
            #     ax.plot(*polygon_sequence[i].exterior.xy, color='blue')
            for p in other_polygons.values():
                ax.plot(*p.exterior.xy, color="black")
            x, y = zip(*[[vertex.x, vertex.y] for vertex in bb_vertices])
            ax.scatter(x, y, marker="x")
            ax.plot(*csv_polygon.exterior.xy, color="green")
            intersection = csv_polygon.intersection(
                other_polygons[collides_with[main_uid][0]]
            )
            ax.plot(*intersection.exterior.xy, color="red")
            ax.axis("equal")
            fig.show()
            print("")

        if len(bb_vertices) >= 2:
            first_half_bb_vertices = bb_vertices[: len(bb_vertices) // 2]
            second_half_bb_vertices = bb_vertices[len(bb_vertices) // 2 :]
            first_half_ids = id_sequence[: len(id_sequence) // 2]
            second_half_ids = id_sequence[len(id_sequence) // 2 :]
            (
                first_half_collides,
                first_half_collides_with,
                _,
                _,
                _,
                _,
            ) = csv_check_collisions(
                main_uid,
                other_polygons,
                polygon_sequence,
                action_sequence,
                first_half_ids,
                aabb_tree=aabb_tree,
                bb_vertices=first_half_bb_vertices,
                ignored_entities=ignored_entities,
                display_debug=display_debug,
                break_at_first=break_at_first,
                bb_type=bb_type,
                csv_polygons=csv_polygons,
                intersections=intersections,
            )
            (
                second_half_collides,
                second_half_collides_with,
                _,
                _,
                _,
                _,
            ) = csv_check_collisions(
                main_uid,
                other_polygons,
                polygon_sequence,
                action_sequence,
                second_half_ids,
                aabb_tree=aabb_tree,
                bb_vertices=second_half_bb_vertices,
                ignored_entities=ignored_entities,
                display_debug=display_debug,
                break_at_first=break_at_first,
                bb_type=bb_type,
                csv_polygons=csv_polygons,
                intersections=intersections,
            )
            collides_with = merge_collides_with(
                first_half_collides_with, second_half_collides_with
            )
            collides = first_half_collides or second_half_collides
            return (
                collides,
                collides_with,
                aabb_tree,
                csv_polygons,
                intersections,
                bb_vertices,
            )
        return (
            True,
            collides_with,
            aabb_tree,
            csv_polygons,
            intersections,
            bb_vertices,
        )

    return False, collides_with, aabb_tree, csv_polygons, intersections, bb_vertices


def csv_simulate_simple_kinematics(
    world: "w.World",
    agent_actions: t.Dict[UID, ba.Action],
    apply: bool = False,
    bb_type: str = "minimum_rotated_rectangle",
    ignore_collisions: bool = False,
) -> t.Dict[UID, t.Set[UID]]:
    # Apply each action to get polygon after for robot and obstacle if relevant
    # and compute CSV for each
    # and check that no CSV intersects with other entities beyond those that are moving this round
    uid_to_csv_polygon = {}
    collides_with = {}
    moving_uids = set(agent_actions.keys()).union(
        {
            world.entity_to_agent.inverse[agent_uid]
            for agent_uid in agent_actions.keys()
            if agent_uid in world.entity_to_agent.inverse
        }
    )
    other_polygons = {
        uid: e.polygon for uid, e in world.entities.items() if uid not in moving_uids
    }
    aabb_tree = polygons_to_aabb_tree(other_polygons)
    if apply:
        new_polygons: t.Dict[UID, Polygon] = {}
        new_poses: t.Dict[UID, PoseModel] = {}

    for agent_uid, action in agent_actions.items():
        agent = world.entities[agent_uid]
        agent_action = action.to_absolute(agent.pose)
        agent_polygon_after = agent_action.apply(agent.polygon)
        agent_csv = csv_from_bb_vertices(
            bounding_boxes_vertices(
                [agent_action], [agent.polygon, agent_polygon_after]
            )
        )
        uid_to_csv_polygon[agent_uid] = agent_csv
        ignored_entities = (
            {action.entity_uid} if isinstance(action, (ba.Release, ba.Grab)) else set()
        )
        agent_collides_with, _ = check_static_collision(
            agent_uid, agent_csv, other_polygons, aabb_tree, ignored_entities
        )

        if agent_uid in agent_collides_with:
            if isinstance(action, ba.Rotation):
                # Extra check for rotation to avoid false positives during transit paths
                actual_colliding_entities = set()
                for other_uid in agent_collides_with[agent_uid]:
                    nearest_distance = Point((agent.pose[0], agent.pose[1])).distance(
                        world.entities[other_uid].polygon
                    )
                    if nearest_distance <= utils.get_circumscribed_radius(
                        agent.polygon
                    ):
                        actual_colliding_entities.add(other_uid)
                if actual_colliding_entities:
                    agent_collides_with = {
                        other_uid: {agent_uid}
                        for other_uid in actual_colliding_entities
                    }
                    agent_collides_with[agent_uid] = actual_colliding_entities
                else:
                    agent_collides_with = {}
            merge_collides_with(collides_with, agent_collides_with)

        if apply:
            if isinstance(action, ba.Advance):
                new_pose = action.predict_pose(agent.pose, agent.pose[2])
            elif isinstance(action, ba.Rotation):
                new_pose = action.predict_pose(
                    agent.pose, (agent.pose[0], agent.pose[1])
                )
            elif isinstance(action, (ba.AbsoluteAction)):
                new_pose = action.predict_pose(agent.pose)
            else:
                raise Exception("Unexpected action type")
            assert new_pose != agent.pose

            new_poses[agent_uid] = new_pose
            new_polygons[agent_uid] = agent_polygon_after

        if (
            not isinstance(action, (ba.Release, ba.Grab))
            and agent_uid in world.entity_to_agent.inverse
        ):
            obs_uid = world.entity_to_agent.inverse[agent_uid]
            obs = world.entities[obs_uid]
            obs_action: ba.AbsoluteAction = action.to_absolute(agent.pose)
            obs_polygon_after = obs_action.apply(obs.polygon)
            obs_csv = csv_from_bb_vertices(
                bounding_boxes_vertices(
                    [obs_action], [obs.polygon, obs_polygon_after], bb_type=bb_type
                )
            )
            uid_to_csv_polygon[obs_uid] = obs_csv
            obs_collides_with, _ = check_static_collision(
                obs_uid, obs_csv, other_polygons, aabb_tree
            )
            merge_collides_with(collides_with, obs_collides_with)

            if apply:
                if isinstance(action, ba.Advance):
                    new_pose = action.predict_pose(obs.pose, agent.pose[2])
                elif isinstance(action, ba.Rotation):
                    new_pose = action.predict_pose(
                        obs.pose, (agent.pose[0], agent.pose[1])
                    )
                elif isinstance(action, (ba.AbsoluteAction)):
                    new_pose = action.predict_pose(obs.pose)
                else:
                    raise Exception("Unexpected action type")

                new_poses[obs_uid] = new_pose
                new_polygons[obs_uid] = obs_polygon_after

    # Check that no CSV intersects with another CSV
    checked_uids = set()
    csv_aabb_tree = polygons_to_aabb_tree(uid_to_csv_polygon)
    for uid, csv_polygon in uid_to_csv_polygon.items():
        checked_uids.add(uid)
        if uid in world.entity_to_agent:
            associated_uid = {world.entity_to_agent[uid]}
        elif uid in world.entity_to_agent.inverse:
            associated_uid = {world.entity_to_agent.inverse[uid]}
        else:
            associated_uid = set()
        merge_collides_with(
            collides_with,
            check_static_collision(
                uid,
                csv_polygon,
                uid_to_csv_polygon,
                csv_aabb_tree,
                ignored_uids=checked_uids.union(associated_uid),
            )[0],
        )

    # If option activated, check that no new polygon's discretized cell is the center cell of another robot in transit
    # utils.accurate_rasterize_in_grid(
    #     new_polygon, self.res, self.grid_pose, self.d_width, self.d_height, fill=fill
    # )
    # discretized_polygons = {uid: utils. for uid, polygon in new_polygons.items()}

    if apply:
        for agent_uid, action in agent_actions.items():
            agent = world.entities[agent_uid]
            if ignore_collisions:
                if agent_uid in world.entity_to_agent.inverse and not isinstance(
                    action, (ba.Release, ba.Grab)
                ):
                    obs_uid = world.entity_to_agent.inverse[agent_uid]
                    obstacle = world.entities[obs_uid]
                    agent.pose, agent.polygon = (
                        new_poses[agent_uid],
                        new_polygons[agent_uid],
                    )
                    obstacle.pose, obstacle.polygon = (
                        new_poses[obs_uid],
                        new_polygons[obs_uid],
                    )
                else:
                    agent.pose, agent.polygon = (
                        new_poses[agent_uid],
                        new_polygons[agent_uid],
                    )
            else:
                if agent_uid not in collides_with:
                    if agent_uid in world.entity_to_agent.inverse and not isinstance(
                        action, (ba.Release, ba.Grab)
                    ):
                        obs_uid = world.entity_to_agent.inverse[agent_uid]
                        if obs_uid not in collides_with:
                            obstacle = world.entities[obs_uid]
                            agent.pose, agent.polygon = (
                                new_poses[agent_uid],
                                new_polygons[agent_uid],
                            )
                            obstacle.pose, obstacle.polygon = (
                                new_poses[obs_uid],
                                new_polygons[obs_uid],
                            )
                    else:
                        agent.pose, agent.polygon = (
                            new_poses[agent_uid],
                            new_polygons[agent_uid],
                        )

    return collides_with
