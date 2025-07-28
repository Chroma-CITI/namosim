import math
import typing as t

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint

import namosim.navigation.basic_actions as ba
import namosim.world.world as w
from namosim.log import logger
from namosim.utils import utils
from namosim.data_models import Pose2D


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
    robot_pose: Pose2D, robot_action: ba.BaseAction, polygon: Polygon
) -> t.List[t.List[t.Tuple[float, float]]]:
    """
    Returns for each action the pointclouds of the bounding boxes that cover each polygon's point trajectory
    during the action.
    :param action_sequence:
    :type action_sequence:
    :param polygon_sequence:
    :type polygon_sequence:
    :return:
    :rtype:
    """

    _, next_polygon = robot_action.apply(robot_pose, polygon=polygon)
    bb_vertices = []
    init_poly_coords = list(polygon.exterior.coords)
    end_poly_coords = list(next_polygon.exterior.coords)
    action_bb_vertices = []
    if isinstance(robot_action, ba.Rotation):
        for point_a, point_b in zip(init_poly_coords, end_poly_coords):
            expected_b = robot_action.apply_to_point(
                center=(robot_pose[0], robot_pose[1]), point=point_a  # type: ignore
            )
            if not np.allclose(point_b, expected_b):
                raise Exception()
            bb = arc_bounding_box(
                point=point_a,  # type: ignore
                degrees=robot_action.angle,
                center=(robot_pose[0], robot_pose[1]),
            )
            for coord in bb:
                action_bb_vertices.append(coord)
    else:
        for coord in init_poly_coords:
            action_bb_vertices.append(coord)
        for coord in end_poly_coords:
            action_bb_vertices.append(coord)
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
    return MultiPoint(all_vertices).convex_hull  # type: ignore


def get_collisions_for_entity(
    entity_polygon: Polygon,
    other_entity_polygons: t.Dict[str, Polygon],
    ignored_entities: t.Set[str] | None = None,
    break_at_first: bool = True,
) -> t.Set[str]:
    """Returns the set of entity ids that collide with the given polygon."""
    collides_with: t.Set[str] = set()
    for uid, polygon in other_entity_polygons.items():
        if ignored_entities and uid in ignored_entities:
            continue
        if entity_polygon.intersects(polygon):
            collides_with.add(uid)
            if break_at_first:
                break
    return collides_with


def merge_collides_with(
    source: t.Dict[str, t.Set[str]], other: t.Dict[str, t.Set[str]]
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


def get_csv_collisions(
    *,
    agent_id: str,
    robot_pose: Pose2D,
    robot_action: ba.Action,
    polygon: Polygon,
    other_polygons: t.Dict[str, Polygon],
    ignored_entities: t.Set[str] | None = None,
):
    bb_vertices = bounding_boxes_vertices(robot_pose, robot_action, polygon)
    csv_polygon = csv_from_bb_vertices(bb_vertices)
    collisions = get_collisions_for_entity(
        csv_polygon,
        other_polygons,
        ignored_entities=ignored_entities,
    )
    assert agent_id not in collisions
    return collisions, csv_polygon


def csv_simulate_simple_kinematics(
    world: "w.World",
    agent_actions: t.Dict[str, ba.Action],
    apply: bool = False,
    ignore_collisions: bool = False,
) -> t.Dict[str, t.Set[str]]:
    # gather all convex swept volumes of all entities
    entity_csv_polygons: t.Dict[str, Polygon] = {}

    # first add all entities static polygons
    for entity in world.dynamic_entities.values():
        entity_csv_polygons[entity.uid] = entity.polygon

    # now handle agents and actions
    for agent in world.agents.values():
        if agent.uid not in agent_actions:
            continue
        action = agent_actions[agent.uid]

        # we assume agent is a circle so rotation does not change the csv
        if not isinstance(action, ba.Rotation):
            agent_csv = csv_from_bb_vertices(
                bounding_boxes_vertices(
                    robot_pose=agent.pose,
                    robot_action=action,
                    polygon=agent.polygon,
                )
            )
            entity_csv_polygons[agent.uid] = agent_csv

        # update csv polygon for any obstacle the agent is holding, if the current action is not a release.
        obs = (
            world.get_agent_held_obstacle(agent.uid)
            if not isinstance(action, ba.Release)
            else None
        )

        if obs:
            obs_csv = csv_from_bb_vertices(
                bounding_boxes_vertices(
                    robot_pose=agent.pose, robot_action=action, polygon=obs.polygon
                )
            )
            entity_csv_polygons[obs.uid] = obs_csv

    collisions: t.Dict[str, t.Set[str]] = {}
    for uid, polygon in entity_csv_polygons.items():
        other_entity_polygons = {
            k: v for k, v in entity_csv_polygons.items() if k != uid
        }
        entity_collisions = get_collisions_for_entity(polygon, other_entity_polygons)
        collisions[uid] = entity_collisions

    for agent_uid, action in agent_actions.items():
        agent = world.agents[agent_uid]
        new_agent_pose, new_agent_polygon = action.apply(agent.pose, agent.polygon)

        if apply:
            obs_id = None
            if isinstance(action, (ba.Grab, ba.Release)):
                obs_id = action.entity_uid

            obs = (
                world.get_agent_held_obstacle(agent_uid)
                if not isinstance(action, ba.Release)
                else None
            )

            if obs:
                obs_id = obs.uid

                new_obs_pose = action.predict_pose(agent.pose, obs.pose)
                new_obs_polygon = action.predict_polygon(agent.pose, obs.polygon)

            def update_poses():
                agent.pose = new_agent_pose
                agent.polygon = new_agent_polygon
                if obs:
                    obs.pose = new_obs_pose
                    obs.polygon = new_obs_polygon

            if obs_id in collisions[agent_uid]:
                collisions[agent_uid].remove(obs_id)
            if obs_id and agent_uid in collisions[obs_id]:
                collisions[obs_id].remove(agent_uid)

            if ignore_collisions:
                update_poses()
            elif collisions[agent_uid] or (obs and collisions[obs.uid]):
                logger.warn(
                    f"Ignored collision between {agent_uid} and {collisions[agent_uid]}"
                )
            else:
                update_poses()

    return collisions
