import typing as t

from shapely.geometry import MultiPolygon, Point, Polygon

import namosim.display.ros2_publisher as rp
from namosim.data_models import Pose2D
from shapely.geometry import JOIN_STYLE


def check_new_local_opening(
    old_osbtacle_polygon: Polygon,
    new_obstacle_polygon: Polygon,
    other_entities_polygons: t.Dict[str, Polygon],
    robot_radius: float,
    goal_pose: Pose2D,
    agent_id: str,
    ros_publisher: t.Optional["rp.RosPublisher"] = None,
) -> bool:
    """Checks if a new local opening exists

    TODO: Add more documentation for this complicated function.
    """
    # Build inflated polygons
    old_obstacle_inflated_polygon = t.cast(
        Polygon,
        old_osbtacle_polygon.buffer(2.0 * robot_radius, join_style=JOIN_STYLE.mitre),
    )
    if old_obstacle_inflated_polygon.intersects(Point(goal_pose[0], goal_pose[1])):
        # Exit early if goal in old_obstacle_inflated_polygon
        return True

    new_obtacle_inflated_by_robot_diameter = new_obstacle_polygon.buffer(
        2.0 * robot_radius, join_style=JOIN_STYLE.mitre
    )
    new_obstacle_inflated_by_robot_radius = t.cast(
        Polygon, new_obstacle_polygon.buffer(robot_radius, join_style=JOIN_STYLE.mitre)
    )
    if new_obstacle_inflated_by_robot_radius.intersects(
        Point(goal_pose[0], goal_pose[1])
    ):
        return False

    if ros_publisher:
        ros_publisher.publish_diameter_inflated_polygons(
            old_obstacle_inflated_polygon,
            new_obtacle_inflated_by_robot_diameter,
            line_width=robot_radius / 10,
            agent_id=agent_id,
        )

    # Build blocking areas
    # Note: Intersection geometry can be either Point, LineString or Polygon
    init_blocking_areas = []

    for polygon in other_entities_polygons.values():
        intersection_geometry = old_obstacle_inflated_polygon.intersection(polygon)
        if not intersection_geometry.is_empty:
            if isinstance(intersection_geometry, Polygon):
                init_blocking_areas.append(intersection_geometry)
            elif isinstance(intersection_geometry, MultiPolygon):
                for sub_intersection_geometry in intersection_geometry.geoms:
                    init_blocking_areas.append(sub_intersection_geometry)

    # If there are no blocking areas to begin with, return True
    if not init_blocking_areas:
        return True

    new_blocking_areas = []

    for polygon in other_entities_polygons.values():
        intersection_geometry = new_obtacle_inflated_by_robot_diameter.intersection(
            polygon
        )
        if not intersection_geometry.is_empty:
            if isinstance(intersection_geometry, Polygon):
                new_blocking_areas.append(intersection_geometry)
            elif isinstance(intersection_geometry, MultiPolygon):
                for sub_intersection_geometry in intersection_geometry.geoms:
                    new_blocking_areas.append(sub_intersection_geometry)

    if ros_publisher:
        ros_publisher.publish_blocking_areas(
            init_blocking_areas, new_blocking_areas, agent_id=agent_id
        )

    # Check if any blocking area has been freed thus a local opening has been created
    for init_blocking_area in init_blocking_areas:
        if not check_still_blocked(init_blocking_area, new_blocking_areas):
            return True
    return False


def check_still_blocked(
    init_blocking_area: Polygon, new_blocking_areas: t.List[Polygon]
):
    try:
        for new_blocking_area in new_blocking_areas:
            if init_blocking_area.intersects(new_blocking_area):
                return True  # If area is still blocked, there is no local opening here
    except Exception:
        print(
            "There was an exception in check_still_blocked function, this is not normal."
        )
    # If initial blocking area does not intersect with any of the target ones, then it is no longer blocked
    return False
