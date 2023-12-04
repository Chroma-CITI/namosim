import typing as t

from aabbtree import AABBTree
from shapely.geometry import MultiPolygon, Point, Polygon

import namosim.utils.collision as collision
from namosim.data_models import PoseModel
from namosim.display.ros2_publisher import RosPublisher


def check_new_local_opening(
    init_entity_polygon: Polygon,
    target_entity_polygon: Polygon,
    other_entities_polygons: t.Dict[int, Polygon],
    other_entities_aabb_tree: AABBTree,
    inflation_radius: float,
    goal_pose: PoseModel,
    ros_publisher: RosPublisher,
    init_blocking_areas: t.List[Polygon] | None = None,
    init_entity_inflated_polygon: t.Optional[Polygon] = None,
    ns: str = "",
):
    """Checks is a new local opening exists

    TODO: Add more documentation for this complicated function.
    """
    # Build inflated polygons
    if not init_entity_inflated_polygon:
        init_entity_inflated_polygon = t.cast(
            Polygon,
            init_entity_polygon.buffer(2.0 * inflation_radius, join_style="mitre"),
        )
        if init_entity_inflated_polygon.intersects(Point(goal_pose[0], goal_pose[1])):
            # Exit early if goal in init_entity_inflated_polygon
            return True, init_blocking_areas, init_entity_inflated_polygon
    target_entity_inflated_polygon = target_entity_polygon.buffer(
        2.0 * inflation_radius, join_style="mitre"
    )
    target_entity_radius_inflated_polygon = target_entity_polygon.buffer(
        inflation_radius, join_style="mitre"
    )
    if target_entity_radius_inflated_polygon.intersects(
        Point(goal_pose[0], goal_pose[1])
    ):
        return False, init_blocking_areas, init_entity_inflated_polygon

    ros_publisher.publish_diameter_inflated_polygons(
        init_entity_inflated_polygon, target_entity_inflated_polygon, ns=ns
    )

    # Build blocking areas
    # Note: Intersection geometry can be either Point, LineString or Polygon
    if not init_blocking_areas:
        init_blocking_areas = []

        init_entity_inflated_polygon_aabb = collision.polygon_to_aabb(
            init_entity_inflated_polygon
        )
        potential_collision_polygons_uids = other_entities_aabb_tree.overlap_values(
            init_entity_inflated_polygon_aabb
        )

        for uid in potential_collision_polygons_uids:
            intersection_geometry = init_entity_inflated_polygon.intersection(
                other_entities_polygons[uid]
            )
            if not intersection_geometry.is_empty:
                if isinstance(intersection_geometry, Polygon):
                    init_blocking_areas.append(intersection_geometry)
                elif isinstance(intersection_geometry, MultiPolygon):
                    for sub_intersection_geometry in intersection_geometry.geoms:
                        init_blocking_areas.append(sub_intersection_geometry)

    # If there are no blocking areas to begin with, return True
    if not init_blocking_areas:
        return True, init_blocking_areas, init_entity_inflated_polygon

    target_blocking_areas = []

    target_entity_inflated_polygon_aabb = collision.polygon_to_aabb(
        target_entity_inflated_polygon
    )
    potential_collision_polygons_uids = other_entities_aabb_tree.overlap_values(
        target_entity_inflated_polygon_aabb
    )

    for uid in potential_collision_polygons_uids:
        intersection_geometry = target_entity_inflated_polygon.intersection(
            other_entities_polygons[uid]
        )
        if not intersection_geometry.is_empty:
            if isinstance(intersection_geometry, Polygon):
                target_blocking_areas.append(intersection_geometry)
            elif isinstance(intersection_geometry, MultiPolygon):
                for sub_intersection_geometry in intersection_geometry.geoms:
                    target_blocking_areas.append(sub_intersection_geometry)

    ros_publisher.publish_blocking_areas(
        init_blocking_areas, target_blocking_areas, ns=ns
    )

    # Check if any blocking area has been freed thus a local opening has been created
    for init_blocking_area in init_blocking_areas:
        if not check_still_blocked(init_blocking_area, target_blocking_areas):
            return True, init_blocking_areas, init_entity_inflated_polygon
    return False, init_blocking_areas, init_entity_inflated_polygon


def check_still_blocked(
    init_blocking_area: Polygon, target_blocking_areas: t.List[Polygon]
):
    try:
        for target_blocking_area in target_blocking_areas:
            if init_blocking_area.intersects(target_blocking_area):
                return True  # If area is still blocked, there is no local opening here
    except Exception:
        print(
            "There was an exception in check_still_blocked function, this is not normal."
        )
    # If initial blocking area does not intersect with any of the target ones, then it is no longer blocked
    return False
