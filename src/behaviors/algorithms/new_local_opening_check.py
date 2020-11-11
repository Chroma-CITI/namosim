from shapely.geometry import Polygon, Point, MultiPolygon
from src.display.ros_publisher import RosPublisher
import src.utils.collision as collision


def check_new_local_opening(init_entity_polygon, target_entity_polygon,
                            other_entities_polygons, other_entities_aabb_tree,
                            inflation_radius, goal_pose,
                            init_blocking_areas=None, init_entity_inflated_polygon=None, ns=''):
    # Build inflated polygons
    if not init_entity_inflated_polygon:
        init_entity_inflated_polygon = init_entity_polygon.buffer(2. * inflation_radius)
        if init_entity_inflated_polygon.intersects(Point(goal_pose[0], goal_pose[1])):
            # Exit early if goal in init_entity_inflated_polygon
            return True, init_blocking_areas, init_entity_inflated_polygon
    target_entity_inflated_polygon = target_entity_polygon.buffer(2. * inflation_radius)
    target_entity_radius_inflated_polygon = target_entity_polygon.buffer(inflation_radius)
    if target_entity_radius_inflated_polygon.intersects(Point(goal_pose[0], goal_pose[1])):
        return False, init_blocking_areas, init_entity_inflated_polygon

    RosPublisher().publish_diameter_inflated_polygons(init_entity_inflated_polygon, target_entity_inflated_polygon, ns=ns)

    # Build blocking areas
    # Note: Intersection geometry can be either Point, LineString or Polygon
    if not init_blocking_areas:
        init_blocking_areas = []

        init_entity_inflated_polygon_aabb = collision.polygon_to_aabb(init_entity_inflated_polygon)
        potential_collision_polygons_uids = other_entities_aabb_tree.overlap_values(init_entity_inflated_polygon_aabb)

        for uid in potential_collision_polygons_uids :
            intersection_geometry = init_entity_inflated_polygon.intersection(other_entities_polygons[uid])
            if not intersection_geometry.is_empty:
                if isinstance(intersection_geometry, Polygon):
                    init_blocking_areas.append(intersection_geometry)
                elif isinstance(intersection_geometry, MultiPolygon):
                    for sub_intersection_geometry in intersection_geometry:
                        init_blocking_areas.append(sub_intersection_geometry)

    # If there are no blocking areas to begin with, return True
    if not init_blocking_areas:
        return True, init_blocking_areas, init_entity_inflated_polygon

    target_blocking_areas = []

    target_entity_inflated_polygon_aabb = collision.polygon_to_aabb(target_entity_inflated_polygon)
    potential_collision_polygons_uids = other_entities_aabb_tree.overlap_values(target_entity_inflated_polygon_aabb)

    for uid in potential_collision_polygons_uids:
        intersection_geometry = target_entity_inflated_polygon.intersection(other_entities_polygons[uid])
        if not intersection_geometry.is_empty:
            if isinstance(intersection_geometry, Polygon):
                target_blocking_areas.append(intersection_geometry)
            elif isinstance(intersection_geometry, MultiPolygon):
                for sub_intersection_geometry in intersection_geometry:
                    target_blocking_areas.append(sub_intersection_geometry)

    RosPublisher().publish_blocking_areas(init_blocking_areas, target_blocking_areas, ns=ns)

    # Check if any blocking area has been freed thus a local opening has been created
    for init_blocking_area in init_blocking_areas:
        if not check_still_blocked(init_blocking_area, target_blocking_areas):
            return True, init_blocking_areas, init_entity_inflated_polygon
    return False, init_blocking_areas, init_entity_inflated_polygon


def check_still_blocked(init_blocking_area, target_blocking_areas):
    try:
        for target_blocking_area in target_blocking_areas:
            if init_blocking_area.intersects(target_blocking_area):
                return True  # If area is still blocked, there is no local opening here
    except Exception as e:
        print('There was an exception in check_still_blocked function, this is not normal.')
    # If initial blocking area does not intersect with any of the target ones, then it is no longer blocked
    return False
