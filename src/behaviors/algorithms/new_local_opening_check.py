from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import cascaded_union
from src.display.ros_publisher import RosPublisher


def check_new_local_opening(init_entity_polygon, target_entity_polygon, other_entities_polygons,
                            inflation_radius, init_blocking_areas=None):
    # Check that all polygonal parameters are actually what they are expected to be
    if not (isinstance(init_entity_polygon, Polygon) and isinstance(target_entity_polygon, Polygon)
            and all([isinstance(other_polygon, Polygon) for other_polygon in other_entities_polygons])
            and (init_blocking_areas is None
                 or (init_blocking_areas is not None
                     and all([isinstance(i_b_a, Polygon) for i_b_a in init_blocking_areas])))):
        raise TypeError("FATAL ERROR : check_new_local_opening method requires shapely Polygons ONLY.")

    # Build inflated polygons
    init_entity_inflated_polygon = init_entity_polygon.buffer(2. * inflation_radius)
    target_entity_inflated_polygon = target_entity_polygon.buffer(2. * inflation_radius)

    RosPublisher().publish_diameter_inflated_polygons(init_entity_inflated_polygon, target_entity_inflated_polygon)

    # Build blocking areas
    # Note: Intersection geometry can be either Point, LineString or Polygon
    if init_blocking_areas is None:
        init_blocking_areas = []
        for other_entity_polygon in other_entities_polygons:
            intersection_geometry = init_entity_inflated_polygon.intersection(other_entity_polygon)
            if not intersection_geometry.is_empty:
                if isinstance(intersection_geometry, Polygon):
                    init_blocking_areas.append(intersection_geometry)
                elif isinstance(intersection_geometry, MultiPolygon):
                    for sub_intersection_geometry in intersection_geometry:
                        init_blocking_areas.append(sub_intersection_geometry)
    target_blocking_areas = []
    for other_entity_polygon in other_entities_polygons:
        intersection_geometry = target_entity_inflated_polygon.intersection(other_entity_polygon)
        if not intersection_geometry.is_empty:
            if isinstance(intersection_geometry, Polygon):
                target_blocking_areas.append(intersection_geometry)
            elif isinstance(intersection_geometry, MultiPolygon):
                for sub_intersection_geometry in intersection_geometry:
                    target_blocking_areas.append(sub_intersection_geometry)

    RosPublisher().publish_blocking_areas(init_blocking_areas, target_blocking_areas)

    # Check if any blocking area has been freed thus a local opening has been created
    for init_blocking_area in init_blocking_areas:
        if not check_still_blocked(init_blocking_area, target_blocking_areas):
            return True, init_blocking_areas
    return False, init_blocking_areas


def check_still_blocked(init_blocking_area, target_blocking_areas):
    for target_blocking_area in target_blocking_areas:
        if init_blocking_area.intersects(target_blocking_area):
            return True  # If area is still blocked, there is no local opening here
    # If initial blocking area does not intersect with any of the target ones, then it is no longer blocked
    return False


def is_move_passing_over_pose(moved_polygons, pose):
    return Point((pose[0], pose[1])).intersects(cascaded_union(moved_polygons).convex_hull)
