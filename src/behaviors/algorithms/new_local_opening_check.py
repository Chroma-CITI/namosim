from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import cascaded_union
from src.display.ros_publisher import RosPublisher
import src.utils.collision as collision


def check_new_local_opening(init_entity_polygon, target_entity_polygon, other_entities_polygons,
                            inflation_radius, init_blocking_areas=None, ns=''):
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

    RosPublisher().publish_diameter_inflated_polygons(init_entity_inflated_polygon, target_entity_inflated_polygon, ns=ns)

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

    # If there are no blocking areas to begin with, return True
    if not init_blocking_areas:
        return True, init_blocking_areas

    target_blocking_areas = []
    for other_entity_polygon in other_entities_polygons:
        intersection_geometry = target_entity_inflated_polygon.intersection(other_entity_polygon)
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
            return True, init_blocking_areas
    return False, init_blocking_areas



def new_check_new_local_opening(init_entity_polygon, target_entity_polygon,
                            other_entities_polygons, other_entities_aabb_tree,
                            inflation_radius, goal_position,
                            init_blocking_areas=None, init_entity_inflated_polygon=None, ns=''):
    # Build inflated polygons
    if init_entity_inflated_polygon is not None:
        init_entity_inflated_polygon = init_entity_polygon.buffer(2. * inflation_radius)
        if init_entity_inflated_polygon.intersects(Point(goal_position)):
            # Exit early if goal in init_entity_inflated_polygon
            return True, init_blocking_areas, init_entity_inflated_polygon
    target_entity_inflated_polygon = target_entity_polygon.buffer(2. * inflation_radius)
    target_entity_radius_inflated_polygon = target_entity_polygon.buffer(inflation_radius)
    if target_entity_radius_inflated_polygon.intersects(Point(goal_position)):
        return False, init_blocking_areas, init_entity_inflated_polygon

    RosPublisher().publish_diameter_inflated_polygons(init_entity_inflated_polygon, target_entity_inflated_polygon, ns=ns)

    # Build blocking areas
    # Note: Intersection geometry can be either Point, LineString or Polygon
    if init_blocking_areas is None:
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


def is_move_passing_over_pose(moved_polygons, pose):
    try:
        union = cascaded_union(moved_polygons)
        convex_hull = union.convex_hull
        value = Point((pose[0], pose[1])).intersects(convex_hull)
    except ValueError as e:
        return False
    return value
