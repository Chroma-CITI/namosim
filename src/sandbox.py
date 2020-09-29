import numpy as np
from shapely.geometry import Polygon, MultiPoint
import shapely.affinity as affinity
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def states_for_points(polygon_states):
    """
    Computes the list (ordered by states) of coordinates of each point of the polygon
    :param polygon_states: sequence of polygons representing the same shape at different
    :type polygon_states: list(shapely.geometry.Polygon)
    :return: List (ordered by states) of coordinates of each point of the polygon
    :rtype: list(list((float, float)))
    """
    return zip(*[list(polygon.exterior.coords) for polygon in polygon_states])


def bounding_boxes_vertices_from_states_for_points(s_for_points):
    """
    Computes the bounding boxes for each point coordinates list of the polygon
    :param s_for_points: List (ordered by states) of coordinates of each point of the polygon
    :type s_for_points: list(list((float, float)))
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.MultiPoint
    """
    bounding_boxes_points_coords = []
    for point_states in s_for_points:
        minx, miny, maxx, maxy = MultiPoint(point_states).bounds
        bounding_boxes_points_coords += [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    return MultiPoint(bounding_boxes_points_coords)


def bounding_boxes_vertices(polygon_states):
    """
    Computes the bounding boxes of corresponding points in a sequence of polygons states of the same shape
    :param polygon_states: sequence of polygons representing the same shape at different
    :type polygon_states: list(shapely.geometry.Polygon)
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.MultiPoint
    """
    s_for_points = states_for_points(polygon_states)
    return bounding_boxes_vertices_from_states_for_points(s_for_points)


def csv_from_polygon_states(polygon_states):
    """
    Computes the CSV (Convex Swept Volume) approximation polygon of the provided polygon states of the same shape
    :param polygon_states: sequence of polygons representing the same shape at different
    :type polygon_states: list(shapely.geometry.Polygon)
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.Polygon
    """
    return bounding_boxes_vertices(polygon_states).convex_hull


def csv_from_bb_vertices(bb_vertices):
    """
    Computes the CSV (Convex Swept Volume) approximation polygon of the provided bounding boxes vertices
    :param bb_vertices: Bounding boxes vertices
    :type bb_vertices: shapely.geometry.MultiPoint
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.Polygon
    """
    return bb_vertices.convex_hull


def csv_check_collisions(polygon, polygon_sequence, polygon_indexes=None, collision_data=None, display_debug=False):
    # TODO : Consider adding verification at first step whether there is a collision or not between polygon and all
    #    polygons in polygon_sequence, it may improve performance ? Or not.

    # Initialize at first recursive iteration
    if not collision_data:
        collision_data = {}
    if not polygon_indexes:
        polygon_indexes = [i for i in range(len(polygon_sequence))]

    # Allow reuse of pre-computed CSV polygons to save perfomance
    if tuple(polygon_indexes) in collision_data:
        bb_vertices = collision_data["bb_vertices"]
        csv_polygon = collision_data["csv_polygon"]
    else:
        bb_vertices = bounding_boxes_vertices(polygon_sequence)
        csv_polygon = csv_from_bb_vertices(bb_vertices)
        collision_data[tuple(polygon_indexes)] = {
            "bb_vertices": bb_vertices,
            "csv_polygon": csv_polygon,
        }

    # Dichotomy-check for collision between polygon and CSV as long as:
    # - there is a collision
    # - AND the CSV envelops more than two consecutive polygons
    if csv_polygon.intersects(polygon):
        collision_data[tuple(polygon_indexes)]["intersection_polygon"] = csv_polygon.intersection(polygon)

        if display_debug:
            fig = plt.figure()
            for p in polygon_sequence:
                plt.plot(*p.exterior.xy, color='blue')
            x, y = zip(*[[vertex.x, vertex.y] for vertex in bb_vertices])
            plt.plot(*polygon.exterior.xy, color='black')
            plt.scatter(x, y)
            plt.plot(*csv_polygon.exterior.xy, color='green')
            plt.plot(*(csv_polygon.intersection(polygon).exterior.xy), color='red')
            plt.axis('equal')
            plt.show()
            print("")

        if len(polygon_sequence) > 2:
            first_half_polygons = polygon_sequence[:len(polygon_sequence)//2 + 1]
            second_half_polygons = polygon_sequence[len(polygon_sequence)//2:]
            first_half_polygon_indexes = polygon_indexes[:len(polygon_indexes)//2 + 1]
            second_half_polygon_indexes = polygon_indexes[len(polygon_indexes)//2:]
            first_half_collides, collision_data = csv_check_collisions(
                polygon, first_half_polygons,
                polygon_indexes=first_half_polygon_indexes,
                collision_data=collision_data, display_debug=display_debug)
            second_half_collides, collision_data = csv_check_collisions(
                polygon, second_half_polygons,
                polygon_indexes=second_half_polygon_indexes,
                collision_data=collision_data, display_debug=display_debug)
            return (first_half_collides or second_half_collides), collision_data
        else:
            return True, collision_data
    else:
        return False, collision_data


if __name__ == '__main__':
    poly = Polygon([(0.,0.), (0.,1.), (1., 1.), (1., 0.75), (0.25, 0.75), (0.25, 0)])
    poly_2 = affinity.translate(affinity.rotate(poly, 45.), 0.4, 1.1)
    poly_3 = affinity.translate(affinity.rotate(poly_2, 20.), 0.6, 1.0)
    polygon_states = [poly, poly_2, poly_3]

    obs = Polygon([(1.45, 1.75), (1.45, 1.25), (1.95, 1.25), (1.95, 1.75)])
    not_obs = affinity.translate(obs, 1., 1.)

    fig = plt.figure()
    plt.plot(*poly.exterior.xy)
    plt.plot(*poly_2.exterior.xy)
    plt.plot(*poly_3.exterior.xy)
    plt.plot(*obs.exterior.xy)
    plt.plot(*not_obs.exterior.xy)

    bb_vertices = bounding_boxes_vertices(polygon_states)
    x, y = zip(*[[vertex.x, vertex.y] for vertex in bb_vertices])
    plt.scatter(x, y)

    csv = csv_from_bb_vertices(bb_vertices)
    plt.plot(*csv.exterior.xy)

    plt.axis('equal')
    # plt.axhline(y=0)
    # plt.axvline(x=0)
    plt.show()

    obs_collides, obs_collision_data = csv_check_collisions(obs, polygon_states, display_debug=True)
    print(str(obs_collides))
    not_obs_collides, not_obs_collision_data = csv_check_collisions(not_obs, polygon_states, display_debug=True)
    print(str(not_obs_collides))

