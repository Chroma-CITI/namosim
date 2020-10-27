import math, cmath
from shapely.geometry import Polygon, MultiPoint, Point, LineString
import shapely.affinity as affinity
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from aabbtree import AABB, AABBTree
import utils


class Action:

    def __init__(self):
        pass


class Rotation(Action):

    def __init__(self, angle, center):
        self.angle = angle
        self.center = center

    def apply(self, polygon):
        return affinity.rotate(geom=polygon, angle=self.angle, origin=self.center, use_radians=False)


class Translation(Action):

    def __init__(self, translation_vector):
        self.translation_vector = translation_vector

    def apply(self, polygon):
        return affinity.translate(geom=polygon, xoff=self.translation_vector[0], yoff=self.translation_vector[1], zoff=0.)


class ActionList:

    def __init__(self, actions):
        self.actions = actions

    def apply(self, polygon):
        polygon_states = [polygon]
        prev_polygon_state = polygon
        for action in self.actions:
            new_polygon_state = action.apply(prev_polygon_state)
            polygon_states.append(new_polygon_state)
            prev_polygon_state = new_polygon_state
        return polygon_states


def states_for_points(polygon_states):
    """
    Computes the list (ordered by states) of coordinates of each point of the polygon
    :param polygon_states: sequence of polygons representing the same shape at different
    :type polygon_states: list(shapely.geometry.Polygon)
    :return: List (ordered by states) of coordinates of each point of the polygon
    :rtype: list(list((float, float)))
    """
    return zip(*[list(polygon.exterior.coords) for polygon in polygon_states])


def bounding_boxes_vertices_from_states_for_points(s_for_points, actions, bb_type='minimum_rotated_rectangle'):
    """
    Computes the bounding boxes for each point coordinates list of the polygon
    :param s_for_points: List (ordered by states) of coordinates of each point of the polygon
    :type s_for_points: list(list((float, float)))
    :param bb_type: either 'minimum_rotated_rectangle' or 'aabbox'
    :type bb_type: str
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.MultiPoint
    """
    bounding_boxes_points_coords = []

    action_subsequences = [[None]]
    subsequences_indexes = [[0]]
    cur_index = 0
    prev_action = actions[0]
    for action_index, action in enumerate(actions):
        cur_seq_still_same_type = (
            isinstance(action, Translation) and isinstance(prev_action, Translation)
            or isinstance(action, Rotation) and isinstance(prev_action, Rotation)
        )
        if cur_seq_still_same_type:
            action_subsequences[cur_index].append(action)
            subsequences_indexes[cur_index].append(action_index + 1)
        else:
            action_subsequences.append([action])
            subsequences_indexes.append([action_index + 1])
            cur_index += 1
        prev_action = action

    for action_subsequence, subsequence_indexes in zip(action_subsequences, subsequences_indexes):
        actions_are_all_translations = (
            (isinstance(action_subsequence[0], Translation))
            or (action_subsequence[0] is None and isinstance(action_subsequence[1], Translation))
        )
        if actions_are_all_translations:
            for point_states in s_for_points[subsequence_indexes[0]: subsequence_indexes[-1] + 1]:
                if bb_type is 'aabbox':
                    minx, miny, maxx, maxy = MultiPoint(point_states).bounds
                    bounding_boxes_points_coords += [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
                elif bb_type is 'minimum_rotated_rectangle':
                    min_rot_rec = MultiPoint(point_states).minimum_rotated_rectangle
                    if isinstance(min_rot_rec, Polygon):
                        bounding_boxes_points_coords += list(min_rot_rec.exterior.coords)[0:4]
                    elif isinstance(min_rot_rec, LineString):
                        bounding_boxes_points_coords += list(min_rot_rec.coords)
                else:
                    raise TypeError('"aabbox" and "minimum_rotated_rectangle" are the only accepted values for bb_type.')
        else:
            prev_points_state = []
            for action, index in zip(action_subsequence, subsequence_indexes):
                points_state = [point_states[index] for point_states in s_for_points]

                if isinstance(action, Rotation):
                    for prev_point_state, point_state in zip(prev_points_state, points_state):
                        bounding_boxes_points_coords += arc_bounding_box(
                            point_a=prev_point_state, point_b=point_state, rot_angle=action.angle,
                            center=action.center, bb_type=bb_type
                        )

                prev_points_state = points_state

    return MultiPoint(bounding_boxes_points_coords)


def bounding_boxes_vertices(polygon_states, actions, bb_type='minimum_rotated_rectangle'):
    """
    Computes the bounding boxes of corresponding points in a sequence of polygons states of the same shape
    :param polygon_states: sequence of polygons representing the same shape at different
    :type polygon_states: list(shapely.geometry.Polygon)
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.MultiPoint
    """
    s_for_points = states_for_points(polygon_states)
    return bounding_boxes_vertices_from_states_for_points(s_for_points, actions)


def csv_from_polygon_states(polygon_states, actions):
    """
    Computes the CSV (Convex Swept Volume) approximation polygon of the provided polygon states of the same shape
    :param polygon_states: sequence of polygons representing the same shape at different
    :type polygon_states: list(shapely.geometry.Polygon)
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.Polygon
    """
    return bounding_boxes_vertices(polygon_states, actions).convex_hull


def csv_from_bb_vertices(bb_vertices):
    """
    Computes the CSV (Convex Swept Volume) approximation polygon of the provided bounding boxes vertices
    :param bb_vertices: Bounding boxes vertices
    :type bb_vertices: shapely.geometry.MultiPoint
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.Polygon
    """
    return bb_vertices.convex_hull


def polygon_to_aabb(polygon):
    xmin, ymin, xmax, ymax = polygon.bounds
    return AABB([(xmin, xmax), (ymin, ymax)])


def polygons_to_aabb_tree(polygons):
    aabb_tree = AABBTree()
    for uid, polygon in polygons.items():
        aabb_tree.add(polygon_to_aabb(polygon), uid)
    return aabb_tree


def arc_bounding_box(point_a, point_b, rot_angle, center, bb_type='minimum_rotated_rectangle'):
    """
    Computes the bounding box of the arc formed by the rotation or combined rotation+translation of a point a
    to another location b
    :param point_a: Initial point state
    :type point_a: (float, float)
    :param point_b: Final point state after rotation (and translation if relevant)
    :type point_b: (float, float)
    :param rot_angle: rotation angle in degrees. MUST BE BETWEEN -360 and 360 degrees !
    :type rot_angle: float
    :param center: rotation origin point
    :type center: (float, float)
    :param bb_type: either 'minimum_rotated_rectangle' or 'aabbox'
    :type bb_type: str
    :return: Return a list of four points coordinates corresponding to the bounding box
    :rtype: [(float, float), (float, float), (float, float), (float, float)]
    """

    if -1.e-10 < rot_angle < 1.e-10:
        # It means that there is no movement, A and B are the same point.
        if bb_type is 'aabbox':
            minx, miny, maxx, maxy = MultiPoint([point_a, point_b]).bounds
            return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
        elif bb_type is 'minimum_rotated_rectangle':
            min_rot_rec = MultiPoint([point_a, point_b]).minimum_rotated_rectangle
            if isinstance(min_rot_rec, Polygon):
                return list(min_rot_rec.exterior.coords)[0:4]
            elif isinstance(min_rot_rec, LineString):
                return list(min_rot_rec.coords)
    elif -180. <= rot_angle <= 180.:
        # If the arc is less than a half circle, we only need to use its middle point C to have the coordinates
        # of the bounding box

        # Compute extra extremal point C and circle terms
        point_a_shapely = Point(point_a)
        point_c = affinity.rotate(point_a_shapely, rot_angle / 2., origin=center).coords[0]
        center_x, center_y, r = utils.find_circle_terms(
            point_a[0], point_a[1], point_b[0], point_b[1], point_c[0], point_c[1])

        if bb_type is 'minimum_rotated_rectangle':
            return list(MultiPoint([point_a, point_b, point_c]).minimum_rotated_rectangle.exterior.coords)[0:4]
        elif bb_type is 'aabbox':
            minx, miny, maxx, maxy = MultiPoint([point_a, point_b, point_c]).bounds
            return list(Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]).exterior.coords)[0:4]
    elif -360. < rot_angle < 360.:
        # If the arc is greater than a half circle, we need to compute the other bounding points;
        # that is, the intersection points D and E between the circle's equation and the perpendicular line
        # to the ray passing through C

        # Compute extra extremal point C and circle terms
        point_a_shapely = Point(point_a)
        point_c = affinity.rotate(point_a_shapely, rot_angle / 2., origin=center).coords[0]
        center_x, center_y, r = utils.find_circle_terms(
            point_a[0], point_a[1], point_b[0], point_b[1], point_c[0], point_c[1])

        # Ray passing through C line terms
        m1 = (point_c[1] - center_y) / (point_c[0] - center_x)
        p1 = center_y - m1 * center_x

        if m1 != 0.:
            # If the line passing by the center and C is not horizontal

            # Perpendicular to the ray line terms
            m2 = 0. if m1 >= 1e15 else -1. / m1
            p2 = center_y - m2 * center_x

            # Terms of the equation to solve for x coordinate of points D and E
            a = 1. + m2 ** 2
            b = m2 * (2. * p2 - 2. * center_y) - 2. * center_x
            c = center_x ** 2 + p2 ** 2 + center_y ** 2 - 2. * p2 * center_y - r ** 2

            # Solve the equation to get the coordinates of points D and E
            discriminant = (b ** 2) - (4. * a * c)

            xd = (-b - math.sqrt(discriminant)) / (2. * a)
            xe = (-b + math.sqrt(discriminant)) / (2. * a)

            yd = xd * m2 + p2
            ye = xe * m2 + p2

            point_d, point_e = (xd, yd), (xe, ye)

            # Now simply return the proper bounding box englobing A, B, C, D and E
            m_lc = m2
            p_lc = point_c[1] - m_lc * point_c[0]

            m_ld = m1
            p_ld = point_d[1] - m_ld * point_d[0]

            m_le = m1
            p_le = point_e[1] - m_le * point_e[0]

            m_lab = m2
            p_lab = point_a[1] - m_lab * point_a[0]

            bb_points_x = [
                (p_lc - p_ld) / (m_ld - m_lc),
                (p_lc - p_le) / (m_le - m_lc),
                (p_lab - p_le) / (m_le - m_lab),
                (p_lab - p_ld) / (m_ld - m_lab)
            ]
            bb_points_y = [
                m_lc * bb_points_x[0] + p_lc,
                m_lc * bb_points_x[1] + p_lc,
                m_lab * bb_points_x[2] + p_lab,
                m_lab * bb_points_x[3] + p_lab
            ]
        else:
            # If the line passing by the center and C IS horizontal

            # Perpendicular to the ray line term (x = p2 is the vertical line equation)
            p2 = center_x

            # Terms of the equation to solve for x coordinate of points D and E
            a = 1.
            b = -2. * center_y
            c = center_x ** 2 + center_y ** 2 + p2 ** 2 - 2. * center_x * p2 - r ** 2

            # Solve the equation to get the coordinates of points D and E
            discriminant = (b ** 2) - (4 * a * c)

            yd = (-b - math.sqrt(discriminant)) / (2 * a)
            ye = (-b + math.sqrt(discriminant)) / (2 * a)

            xd = center_x
            xe = center_x

            point_d, point_e = (xd, yd), (xe, ye)

            # Now simply return the proper bounding box englobing A, B, C, D and E
            bb_points_x = [
                point_c[0],
                point_c[0],
                point_a[0],
                point_a[0]
            ]
            bb_points_y = [
                point_d[1],
                point_e[1],
                point_e[1],
                point_d[1]
            ]

        return list(zip(bb_points_x, bb_points_y))


def csv_check_collisions(other_polygons, polygon_sequence, action_sequence, bb_type='minimum_rotated_rectangle',
                         aabb_tree=None, indexes=None, collision_data=None, display_debug=False):
    # TODO: Separate collision_data and csv_polygon. Only return bb_vertices if debug option is active.
    # TODO: Make object for collision data, not dict.

    # Initialize at first recursive iteration
    if not collision_data:
        collision_data = {}
    if not indexes:
        indexes = [i for i in range(len(polygon_sequence))]
    if not aabb_tree:
        aabb_tree = polygons_to_aabb_tree(other_polygons)

    # Allow reuse of pre-computed CSV polygons to save perfomance
    indexes_tuple = tuple(indexes)
    if indexes_tuple in collision_data:
        bb_vertices = collision_data[indexes_tuple]["bb_vertices"]
        csv_polygon = collision_data[indexes_tuple]["csv_polygon"]
    else:
        bb_vertices = bounding_boxes_vertices(
            [polygon_sequence[index] for index in indexes],
            [action_sequence[index] for index in indexes[:-1]]
        )
        csv_polygon = csv_from_bb_vertices(bb_vertices)
        collision_data[indexes_tuple] = {
            "bb_vertices": bb_vertices,
            "csv_polygon": csv_polygon,
        }

    # Dichotomy-check for collision between polygon and CSV as long as:
    # - there is a collision
    # - AND the CSV envelops more than two consecutive polygons
    csv_aabb = polygon_to_aabb(csv_polygon)
    potential_collision_polygons_indexes = aabb_tree.overlap_values(csv_aabb)
    intersects = False
    for uid in potential_collision_polygons_indexes:
        polygon = other_polygons[uid]
        if csv_polygon.intersects(polygon):
            intersects = True
            collision_data[indexes_tuple]['colliding_polygon_uid'] = uid
            intersection = csv_polygon.intersection(polygon)
            collision_data[indexes_tuple]["intersection_polygon"] = intersection

            if display_debug and len(indexes) == 2:
                fig, ax = plt.subplots()
                for p in polygon_sequence:
                    ax.plot(*p.exterior.xy, color='grey')
                for i in indexes:
                    ax.plot(*polygon_sequence[i].exterior.xy, color='blue')
                for p in other_polygons.values():
                    ax.plot(*p.exterior.xy, color='black')
                x, y = zip(*[[vertex.x, vertex.y] for vertex in bb_vertices])
                ax.scatter(x, y, marker='x')
                ax.plot(*csv_polygon.exterior.xy, color='green')
                intersection = csv_polygon.intersection(polygon)
                ax.plot(*intersection.exterior.xy, color='red')
                ax.axis('equal')
                fig.show()
                print("")

            break

    if intersects:
        if len(indexes) > 2:
            first_half_indexes = indexes[:len(indexes) // 2 + 1]
            second_half_indexes = indexes[len(indexes) // 2:]
            first_half_collides, collision_data, aabb_tree = csv_check_collisions(
                other_polygons, polygon_sequence, action_sequence,
                indexes=first_half_indexes,
                collision_data=collision_data, display_debug=display_debug)
            second_half_collides, collision_data, aabb_tree = csv_check_collisions(
                other_polygons, polygon_sequence, action_sequence,
                indexes=second_half_indexes,
                collision_data=collision_data, display_debug=display_debug)
            return (first_half_collides or second_half_collides), collision_data, aabb_tree
        else:
            return True, collision_data, aabb_tree
    else:
        return False, collision_data, aabb_tree


if __name__ == '__main__':
    poly = Polygon([(0., 0.), (0., 1.), (1., 1.), (1., 0.75), (0.25, 0.75), (0.25, 0)])
    action_list = ActionList([
        LinearMovement(translation_vector=(0.4, 1.1), angle=45., center='center'),
        LinearMovement(translation_vector=(0.6, 1.0), angle=20., center='center')
    ])
    polygon_states = action_list.apply(poly)

    obs = Polygon([(1.45, 1.75), (1.45, 1.25), (1.95, 1.25), (1.95, 1.75)])
    not_obs = affinity.translate(obs, 1., 1.)

    _fig, _ax = plt.subplots()
    for p in polygon_states:
        _ax.plot(*p.exterior.xy)
    _ax.plot(*obs.exterior.xy)
    _ax.plot(*not_obs.exterior.xy)

    _bb_vertices = bounding_boxes_vertices(polygon_states, action_list.actions, bb_type='aabbox')
    _x, _y = zip(*[[_vertex.x, _vertex.y] for _vertex in _bb_vertices])
    _ax.scatter(_x, _y)

    csv = csv_from_bb_vertices(_bb_vertices)
    _ax.plot(*csv.exterior.xy)

    # Test circle finding method
    poly_mi_2 = affinity.translate(affinity.rotate(poly, 22.5), 0.2, 0.55)
    _ax.plot(*poly_mi_2.exterior.xy)

    synced_coords = zip(
        list(poly.exterior.coords),
        list(poly_mi_2 .exterior.coords),
        list(polygon_states[1].exterior.coords)
    )

    circles_terms = [
        utils.find_circle_terms(
            coords[0][0], coords[0][1],
            coords[1][0], coords[1][1],
            coords[2][0], coords[2][1]
        )
        for coords in synced_coords
    ]

    plt_circles = [plt.Circle((circle_terms[0], circle_terms[1]), circle_terms[2], fill=False) for circle_terms in circles_terms]

    for circle in plt_circles:
        _ax.add_artist(circle)
    # _ax.add_artist(plt_circles[1])

    angles = [
        (
            utils.points_to_angle(
                coords[0][0], coords[0][1], circle_terms[0], circle_terms[1], coords[1][0], coords[1][1]
            ),
            utils.points_to_angle(
                coords[1][0], coords[1][1], circle_terms[0], circle_terms[1], coords[2][0], coords[2][1]
            )
        )
        for circle_terms, coords in zip(circles_terms, synced_coords)
    ]

    deg_angles = [math.degrees(angle_1 + angle_2) for angle_1, angle_2 in angles]

    # Display everything
    _ax.axis('equal')
    # plt.axhline(y=0)
    # plt.axvline(x=0)
    _fig.show()

    obs_collides, obs_collision_data, _aabb_tree = csv_check_collisions(
        {1: obs}, polygon_states, action_list.actions, display_debug=True)
    print(str(obs_collides))
    not_obs_collides, not_obs_collision_data, not_obs_aabb_tree = csv_check_collisions(
        {2: not_obs}, polygon_states, action_list.actions, display_debug=True)
    print(str(not_obs_collides))
