import math
from shapely.geometry import MultiPoint
import shapely.affinity as affinity
import matplotlib.pyplot as plt
from aabbtree import AABB, AABBTree
import snamosim.behaviors.plan.basic_actions as ba


class Action:

    def __init__(self):
        pass


class Rotation(Action):

    def __init__(self, angle, center):
        Action.__init__(self)
        self.angle = angle
        self.center = center

    def apply(self, polygon):
        return affinity.rotate(geom=polygon, angle=self.angle, origin=self.center, use_radians=False)


class Translation(Action):

    def __init__(self, translation_vector):
        Action.__init__(self)
        self.translation_vector = translation_vector

    def apply(self, polygon):
        return affinity.translate(geom=polygon, xoff=self.translation_vector[0], yoff=self.translation_vector[1], zoff=0.)


def convert_action(action, robot_pose):
    # TODO Deprecate this and the specific actions by changing the API of the following functions
    if isinstance(action, ba.Translation):
        translation_vector = action.compute_translation_vector(robot_pose[2])
        return Translation(translation_vector)
    elif isinstance(action, ba.Rotation):
        return Rotation(action.angle, (robot_pose[0], robot_pose[1]))


def bounds(points):
    minx, miny, maxx, maxy = float("inf"), float("inf"), -float("inf"), -float("inf")
    for point in points:
        minx, miny, maxx, maxy = min(minx, point[0]), min(miny, point[1]), max(maxx, point[0]), max(maxy, point[1])
    return minx, miny, maxx, maxy


def rotate(point, angle, center, radius=None, radians=False):
    if not radius:
        radius = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
    if not radians:
        angle = math.radians(angle)
    angle += math.atan2((point[1] - center[1]), (point[0] - center[0]))
    return center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)


def arc_bounding_box(point_a, rot_angle, center, point_b=None, point_c=None, bb_type='minimum_rotated_rectangle'):
    """
    Computes the bounding box of the arc formed by the rotation of a point A around a given center
    :param point_a: Initial point state
    :type point_a: (float, float)
    :param rot_angle: rotation angle in degrees.
    :type rot_angle: float
    :param center: rotation origin point
    :type center: (float, float)
    :param point_b: Final point state after rotation, can be provided to accelerate computation
    :type point_b: (float, float)
    :param point_c: Middle point state after rotation, can be provided to accelerate computation
    :type point_c: (float, float)
    :param bb_type: Type of bounding box, either 'minimum_rotated_rectangle' or 'aabbox', first one is most accurate
    :type bb_type: str
    :return: Return a list of four points coordinates corresponding to the bounding box
    :rtype: [(float, float), (float, float), (float, float), (float, float)]
    """
    if not point_b:
        r = math.sqrt((point_a[0] - center[0]) ** 2 + (point_a[1] - center[1]) ** 2)
        point_b = rotate(point_a, rot_angle, center, radius=r)
    else:
        r = None

    if -1.e-15 < rot_angle < 1.e-15:
        # It means that there is no movement, return only A
        return [point_a]
    elif -180. <= rot_angle <= 180.:
        # If the arc is less than a half circle

        # Compute middle point C
        if not point_c:
            point_c = rotate(point_a, rot_angle / 2., center)

        if bb_type is 'minimum_rotated_rectangle':
            # The minimum rotated rectangle's corners are points A, B, D and E
            # D and E are the intersection points between the line parallel to [AB] passing by C, and respectively,
            # the lines perpendicular to [AB] passing by A and B.
            x_b_min_a, y_b_min_a = (point_b[0] - point_a[0]), (point_b[1] - point_a[1])
            if -1.e-15 < x_b_min_a < 1.e-15:
                # Special case where [AB] is vertical
                point_d, point_e = (point_c[0], point_a[1]), (point_c[0], point_b[1])
            else:
                # General case
                m_ab = y_b_min_a / x_b_min_a  # [AB]'s slope = [DC]'s slope
                if -1.e-15 < m_ab < 1.e-15:
                    # Special case where [AB] is horizontal
                    point_d, point_e = (point_a[0], point_c[1]), (point_b[0], point_c[1])
                else:
                    b_dc = point_c[1] - m_ab * point_c[0]
                    m_ad = 0. if m_ab >= 1e15 else -1. / m_ab
                    b_ad = point_a[1] - m_ad * point_a[0]
                    xd = (b_ad - b_dc) / (m_ab - m_ad)
                    yd = xd * m_ab + b_dc
                    point_d = (xd, yd)
                    # C is the midpoint between D and E, allowing us to compute E
                    point_e = (2. * point_c[0] - point_d[0], 2. * point_c[1] - point_d[1])
            return [point_a, point_b, point_d, point_e]
        elif bb_type is 'aabbox':
            # The aabb corners are simply the bounds of points A, B and C.
            minx, miny, maxx, maxy = bounds([point_a, point_b, point_c])
            return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    elif -360. < rot_angle < 360.:
        # If the arc is greater than a half circle but not a circle
        # then we have 5 extremal points : A, B, C, D and E.
        # C is the arc middle point
        # D and E are the intersection points between the circle's equation and the ray that is perpendicular
        # to the ray passing through C

        # Compute middle point C and the radius if not already computed
        if not r:
            r = math.sqrt((point_a[0] - center[0]) ** 2 + (point_a[1] - center[1]) ** 2)
        if not point_c:
            point_c = rotate(point_a, rot_angle / 2., center, radius=r)

        # Compute the slope of the ray passing through C
        m1 = (point_c[1] - center[1]) / (point_c[0] - center[0])

        if -1.e-15 < m1 < 1.e-15:
            # If the ray passing through C IS horizontal

            # Line terms of the ray that is perpendicular to the ray passing through C (x=p2 is vertical line equation)
            p2 = center[0]

            # Terms of the equation to solve for x coordinate of points D and E
            a = 1.
            b = -2. * center[1]
            c = center[0] ** 2 + center[1] ** 2 + p2 ** 2 - 2. * center[0] * p2 - r ** 2

            # Solve the equation to get the coordinates of points D and E
            discriminant = (b ** 2) - (4 * a * c)

            yd = (-b - math.sqrt(discriminant)) / (2 * a)
            ye = (-b + math.sqrt(discriminant)) / (2 * a)

            xd = center[0]
            xe = center[0]

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
            if bb_type is 'minimum_rotated_rectangle':
                return list(zip(bb_points_x, bb_points_y))
            elif bb_type is 'aabbox':
                minx, miny, maxx, maxy = bounds(list(zip(bb_points_x, bb_points_y)))
                return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
        else:
            # If the ray passing through C is not horizontal (GENERAL CASE)

            # Line terms of the ray that is perpendicular to the ray passing through C
            m2 = 0. if m1 >= 1e15 else -1. / m1  # If ray passing through C is vertical, perpendicular is horizontal
            p2 = center[1] - m2 * center[0]

            # Terms of the equation to solve for x coordinate of points D and E
            a = 1. + m2 ** 2
            b = m2 * (2. * p2 - 2. * center[1]) - 2. * center[0]
            c = center[0] ** 2 + p2 ** 2 + center[1] ** 2 - 2. * p2 * center[1] - r ** 2

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
            if bb_type is 'minimum_rotated_rectangle':
                return list(zip(bb_points_x, bb_points_y))
            elif bb_type is 'aabbox':
                minx, miny, maxx, maxy = bounds(list(zip(bb_points_x, bb_points_y)))
                return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    else:
        # Beyond 360 degrees, the arc is a circle: its bounding box is necessarily a square aabb
        r = math.sqrt((point_a[0] - center[0]) ** 2 + (point_a[1] - center[1]) ** 2)
        return [
            (center[0] - r, center[1] - r), (center[0] + r, center[1] - r),
            (center[0] + r, center[1] + r), (center[0] - r, center[1] + r)
        ]


def bounding_boxes_vertices(action_sequence, polygon_sequence, bb_type='minimum_rotated_rectangle'):
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
        if isinstance(action, Translation):
            for coord in init_poly_coords:
                action_bb_vertices.append(coord)
            for coord in end_poly_coords:
                action_bb_vertices.append(coord)
        elif isinstance(action, Rotation):
            for point_a, point_b in zip(init_poly_coords, end_poly_coords):
                bb = arc_bounding_box(point_a=point_a, point_b=point_b, rot_angle=action.angle, center=action.center, bb_type=bb_type)
                for coord in bb:
                    action_bb_vertices.append(coord)
        else:
            raise TypeError("Actions must be pure Translation or Rotation.")
        bb_vertices.append(action_bb_vertices)
    return bb_vertices


def csv_from_bb_vertices(bb_vertices):
    """
    Computes the CSV (Convex Swept Volume) approximation polygon of the provided bounding boxes vertices
    :param bb_vertices: Bounding boxes vertices
    :type bb_vertices: shapely.geometry.MultiPoint
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.Polygon
    """
    all_vertices = [vertex for vertices in bb_vertices for vertex in vertices]
    return MultiPoint(all_vertices).convex_hull


def polygon_to_aabb(polygon):
    xmin, ymin, xmax, ymax = polygon.bounds
    return AABB([(xmin, xmax), (ymin, ymax)])


def polygons_to_aabb_tree(polygons):
    aabb_tree = AABBTree()
    for uid, polygon in polygons.items():
        aabb_tree.add(polygon_to_aabb(polygon), uid)
    return aabb_tree


def check_static_collision(main_uid, polygon, other_entities_polygons, aabb_tree, ignored_uids=None, break_at_first=True):
    aabb = polygon_to_aabb(polygon)
    potential_collision_uids = aabb_tree.overlap_values(aabb)
    if ignored_uids:
        potential_collision_uids = set(potential_collision_uids).difference(set(ignored_uids))
    if break_at_first:
        for uid in potential_collision_uids:
            if polygon.intersects(other_entities_polygons[uid]):
                return {main_uid: {uid}, uid: {main_uid}}
        return {}
    else:
        collides_with = {}
        for uid in potential_collision_uids:
            if polygon.intersects(other_entities_polygons[uid]):
                if main_uid in collides_with:
                    collides_with[main_uid].add(uid)
                else:
                    collides_with[main_uid] = {uid}

                if uid in collides_with:
                    collides_with[uid].add(main_uid)
                else:
                    collides_with[uid] = {main_uid}
        return collides_with


def csv_check_collisions(main_uid, other_polygons, polygon_sequence, action_sequence, bb_type='minimum_rotated_rectangle',
                         aabb_tree=None, bb_vertices=None, ignored_entities=None, display_debug=False):
    # Initialize at first recursive iteration
    if not aabb_tree:
        aabb_tree = polygons_to_aabb_tree(other_polygons)
    if not bb_vertices:
        bb_vertices = bounding_boxes_vertices(action_sequence, polygon_sequence, bb_type)

    csv_polygon = csv_from_bb_vertices(bb_vertices)

    # Dichotomy-check for collision between polygon and CSV as long as:
    # - there is no collision
    # - AND the CSV envelops more than one action (two consecutive polygons)
    collides_with = check_static_collision(main_uid, csv_polygon, other_polygons, aabb_tree, ignored_entities, True)

    if collides_with:
        if display_debug:
            fig, ax = plt.subplots()
            for p in polygon_sequence:
                ax.plot(*p.exterior.xy, color='grey')
            # for i in indexes:
            #     ax.plot(*polygon_sequence[i].exterior.xy, color='blue')
            for p in other_polygons.values():
                ax.plot(*p.exterior.xy, color='black')
            x, y = zip(*[[vertex.x, vertex.y] for vertex in bb_vertices])
            ax.scatter(x, y, marker='x')
            ax.plot(*csv_polygon.exterior.xy, color='green')
            intersection = csv_polygon.intersection(other_polygons[collides_with[main_uid][0]])
            ax.plot(*intersection.exterior.xy, color='red')
            ax.axis('equal')
            fig.show()
            print("")

        if len(bb_vertices) >= 2:
            first_half_bb_vertices = bb_vertices[:len(bb_vertices) // 2]
            second_half_bb_vertices = bb_vertices[len(bb_vertices) // 2:]
            first_half_collides, collides_with, aabb_tree = csv_check_collisions(
                main_uid, other_polygons, polygon_sequence, action_sequence, aabb_tree=aabb_tree,
                bb_vertices=first_half_bb_vertices, display_debug=display_debug)
            second_half_collides, collides_with, aabb_tree = csv_check_collisions(
                main_uid, other_polygons, polygon_sequence, action_sequence, aabb_tree=aabb_tree,
                bb_vertices=second_half_bb_vertices, display_debug=display_debug)
            return (first_half_collides or second_half_collides), collides_with, aabb_tree
        else:
            return True, collides_with, aabb_tree
    else:
        return False, collides_with, aabb_tree
