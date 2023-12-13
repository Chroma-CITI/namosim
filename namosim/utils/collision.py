import math
import typing as t

import matplotlib.pyplot as plt
import shapely.affinity as affinity
from aabbtree import AABB, AABBTree
from shapely import Polygon
from shapely.geometry import MultiPoint, Point

import namosim.navigation.basic_actions as ba
from namosim.data_models import PoseModel
from namosim.utils import utils
from namosim.world.world import World


class Action:
    def __init__(self):
        pass

    def apply(self, polygon: Polygon) -> Polygon:
        raise NotImplementedError()


class Rotation(Action):
    def __init__(self, angle: float, center: t.Tuple[float, float]):
        Action.__init__(self)
        self.angle = angle
        self.center = center

    def apply(self, polygon: Polygon) -> Polygon:
        return affinity.rotate(
            geom=polygon,
            angle=self.angle,
            origin=self.center,  # type: ignore
            use_radians=False,
        )


class Translation(Action):
    def __init__(self, translation_vector: t.Tuple[float, float]):
        Action.__init__(self)
        self.translation_vector = translation_vector

    def apply(self, polygon: Polygon) -> Polygon:
        return affinity.translate(
            geom=polygon,
            xoff=self.translation_vector[0],
            yoff=self.translation_vector[1],
            zoff=0.0,
        )


def convert_action(action: ba.BasicAction, robot_pose: PoseModel) -> Action:
    # TODO Deprecate this and the specific actions by changing the API of the following functions
    if isinstance(action, ba.Translation):
        translation_vector = action.compute_translation_vector(robot_pose[2])
        return Translation(translation_vector)
    elif isinstance(action, ba.Rotation):
        return Rotation(action.angle, (robot_pose[0], robot_pose[1]))
    raise Exception("Unable to convert action")


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


def rotate(
    point: t.Tuple[float, float],
    angle: float,
    center: t.Tuple[float, float],
    radius: float | None = None,
    radians: bool = False,
):
    if not radius:
        radius = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
    if not radians:
        angle = math.radians(angle)
    angle += math.atan2((point[1] - center[1]), (point[0] - center[0]))
    return center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)


def arc_bounding_box(
    point_a: t.Tuple[float, float],
    rot_angle: float,
    center: t.Tuple[float, float],
    point_b: t.Tuple[float, float] | None = None,
    point_c: t.Tuple[float, float] | None = None,
    bb_type: str = "minimum_rotated_rectangle",
):
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

    if -1.0e-15 < rot_angle < 1.0e-15:
        # It means that there is no movement, return only A
        return [point_a]
    elif -180.0 <= rot_angle <= 180.0:
        # If the arc is less than a half circle

        # Compute middle point C
        if not point_c:
            point_c = rotate(point_a, rot_angle / 2.0, center)

        if bb_type == "minimum_rotated_rectangle":
            # The minimum rotated rectangle's corners are points A, B, D and E
            # D and E are the intersection points between the line parallel to [AB] passing by C, and respectively,
            # the lines perpendicular to [AB] passing by A and B.
            x_b_min_a, y_b_min_a = (point_b[0] - point_a[0]), (point_b[1] - point_a[1])
            if -1.0e-15 < x_b_min_a < 1.0e-15:
                # Special case where [AB] is vertical
                point_d, point_e = (point_c[0], point_a[1]), (point_c[0], point_b[1])
            else:
                # General case
                m_ab = y_b_min_a / x_b_min_a  # [AB]'s slope = [DC]'s slope
                if -1.0e-15 < m_ab < 1.0e-15:
                    # Special case where [AB] is horizontal
                    point_d, point_e = (
                        (point_a[0], point_c[1]),
                        (point_b[0], point_c[1]),
                    )
                else:
                    b_dc = point_c[1] - m_ab * point_c[0]
                    m_ad = 0.0 if m_ab >= 1e15 else -1.0 / m_ab
                    b_ad = point_a[1] - m_ad * point_a[0]
                    xd = (b_ad - b_dc) / (m_ab - m_ad)
                    yd = xd * m_ab + b_dc
                    point_d = (xd, yd)
                    # C is the midpoint between D and E, allowing us to compute E
                    point_e = (
                        2.0 * point_c[0] - point_d[0],
                        2.0 * point_c[1] - point_d[1],
                    )
            return [point_a, point_b, point_d, point_e]
        elif bb_type == "aabbox":
            # The aabb corners are simply the bounds of points A, B and C.
            minx, miny, maxx, maxy = bounds([point_a, point_b, point_c])
            return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    elif -360.0 < rot_angle < 360.0:
        # If the arc is greater than a half circle but not a circle
        # then we have 5 extremal points : A, B, C, D and E.
        # C is the arc middle point
        # D and E are the intersection points between the circle's equation and the ray that is perpendicular
        # to the ray passing through C

        # Compute middle point C and the radius if not already computed
        if not r:
            r = math.sqrt((point_a[0] - center[0]) ** 2 + (point_a[1] - center[1]) ** 2)
        if not point_c:
            point_c = rotate(point_a, rot_angle / 2.0, center, radius=r)

        # Compute the slope of the ray passing through C
        m1 = (point_c[1] - center[1]) / (point_c[0] - center[0])

        if -1.0e-15 < m1 < 1.0e-15:
            # If the ray passing through C IS horizontal

            # Line terms of the ray that is perpendicular to the ray passing through C (x=p2 is vertical line equation)
            p2 = center[0]

            # Terms of the equation to solve for x coordinate of points D and E
            a = 1.0
            b = -2.0 * center[1]
            c = center[0] ** 2 + center[1] ** 2 + p2**2 - 2.0 * center[0] * p2 - r**2

            # Solve the equation to get the coordinates of points D and E
            discriminant = (b**2) - (4 * a * c)

            yd = (-b - math.sqrt(discriminant)) / (2 * a)
            ye = (-b + math.sqrt(discriminant)) / (2 * a)

            xd = center[0]
            xe = center[0]

            point_d, point_e = (xd, yd), (xe, ye)

            # Now simply return the proper bounding box englobing A, B, C, D and E
            bb_points_x = [point_c[0], point_c[0], point_a[0], point_a[0]]
            bb_points_y = [point_d[1], point_e[1], point_e[1], point_d[1]]
            if bb_type == "minimum_rotated_rectangle":
                return list(zip(bb_points_x, bb_points_y))
            elif bb_type == "aabbox":
                minx, miny, maxx, maxy = bounds(list(zip(bb_points_x, bb_points_y)))
                return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
            else:
                raise Exception("Invalid bb_type arg")
        else:
            # If the ray passing through C is not horizontal (GENERAL CASE)

            # Line terms of the ray that is perpendicular to the ray passing through C
            m2 = (
                0.0 if m1 >= 1e15 else -1.0 / m1
            )  # If ray passing through C is vertical, perpendicular is horizontal
            p2 = center[1] - m2 * center[0]

            # Terms of the equation to solve for x coordinate of points D and E
            a = 1.0 + m2**2
            b = m2 * (2.0 * p2 - 2.0 * center[1]) - 2.0 * center[0]
            c = center[0] ** 2 + p2**2 + center[1] ** 2 - 2.0 * p2 * center[1] - r**2

            # Solve the equation to get the coordinates of points D and E
            discriminant = (b**2) - (4.0 * a * c)

            xd = (-b - math.sqrt(discriminant)) / (2.0 * a)
            xe = (-b + math.sqrt(discriminant)) / (2.0 * a)

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
                (p_lab - p_ld) / (m_ld - m_lab),
            ]
            bb_points_y = [
                m_lc * bb_points_x[0] + p_lc,
                m_lc * bb_points_x[1] + p_lc,
                m_lab * bb_points_x[2] + p_lab,
                m_lab * bb_points_x[3] + p_lab,
            ]
            if bb_type == "minimum_rotated_rectangle":
                return list(zip(bb_points_x, bb_points_y))
            elif bb_type == "aabbox":
                minx, miny, maxx, maxy = bounds(list(zip(bb_points_x, bb_points_y)))
                return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    else:
        # Beyond 360 degrees, the arc is a circle: its bounding box is necessarily a square aabb
        r = math.sqrt((point_a[0] - center[0]) ** 2 + (point_a[1] - center[1]) ** 2)
        return [
            (center[0] - r, center[1] - r),
            (center[0] + r, center[1] - r),
            (center[0] + r, center[1] + r),
            (center[0] - r, center[1] + r),
        ]


def bounding_boxes_vertices(
    action_sequence: t.List[Action],
    polygon_sequence: t.List[Polygon],
    bb_type: str = "minimum_rotated_rectangle",
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
        if isinstance(action, Translation):
            for coord in init_poly_coords:
                action_bb_vertices.append(coord)
            for coord in end_poly_coords:
                action_bb_vertices.append(coord)
        elif isinstance(action, Rotation):
            for point_a, point_b in zip(init_poly_coords, end_poly_coords):
                bb = arc_bounding_box(
                    point_a=point_a,
                    point_b=point_b,
                    rot_angle=action.angle,
                    center=action.center,
                    bb_type=bb_type,
                )
                for coord in bb:
                    action_bb_vertices.append(coord)
        else:
            raise TypeError("Actions must be pure Translation or Rotation.")
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


def polygons_to_aabb_tree(polygons: t.Dict[int, Polygon]):
    aabb_tree = AABBTree()
    for uid, polygon in polygons.items():
        aabb_tree.add(polygon_to_aabb(polygon), uid)
    return aabb_tree


def check_static_collision(
    main_uid: int,
    polygon: Polygon,
    other_entities_polygons: t.Dict[int, Polygon],
    aabb_tree: AABBTree,
    ignored_uids: t.Iterable[int] | None = None,
    break_at_first: bool = True,
    save_intersections: bool = False,
):
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
                else:
                    return {main_uid: {uid}, uid: {main_uid}}
        return {}
    else:
        collides_with: t.Dict[int, t.Set[int]] = {}
        intersections: t.Dict[t.Tuple[int, int], Polygon] = {}

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

        if save_intersections:
            return collides_with, intersections
        else:
            return collides_with


def merge_collides_with(
    source: t.Dict[int, t.Set[int]], other: t.Dict[int, t.Set[int]]
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
    main_uid: int,
    other_polygons: t.Dict[int, Polygon],
    polygon_sequence: t.List[Polygon],
    action_sequence: t.List[Action],
    id_sequence: t.List[int] | None = None,
    bb_type: str = "minimum_rotated_rectangle",
    aabb_tree: AABBTree | None = None,
    bb_vertices: t.List[t.List[t.Tuple[float, float]]] | None = None,
    csv_polygons: t.Dict[t.Sequence[int], Polygon] | None = None,
    intersections: t.Dict[t.Tuple[int, int], Polygon] | None = None,
    ignored_entities: t.Set[int] | None = None,
    display_debug: bool = False,
    break_at_first: bool = True,
    save_intersections: bool = False,
) -> t.Tuple[
    bool,
    t.Dict[int, t.Set[int]],
    AABBTree,
    t.Dict[t.Sequence[int], Polygon],
    t.Dict[t.Tuple[int, int], Polygon],
    t.List[t.List[t.Tuple[float, float]]],
]:
    # Initialize at first recursive iteration
    if not aabb_tree:
        aabb_tree = polygons_to_aabb_tree(other_polygons)
    if not bb_vertices:
        bb_vertices = bounding_boxes_vertices(
            action_sequence, polygon_sequence, bb_type
        )
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
        collides_with = check_static_collision(
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
        else:
            return (
                True,
                collides_with,
                aabb_tree,
                csv_polygons,
                intersections,
                bb_vertices,
            )
    else:
        return False, collides_with, aabb_tree, csv_polygons, intersections, bb_vertices


def csv_simulate_simple_kinematics(
    world: World,
    agent_uid_to_next_action: t.Dict[int, ba.BasicAction],
    apply: bool = False,
    bb_type: str = "minimum_rotated_rectangle",
    ignore_collisions: bool = False,
    extra_transit_check: bool = False,
):
    # Apply each action to get polygon after for robot and obstacle if relevant
    # and compute CSV for each
    # and check that no CSV intersects with other entities beyond those that are moving this round
    uid_to_csv_polygon = {}
    collides_with = {}
    moving_uids = set(agent_uid_to_next_action.keys()).union(
        {
            world.entity_to_agent.inverse[agent_uid]
            for agent_uid in agent_uid_to_next_action.keys()
            if agent_uid in world.entity_to_agent.inverse
        }
    )
    other_polygons = {
        uid: e.polygon for uid, e in world.entities.items() if uid not in moving_uids
    }
    aabb_tree = polygons_to_aabb_tree(other_polygons)
    if apply:
        new_polygons = {}
        new_poses = {}
    for agent_uid, action in agent_uid_to_next_action.items():
        agent = world.entities[agent_uid]
        agent_action = convert_action(action, agent.pose)
        agent_polygon_after = action.apply(agent.polygon, agent.pose)
        agent_csv = csv_from_bb_vertices(
            bounding_boxes_vertices(
                [agent_action], [agent.polygon, agent_polygon_after], bb_type=bb_type
            )
        )
        uid_to_csv_polygon[agent_uid] = agent_csv
        ignored_entities = (
            {action.entity_uid} if isinstance(action, (ba.Release, ba.Grab)) else set()
        )
        agent_collides_with = check_static_collision(
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
            new_polygons[agent_uid] = agent_polygon_after
            if isinstance(action, ba.Translation):
                new_poses[agent_uid] = action.predict_pose(agent.pose, agent.pose[2])
            else:
                new_poses[agent_uid] = action.predict_pose(
                    agent.pose, (agent.pose[0], agent.pose[1])
                )
        if (
            not isinstance(action, (ba.Release, ba.Grab))
            and agent_uid in world.entity_to_agent.inverse
        ):
            obs_uid = world.entity_to_agent.inverse[agent_uid]
            obs = world.entities[obs_uid]
            obs_action = convert_action(action, agent.pose)
            obs_polygon_after = action.apply(obs.polygon, agent.pose)
            obs_csv = csv_from_bb_vertices(
                bounding_boxes_vertices(
                    [obs_action], [obs.polygon, obs_polygon_after], bb_type=bb_type
                )
            )
            uid_to_csv_polygon[obs_uid] = obs_csv
            obs_collides_with = check_static_collision(
                obs_uid, obs_csv, other_polygons, aabb_tree
            )
            merge_collides_with(collides_with, obs_collides_with)
            if apply:
                new_polygons[obs_uid] = obs_polygon_after
                if isinstance(action, ba.Translation):
                    new_poses[obs_uid] = action.predict_pose(obs.pose, agent.pose[2])
                else:
                    new_poses[obs_uid] = action.predict_pose(
                        obs.pose, (agent.pose[0], agent.pose[1])
                    )

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
            ),
        )

    # If option activated, check that no new polygon's discretized cell is the center cell of another robot in transit
    # utils.accurate_rasterize_in_grid(
    #     new_polygon, self.res, self.grid_pose, self.d_width, self.d_height, fill=fill
    # )
    # discretized_polygons = {uid: utils. for uid, polygon in new_polygons.items()}

    if apply:
        for agent_uid, action in agent_uid_to_next_action.items():
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
    return collides_with
    return collides_with
