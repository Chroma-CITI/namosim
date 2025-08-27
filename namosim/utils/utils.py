import json
import math
import os
import random
import typing as t
from collections.abc import MutableSet
from datetime import datetime

import numpy as np
import numpy.typing as npt
import shapely.affinity as affinity
import typing_extensions as tx
from matplotlib import colors
from shapely.geometry import LineString, Polygon
from rclpy.impl.rcutils_logger import RcutilsLogger

from namosim.data_models import Pose2D, VertexModel

# Constants
SQRT_OF_2 = math.sqrt(2.0)
TWO_PI = 2.0 * math.pi

TAXI_NEIGHBORHOOD = ((0, 1), (0, -1), (1, 0), (-1, 0))
CHESSBOARD_NEIGHBORHOOD = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
CHESSBOARD_NEIGHBORHOOD_EXTRAS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
CHESSBOARD_NEIGHBORHOOD_EXTRAS_SET = set(CHESSBOARD_NEIGHBORHOOD_EXTRAS)


def timestamp_string():
    return datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f")


class NamosimLog:
    def __init__(self, message: str, step, timestamp=timestamp_string()):
        self.message = message
        self.step = step
        self.timestamp = timestamp

    def __str__(self):
        # return "At step {}: '{}' - Timestamp: {}".format(self.step, self.message, self.timestamp)
        return "At step {}: '{}'".format(self.step, self.message)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class NamosimLogger(list[NamosimLog]):
    def __init__(self, printout: bool = True, ros2_logger: RcutilsLogger | None = None):
        super(NamosimLogger, self).__init__(self)
        self.printout = printout
        self.ros2_logger = ros2_logger

    def append(self, log: NamosimLog):
        super(NamosimLogger, self).append(log)
        if self.printout:
            print(log)
        if self.ros2_logger:
            self.ros2_logger.info(f"[namosim]:[step={log.step}]: {log.message})")


def euclidean_distance(a: t.Sequence[float], b: t.Sequence[float]):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def euclidean_distance_squared_heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def manhattan_distance(a, b, c_cost=1.0):
    return c_cost * (abs(b[0] - a[0]) + abs(b[1] - a[1]))


def chebyshev_distance(a, b, c_cost=1.0, d_cost=SQRT_OF_2):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return c_cost * (dx + dy) + (d_cost - 2.0 * c_cost) * min(dx, dy)


def sum_of_euclidean_distances(poses):
    if len(poses) == 0:
        return float("inf")
    if len(poses) == 1:
        return 0.0

    total = 0.0
    prev_pose = poses[0]
    for cur_pose in poses[1 : len(poses)]:
        total += euclidean_distance(cur_pose, prev_pose)
        prev_pose = cur_pose

    return total


def get_neighbors(cell, width, height, neighborhood=TAXI_NEIGHBORHOOD):
    neighbors = set()
    for i, j in neighborhood:
        neighbor = cell[0] + i, cell[1] + j
        if is_in_matrix(neighbor, width, height):
            neighbors.add(neighbor)
    return neighbors


def get_neighbors_no_checks(cell, neighborhood=TAXI_NEIGHBORHOOD):
    return {(cell[0] + i, cell[1] + j) for i, j in neighborhood}


def get_neighbors_no_coll(cell, grid, width, height, neighborhood=TAXI_NEIGHBORHOOD):
    neighbors = set()
    for i, j in neighborhood:
        neighbor = cell[0] + i, cell[1] + j
        if (
            is_in_matrix(neighbor, width, height)
            and grid[neighbor[0]][neighbor[1]] == 0
        ):
            neighbors.add(neighbor)
    return neighbors


def get_set_neighbors(
    cell_set, width, height, neighborhood=TAXI_NEIGHBORHOOD, previous_cell_set=None
):
    neighbor_set = set()
    for cell in cell_set:
        neighbor_set.update(get_neighbors(cell, width, height, neighborhood))
    neighbor_set.difference_update(cell_set)
    if previous_cell_set is not None:
        neighbor_set.difference_update(previous_cell_set)
    return neighbor_set


def get_set_neighbors_no_coll(
    cell_set, grid, neighborhood=TAXI_NEIGHBORHOOD, previous_cell_set=None
):
    neighbor_set = set()
    width, height = grid.shape
    for cell in cell_set:
        neighbor_set.update(
            get_neighbors_no_coll(cell, grid, width, height, neighborhood)
        )
    neighbor_set.difference_update(cell_set)
    if previous_cell_set is not None:
        neighbor_set.difference_update(previous_cell_set)
    return neighbor_set


def get_set_neighbors_no_checks(cell_set, neighborhood=TAXI_NEIGHBORHOOD):
    neighbor_set = set()
    for cell in cell_set:
        neighbor_set.update(get_neighbors_no_checks(cell, neighborhood))
    neighbor_set.difference_update(cell_set)
    return neighbor_set


def is_in_matrix(cell, width, height):
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def real_to_grid(real_x: float, real_y: float, res: float, grid_pose: Pose2D):
    return int(math.floor((real_x - grid_pose[0]) / res)), int(
        math.floor((real_y - grid_pose[1]) / res)
    )


def grid_to_real(
    cell_x: int, cell_y: int, res: float, grid_pose: Pose2D
) -> t.Tuple[float, float]:
    """
    Converts a grid cell's (x,y) coordinates into continuous world (x,y) coordinates
    """
    return (
        res * float(cell_x) + grid_pose[0] + res * 0.5,
        res * float(cell_y) + grid_pose[1] + res * 0.5,
    )


def real_pose_to_grid_pose(real_pose, res, grid_pose, clamp_angle=None):
    return (
        int(math.floor((real_pose[0] - grid_pose[0]) / res)),
        int(math.floor((real_pose[1] - grid_pose[1]) / res)),
        (
            real_pose[2]
            if clamp_angle is None
            else int(round(real_pose[2] / clamp_angle) * clamp_angle)
        ),
    )


def grid_pose_to_real_pose(grid_pose, res, parent_grid_pose):
    return Pose2D(
        res * float(grid_pose[0]) + parent_grid_pose[0] + res * 0.5,
        res * float(grid_pose[1]) + parent_grid_pose[1] + res * 0.5,
        float(grid_pose[2]),
    )


def real_pose_to_fixed_precision_pose(
    real_pose: Pose2D, trans_mult: float, rot_mult: float
) -> t.Tuple[int, int, int]:
    """
    Takes a regular real-valued pose and converts to an integer-valued pose with a fixed degree of precision
    determined by the given multipler values.
    """
    angle = normalize_angle_degrees(real_pose[2])
    return (
        int(real_pose[0] * trans_mult),
        int(real_pose[1] * trans_mult),
        int(angle * rot_mult),
    )


def yaw_from_direction(
    direction_vector: t.Tuple[float, float], radians: bool = False
) -> float:
    """
    Takes an (x,y) direction vector and converts it to a `yaw` angle in either degrees or radians
    """
    yaw = math.atan2(direction_vector[1], direction_vector[0])
    if radians:
        return yaw
    return math.degrees(yaw)


def direction_from_yaw(yaw, radians=False):
    if radians:
        return math.cos(yaw), math.sin(yaw)
    return math.cos(math.radians(yaw)), math.sin(math.radians(yaw))


def grid_path_to_real_path(grid_path, start_pose, goal_pose, res, grid_pose):
    if not grid_path:
        return []
    real_path = [start_pose]
    previous_pose = start_pose
    for cell in grid_path[1:]:
        real_x, real_y = grid_to_real(cell[0], cell[1], res, grid_pose)
        direction_vector = (real_x - previous_pose[0], real_y - previous_pose[1])
        real_yaw = yaw_from_direction(direction_vector)
        new_pose = (real_x, real_y, real_yaw)
        has_rotation = not angle_is_close(new_pose[2], previous_pose[2], abs_tol=1e-6)
        has_translation = not is_close(
            new_pose[0], previous_pose[0], abs_tol=1e-6
        ) or not is_close(new_pose[1], previous_pose[1], abs_tol=1e-6)

        if has_rotation or has_translation:
            if has_rotation and has_translation:
                real_path.append((previous_pose[0], previous_pose[1], new_pose[2]))
                real_path.append(new_pose)
            else:
                real_path.append(new_pose)
        previous_pose = new_pose

    if goal_pose:
        last_direction_vector = (
            goal_pose[0] - real_path[-1][0],
            goal_pose[1] - real_path[-1][1],
        )
        last_real_yaw = yaw_from_direction(last_direction_vector)
        real_path.append((real_path[-1][0], real_path[-1][1], last_real_yaw))
        real_path.append((goal_pose[0], goal_pose[1], last_real_yaw))
        real_path.append(goal_pose)
    return real_path


def is_within_interchangeable_interval(eval_value, value_a, value_b):
    if value_a <= value_b:
        return value_a <= eval_value <= value_b
    return value_b <= eval_value <= value_a


def is_cells_set_colliding_in_grid(cells_set, grid):
    for cell in cells_set:
        if grid[cell[0]][cell[1]] != 0:
            return True
    return False


def get_circumscribed_radius(polygon: Polygon) -> float:
    return polygon.hausdorff_distance(polygon.centroid)


def get_inscribed_radius(polygon: Polygon) -> float:
    return polygon.centroid.distance(LineString(polygon.exterior.coords))


def get_translation(start_pose: Pose2D, end_pose: Pose2D):
    return end_pose[0] - start_pose[0], end_pose[1] - start_pose[1]


def get_rotation(start_pose: Pose2D, end_pose: Pose2D):
    return normalize_angle_degrees(end_pose[2] - start_pose[2])


def add_angles(a: float, b: float):
    return normalize_angle_degrees(a + b)


def subtract_angles(a: float, b: float):
    return normalize_angle_degrees(a - b)


def get_translation_and_rotation(start_pose: Pose2D, end_pose: Pose2D):
    translation = get_translation(start_pose, end_pose)
    rotation = get_rotation(start_pose, end_pose)
    if math.isnan(rotation):
        pass
    return translation, rotation


def set_polygon_pose(
    polygon: Polygon,
    init_polygon_pose: Pose2D,
    end_polygon_pose: Pose2D,
    rotation_center: t.Union[str, t.Tuple[float, float]] = "center",
) -> Polygon:
    translation, rotation = get_translation_and_rotation(
        init_polygon_pose, end_polygon_pose
    )
    return rotate_then_translate_polygon(
        polygon, translation, rotation, rotation_center
    )


def rotate_then_translate_polygon(
    polygon: Polygon,
    translation: t.Tuple[float, float],
    rotation: float,
    rotation_center: t.Union[str, t.Tuple[float, float]] = "center",
) -> Polygon:
    return affinity.translate(
        affinity.rotate(polygon, rotation, origin=rotation_center),  # type: ignore
        *translation,
    )


def append_suffix(filename, suffix):
    return "{0}_{2}{1}".format(*os.path.splitext(filename) + (suffix,))


def find_circle_terms(x1, y1, x2, y2, x3, y3):
    """
    Computes the circle's center coordinates and radius from three points on the circle.
    Code by Geeksforgeeks user Gyanendra Singh Panwar (gyanendra371), available here:
    https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/.
    Fixed the mistaken "//" operators into plain "/" ones (otherwise the float get cast to int, inducing errors)
    :param x1: x coordinate of first point
    :type x1: float
    :param y1: y coordinate of first point
    :type y1: float
    :param x2: x coordinate of second point
    :type x2: float
    :param y2: y coordinate of second point
    :type y2: float
    :param x3: x coordinate of third point
    :type x3: float
    :param y3: y coordinate of third point
    :type y3: float
    :return: circle's center coordinates (x-axis, then y-axis) and radius
    :rtype: float, float, float
    """
    if x1 == x2 == x3 and y1 == y2 == y3:
        # Manage special case where the point does not move
        return x1, y1, 0.0

    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = (sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) / (
        2 * (y31 * x12 - y21 * x13)
    )

    g = (sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) / (
        2.0 * (x31 * y12 - x21 * y13)
    )

    c = -pow(x1, 2) - pow(y1, 2) - 2.0 * g * x1 - 2.0 * f * y1

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c

    # r is the radius
    r = math.sqrt(sqr_of_r)

    return h, k, r


def points_to_angle(x1, y1, x2, y2, x3, y3):
    """
    Compute angle in radians (< pi !) between three points A(x1, y1), B(x2, y2), C(x3, y3), in this order
    :param x1: x coordinate of first point
    :type x1: float
    :param y1: y coordinate of first point
    :type y1: float
    :param x2: x coordinate of second point
    :type x2: float
    :param y2: y coordinate of second point
    :type y2: float
    :param x3: x coordinate of third point
    :type x3: float
    :param y3: y coordinate of third point
    :type y3: float
    :return: angle between points in radians, is always < pi !
    :rtype: float
    """
    scalar_product = (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)
    product_of_norms = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * math.sqrt(
        (x3 - x2) ** 2 + (y3 - y2) ** 2
    )
    term = scalar_product / product_of_norms
    term = max(-1.0, term)
    term = min(1.0, term)
    return math.acos(term)


def map_bounds(polygons: t.Iterable[Polygon]):
    if not polygons:
        raise ValueError(
            "There are no entities to populate the grid, it can't be created!"
        )

    map_min_x, map_min_y, map_max_x, map_max_y = (
        float("inf"),
        float("inf"),
        -float("inf"),
        -float("inf"),
    )

    for polygon in polygons:
        min_x, min_y, max_x, max_y = polygon.bounds  # type: ignore
        map_min_x, map_min_y = min(map_min_x, min_x), min(map_min_y, min_y)
        map_max_x, map_max_y = max(map_max_x, max_x), max(map_max_y, max_y)
    return map_min_x, map_min_y, map_max_x, map_max_y


def are_points_on_opposite_sides(ax, ay, bx, by, x1, y1, x2, y2):
    """
    Method inspired by answer of Stackoverflow use copper.har at link :
    https://math.stackexchange.com/questions/162728/how-to-determine-if-2-points-are-on-opposite-sides-of-a-line
    :param ax: X coordinate of one of the points
    :type ax: float
    :param ay: Y coordinate of one of the points
    :type ay: float
    :param bx: X coordinate of the other point
    :type bx: float
    :param by: Y coordinate of the other point
    :type by: float
    :param x1: X coordinate of one the line's points
    :type x1: float
    :param y1: Y coordinate of one the line's points
    :type y1: float
    :param x2: X coordinate of the other point of the line
    :type x2: float
    :param y2: Y coordinate of the other point of the line
    :type y2: float
    :return: True if the points are on opposite sides of the line, False otherwise
    :rtype: bool
    """
    return ((y1 - y2) * (ax - x1) + (x2 - x1) * (ay - y1)) * (
        (y1 - y2) * (bx - x1) + (x2 - x1) * (by - y1)
    ) < 0.0


def sample_poses_at_middle_of_inflated_sides(
    polygon: Polygon, dist_from_sides: float, close_to_zero_atol: float = 1e-06
) -> t.List[Pose2D]:
    """
    Computes and returns the manipulation poses that are at a distance dist_from_border from the sides,
    and facing their middle.
    :param dist_from_sides: distance from the obstacle's sides at which the manipulation poses are computed [m]
    :type dist_from_sides: float
    :return: list of manipulation poses
    :rtype: list(tuple(float, float, float))
    """
    poses = []

    # METHOD BY CHANGING CARTESIAN REFERENTIAL
    poly_center = polygon.centroid.coords[0]
    for i in range(len(polygon.exterior.coords) - 1):
        d = dist_from_sides
        x_a, y_a = polygon.exterior.coords[i]  # First side segment point
        x_b, y_b = polygon.exterior.coords[i + 1]  # Second side segment point
        x_m, y_m = ((x_a + x_b) / 2.0, (y_a + y_b) / 2.0)  # Middle of side segment
        norm_a_b = np.linalg.norm([x_b - x_a, y_b - y_a])  # Side segment length
        if norm_a_b != 0.0:
            # Compute candidate manip points obtained by cartesian referential change
            points = [
                (x_m + d * (y_b - y_a) / norm_a_b, y_m + d * (x_b - x_a) / norm_a_b),
                (x_m + d * (y_b - y_a) / norm_a_b, y_m - d * (x_b - x_a) / norm_a_b),
                (x_m - d * (y_b - y_a) / norm_a_b, y_m + d * (x_b - x_a) / norm_a_b),
                (x_m - d * (y_b - y_a) / norm_a_b, y_m - d * (x_b - x_a) / norm_a_b),
            ]
            manip_point = (0.0, 0.0)
            max_dist = 0.0
            # Iterate over candidate manip points to select only the closest one orthogonal to side segment
            for x_r, y_r in points:
                scalar_product = (x_b - x_a) * (x_r - x_m) + (y_b - y_a) * (y_r - y_m)
                if abs(scalar_product - 0.0) <= close_to_zero_atol:
                    norm_r_poly_center = float(
                        np.linalg.norm([poly_center[0] - x_r, poly_center[1] - y_r])
                    )
                    if norm_r_poly_center > max_dist:
                        manip_point = (x_r, y_r)
                        max_dist = norm_r_poly_center

            # Save selected manip point in returned list
            direction = (x_m - manip_point[0], y_m - manip_point[1])
            manip_pose = (manip_point[0], manip_point[1], yaw_from_direction(direction))  # type: ignore
            poses.append(manip_pose)

    return poses


def generate_random_polygon(
    ctr_x: float,
    ctr_y: float,
    ave_radius: float,
    irregularity: float,
    spikeyness: float,
    num_verts: int,
) -> t.List[t.Tuple[float, float]]:
    """
    Random polygon generator copied from Stackoverflow user Mike Ounsworth:
    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
    Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.
    :param ctr_x: polygon center x-coordinate
    :type ctr_x: float
    :param ctr_y: polygon center y-coordinate
    :type ctr_y: float
    :param ave_radius: the average radius of this polygon, this roughly controls how large the polygon is,
        really only useful for order of magnitude.
    :type ave_radius: float
    :param irregularity: [0,1] indicating how much variance there is in the angular spacing of vertices.
        [0,1] will map to [0, 2pi/num_verts]
    :type irregularity: float
    :param spikeyness: [0,1] indicating how much variance there is in each vertex from the circle of radius ave_radius.
        [0,1] will map to [0, ave_radius]
    :type spikeyness: float
    :param num_verts: number of vertices
    :type num_verts: int
    :return: a list of vertices, in counter-clockwise order
    :rtype: list(tuple(float, float))
    """
    irregularity = np.clip(irregularity, 0.0, 1.0) * TWO_PI / num_verts
    spikeyness = np.clip(spikeyness, 0.0, 1.0) * ave_radius

    # generate n angle steps
    angle_steps = []
    lower = (TWO_PI / num_verts) - irregularity
    upper = (TWO_PI / num_verts) + irregularity
    _sum = 0.0
    for i in range(num_verts):
        tmp = random.uniform(lower, upper)
        angle_steps.append(tmp)
        _sum = _sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = _sum / TWO_PI
    for i in range(num_verts):
        angle_steps[i] = angle_steps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0.0, 2.0 * math.pi)
    for i in range(num_verts):
        r_i = np.clip(random.gauss(ave_radius, spikeyness), 0.0, 2.0 * ave_radius)
        x = ctr_x + r_i * math.cos(angle)
        y = ctr_y + r_i * math.sin(angle)
        points.append((x, y))

        angle = angle + angle_steps[i]

    return points


def polygon_to_subgrid_polygon_and_parameters(polygon: Polygon, res, grid_pose):
    # Compute real min point and max point of projected polygon grid-axis-aligned bounding box
    min_x, min_y, max_x, max_y = polygon.bounds  # type: ignore

    # Clamp the values to their appropriate cell
    try:
        min_d_x, min_d_y = (
            int((min_x - grid_pose[0]) / res),
            int((min_y - grid_pose[1]) / res),
        )
        max_d_x, max_d_y = (
            int(math.ceil((max_x - grid_pose[0]) / res)),
            int(math.ceil((max_y - grid_pose[1]) / res)),
        )

        # Compute cell width and height of subgrid
        d_width, d_height = max_d_x - min_d_x + 1, max_d_y - min_d_y + 1

        min_x_bi1s, min_y_bis = (
            grid_pose[0] + res * float(min_d_x),
            grid_pose[1] + res * float(min_d_y),
        )
        subgrid_projected_polygon = affinity.translate(
            polygon, -grid_pose[0] - min_d_x * res, -grid_pose[1] - min_d_y * res
        )

        return subgrid_projected_polygon, d_width, d_height, min_d_x, min_d_y

    except ValueError as e:
        raise e


NORTH_EAST_CORNER_NEIGHBORS = ((0, 1), (1, 1), (1, 0))
NORTH_WEST_CORNER_NEIGHBORS = ((0, 1), (-1, 1), (-1, 0))
SOUTH_WEST_CORNER_NEIGHBORS = ((0, -1), (-1, -1), (-1, 0))
SOUTH_EAST_CORNER_NEIGHBORS = ((0, -1), (1, -1), (1, 0))
NORTH_NEIGBHBOR = ((0, 1),)
SOUTH_NEIGHBOR = ((0, -1),)
EAST_NEIGHBOR = ((1, 0),)
WEST_NEIGHBOR = ((-1, 0),)


def same_side(line_p1, dx, dy, a, b, c, d):
    """
    @param line_p1 first point of the line
    @type line_p1 tuple(float, float)
    @param line_p2 second point of the line
    @type line_p2 tuple(float, float)
    @param a first point to check
    @type a tuple(float, float)
    @param b second point to check
    @type b tuple(float, float)
    """
    a_term = -dy * (a[0] - line_p1[0]) + dx * (a[1] - line_p1[1])
    b_term = -dy * (b[0] - line_p1[0]) + dx * (b[1] - line_p1[1])
    c_term = -dy * (c[0] - line_p1[0]) + dx * (c[1] - line_p1[1])
    d_term = -dy * (d[0] - line_p1[0]) + dx * (d[1] - line_p1[1])
    return a_term * b_term >= 0.0 and a_term * c_term >= 0.0 and a_term * d_term >= 0.0


def shapely_geom_to_local(global_geom, local_cs_pose_in_global):
    translated_geometry = affinity.translate(
        global_geom, -local_cs_pose_in_global[0], -local_cs_pose_in_global[1]
    )
    final_geometry = affinity.rotate(
        translated_geometry,
        angle=-local_cs_pose_in_global[2],
        origin=(0.0, 0.0),  # type: ignore
    )
    return final_geometry


def shapely_geom_to_global(local_geom, local_cs_pose_in_global):
    rotated_geometry = affinity.rotate(
        local_geom,
        angle=local_cs_pose_in_global[2],
        origin=(0.0, 0.0),  # type: ignore
    )
    final_geometry = affinity.translate(
        rotated_geometry, local_cs_pose_in_global[0], local_cs_pose_in_global[1]
    )
    return final_geometry


def coords(polygon: Polygon):
    return polygon.exterior.coords[:-1]


def normalize_angle_radians(radians: float):
    """Normalize angle to [-pi, pi]"""
    return math.atan2(math.sin(radians), math.cos(radians))


def normalize_angle_degrees(degrees: float):
    """Normalize angle to [-180, 180]"""
    radians = math.radians(degrees)
    return math.degrees(normalize_angle_radians(radians))


def is_close(a: float, b: float, abs_tol: float = 1e-09):
    return np.abs(a - b) <= abs_tol


def angle_is_close(a: float, b: float, abs_tol: float = 1e-09):
    diff = abs(normalize_angle_degrees(a - b))
    return diff <= abs_tol


class Circle:
    def __init__(self, x: float, y: float, r: float):
        self.x = x
        self.y = y
        self.r = r

    def intersects(self, x: float, y: float):
        return euclidean_distance((self.x, self.y), (x, y)) <= self.r

    def tuple_intersects(self, position: VertexModel):
        return euclidean_distance((self.x, self.y), position) <= self.r


def cmp(a: float | int, b: float | int):
    return (a > b) - (a < b)


def rotate_2d_vector(vector: t.Tuple[float, float], degrees: float):
    radians = math.radians(degrees)
    a, b = vector
    x = a * math.cos(radians) - b * math.sin(radians)
    y = a * math.sin(radians) + b * math.cos(radians)
    return (x, y)


def signed_angle_between(v1: npt.NDArray[t.Any], v2: npt.NDArray[t.Any]) -> float:
    # Normalize the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    v1_norm = v1 / norm_v1
    v2_norm = v2 / norm_v2

    # Calculate the dot product
    dot = max(-1.0, min(1.0, np.dot(v1_norm, v2_norm)))

    # Calculate the cross product
    cross = np.cross(v1_norm, v2_norm)

    # Determine the sign of the angle
    if abs(cross) < 1e-10:  # Handle collinear vectors
        if dot < -0.999999:  # Vectors are opposite
            return 180.0
        elif dot > 0.999999:  # Vectors are aligned
            return 0.0
        else:
            return 0.0  # Fallback for numerical stability

    sign = np.sign(cross)

    # Combine the dot product and sign to get the signed angle
    signed_angle = np.arccos(dot) * sign

    if math.isnan(signed_angle):
        raise Exception("Angle is NaN")

    return np.degrees(normalize_angle_radians(signed_angle))


def get_angle_to_turn(a: t.Sequence[float], b: t.Sequence[float]) -> float:
    """Computes the number of degrees a robot in pose `a` must rotate to be facing pose `b`"""
    v_a = np.array(rotate_2d_vector((1, 0), a[2]))
    v_a_to_b = np.array((b[0] - a[0], b[1] - a[1]))
    angle = signed_angle_between(v_a, v_a_to_b)
    return angle


def path_to_polygon(
    points: t.List[npt.NDArray[np.float32]],
    line_width: float,
) -> Polygon:
    """Converts a sequence of points representing a navigation path into a polygonal "line strip".

    :param points: A sequence of points
    :type points: t.List[npt.NDArray[np.float32]
    :param line_width: width to use for the polygonal line strip
    :type line_width: float
    :raises Exception: if less than two points are in the path
    :return: a polygonal line strip
    :rtype: Polygon
    """

    if len(points) < 2:
        raise Exception("Less than two points")

    # remove z-coord, if any
    points = [x[:2] for x in points]

    # remove duplicates
    dedup_points = []
    seen = set()
    for p in points:
        p = (p[0], p[1])
        if p not in seen:
            seen.add(p)
            dedup_points.append(p)

    linestr = LineString(coordinates=dedup_points)
    buf = linestr.buffer(distance=line_width / 2.0)
    return t.cast(Polygon, buf)


def generate_distinct_colors(num_colors: int):
    # Generate evenly spaced values for hue
    hues = np.linspace(0, 1, num_colors + 1)[:-1]

    # Convert HSV to RGB
    hsv_colors = [(hue, 0.7, 0.9) for hue in hues]  # Saturation and value are fixed
    rgb_colors = list(map(lambda x: colors.hsv_to_rgb(x), hsv_colors))  # type: ignore

    # Convert RGB to hex
    hex_colors = [colors.to_hex(color) for color in rgb_colors]  # type: ignore

    return hex_colors


def hash_to_32_bit_int(s: str) -> int:
    # Get the hash value of the string using Python's built-in hash function
    hash_value = hash(s)

    # Convert the hash value to a 32-bit integer
    # Using bitwise AND with 0xFFFFFFFF to truncate to 32 bits
    bit_mask = 0xFFFFFFFF  # This is 2**32 - 1
    truncated_hash = hash_value & bit_mask

    # If the result is a negative number, convert it to unsigned 32-bit integer
    # by adding 2**32 to it
    if truncated_hash & (1 << 31):
        truncated_hash = -((truncated_hash ^ bit_mask) + 1)

    return truncated_hash


def get_box_orientation(box: Polygon) -> float:
    bbox = box.minimum_rotated_rectangle

    if isinstance(bbox, LineString):
        coords = np.array(bbox.coords)
    else:
        coords = np.array(bbox.exterior.coords)  # type: ignore

    # Get the vectors along the sides of the bounding box
    edges = np.diff(coords, axis=0)

    # Find the angle of the minimum area bounding box by taking the angle of the longer side
    angle = np.arctan2(edges[0, 1], edges[0, 0])

    return angle


def distance_between_poses(a: Pose2D, b: Pose2D) -> float:
    """
    Calculate the distance between two poses, considering position and orientation.

    Args:
        pose1: Tuple of (x, y, theta) where theta is in degrees
        pose2: Tuple of (x, y, theta) where theta is in degrees

    Returns:
        float: Weighted distance combining position and orientation differences
    """
    # Extract coordinates and angles
    x1, y1, theta1 = a
    x2, y2, theta2 = b

    # Euclidean distance between positions
    position_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Calculate smallest angle difference (in radians)
    theta1_rad = math.radians(theta1)
    theta2_rad = math.radians(theta2)
    angle_diff = abs((theta1_rad - theta2_rad + math.pi) % (2 * math.pi) - math.pi)

    # Combine position and orientation distances with weights
    # Weight of 0.1 for angle (in radians) balances it with position (in meters)
    total_distance = position_distance + 0.1 * angle_diff

    return total_distance


def save_image(image, path):
    # Extract the directory path from the full file path
    directory = os.path.dirname(path)

    # Create directories if they don't exist
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save the image
    image.save(path)
