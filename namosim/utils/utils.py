import json
import math
import os
import random
import typing as t
from collections.abc import MutableSet
from datetime import datetime

import mapbox_earcut as earcut
import numpy as np
import shapely.affinity as affinity
import typing_extensions as tx
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon

from namosim.data_models_v2 import PoseModel, VertexModel

# Constants
SQRT_OF_2 = math.sqrt(2.0)
SQRT_OF_2_MIN_1 = SQRT_OF_2 - 1.0
SQRT_OF_2_MIN_2 = SQRT_OF_2 - 2.0
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

OMNI_ROBOT_TAXI_TRANS_VECTORS = TAXI_NEIGHBORHOOD
OMNI_ROBOT_TAXI_ROT_ANGLES = (90.0, 180.0, 270.0, -90.0, -180, -270.0)
OMNI_ROBOT_CHESSBOARD_TRANS_VECTORS = CHESSBOARD_NEIGHBORHOOD
OMNI_ROBOT_CHESSBOARD_ROT_ANGLES = OMNI_ROBOT_TAXI_ROT_ANGLES

DIFF_ROBOT_TAXI_TRANS_VECTORS = ((1, 0),)
DIFF_ROBOT_TAXI_ROT_ANGLES = OMNI_ROBOT_TAXI_ROT_ANGLES
DIFF_ROBOT_CHESSBOARD_TRANS_VECTORS = DIFF_ROBOT_TAXI_TRANS_VECTORS
DIFF_ROBOT_CHESSBOARD_ROT_ANGLES = OMNI_ROBOT_CHESSBOARD_ROT_ANGLES

ROBOT_ANGLES_AT_60 = (
    60.0,
    120.0,
    180.0,
    240.0,
    300.0,
    -60.0,
    -120.0,
    -180.0,
    -240.0,
    -300.0,
)

ROBOT_ANGLES_AT_45 = (
    45.0,
    90.0,
    135.0,
    180.0,
    225.0,
    270.0,
    315.0,
    -45.0,
    -90.0,
    -135.0,
    -180.0,
    -225.0,
    -270.0,
    -315.0,
)

ROBOT_ANGLES_AT_30 = (
    30.0,
    60.0,
    90.0,
    120.0,
    150.0,
    180.0,
    210.0,
    240.0,
    270.0,
    300.0,
    330.0,
    -30.0,
    -60.0,
    -90.0,
    -120.0,
    -150.0,
    -180.0,
    -210.0,
    -240.0,
    -270.0,
    -300.0,
    -330.0,
)

ROBOT_ANGLES_AT_15 = (
    15.0,
    30.0,
    45.0,
    60.0,
    75.0,
    90.0,
    105.0,
    120.0,
    135.0,
    150.0,
    165.0,
    180.0,
    195.0,
    210.0,
    225.0,
    240.0,
    255.0,
    270.0,
    285.0,
    300.0,
    315.0,
    330.0,
    345.0 - 15.0,
    -30.0,
    -45.0,
    -60.0,
    -75.0,
    -90.0,
    -105.0,
    -120.0,
    -135.0,
    -150.0,
    -165.0,
    -180.0,
    -195.0,
    -210.0,
    -225.0,
    -240.0,
    -255.0,
    -270.0,
    -285.0,
    -300.0,
    -315.0,
    -330.0,
    -345.0,
)

ROBOT_ANGLES_AT_10 = (
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    110.0,
    120.0,
    130.0,
    140.0,
    150.0,
    160.0,
    170.0,
    180.0,
    190.0,
    200.0,
    210.0,
    220.0,
    230.0,
    240.0,
    250.0,
    260.0,
    270.0,
    280.0,
    290.0,
    300.0,
    310.0,
    320.0,
    330.0,
    340.0,
    350.0,
    -10.0,
    -20.0,
    -30.0,
    -40.0,
    -50.0,
    -60.0,
    -70.0,
    -80.0,
    -90.0,
    -100.0,
    -110.0,
    -120.0,
    -130.0,
    -140.0,
    -150.0,
    -160.0,
    -170.0,
    -180.0,
    -190.0,
    -200.0,
    -210.0,
    -220.0,
    -230.0,
    -240.0,
    -250.0,
    -260.0,
    -270.0,
    -280.0,
    -290.0,
    -300.0,
    -310.0,
    -320.0,
    -330.0,
    -340.0,
    -350.0,
)

ROBOT_ANGLES_AT_5 = (
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
    95.0,
    100.0,
    105.0,
    110.0,
    115.0,
    120.0,
    125.0,
    130.0,
    135.0,
    140.0,
    145.0,
    150.0,
    155.0,
    160.0,
    165.0,
    170.0,
    175.0,
    180.0,
    185.0,
    190.0,
    195.0,
    200.0,
    205.0,
    210.0,
    215.0,
    220.0,
    225.0,
    230.0,
    235.0,
    240.0,
    245.0,
    250.0,
    255.0,
    260.0,
    265.0,
    270.0,
    275.0,
    280.0,
    285.0,
    290.0,
    295.0,
    300.0,
    305.0,
    310.0,
    315.0,
    320.0,
    325.0,
    330.0,
    335.0,
    340.0,
    345.0,
    350.0,
    355.0,
    -5.0,
    -10.0,
    -15.0,
    -20.0,
    -25.0,
    -30.0,
    -35.0,
    -40.0,
    -45.0,
    -50.0,
    -55.0,
    -60.0,
    -65.0,
    -70.0,
    -75.0,
    -80.0,
    -85.0,
    -90.0,
    -95.0,
    -100.0,
    -105.0,
    -110.0,
    -115.0,
    -120.0,
    -125.0,
    -130.0,
    -135.0,
    -140.0,
    -145.0,
    -150.0,
    -155.0,
    -160.0,
    -165.0,
    -170.0,
    -175.0,
    -180.0,
    -185.0,
    -190.0,
    -195.0,
    -200.0,
    -205.0,
    -210.0,
    -215.0,
    -220.0,
    -225.0,
    -230.0,
    -235.0,
    -240.0,
    -245.0,
    -250.0,
    -255.0,
    -260.0,
    -265.0,
    -270.0,
    -275.0,
    -280.0,
    -285.0,
    -290.0,
    -295.0,
    -300.0,
    -305.0,
    -310.0,
    -315.0,
    -320.0,
    -325.0,
    -330.0,
    -335.0,
    -340.0,
    -345.0,
    -350.0,
    -355.0,
)

DIRECTIONS = [["NW", "N", "NE"], ["W", "X", "E"], ["SW", "S", "SE"]]

HALF_ONE_UP_TIMES = (0.45, 0.70, 0.90, 1.20)


def timestamp_string():
    return datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f")


class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]  # sentinel node for doubly linked list
        self.map = {}  # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError("set is empty")
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))

    def __eq__(self, other: tx.Self | t.Iterable[t.Any]):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


class BasicLog:
    def __init__(self, message, step, timestamp=timestamp_string()):
        self.message = message
        self.step = step
        self.timestamp = timestamp

    def __str__(self):
        # return "At step {}: '{}' - Timestamp: {}".format(self.step, self.message, self.timestamp)
        return "At step {}: '{}'".format(self.step, self.message)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class CustomLogger(list[BasicLog]):
    def __init__(self, printout: bool = True):
        super(CustomLogger, self).__init__(self)
        self.printout = printout

    def append(self, log: BasicLog):
        super(CustomLogger, self).append(log)
        if self.printout:
            print(log)


def euclidean_distance(a, b):
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
    elif len(poses) == 1:
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
    # # width_m_1, height_m_1= width - 1, height - 1
    # if 0 < cell[0] < width - 1:
    #     if 0 < cell[1] < height - 1:
    #         # If cell in grid center, return all neighbors in neighborhood
    #         return {(cell[0] + i, cell[1] + j) for i, j in neighborhood}
    #     elif cell[1] == 0:
    #         # If cell in top row
    #         pass
    #     elif cell[1] == height - 1:
    #         pass
    #     else:
    #         return set()
    # elif cell[0] == 0:
    #     pass
    # elif cell[0] == width - 1:
    #     pass
    # else:
    #     return set()

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


def real_to_grid(real_x: float, real_y: float, res: float, grid_pose: PoseModel):
    return int(math.floor((real_x - grid_pose[0]) / res)), int(
        math.floor((real_y - grid_pose[1]) / res)
    )


def grid_to_real(
    cell_x: int, cell_y: int, res: float, grid_pose: PoseModel
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
        real_pose[2]
        if clamp_angle is None
        else int(round(real_pose[2] / clamp_angle) * clamp_angle),
    )


def grid_pose_to_real_pose(grid_pose, res, parent_grid_pose):
    return (
        res * float(grid_pose[0]) + parent_grid_pose[0] + res * 0.5,
        res * float(grid_pose[1]) + parent_grid_pose[1] + res * 0.5,
        float(grid_pose[2]),
    )


def real_pose_to_fixed_precision_pose(
    real_pose: PoseModel, trans_mult: float, rot_mult: float
) -> t.Tuple[int, int, int]:
    """
    Takes a regular real-valued pose and converts to an integer-valued pose with a fixed degree of precision
    determined by the given multipler values.
    """
    return (
        round(real_pose[0] * trans_mult),
        round(real_pose[1] * trans_mult),
        round(real_pose[2] * rot_mult),
    )


def yaw_from_direction(
    direction_vector: t.Tuple[float, float], radians: bool = False
) -> float:
    """Takes an (x,y) direction vector and converts it to a `yaw` angle in either degrees or radians"""
    yaw = math.atan2(direction_vector[1], direction_vector[0])
    if radians:
        return yaw
    else:
        return math.degrees(yaw)


def direction_from_yaw(yaw, radians=False):
    if radians:
        return math.cos(yaw), math.sin(yaw)
    else:
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
        has_rotation = not angle_is_close(new_pose[2], previous_pose[2], rel_tol=1e-6)
        has_translation = not is_close(
            new_pose[0], previous_pose[0], rel_tol=1e-6
        ) or not is_close(new_pose[1], previous_pose[1], rel_tol=1e-6)

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
    else:
        return value_b <= eval_value <= value_a


def is_cells_set_colliding_in_grid(cells_set, grid):
    for cell in cells_set:
        if grid[cell[0]][cell[1]] != 0:
            return True
    return False


# region DEPRECATED
def polygon_to_grid(polygon, res, fill=True):
    # Compute real min point and max point of polygon bounding box (subgrid)
    min_x, min_y, max_x, max_y = polygon.bounds

    # Compute real width and height of subgrid
    width, height = max_x - min_x, max_y - min_y

    # Compute cell width and height of subgrid
    d_width, d_height = int(round(width / res)), int(round(height / res))

    # Use PIL to discretize polygon
    # - Create PIL image
    img = Image.new("L", (d_width, d_height), 0)
    # - Transform real polygon coordinates in image coordinate system
    poly_coordinates_in_image = [
        ((x - min_x) / res, (y - min_y) / res) for x, y in polygon.exterior.coords
    ]
    # - Discretize polygon into image
    ImageDraw.Draw(img).polygon(
        poly_coordinates_in_image, outline=1, fill=1 if fill else 0
    )
    # - Transform image back into polygon coordinate system
    subgrid = np.flipud(np.rot90(np.array(img)))

    return subgrid, (min_x, min_y, 0.0)


def subgrid_to_discrete_cells_set(
    subgrid, subgrid_pose, res, grid_pose, grid_d_width, grid_d_height
):
    # Compute subgrid corner coordinate in parent grid
    d_min_x, d_min_y = real_to_grid(subgrid_pose[0], subgrid_pose[1], res, grid_pose)

    x_coords, y_coords = np.where(subgrid == 1)
    x_coords += d_min_x
    y_coords += d_min_y
    unchecked_cells = zip(x_coords, y_coords)
    discrete_cells_set = {
        cell
        for cell in unchecked_cells
        if is_in_matrix(cell, grid_d_width, grid_d_height)
    }

    return discrete_cells_set


# endregion


def reference_polygon_to_discrete_cells_set(
    polygon, res, grid_pose, grid_d_width, grid_d_height, fill=True
):
    subgrid, subgrid_min_x, subgrid_min_y = reference_polygon_to_subgrid(
        polygon, res, grid_pose, fill
    )
    cells_set = reference_subgrid_to_grid_cells_set(
        subgrid, subgrid_min_x, subgrid_min_y, grid_d_width, grid_d_height
    )
    return cells_set


def reference_subgrid_to_grid_cells_set(
    subgrid, subgrid_min_x, subgrid_min_y, grid_d_width, grid_d_height
):
    x_coords, y_coords = np.where(subgrid == 1)
    x_coords += subgrid_min_x
    y_coords += subgrid_min_y
    unchecked_cells = zip(x_coords, y_coords)
    discrete_cells_set = {
        cell
        for cell in unchecked_cells
        if is_in_matrix(cell, grid_d_width, grid_d_height)
    }
    return discrete_cells_set


def reference_subgrid_to_cells_set(subgrid):
    x_coords, y_coords = np.where(subgrid == 1)
    cells = set(zip(x_coords, y_coords))
    return cells


def reference_polygon_to_subgrid(polygon, res, grid_pose, fill=True):
    # TODO implement rotation when it may prove useful

    # Compute real min point and max point of projected polygon grid-axis-aligned bounding box
    min_x, min_y, max_x, max_y = polygon.bounds

    # Clamp the values to their appropriate cell
    min_d_x, min_d_y = (
        int(math.floor((min_x - grid_pose[0]) / res)),
        int(math.floor((min_y - grid_pose[1]) / res)),
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

    new_subgrid = np.zeros((d_width, d_height), dtype=int)
    # For each cell in subgrid, create a shapely square polygon and check
    for i in range(d_width):
        for j in range(d_height):
            coordinates = [
                (i * res, j * res),
                ((i + 1) * res, j * res),
                ((i + 1) * res, (j + 1) * res),
                (i * res, (j + 1) * res),
            ]
            cell_poly = Polygon(coordinates)
            if cell_poly.intersects(subgrid_projected_polygon):
                new_subgrid[i][j] = 1

    return new_subgrid, min_d_x, min_d_y


# def get_circumscribed_radius(polygon):
#     center = list(polygon.centroid.coords)[0]
#     points = list(polygon.exterior.coords)
#     circumscribed_radius = 0.
#     for point in points:
#         circumscribed_radius = max(circumscribed_radius, euclidean_distance(center, point))
#     return circumscribed_radius


def get_circumscribed_radius(polygon: Polygon) -> float:
    return polygon.hausdorff_distance(polygon.centroid)


# def get_inscribed_radius(polygon):
#     center = list(polygon.centroid.coords)[0]
#     points = list(polygon.exterior.coords)
#     inscribed_radius = euclidean_distance(center, points[0])
#     for i in range(len(points) - 1):
#         point_a, point_b = points[i], points[i + 1]
#         inscribed_radius = min(inscribed_radius, euclidean_distance(center, point_b))
#         middle_point = ((point_a[0] + point_b[0]) / 2., (point_a[1] + point_b[1]) / 2.)
#         inscribed_radius = min(inscribed_radius, euclidean_distance(center, middle_point))
#
#     return inscribed_radius


def get_inscribed_radius(polygon):
    return polygon.centroid.distance(LineString(polygon.exterior.coords))


def get_inscribed_square_sidelength(radius):
    return math.sqrt(radius**2 * 2)


def get_translation(start_pose, end_pose):
    return end_pose[0] - start_pose[0], end_pose[1] - start_pose[1]


def get_rotation(start_pose, end_pose):
    return angle_to_360_interval(end_pose[2] - start_pose[2])


def get_translation_and_rotation(start_pose, end_pose):
    translation = get_translation(start_pose, end_pose)
    rotation = get_rotation(start_pose, end_pose)
    return translation, rotation


def set_polygon_pose(
    polygon, init_polygon_pose, end_polygon_pose, rotation_center="center"
):
    translation, rotation = get_translation_and_rotation(
        init_polygon_pose, end_polygon_pose
    )
    return rotate_then_translate_polygon(
        polygon, translation, rotation, rotation_center
    )


def rotate_then_translate_polygon(
    polygon, translation, rotation, rotation_center="center"
):
    return affinity.translate(
        affinity.rotate(polygon, rotation, origin=rotation_center), *translation
    )


def polygon_collides_with_entities(polygon, entities, aabb_tree=None):
    for entity in entities:
        if entity.polygon.intersects(polygon):
            return True
    return False


def append_suffix(filename, suffix):
    return "{0}_{2}{1}".format(*os.path.splitext(filename) + (suffix,))


def shapely_polygon_to_shapely_triangles(polygon):
    return [
        Polygon(triangle_coords)
        for triangle_coords in shapely_polygon_to_triangles_coords(polygon)
    ]


def shapely_polygon_to_triangles_coords(polygon):
    return polygon_coords_to_triangles_coords(list(polygon.exterior.coords))


def polygon_coords_to_triangles_coords(polygon):
    verts = np.array(polygon).reshape(-1, 2)
    rings = np.array([verts.shape[0]])
    triangles_vertices = verts[earcut.triangulate_float64(verts, rings)]
    triangles_vertices_as_tuples = [
        tuple(triangle_vertices) for triangle_vertices in triangles_vertices
    ]
    triangles = [
        triangles_vertices_as_tuples[n : n + 3]
        for n in range(0, len(triangles_vertices_as_tuples), 3)
    ]
    return triangles


def is_convex(polygon: Polygon):
    return polygon.convex_hull.equals(polygon)


def convert_to_convex_polygons_list(polygon):
    if is_convex(polygon):
        return [polygon]
    else:
        return shapely_polygon_to_shapely_triangles(polygon)


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
        min_x, min_y, max_x, max_y = polygon.bounds
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
    polygon, dist_from_sides, close_to_zero_atol=1e-06
):
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
            manip_pose = (manip_point[0], manip_point[1], yaw_from_direction(direction))
            poses.append(manip_pose)

    return poses


def generate_random_polygon(
    ctr_x, ctr_y, ave_radius, irregularity, spikeyness, num_verts
):
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


def polygon_to_subgrid_polygon_and_parameters(polygon, res, grid_pose):
    # Compute real min point and max point of projected polygon grid-axis-aligned bounding box
    min_x, min_y, max_x, max_y = polygon.bounds

    # Clamp the values to their appropriate cell
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


def accurate_rasterize_to_cells(projected_polygon, d_width, d_height, res, fill=True):
    projected_poly_coords = list(projected_polygon.exterior.coords)
    cells = set()
    # subgrid = np.zeros((d_width, d_height), dtype=np.uint8)  # DEBUG

    point_iter = iter(projected_poly_coords)
    prev_point = next(point_iter)
    for cur_point in point_iter:
        start_cell = int((prev_point[0]) / res), int((prev_point[1]) / res)
        if is_in_matrix(start_cell, d_width, d_height):
            cells.add(start_cell)
            # subgrid[start_cell[0]][start_cell[1]] = 1  # DEBUG
        end_cell = int((cur_point[0]) / res), int((cur_point[1]) / res)
        if is_in_matrix(end_cell, d_width, d_height):
            cells.add(end_cell)
            # subgrid[end_cell[0]][end_cell[1]] = 1  # DEBUG

        if start_cell == end_cell:
            prev_point = cur_point
            continue

        dx, dy = cur_point[0] - prev_point[0], cur_point[1] - prev_point[1]
        if dx > 0:
            if dy > 0:
                neighbors = NORTH_EAST_CORNER_NEIGHBORS
            elif dy < 0:
                neighbors = SOUTH_EAST_CORNER_NEIGHBORS
            else:
                # dy == 0
                neighbors = EAST_NEIGHBOR
        elif dx < 0:
            if dy > 0:
                neighbors = NORTH_WEST_CORNER_NEIGHBORS
            elif dy < 0:
                neighbors = SOUTH_WEST_CORNER_NEIGHBORS
            else:
                # dy == 0
                neighbors = WEST_NEIGHBOR
        else:
            # dx == 0
            if dy > 0:
                neighbors = NORTH_NEIGBHBOR
            elif dy < 0:
                neighbors = SOUTH_NEIGHBOR
            else:
                # dy == 0
                prev_point = cur_point
                continue

        current_cells_to_visit = [
            (start_cell[0] + i, start_cell[1] + j) for i, j in neighbors
        ]
        next_cells_to_visit = []
        found_end_cell = end_cell in current_cells_to_visit

        if (
            found_end_cell
            and (end_cell[0] - start_cell[0], end_cell[1] - start_cell[1])
            in TAXI_NEIGHBORHOOD
        ):
            # If we have two neighbouring cells in 4-connectivity, do not try to add more cells or
            # it will generate noise in rounded corners (because the algorithm will evaluate cells that are
            # beyond the end cell one and it should not in this case).
            prev_point = cur_point
            continue

        while current_cells_to_visit:
            cur_cell = current_cells_to_visit.pop(0)

            if is_in_matrix(cur_cell, d_width, d_height):
                a = cur_cell[0] * res, cur_cell[1] * res
                c = a[0] + res, a[1] + res
                b = c[0], a[1]
                d = a[0], c[1]
                if not same_side(prev_point, dx, dy, a, b, c, d):
                    cells.add(cur_cell)
                    # subgrid[cur_cell[0]][cur_cell[1]] = 1  # DEBUG
                    next_cells_to_visit += [
                        (cur_cell[0] + i, cur_cell[1] + j) for i, j in neighbors
                    ]

            if not found_end_cell:
                found_end_cell = cur_cell == end_cell

            if not current_cells_to_visit and not found_end_cell:
                current_cells_to_visit = next_cells_to_visit
                next_cells_to_visit = []

        prev_point = cur_point

    if fill:
        # custom_fill_start = time.time()
        # all_cells_in_subgrid = [(i, j) for i in range(d_width) for j in range(d_height)]
        # corners_to_check = [(i * res, j * res) for i, j in all_cells_in_subgrid]
        # projected_poly_coords = list(projected_polygon.exterior.coords)
        # poly_path = Path(projected_poly_coords)
        # mask = poly_path.contains_points(corners_to_check)
        # cells.update({cell for cell, is_inside_polygon in zip(all_cells_in_subgrid, mask) if is_inside_polygon})
        # custom_fill_duration = time.time() - custom_fill_start

        # Use PIL to compute fill, it's 20x faster than naive custom implementation above, and 10x faster than Skimage
        # - Create PIL image
        # pil_fill_start = time.time()
        img = Image.new("L", (d_width, d_height), 0)
        # - Transform real polygon coordinates in image coordinate system
        poly_coordinates_in_image = [
            (x / res, y / res) for x, y in projected_poly_coords
        ]
        # - Discretize polygon into image
        ImageDraw.Draw(img).polygon(poly_coordinates_in_image, outline=1, fill=1)
        # - Transform image back into polygon coordinate system
        subgrid = np.flipud(np.rot90(np.array(img, dtype=np.uint8)))
        x_coords, y_coords = np.where(subgrid == 1)
        cells.update(set(zip(x_coords, y_coords)))
        # pill_fill_duration = time.time() - pil_fill_start

    return cells


def accurate_rasterize_to_subgrid(projected_polygon, d_width, d_height, res, fill=True):
    subgrid = np.zeros((d_width, d_height), dtype=np.uint8)
    cells = accurate_rasterize_to_cells(projected_polygon, d_width, d_height, res)
    for cell in cells:
        subgrid[cell[0]][cell[1]] = 1
    return subgrid


def accurate_rasterize_in_grid(
    polygon: Polygon,
    res: float,
    grid_pose: PoseModel,
    d_width: int,
    d_height: int,
    fill: bool = True,
):
    (
        projected_polygon,
        subgrid_d_width,
        subgrid_d_height,
        subgrid_min_d_x,
        subgrid_min_d_y,
    ) = polygon_to_subgrid_polygon_and_parameters(polygon, res, grid_pose)
    subgrid_cells = accurate_rasterize_to_cells(
        projected_polygon, subgrid_d_width, subgrid_d_height, res, fill
    )
    grid_cells = set()
    for cell in subgrid_cells:
        grid_cell = (cell[0] + subgrid_min_d_x, cell[1] + subgrid_min_d_y)
        if is_in_matrix(grid_cell, d_width, d_height):
            grid_cells.add(grid_cell)
    return grid_cells


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


def angle_to_360_interval(angle: float):
    final_angle = angle % 360.0
    final_angle = final_angle if final_angle >= 0.0 else final_angle + 360.0
    return final_angle


def is_close(a: float, b: float, rel_tol: float = 1e-09):
    return b - rel_tol <= a <= b + rel_tol or a - rel_tol <= b <= a + rel_tol


def angle_is_close(a: float, b: float, rel_tol: float = 1e-09):
    return (
        is_close(a, b, rel_tol)
        or is_close(a - 360.0, b, rel_tol)
        or is_close(a, b - 360.0, rel_tol)
    )


# def circle_to_cells(x, y, r, res, grid_pose, neighborhood=CHESSBOARD_NEIGHBORHOOD):
#     start_cell = real_to_grid(x, y, res, grid_pose)
#


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


def get_ros_version():
    import rospkg

    # Initialize the ROS package database
    rospack = rospkg.RosPack()

    # Check if the 'rospy' package (ROS1) is available
    if rospack.get_manifest("rospy"):
        return "ROS1"

    # Check if the 'rclpy' package (ROS2) is available
    if rospack.get_manifest("rclpy"):
        return "ROS2"

    return None
