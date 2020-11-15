import math
from matplotlib.path import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import os
import shapely.affinity as affinity
import mapbox_earcut as earcut
from shapely.geometry import Polygon
import time
import rasterio.features
import skimage.draw

# Constants
SQRT_OF_2 = math.sqrt(2.)
SQRT_OF_2_MIN_1 = SQRT_OF_2 - 1.
SQRT_OF_2_MIN_2 = SQRT_OF_2 - 2.
TWO_PI = 2. * math.pi


TAXI_NEIGHBORHOOD = ((0, 1), (0, -1), (1, 0), (-1, 0))
CHESSBOARD_NEIGHBORHOOD = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
CHESSBOARD_NEIGHBORHOOD_EXTRAS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

OMNI_ROBOT_TAXI_TRANS_VECTORS = TAXI_NEIGHBORHOOD
OMNI_ROBOT_TAXI_ROT_ANGLES = (90., 180., 270.,
                              -90., -180, -270.)
OMNI_ROBOT_CHESSBOARD_TRANS_VECTORS = CHESSBOARD_NEIGHBORHOOD
OMNI_ROBOT_CHESSBOARD_ROT_ANGLES = OMNI_ROBOT_TAXI_ROT_ANGLES

DIFF_ROBOT_TAXI_TRANS_VECTORS = ((1, 0),)
DIFF_ROBOT_TAXI_ROT_ANGLES = OMNI_ROBOT_TAXI_ROT_ANGLES
DIFF_ROBOT_CHESSBOARD_TRANS_VECTORS = DIFF_ROBOT_TAXI_TRANS_VECTORS
DIFF_ROBOT_CHESSBOARD_ROT_ANGLES = OMNI_ROBOT_CHESSBOARD_ROT_ANGLES

ROBOT_ANGLES_AT_60 = (
    60.0, 120.0, 180.0, 240.0, 300.0,
    -60.0, -120.0, -180.0, -240.0, -300.0
)

ROBOT_ANGLES_AT_45 = (
    45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0,
    -45.0, -90.0, -135.0, -180.0, -225.0, -270.0, -315.0
)

ROBOT_ANGLES_AT_30 = (
    30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
    -30.0, -60.0, -90.0, -120.0, -150.0, -180.0, -210.0, -240.0, -270.0, -300.0, -330.0
)

ROBOT_ANGLES_AT_15 = (
    15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0, 195.0, 210.0, 225.0, 240.0,
    255.0, 270.0, 285.0, 300.0, 315.0, 330.0, 345.0
    -15.0, -30.0, -45.0, -60.0, -75.0, -90.0, -105.0, -120.0, -135.0, -150.0, -165.0, -180.0, -195.0, -210.0,
    -225.0, -240.0, -255.0, -270.0, -285.0, -300.0, -315.0, -330.0, -345.0
)

ROBOT_ANGLES_AT_10 = (
    10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0,
    170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0,
    320.0, 330.0, 340.0, 350.0,
    -10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0, -100.0, -110.0, -120.0, -130.0, -140.0,
    -150.0, -160.0, -170.0, -180.0, -190.0, -200.0, -210.0, -220.0, -230.0, -240.0, -250.0, -260.0, -270.0,
    -280.0, -290.0, -300.0, -310.0, -320.0, -330.0, -340.0, -350.0
)

ROBOT_ANGLES_AT_5 = (
    5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0,
    95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0,
    170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 230.0, 235.0, 240.0,
    245.0, 250.0, 255.0, 260.0, 265.0, 270.0, 275.0, 280.0, 285.0, 290.0, 295.0, 300.0, 305.0, 310.0, 315.0,
    320.0, 325.0, 330.0, 335.0, 340.0, 345.0, 350.0, 355.0,
    -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -45.0, -50.0, -55.0, -60.0, -65.0, -70.0, -75.0,
    -80.0, -85.0, -90.0, -95.0, -100.0, -105.0, -110.0, -115.0, -120.0, -125.0, -130.0, -135.0, -140.0, -145.0,
    -150.0, -155.0, -160.0, -165.0, -170.0, -175.0, -180.0, -185.0, -190.0, -195.0, -200.0, -205.0, -210.0,
    -215.0, -220.0, -225.0, -230.0, -235.0, -240.0, -245.0, -250.0, -255.0, -260.0, -265.0, -270.0, -275.0,
    -280.0, -285.0, -290.0, -295.0, -300.0, -305.0, -310.0, -315.0, -320.0, -325.0, -330.0, -335.0, -340.0,
    -345.0, -350.0, -355.0
)

DIRECTIONS = [['NW', 'N', 'NE'],
              ['W', 'X', 'E'],
              ['SW', 'S', 'SE']]

HALF_ONE_UP_TIMES = (0.45, 0.70, 0.90, 1.20)


def euclidean_distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def euclidean_distance_squared_heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def manhattan_distance(a, b, c_cost=1.):
    return c_cost * (abs(b[0] - a[0]) + abs(b[1] - a[1]))


def chebyshev_distance(a, b, c_cost=1., d_cost=SQRT_OF_2):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return c_cost * (dx + dy) + (d_cost - 2. * c_cost) * min(dx, dy)


def sum_of_euclidean_distances(poses):
    if len(poses) == 0:
        return float("inf")
    elif len(poses) == 1:
        return 0.

    total = 0.
    prev_pose = poses[0]
    for cur_pose in poses[1:len(poses)]:
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
        if is_in_matrix(neighbor, width, height) and grid[neighbor[0]][neighbor[1]] == 0:
            neighbors.add(neighbor)
    return neighbors


def get_set_neighbors(cell_set, width, height, neighborhood=TAXI_NEIGHBORHOOD, previous_cell_set=None):
    neighbor_set = set()
    for cell in cell_set:
        neighbor_set.update(get_neighbors(cell, width, height, neighborhood))
    neighbor_set.difference_update(cell_set)
    if previous_cell_set is not None:
        neighbor_set.difference_update(previous_cell_set)
    return neighbor_set


def get_set_neighbors_no_coll(cell_set, grid, neighborhood=TAXI_NEIGHBORHOOD, previous_cell_set=None):
    neighbor_set = set()
    width, height = grid.shape
    for cell in cell_set:
        neighbor_set.update(get_neighbors_no_coll(cell, grid, width, height, neighborhood))
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


def real_to_grid(real_x, real_y, res, grid_pose):
    return int(math.floor((real_x - grid_pose[0]) / res)), int(math.floor((real_y - grid_pose[1]) / res))


def grid_to_real(cell_x, cell_y, res, grid_pose):
    return res * float(cell_x) + grid_pose[0] + res * 0.5, res * float(cell_y) + grid_pose[1] + res * 0.5


def real_pose_to_grid_pose(real_pose, res, grid_pose, clamp_angle=None):
    return (int(math.floor((real_pose[0] - grid_pose[0]) / res)),
            int(math.floor((real_pose[1] - grid_pose[1]) / res)),
            real_pose[2] if clamp_angle is None else int(round(real_pose[2] / clamp_angle) * clamp_angle))


def grid_pose_to_real_pose(grid_pose, res, parent_grid_pose):
    return res * float(grid_pose[0]) + parent_grid_pose[0] + res * 0.5, res * float(grid_pose[1]) + parent_grid_pose[1] + res * 0.5, float(grid_pose[2])


def real_pose_to_fixed_precision_pose(real_pose, trans_mult, rot_mult):
    return (
        round(real_pose[0] * trans_mult),
        round(real_pose[1] * trans_mult),
        round(real_pose[2] * rot_mult)
    )


def yaw_from_direction(direction_vector):
    if direction_vector[1] < 0:
        yaw = 2 * math.pi - math.acos(
            direction_vector[0] / math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
    else:
        yaw = math.acos(
            direction_vector[0] / math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
    return math.degrees(yaw)


def direction_from_yaw(yaw):
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
        real_path.append(new_pose)
        previous_pose = new_pose
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


def matplotlib_show_grid(grid):
    plt.imshow(grid)
    plt.show()


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
    img = Image.new('L', (d_width, d_height), 0)
    # - Transform real polygon coordinates in image coordinate system
    poly_coordinates_in_image = [((x - min_x) / res, (y - min_y) / res) for x, y in polygon.exterior.coords]
    # - Discretize polygon into image
    ImageDraw.Draw(img).polygon(poly_coordinates_in_image, outline=1, fill=1 if fill else 0)
    # - Transform image back into polygon coordinate system
    subgrid = np.flipud(np.rot90(np.array(img)))

    return subgrid, (min_x, min_y, 0.)


def subgrid_to_discrete_cells_set(subgrid, subgrid_pose, res, grid_pose, grid_d_width, grid_d_height):
    # Compute subgrid corner coordinate in parent grid
    d_min_x, d_min_y = real_to_grid(subgrid_pose[0], subgrid_pose[1], res, grid_pose)

    x_coords, y_coords = np.where(subgrid == 1)
    x_coords += d_min_x
    y_coords += d_min_y
    unchecked_cells = zip(x_coords, y_coords)
    discrete_cells_set = {cell for cell in unchecked_cells if is_in_matrix(cell, grid_d_width, grid_d_height)}

    return discrete_cells_set
# endregion


def polygon_to_discrete_cells_set(polygon, res, grid_pose, grid_d_width, grid_d_height, grid_r_width, grid_r_height, fill=True):
    subgrid, subgrid_min_x, subgrid_min_y = polygon_to_subgrid(polygon, res, grid_pose, grid_r_width, grid_r_height, fill)
    cells_set = subgrid_to_cells_set(subgrid, subgrid_min_x, subgrid_min_y, grid_d_width, grid_d_height)
    return cells_set


def subgrid_to_cells_set(subgrid, subgrid_min_x, subgrid_min_y, grid_d_width, grid_d_height):
    x_coords, y_coords = np.where(subgrid == 1)
    x_coords += subgrid_min_x
    y_coords += subgrid_min_y
    unchecked_cells = zip(x_coords, y_coords)
    discrete_cells_set = {cell for cell in unchecked_cells if is_in_matrix(cell, grid_d_width, grid_d_height)}
    return discrete_cells_set


def polygon_to_subgrid(polygon, res, grid_pose, grid_r_width, grid_r_height, fill=True):
    # TODO implement rotation when it may prove useful

    # Compute real min point and max point of projected polygon grid-axis-aligned bounding box
    min_x, min_y, max_x, max_y = polygon.bounds

    # Clamp the values to their appropriate cell
    min_d_x, min_d_y = int(math.floor((min_x - grid_pose[0]) / res)), int(math.floor((min_y - grid_pose[1]) / res))
    max_d_x, max_d_y = int(math.ceil((max_x - grid_pose[0]) / res)), int(math.ceil((max_y - grid_pose[1]) / res))

    # Compute cell width and height of subgrid
    d_width, d_height = max_d_x - min_d_x + 1, max_d_y - min_d_y + 1

    min_x_bi1s, min_y_bis = grid_pose[0] + res * float(min_d_x), grid_pose[1] + res * float(min_d_y)
    subgrid_projected_polygon = affinity.translate(polygon, -grid_pose[0] - min_d_x * res, -grid_pose[1] - min_d_y * res)

    custom_start_time = time.time()
    new_subgrid = np.zeros((d_width, d_height), dtype=int)
    # For each cell in subgrid, create a shapely square polygon and check
    for i in range(d_width):
        for j in range(d_height):
            coordinates = [
                    (i * res, j * res),
                    ((i + 1) * res, j * res),
                    ((i + 1) * res, (j + 1) * res),
                    (i * res, (j + 1) * res)
                ]
            cell_poly = Polygon(coordinates)
            if cell_poly.intersects(subgrid_projected_polygon):
                new_subgrid[i][j] = 1
    custom_duration = time.time() - custom_start_time
    poly_coordinates_in_image = list(subgrid_projected_polygon.exterior.coords)

    # # Use PIL to discretize polygon DEPRECATED BECAUSE NOT TRUSTWORTHY
    # # - Create PIL image
    # pil_start_time = time.time()
    # img = Image.new('L', (d_width, d_height), 0)
    # # - Transform real polygon coordinates in image coordinate system
    # # - Discretize polygon into image
    # ImageDraw.Draw(img).polygon(poly_coordinates_in_image, outline=1, fill=1 if fill else 0)
    # # - Transform image back into polygon coordinate system
    # subgrid = np.flipud(np.rot90(np.array(img)))
    # pil_duration = time.time() - pil_start_time

    # Use matplotlib to discretize polygon
    mpl_start_time = time.time()
    poly_path = Path(poly_coordinates_in_image)
    x_g, y_g = np.mgrid[:d_width, :d_height]
    x, y = (x_g.reshape(-1, 1), y_g.reshape(-1, 1))
    coors = np.hstack((x, y))
    mask = poly_path.contains_points(coors)
    for dx, dy in [(0., 1.), (1., 1.), (1., 0.)]:
        x2, y2 = x + dx, y + dy
        coors = np.hstack((x2, y2))
        mask += poly_path.contains_points(coors)
    mpl_subgrid = mask.reshape(d_width, d_height)
    mpl_duration = time.time() - mpl_start_time

    # # Use rasterio to discretize polygon
    # ras_start_time = time.time()
    # ras_subgrid = rasterio.features.rasterize([subgrid_projected_polygon], (d_width, d_height))
    # ras_duration = time.time() - ras_start_time
    #
    # # Use skimage to discretize polygon
    # sk_start_time = time.time()
    # in_rr, in_cc = skimage.draw.polygon(*zip(*poly_coordinates_in_image), shape=(d_width, d_height))
    # out_rr, out_cc = skimage.draw.polygon_perimeter(*zip(*poly_coordinates_in_image), shape=(d_width, d_height))
    # sk_subgrid = np.zeros((d_width, d_height), dtype=np.uint8)
    # sk_subgrid[in_rr, in_cc] = 1
    # sk_subgrid[out_rr, out_cc] = 1
    # sk_duration = time.time() - sk_start_time
    #
    # # Reduce +
    # c2_start_time = time.time()
    # coordinates_cells = []
    # coordinates_cells_set = set()
    # for real_coords in poly_coordinates_in_image[:-1]:
    #     grid_coords = int(math.floor(real_coords[0] / res)), int(math.floor(real_coords[1] / res))
    #     if grid_coords not in coordinates_cells_set:
    #         coordinates_cells_set.add(grid_coords)
    #         coordinates_cells.append(grid_coords)
    # coordinates_cells.append((int(math.floor(poly_coordinates_in_image[-1][0] / res)), int(math.floor(poly_coordinates_in_image[-1][1] / res))))
    # c2_duration = time.time() - c2_start_time

    return new_subgrid, min_d_x, min_d_y


# def is_point_inside_polygon(point, polygon):
#     int cross = 0
#     for (int i = 0, j = vertices_.size() - 1; i < vertices_.size(); j = i++)
#     i, polygon.coords = 0
#
# """
# bool Polygon::isInside(const Position& point) const
# {
#   int cross = 0;
#   for (int i = 0, j = vertices_.size() - 1; i < vertices_.size(); j = i++) {
#     if ( ((vertices_[i].y() > point.y()) != (vertices_[j].y() > point.y()))
#            && (point.x() < (vertices_[j].x() - vertices_[i].x()) * (point.y() - vertices_[i].y()) /
#             (vertices_[j].y() - vertices_[i].y()) + vertices_[i].x()) )
#     {
#       cross++;
#     }
#   }
#   return bool(cross % 2);
# }
# """


def get_circumscribed_radius(polygon):
    center = list(polygon.centroid.coords)[0]
    points = list(polygon.exterior.coords)
    circumscribed_radius = 0.
    for point in points:
        circumscribed_radius = max(circumscribed_radius, euclidean_distance(center, point))
    return circumscribed_radius


def get_inscribed_radius(polygon):
    center = list(polygon.centroid.coords)[0]
    points = list(polygon.exterior.coords)
    inscribed_radius = float("inf")
    for i in range(len(points) - 1):
        point_a, point_b = points[i], points[i + 1]
        middle_point = ((point_a[0] + point_b[0]) / 2., (point_a[1] + point_b[1]) / 2.)
        inscribed_radius = min((inscribed_radius, euclidean_distance(center, middle_point)))
    return inscribed_radius


def get_inscribed_square_sidelength(radius):
    return math.sqrt(radius ** 2 * 2)


def get_translation(start_pose, end_pose):
    return end_pose[0] - start_pose[0], end_pose[1] - start_pose[1]


def get_rotation(start_pose, end_pose):
    return end_pose[2] - start_pose[2]


def get_translation_and_rotation(start_pose, end_pose):
    translation = get_translation(start_pose, end_pose)
    rotation = get_rotation(start_pose, end_pose)
    return translation, rotation


def set_polygon_pose(polygon, init_polygon_pose, end_polygon_pose, rotation_center='center'):
    translation, rotation = get_translation_and_rotation(init_polygon_pose, end_polygon_pose)
    return affinity.rotate(affinity.translate(polygon,*translation), rotation, origin=rotation_center)


def polygon_collides_with_entities(polygon, entities):
    for entity in entities:
        if entity.polygon.intersects(polygon):
            return True
    return False


def append_suffix(filename, suffix):
  return "{0}_{2}{1}".format(*os.path.splitext(filename) + (suffix,))


def shapely_polygon_to_triangles_coords(polygon):
    return polygon_coords_to_triangles_coords(list(polygon.exterior.coords))


def polygon_coords_to_triangles_coords(polygon):
    verts = np.array(polygon).reshape(-1, 2)
    rings = np.array([verts.shape[0]])
    triangles_vertices = verts[earcut.triangulate_float64(verts, rings)]
    triangles = [triangles_vertices[n:n + 3] for n in range(0, len(triangles_vertices), 3)]
    return triangles


def is_shapely_polygon_convex(polygon):
    if not isinstance(polygon, Polygon):
        raise TypeError(
            "is_shapely_polygon_convex method expects a shapely.geometry.Polygon object, received: {}".format(
                str(type(polygon))
            ))
    return is_convex_polygon(list(polygon.exterior.coords)[:-1])


def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

    WARNING : The first point must not be repeated at the end of the
        sequence, i.e. the triangle defined by the sequence
        [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
        should be [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)].

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.

    Code by StackOverflow user Rory Daulton, available here:
    https://stackoverflow.com/questions/471962/how-do-i-efficiently-determine-if-a-polygon-is-convex-non-convex-or-complex/472001#472001
    """
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = math.atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = math.atan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -math.pi:
                angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
            elif angle > math.pi:
                angle -= TWO_PI
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon


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
        return x1, y1, 0.

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

    f = (sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) / (2 * (y31 * x12 - y21 * x13))

    g = (sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) / (2. * (x31 * y12 - x21 * y13))

    c = (-pow(x1, 2) - pow(y1, 2) - 2. * g * x1 - 2. * f * y1)

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
    product_of_norms = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    term = scalar_product / product_of_norms
    term = max(-1., term)
    term = min(1., term)
    return math.acos(term)


def map_bounds(polygons):
    if not polygons:
        raise ValueError("There are no entities to populate the grid, it can't be created !")

    map_min_x, map_min_y, map_max_x, map_max_y = float("inf"), float("inf"), -float("inf"), -float("inf")

    for uid, polygon in polygons.items():
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
    return ((y1 - y2) * (ax - x1) + (x2 - x1) * (ay - y1)) * ((y1 - y2) * (bx - x1) + (x2 - x1) * (by - y1)) < 0.


def sample_poses_at_middle_of_inflated_sides(polygon, dist_from_sides, close_to_zero_atol=1e-06):
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
        if norm_a_b != 0.:
            # Compute candidate manip points obtained by cartesian referential change
            points = [(x_m + d * (y_b - y_a) / norm_a_b, y_m + d * (x_b - x_a) / norm_a_b),
                      (x_m + d * (y_b - y_a) / norm_a_b, y_m - d * (x_b - x_a) / norm_a_b),
                      (x_m - d * (y_b - y_a) / norm_a_b, y_m + d * (x_b - x_a) / norm_a_b),
                      (x_m - d * (y_b - y_a) / norm_a_b, y_m - d * (x_b - x_a) / norm_a_b)]
            manip_point = (0., 0.)
            max_dist = 0.0
            # Iterate over candidate manip points to select only the closest one orthogonal to side segment
            for x_r, y_r in points:
                scalar_product = (x_b - x_a) * (x_r - x_m) + (y_b - y_a) * (y_r - y_m)
                if abs(scalar_product - 0.) <= close_to_zero_atol:
                    norm_r_poly_center = np.linalg.norm([poly_center[0] - x_r, poly_center[1] - y_r])
                    if norm_r_poly_center > max_dist:
                        manip_point = (x_r, y_r)
                        max_dist = norm_r_poly_center

            # Save selected manip point in returned list
            direction = (x_m - manip_point[0], y_m - manip_point[1])
            manip_pose = (manip_point[0], manip_point[1], yaw_from_direction(direction))
            poses.append(manip_pose)

    return poses