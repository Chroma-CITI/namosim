import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

# Constants
SQRT_OF_2 = math.sqrt(2)

TAXI_NEIGHBORHOOD = ((0, 1), (0, -1), (1, 0), (-1, 0))
CHESSBOARD_NEIGHBORHOOD = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))

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


def manhattan_distance(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def get_neighbors(cell, width, height, neighborhood=TAXI_NEIGHBORHOOD):
    neighbors = set()
    for i, j in neighborhood:
        neighbor = cell[0] + i, cell[1] + j
        if is_in_matrix(neighbor, width, height):
            neighbors.add(neighbor)
    return neighbors


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


def is_in_matrix(cell, width, height):
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def real_to_grid(real_x, real_y, res, grid_pose):
    return int(round((real_x - grid_pose[0]) / res)), int(round((real_y - grid_pose[1]) / res))


def grid_to_real(cell_x, cell_y, res, grid_pose):
    return res * float(cell_x) + grid_pose[0] + res * 0.5, res * float(cell_y) + grid_pose[1] + res * 0.5


def real_pose_to_grid_pose(real_pose, res, grid_pose, clamp_angle=None):
    return (int(round((real_pose[0] - grid_pose[0]) / res)),
            int(round((real_pose[1] - grid_pose[1]) / res)),
            real_pose[2] if clamp_angle is None else int(round(real_pose[2] / clamp_angle) * clamp_angle))


def grid_pose_to_real_pose(grid_pose, res, parent_grid_pose):
    return res * float(grid_pose[0]) + parent_grid_pose[0] + res * 0.5, res * float(grid_pose[1]) + parent_grid_pose[1] + res * 0.5, float(grid_pose[2])


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
    for cell in grid_path[1:len(grid_path) - 1]:
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


def polygon_to_discrete_cells_set(polygon, res, grid_pose, grid_d_width, grid_d_height, fill=True):
    subgrid, subgrid_pose, = polygon_to_grid(polygon, res, fill)
    cells_set = subgrid_to_discrete_cells_set(subgrid, subgrid_pose, res, grid_pose, grid_d_width, grid_d_height)
    return cells_set


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


def polygon_collides_with_entities(polygon, entities):
    for entity in entities:
        if entity.polygon.intersects(polygon):
            return True
    return False
