import math

# Constants
TAXI_NEIGHBORHOOD = ((0, 1), (0, -1), (1, 0), (-1, 0))
CHESSBOARD_NEIGHBORHOOD = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


def get_neighbors(cell, width, height, neighborhood=CHESSBOARD_NEIGHBORHOOD):
    neighbors = set()
    for i, j in neighborhood:
        neighbor = cell[0] + i, cell[1] + j
        if is_in_matrix(neighbor, width, height):
            neighbors.add(neighbor)
    return neighbors


def is_in_matrix(cell, width, height):
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def real_to_grid(real_x, real_y, dd):
    return int((real_x - dd.grid_pose[0]) / dd.res), int((real_y - dd.grid_pose[1]) / dd.res)


def grid_to_real(cell_x, cell_y, dd):
    return dd.res * float(cell_x) + dd.grid_pose[0] + dd.res * 0.5, dd.res * float(cell_y) + dd.grid_pose[1] + dd.res * 0.5


def yaw_from_direction(direction_vector):
    if direction_vector[1] < 0:
        yaw = 2 * math.pi - math.acos(
            direction_vector[0] / math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
    else:
        yaw = math.acos(
            direction_vector[0] / math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
    return math.degrees(yaw)


def grid_path_to_real_path(grid_path, start_pose, goal_pose, dd):
    if not grid_path:
        return []
    real_path = [start_pose]
    previous_pose = start_pose
    for cell in grid_path[1:len(grid_path) - 1]:
        real_x, real_y = grid_to_real(cell[0], cell[1], dd)
        direction_vector = (real_x - previous_pose[0], real_y - previous_pose[1])
        real_yaw = yaw_from_direction(direction_vector)
        new_pose = (real_x, real_y, real_yaw)
        real_path.append(new_pose)
        previous_pose = new_pose
    real_path.append(goal_pose)
    return real_path
