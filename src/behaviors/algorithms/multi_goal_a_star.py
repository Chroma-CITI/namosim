from heapq import heappush, heappop
import copy
from src.utils import utils
from a_star import heuristic_cost_estimate, dist_between, CellHeapNode
from src.display.ros_publisher import RosPublisher


def two_way_multi_goal_a_star(grid, start_pose, intermediate_cells_and_poses, goal_pose, dd,
                              restrict_4_neighbors=False, authorize_goal_in_occupied_zone = False):
    start_cell = utils.real_to_grid(start_pose[0], start_pose[1], dd)
    goal_cell = utils.real_to_grid(goal_pose[0], goal_pose[1], dd)

    # Execute Multi-Goal A* from start pose to intermediate poses
    paths_q_r_q_l = multi_goal_astar(
        grid, start_cell, intermediate_cells_and_poses.keys(), dd, True, restrict_4_neighbors)

    # Execute Multi-Goal A* from goal pose to intermediate poses
    paths_q_l_q_manip = multi_goal_astar(
        grid, goal_cell, intermediate_cells_and_poses.keys(), dd, False, restrict_4_neighbors)

    min_c_0_c_1_cost, min_c_0_disc_path, min_c_1_disc_path, best_obs_pose = float("inf"), None, None, None
    for cell, pose in intermediate_cells_and_poses.items():
        c_0_c1_cost = paths_q_r_q_l[cell][0] + paths_q_l_q_manip[cell][0]
        if c_0_c1_cost < min_c_0_c_1_cost:
            min_c_0_c_1_cost = c_0_c1_cost
            min_c_0_disc_path = paths_q_r_q_l[cell][1]
            min_c_1_disc_path = paths_q_l_q_manip[cell][1]
            best_obs_pose = pose
    real_c0_path = utils.grid_path_to_real_path(min_c_0_disc_path, start_pose, best_obs_pose, dd)
    real_c1_path = utils.grid_path_to_real_path(min_c_1_disc_path, best_obs_pose, goal_pose, dd)
    return real_c0_path, real_c1_path


def _multi_heuristic_cost_estimate(cur_cell, goal_cells):
    min_dist = float("inf")
    for goal_cell in goal_cells:
        min_dist = min(min_dist, heuristic_cost_estimate(cur_cell, goal_cell))
    return min_dist


def _shortest_path(start, end, came_from, reverse):
    path = []
    cur_cell = end
    while 1:
        path.append(cur_cell)
        if cur_cell == start:
            break
        cur_cell = came_from[cur_cell]
    if reverse:
        path.reverse()
    return path


def multi_goal_astar(grid, start_cell, goal_s, dd, reverse=True, restrict_4_neighbors=True, threshold_obstacle_value=1):
    rp = RosPublisher()

    # Acceptable transitions from current grid element to neighbors
    if restrict_4_neighbors:
        neighborhood = utils.TAXI_NEIGHBORHOOD
    else:
        neighborhood = utils.CHESSBOARD_NEIGHBORHOOD

    # The set of nodes that need to be evaluated, add start if it is not already in goal_s
    to_evaluate_set = copy.deepcopy(goal_s)

    # The set of nodes already evaluated
    close_set = set()

    # The dictionary that remembers for each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    came_from = {}

    # The dictionary that remembers for each node, the cost of getting from the start node to that node.
    # The cost of going from start to start is zero.
    gscore = {start_cell: 0}

    # The dictionary that remembers for each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fscore = {start_cell: _multi_heuristic_cost_estimate(start_cell, to_evaluate_set)}

    # The set of currently discovered nodes that are not evaluated yet.
    open_heap = []
    # Initially, only the start node is known.
    heappush(open_heap, CellHeapNode(fscore[start_cell], start_cell))

    rp.publish_multigoal_a_star_open_heap(open_heap, dd)

    # While open_heap is not empty == While there are discovered nodes that have not been evaluated
    while open_heap:

        # The node in open_heap having the lowest fScore[] value
        current = heappop(open_heap).cell
        rp.publish_multigoal_a_star_open_heap(open_heap, dd)

        # Exit early if goal set has been reached
        if not to_evaluate_set:
            break

        close_set.add(current)
        try:
            to_evaluate_set.remove(current)
        except ValueError:
            pass
        rp.publish_multigoal_a_star_close_set(close_set, dd)

        # For each neighbor of current node in the defined neighborhood
        for i, j in neighborhood:
            neighbor = current[0] + i, current[1] + j

            # Check that neighbor exists within the map, has not already been evaluated, is not an obstacle (except if
            # the neighbor is the goal cell)
            if (utils.is_in_matrix(neighbor, grid.shape[0], grid.shape[1])
                    and neighbor not in close_set
                    and (grid[neighbor[0]][neighbor[1]] < threshold_obstacle_value or neighbor not in goal_s)):

                cost_between_current_and_neighbor = dist_between(current, neighbor)

                # The cost from start to a neighbor.
                tentative_g_score = gscore[current] + cost_between_current_and_neighbor

                # Discover a new node or update info about known one :
                if tentative_g_score < gscore[neighbor] or neighbor not in [i.cell for i in open_heap]:
                    # This path is the best until now. Record it!
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + _multi_heuristic_cost_estimate(neighbor, to_evaluate_set)
                    heappush(open_heap, CellHeapNode(fscore[neighbor], neighbor))
                    rp.publish_multigoal_a_star_open_heap(open_heap, dd)

    paths = dict()
    for goal_cell in goal_s:
        try:
            paths[goal_cell] = (gscore[goal_cell], _shortest_path(start_cell, goal_cell, came_from, reverse))
        except KeyError:
            paths[goal_cell] = (float("inf"), [])
    return paths
