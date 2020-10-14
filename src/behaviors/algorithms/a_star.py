"""
A* algorithm

Bootstraped by a python implementation under MIT License from:
Christian Careaga (christian.careaga7@gmail.com)

Available at:
http://code.activestate.com/recipes/578919-python-a-pathfinding-with-binary-heap/

Documented and fixed using the pseudocode in A* Wikipedia page:
https://en.wikipedia.org/wiki/A_star

And augmented to support:
- Python 3
- Non-binary occupation grids
- Manhattan distance
plan_for_obstacle
TODO : PARAMETERIZE MORE AND MOVE CONSTANTS OUT OF THE WAY

By:
Benoit Renault (benoit.renault@inria.fr)
"""

import heapq
from src.utils import utils
from src.display.ros_publisher import RosPublisher
from cell_heap_node import CellHeapNode


def dist_between(a, b):
    return utils.euclidean_distance(a, b)


def heuristic_cost_estimate(a, b):
    return utils.euclidean_distance(a, b)


def reconstruct_path(came_from, end, reverse=True):
    path = [end]
    current = end
    while current in came_from:
        current = came_from[current]
        path.append(current)
    if reverse:
        path.reverse()
    return path


def astar(grid, start_cell, goal_cell, res, grid_pose,
          neighborhood = utils.CHESSBOARD_NEIGHBORHOOD, threshold_obstacle_value=1, ns=''):
    rp = RosPublisher()

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
    fscore = {start_cell: heuristic_cost_estimate(start_cell, goal_cell)}

    # The set of currently discovered nodes that are not evaluated yet.
    open_heap = []
    # Initially, only the start node is known.
    heapq.heappush(open_heap, CellHeapNode(fscore[start_cell], start_cell))

    # rp.publish_a_star_open_heap(open_heap, res, grid_pose, ns=ns)

    # While open_heap is not empty == While there are discovered nodes that have not been evaluated
    while open_heap:

        # The node in open_heap having the lowest fScore[] value
        current = heapq.heappop(open_heap).cell
        # rp.publish_a_star_open_heap(open_heap, res, grid_pose, ns=ns)

        # Exit early if goal is reached
        if current == goal_cell:
            # rp.cleanup_a_star_open_heap(ns=ns)
            rp.cleanup_a_star_close_set(ns=ns)
            return reconstruct_path(came_from, goal_cell)

        close_set.add(current)
        rp.publish_a_star_close_set(close_set, res, grid_pose, ns=ns)

        # For each neighbor of current node in the defined neighborhood
        for i, j in neighborhood:
            neighbor = current[0] + i, current[1] + j

            # If neighbor's g score has not been computed yet, assign +inf
            if neighbor not in gscore:
                gscore[neighbor] = float("inf")

            # Check that neighbor exists within the map, has not already been evaluated, is not an obstacle (except if
            # the neighbor is the goal cell)
            if (utils.is_in_matrix(neighbor, grid.shape[0], grid.shape[1])
                    and neighbor not in close_set
                    and (grid[neighbor[0]][neighbor[1]] < threshold_obstacle_value or neighbor == goal_cell)):

                cost_between_current_and_neighbor = dist_between(current, neighbor)

                # The cost from start to a neighbor.
                tentative_g_score = gscore[current] + cost_between_current_and_neighbor

                # Discover a new node or update info about known one :
                if tentative_g_score < gscore[neighbor] or neighbor not in [i.cell for i in open_heap]:
                    # This path is the best until now. Record it!
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic_cost_estimate(neighbor, goal_cell)
                    heapq.heappush(open_heap, CellHeapNode(fscore[neighbor], neighbor))
                    # rp.publish_a_star_open_heap(open_heap, res, grid_pose, ns=ns)

    # rp.cleanup_a_star_open_heap(ns=ns)
    rp.publish_a_star_close_set(close_set, res, grid_pose, ns=ns)
    # rp.cleanup_a_star_close_set(ns=ns)
    return []


def a_star_real_path(grid, start_pose, goal_pose, res, grid_pose,
                     restrict_4_neighbors=False, authorize_goal_in_occupied_zone = False, ns=''):
    start_cell = utils.real_to_grid(start_pose[0], start_pose[1], res, grid_pose)
    # first_start_cell_considered = start_cell
    goal_cell = utils.real_to_grid(goal_pose[0], goal_pose[1], res, grid_pose)

    #
    if grid[start_cell[0]][start_cell[1]] != 0:
        straight_dist = res
        diag_dist = res * utils.SQRT_OF_2
        width, height = grid.shape

        frontier = []
        heapq.heappush(frontier, (0., start_cell))
        cost_so_far = {start_cell: 0.}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if grid[current[0]][current[1]] == 0:
                start_cell = current
                break
            neighbors = utils.get_neighbors(current, width, height, utils.CHESSBOARD_NEIGHBORHOOD)
            for next in neighbors:
                extra_cost = straight_dist if current[0] == next[0] or current[1] == next[1] else diag_dist
                new_cost = cost_so_far[current] + extra_cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    heapq.heappush(frontier, (new_cost, next))

    # Execute A*
    astar_path = astar(grid, start_cell, goal_cell, res, grid_pose, utils.CHESSBOARD_NEIGHBORHOOD, ns=ns)
    # rp = RosPublisher()
    # rp.publish_grid_path(astar_path, res, grid_pose, ns=ns)

    # Convert A* output to standard ROS path
    real_path = utils.grid_path_to_real_path(astar_path, start_pose, goal_pose, res, grid_pose)

    return real_path


def new_generic_a_star(start, goal, exit_condition, get_neighbors, heuristic):
    close_set = set()
    came_from = dict()

    if isinstance(start, list) or isinstance(start, set):
        gscore = {element for element in start}
        open_heap = []
        for element in start:
            heapq.heappush(open_heap, (0., element))
    elif isinstance(start, dict):
        gscore = {element: cost for element, cost in start.items()}
        open_heap = []
        for element, cost in start.items():
            heapq.heappush(open_heap, (cost, element))
    else:
        gscore = {start}
        open_heap = []
        for element in start:
            heapq.heappush(open_heap, (0., element))

    while open_heap:
        # The node in open_heap having the lowest fScore[] value
        current = heapq.heappop(open_heap)[1]

        # Exit early if goal is reached
        if exit_condition(current, goal):
            return True, came_from, close_set, gscore, open_heap

        # Add current to the close set to prevent unneeded future re-evaluation
        close_set.add(current)

        # For each neighbor of current node in the defined neighborhood
        neighbors, tentative_g_scores = get_neighbors(current, gscore, close_set)
        for neighbor, tentative_g_score in neighbors, tentative_g_scores:
            # Discover a new node or update info about known one :
            if neighbor not in gscore or (neighbor in gscore and tentative_g_score < gscore[neighbor]):
                # This path is the best until now. Record it!
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore_neighbor = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in open_heap:
                    heapq.heappush(open_heap, (fscore_neighbor, neighbor))

    # If goal could not be reached despite exploring the full search space
    return False, close_set, came_from, gscore, open_heap