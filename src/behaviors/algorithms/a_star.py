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


def reconstruct_path(came_from, end):
    total_path = [end]
    current = end
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path


def astar(grid, start_cell, goal_cell, res, grid_pose, restrict_4_neighbors=False, threshold_obstacle_value=1):
    rp = RosPublisher()

    # Acceptable transitions from current grid element to neighbors
    if restrict_4_neighbors:
        neighborhood = utils.TAXI_NEIGHBORHOOD
    else:
        neighborhood = utils.CHESSBOARD_NEIGHBORHOOD

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

    rp.publish_a_star_open_heap(open_heap, res, grid_pose)

    # While open_heap is not empty == While there are discovered nodes that have not been evaluated
    while open_heap:

        # The node in open_heap having the lowest fScore[] value
        current = heapq.heappop(open_heap).cell
        rp.publish_a_star_open_heap(open_heap, res, grid_pose)

        # Exit early if goal is reached
        if current == goal_cell:
            return reconstruct_path(came_from, goal_cell)

        close_set.add(current)
        rp.publish_a_star_close_set(close_set, res, grid_pose)

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
                    rp.publish_a_star_open_heap(open_heap, res, grid_pose)
    return []


def a_star_real_path(grid, start_pose, goal_pose, res, grid_pose,
                     restrict_4_neighbors=False, authorize_goal_in_occupied_zone = False):
    start_cell = utils.real_to_grid(start_pose[0], start_pose[1], res, grid_pose)
    goal_cell = utils.real_to_grid(goal_pose[0], goal_pose[1], res, grid_pose)

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

            for next in utils.get_neighbors_no_coll(current, grid, width, height, utils.CHESSBOARD_NEIGHBORHOOD):
                extra_cost = straight_dist if current[0] == next[0] or current[1] == next[1] else diag_dist
                new_cost = cost_so_far[current] + extra_cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    heapq.heappush(frontier, (new_cost, next))

        to_evaluate = [start_cell]
        evaluated = set()
        while to_evaluate:
            current = to_evaluate.pop()
            if grid[current[0]][current[1]] == 0:
                start_cell = current
                break
            evaluated.add(current)
            to_evaluate += list(utils.get_neighbors(
                current, grid.shape[0], grid.shape[1], utils.TAXI_NEIGHBORHOOD).difference(evaluated))

    # Execute A*
    astar_path = astar(grid, start_cell, goal_cell, res, grid_pose, restrict_4_neighbors)
    rp = RosPublisher()
    rp.publish_grid_path(astar_path, res, grid_pose)

    # Convert A* output to standard ROS path
    real_path = utils.grid_path_to_real_path(astar_path, start_pose, goal_pose, res, grid_pose)

    return real_path
