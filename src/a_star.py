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
- Zig-zag suppression when using only N, S, E and W directions
plan_for_obstacle
TODO : PARAMETERIZE MORE AND MOVE CONSTANTS OUT OF THE WAY

By:
Benoit Renault (benoit.renault@inria.fr)
"""

import copy
from heapq import *
import math
import conversion


def _dist_between(a, b):
    return _euclidean_distance(a, b)


def _heuristic_cost_estimate(a, b):
    return _euclidean_distance(a, b)


def _euclidean_distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def _manhattan_distance(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def _reconstruct_path(came_from, end):
    total_path = [end]
    current = end
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path


def print_path(nmap, path):
    matrix = copy.deepcopy(nmap)
    for point in path:
        matrix[point[0]][point[1]] = -1
    print(matrix)


# from Queue import PriorityQueue
#
#
# def heuristic(a, b):
#     (x1, y1) = a
#     (x2, y2) = b
#     return abs(x1 - x2) + abs(y1 - y2)
#
#
# def astar(grid, start, goal, dd, rp, restrict_4_neighbors = True):
#     if restrict_4_neighbors:
#         neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#     else:
#         neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
#
#     frontier = PriorityQueue()
#     frontier.put(start, 0)
#     rp.publish_a_star_open_heap(frontier, dd)
#     came_from = {}
#     cost_so_far = {}
#     cost_so_far[start] = 0
#
#     while not frontier.empty():
#         current = frontier.get()
#
#         if current == goal:
#             break
#
#         for i, j in neighborhood:
#             neighbor = current[0] + i, current[1] + j
#
#             # Check that neighbor exists within the map
#             if 0 <= neighbor[0] < grid.shape[0]:
#                 if 0 <= neighbor[1] < grid.shape[1]:
#                     if grid[neighbor[0]][neighbor[1]] > 0.0:
#                         continue
#                     new_cost = cost_so_far[current] + _manhattan_distance(current, neighbor)
#                     if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
#                         cost_so_far[neighbor] = new_cost
#                         priority = new_cost + heuristic(goal, neighbor)
#                         frontier.put(neighbor, priority)
#                         rp.publish_a_star_open_heap(frontier, dd)
#                         came_from[neighbor] = current
#
#     return _reconstruct_path(came_from, goal)


def astar(grid, start_cell, goal_cell, dd, rp, restrict_4_neighbors=True):
    # Acceptable transitions from current grid element to neighbors
    if restrict_4_neighbors:
        neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    else:
        neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Directions
    directions = [['NW', 'N', 'NE'],
                  ['W' , 'X', 'E' ],
                  ['SW', 'S', 'SE']]

    # The set of nodes already evaluated
    close_set = set()

    # The dictionary that remembers for each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    came_from = {}
    came_from_direction = {}

    # The dictionary that remembers for each node, the cost of getting from the start node to that node.
    # The cost of going from start to start is zero.
    gscore = {start_cell: 0}

    # The dictionary that remembers for each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fscore = {start_cell: _heuristic_cost_estimate(start_cell, goal_cell)}

    # The set of currently discovered nodes that are not evaluated yet.
    open_heap = []
    # Initially, only the start node is known.
    heappush(open_heap, (fscore[start_cell], start_cell))
    rp.publish_a_star_open_heap(open_heap, dd)

    # While open_heap is not empty == While there are discovered nodes that have not been evaluated
    while open_heap:

        # The node in open_heap having the lowest fScore[] value
        current = heappop(open_heap)[1]
        rp.publish_a_star_open_heap(open_heap, dd)

        # Exit early if goal is reached
        if current == goal_cell:
            return _reconstruct_path(came_from, current)

        close_set.add(current)
        rp.publish_a_star_close_set(close_set, dd)

        # For each neighbor of current node in the defined neighborhood
        for i, j in neighborhood:
            neighbor = current[0] + i, current[1] + j
            new_direction = directions[1 + i][1 + j]

            # Check that neighbor exists within the map
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    # Do not consider traversing neighbor if it is an obstacle
                    if grid[neighbor[0]][neighbor[1]] >= dd.cost_circumscribed:
                        if neighbor != goal_cell:
                            continue
                    cost_between_current_and_neighbor = _dist_between(current, neighbor)
                else:
                    # Neighbor is outside of map in y axis
                    continue
            else:
                # Neighbor is outside of map in x axis
                continue

            if restrict_4_neighbors:
                try:
                    previous_direction = came_from_direction[current]
                except KeyError:
                    previous_direction = new_direction
                rotation_cost = (1.5 if new_direction != previous_direction else 0.0)
                cost_between_current_and_neighbor = cost_between_current_and_neighbor + rotation_cost

            # The cost from start to a neighbor.
            tentative_g_score = gscore[current] + cost_between_current_and_neighbor

            if neighbor in close_set:
                continue  # Ignore the neighbor which is already evaluated.

            # Discover a new node or update info about known one :
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in open_heap]:
                # This path is the best until now. Record it!
                came_from[neighbor] = current
                came_from_direction[neighbor] = new_direction
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + _heuristic_cost_estimate(neighbor, goal_cell)
                heappush(open_heap, (fscore[neighbor], neighbor))
                rp.publish_a_star_open_heap(open_heap, dd)

    return []


def a_star_real_path(grid, start_pose, goal_pose, dd, rp,
                     restrict_4_neighbors=False, authorize_goal_in_occupied_zone = False):
    start_cell = conversion.real_to_grid(start_pose[0], start_pose[1], dd)
    goal_cell = conversion.real_to_grid(goal_pose[0], goal_pose[1], dd)

    # Execute A*
    astar_path = astar(grid, start_cell, goal_cell, dd, rp, restrict_4_neighbors)
    rp.publish_grid_path(astar_path, dd)

    # Convert A* output to standard ROS path
    real_path = conversion.grid_path_to_real_path(astar_path, start_pose, goal_pose, dd)

    return real_path
