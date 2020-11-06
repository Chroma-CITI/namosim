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

By:
Benoit Renault (benoit.renault@inria.fr)
"""

import heapq
from src.utils import utils
try:
    from src.display.ros_publisher import RosPublisher
    USE_ROS = True
except ImportError:
    USE_ROS = False

from cell_heap_node import CellHeapNode


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.elements_to_heap_nodes_uids = {}
        self.next_uid = 1

    def push(self, cost, element):
        new_heap_node = HeapNode(cost, element, self.next_uid)
        self.next_uid += 1
        if element in self.elements_to_heap_nodes_uids:
            self.elements_to_heap_nodes_uids[element].append(new_heap_node.uid)
        else:
            self.elements_to_heap_nodes_uids[element] = [new_heap_node.uid]
        heapq.heappush(self.heap, new_heap_node)

    def pop(self):
        while self:
            candidate_heap_node = heapq.heappop(self.heap)
            corresponding_element = candidate_heap_node.element
            corresponding_uids = self.elements_to_heap_nodes_uids[corresponding_element]
            if corresponding_uids[-1] == candidate_heap_node.uid:
                corresponding_uids.pop()
                if not corresponding_uids:
                    del self.elements_to_heap_nodes_uids[corresponding_element]
                return corresponding_element
            else:
                corresponding_uids.remove(candidate_heap_node.uid)
        return None

    def __nonzero__(self):
        return bool(self.heap)


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


class HeapNode:
    def __init__(self, cost, element, uid):
        self.cost = cost
        self.element= element
        self.uid = uid

    def __cmp__(self, other):
        # Meant for allowing heapq to properly order the heap's elements according to lowest cost
        return cmp(self.cost, other.cost)

    def __lt__(self, other):
        # Meant for allowing heapq to properly order the heap's elements according to lowest cost
        return self.cost < other.cost

    def __eq__(self, other):
        # Meant for fast check whether a configuration is in open heap or not
        if isinstance(other, tuple):
            return self.element == other
        else:
            return self.element == other.element


def new_generic_a_star(start, goal, exit_condition, get_neighbors, heuristic):
    close_set = {start}
    came_from = dict()
    gscore = {start: 0.}
    open_queue = PriorityQueue()
    open_queue.push(heuristic_cost_estimate(start, goal), start)
    current = None

    while open_queue:
        # The first node in open_queue
        current = open_queue.pop()

        # Exit early if goal is reached
        if exit_condition(current, goal):
            return True, current, came_from, close_set, gscore, open_queue

        # Add current to the close set to prevent unneeded future re-evaluation
        close_set.add(current)

        # For each neighbor of current node in the defined neighborhood
        neighbors, tentative_g_scores = get_neighbors(current, gscore, close_set)
        for neighbor, tentative_g_score in zip(neighbors, tentative_g_scores):
            if neighbor not in gscore or (neighbor in gscore and tentative_g_score < gscore[neighbor]):
                # This path is the best until now. Record it!
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                # TODO: Check if saving the heuristic in a hscores dict would bring a significant perf improvement
                fscore_neighbor = tentative_g_score + heuristic(neighbor, goal)
                open_queue.push(fscore_neighbor, neighbor)

    # If goal could not be reached despite exploring the full search space
    return False, current, came_from, close_set, gscore, open_queue
    # if isinstance(start, list) or isinstance(start, set):
    #     gscore = {element: 0. for element in start}
    #     open_queue = []
    #     for element in start:
    #         heapq.heappush(open_queue, HeapNode(0., element))
    # elif isinstance(start, dict):
    #     gscore = {element: cost for element, cost in start.items()}
    #     open_queue = []
    #     for element, cost in start.items():
    #         heapq.heappush(open_queue, HeapNode(cost, element))
    # else:
    #     gscore = {start: 0.}
    #     open_queue = []
    #     heapq.heappush(open_queue, HeapNode(0., start))


def basic_exit_condition(current, goal):
    """
    Simple exit condition that checks whether the goal is the current cell.
    :param current:
    :type current: any type that has an __eq__ function compatible with the type of the goal parameter
    :param goal:
    :type goal: any type that has an __eq__ function compatible with the type of the goal parameter
    :return: True if current == goal, False otherwise. Exception if no __eq__ operator is found
    :rtype: bool
    """
    return current == goal


def grid_get_neighbors(current, gscore, close_set, grid, width, height, chess_neighborhood=False):
    neighbors, gscores = [], []
    current_gscore = gscore[current]
    for i, j in utils.TAXI_NEIGHBORHOOD:
        neighbor = current[0] + i, current[1] + j
        neighbor_is_valid = (
            neighbor not in close_set
            and utils.is_in_matrix(neighbor, width, height)
            and grid[neighbor[0]][neighbor[1]] == 0
        )
        if neighbor_is_valid:
            neighbors.append(neighbor)
            gscores.append(current_gscore + 1.)
    if chess_neighborhood:
        for i, j in utils.CHESSBOARD_NEIGHBORHOOD_EXTRAS:
            neighbor = current[0] + i, current[1] + j
            neighbor_is_valid = (
                neighbor not in close_set
                and utils.is_in_matrix(neighbor, width, height)
                and grid[neighbor[0]][neighbor[1]] == 0
                and grid[current[0]][neighbor[1]] == 0
                and grid[neighbor[0]][current[1]] == 0
            )
            if neighbor_is_valid:
                neighbors.append(neighbor)
                gscores.append(current_gscore + utils.SQRT_OF_2)
    return neighbors, gscores


def grid_search_a_star(start, goal, grid, width, height, chess_neighborhood=False):

    def grid_get_neighbors_instance(current, gscore, close_set):
        return grid_get_neighbors(current, gscore, close_set, grid, width, height, chess_neighborhood)

    if chess_neighborhood:
        return new_generic_a_star(
            start, goal, basic_exit_condition, grid_get_neighbors_instance, utils.chebyshev_distance
        )
    else:
        return new_generic_a_star(
            start, goal, basic_exit_condition, grid_get_neighbors_instance, utils.manhattan_distance
        )


def new_generic_dijkstra(start, goal, exit_condition, get_neighbors):
    close_set = {start}
    came_from = dict()
    gscore = {start: 0.}
    open_queue = PriorityQueue()
    open_queue.push(0., start)
    current = None

    while open_queue:
        # The first node in open_queue
        current = open_queue.pop()

        # Exit early if goal is reached
        if exit_condition(current, goal):
            return True, current, came_from, close_set, gscore, open_queue

        # Add current to the close set to prevent unneeded future re-evaluation
        close_set.add(current)

        # For each neighbor of current node in the defined neighborhood
        neighbors, tentative_g_scores = get_neighbors(current, gscore, close_set)
        for neighbor, tentative_g_score in zip(neighbors, tentative_g_scores):
            if neighbor not in gscore or (neighbor in gscore and tentative_g_score < gscore[neighbor]):
                # This path is the best until now. Record it!
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                open_queue.push(tentative_g_score, neighbor)

    # If goal could not be reached despite exploring the full search space
    return False, current, came_from, close_set, gscore, open_queue


def grid_search_dijkstra(start, goal, grid, width, height, chess_neighborhood=False):

    def grid_get_neighbors_instance(current, gscore, close_set):
        return grid_get_neighbors(current, gscore, close_set, grid, width, height, chess_neighborhood)

    if chess_neighborhood:
        return new_generic_dijkstra(start, goal, basic_exit_condition, grid_get_neighbors_instance)
    else:
        return new_generic_dijkstra(start, goal, basic_exit_condition, grid_get_neighbors_instance)
