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
from snamosim.utils import utils
from snamosim.display.ros_publisher import RosPublisher


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


def reconstruct_path(came_from, end, reverse=True):
    path = [end]
    current = end
    while current in came_from:
        current = came_from[current]
        path.append(current)
    if reverse:
        path.reverse()
    return path


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

    came_from = dict()
    current = None
    open_queue = PriorityQueue()

    if isinstance(start, list) or isinstance(start, set):
        close_set = set(start)
        gscore = {element: 0. for element in start}
        for element in start:
            open_queue.push(heuristic(element, goal), element)
    elif isinstance(start, dict):
        close_set = set(start.keys())
        gscore = {element: cost for element, cost in start.items()}
        for element, cost in start.items():
            open_queue.push(heuristic(element, goal), element)
    else:
        close_set = {start}
        gscore = {start: 0.}
        open_queue.push(heuristic(start, goal), start)

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


def grid_get_neighbors_taxi(current, gscore, close_set, grid, width, height):
    neighbors, tentative_gscores = [], []
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
            tentative_gscores.append(current_gscore + 1.)

    return neighbors, tentative_gscores


def grid_get_neighbors_chessboard_simple(current, gscore, close_set, grid, width, height):
    neighbors, tentative_gscores = grid_get_neighbors_taxi(current, gscore, close_set, grid, width, height)
    current_gscore = gscore[current]

    for i, j in utils.CHESSBOARD_NEIGHBORHOOD_EXTRAS:
        neighbor = current[0] + i, current[1] + j
        neighbor_is_valid = (
                neighbor not in close_set
                and utils.is_in_matrix(neighbor, width, height)
                and grid[neighbor[0]][neighbor[1]] == 0
        )
        if neighbor_is_valid:
            neighbors.append(neighbor)
            tentative_gscores.append(current_gscore + utils.SQRT_OF_2)
    return neighbors, tentative_gscores


def grid_get_neighbors_chessboard_check_diag_neighbors(current, gscore, close_set, grid, width, height):
    neighbors, tentative_gscores = grid_get_neighbors_taxi(current, gscore, close_set, grid, width, height)
    current_gscore = gscore[current]

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
            tentative_gscores.append(current_gscore + utils.SQRT_OF_2)
    return neighbors, tentative_gscores


def grid_search_a_star(start, goal, grid, width, height, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD, check_diag_neighbors=False):

    is_chess_neighborhood = neighborhood==utils.CHESSBOARD_NEIGHBORHOOD

    if is_chess_neighborhood:
        if check_diag_neighbors:
            def grid_get_neighbors_instance(current, gscore, close_set):
                return grid_get_neighbors_chessboard_check_diag_neighbors(
                    current, gscore, close_set, grid, width, height
                )
        else:
            def grid_get_neighbors_instance(current, gscore, close_set):
                return grid_get_neighbors_chessboard_simple(current, gscore, close_set, grid, width, height)

        heuristic = utils.chebyshev_distance
    else:
        def grid_get_neighbors_instance(current, gscore, close_set):
            return grid_get_neighbors_taxi(current, gscore, close_set, grid, width, height)

        heuristic = utils.manhattan_distance

    return new_generic_a_star(start, goal, basic_exit_condition, grid_get_neighbors_instance, heuristic)


def real_to_grid_search_a_star(start_pose, goal_pose, grid):
        start_cell = utils.real_to_grid(start_pose[0], start_pose[1], grid.res, grid.grid_pose)
        goal_cell = utils.real_to_grid(goal_pose[0], goal_pose[1], grid.res, grid.grid_pose)

        if grid.grid[start_cell[0]][start_cell[1]] > 0 or grid.grid[goal_cell[0]][goal_cell[1]] > 0:
            return []

        path_found, last_cell, came_from, _, _, _ = grid_search_a_star(
            start_cell, goal_cell, grid.grid, grid.d_width, grid.d_height, grid.neighborhood, check_diag_neighbors=False
        )

        if path_found:
            raw_path = reconstruct_path(came_from, last_cell)
            return utils.grid_path_to_real_path(raw_path, start_pose, goal_pose, grid.res, grid.grid_pose)
        else:
            return []


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


def grid_search_dijkstra(start, goal, grid, width, height, neighborhood=False):

    is_chess_neighborhood = neighborhood==utils.CHESSBOARD_NEIGHBORHOOD

    def grid_get_neighbors_instance(current, gscore, close_set):
        return grid_get_neighbors(current, gscore, close_set, grid, width, height, is_chess_neighborhood)

    if is_chess_neighborhood:
        return new_generic_dijkstra(start, goal, basic_exit_condition, grid_get_neighbors_instance)
    else:
        return new_generic_dijkstra(start, goal, basic_exit_condition, grid_get_neighbors_instance)
