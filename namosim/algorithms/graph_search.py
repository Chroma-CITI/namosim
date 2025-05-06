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
import typing as t

import numpy.typing as npt

from namosim.utils import utils


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

            corresponding_uids.remove(candidate_heap_node.uid)
        return None

    def __nonzero__(self):
        return bool(self.heap)

    def __bool__(self):
        return bool(self.heap)


T = t.TypeVar("T")


def reconstruct_path(
    came_from: t.Dict[T, T], end: T, reverse: bool = True
) -> t.List[T]:
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
        self.element = element
        self.uid = uid

    def __cmp__(self, other):
        # Meant for allowing heapq to properly order the heap's elements according to lowest cost
        return utils.cmp(self.cost, other.cost)

    def __lt__(self, other):
        # Meant for allowing heapq to properly order the heap's elements according to lowest cost
        return self.cost < other.cost

    def __eq__(self, other):
        # Meant for fast check whether a configuration is in open heap or not
        if isinstance(other, tuple):
            return self.element == other
        return self.element == other.element


def new_generic_a_star(start, goal, exit_condition, get_neighbors, heuristic):
    came_from = dict()
    current = None
    open_queue = PriorityQueue()
    close_set = set()

    if isinstance(start, list) or isinstance(start, set):
        gscore = {element: 0.0 for element in start}
        for element in start:
            open_queue.push(heuristic(element, goal), element)
    elif isinstance(start, dict):
        gscore = {element: cost for element, cost in start.items()}
        for element, cost in start.items():
            open_queue.push(cost + heuristic(element, goal), element)
    else:
        gscore = {start: 0.0}
        open_queue.push(heuristic(start, goal), start)

    while open_queue:
        # The first node in open_queue
        current = open_queue.pop()

        # Exit early if goal is reached
        if exit_condition(current, goal):
            return True, current, came_from, close_set, gscore, open_queue

        # Add current to the close set to prevent unneeded future re-evaluation
        if current in close_set:
            continue
        else:
            close_set.add(current)

        # For each neighbor of current node in the defined neighborhood
        neighbors, tentative_g_scores = get_neighbors(
            current, gscore, close_set, open_queue, came_from
        )
        for neighbor, tentative_g_score in zip(neighbors, tentative_g_scores):
            if neighbor not in gscore or (
                neighbor in gscore and tentative_g_score < gscore[neighbor]
            ):
                # This path is the best until now. Record it!
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                # TODO: Check if saving the heuristic in a hscores dict would bring a significant perf improvement
                fscore_neighbor = tentative_g_score + heuristic(neighbor, goal)
                open_queue.push(fscore_neighbor, neighbor)

    # If goal could not be reached despite exploring the full search space
    return False, current, came_from, close_set, gscore, open_queue


def basic_exit_condition(current: t.Any, goal: t.Any) -> bool:
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


def grid_get_neighbors_taxi(
    current,
    gscore,
    close_set,
    open_queue,
    came_from,
    grid,
    width,
    height,
    check=(lambda x: x == 0),
):
    neighbors, tentative_gscores = [], []
    current_gscore = gscore[current]
    for i, j in utils.TAXI_NEIGHBORHOOD:
        neighbor = current[0] + i, current[1] + j
        neighbor_is_valid = (
            neighbor not in close_set
            and utils.is_in_matrix(neighbor, width, height)
            and check(grid[neighbor[0]][neighbor[1]])
        )
        if neighbor_is_valid:
            neighbors.append(neighbor)
            tentative_gscores.append(current_gscore + 1.0)

    return neighbors, tentative_gscores


def grid_get_neighbors_chessboard_simple(
    current,
    gscore,
    close_set,
    open_queue,
    came_from,
    grid,
    width,
    height,
    check=(lambda x: x == 0),
):
    neighbors, tentative_gscores = grid_get_neighbors_taxi(
        current, gscore, close_set, open_queue, came_from, grid, width, height
    )
    current_gscore = gscore[current]

    for i, j in utils.CHESSBOARD_NEIGHBORHOOD_EXTRAS:
        neighbor = current[0] + i, current[1] + j
        neighbor_is_valid = (
            neighbor not in close_set
            and utils.is_in_matrix(neighbor, width, height)
            and check(grid[neighbor[0]][neighbor[1]])
        )
        if neighbor_is_valid:
            neighbors.append(neighbor)
            tentative_gscores.append(current_gscore + utils.SQRT_OF_2)
    return neighbors, tentative_gscores


def grid_get_neighbors_chessboard_check_diag_neighbors(
    current,
    gscore,
    close_set,
    open_queue,
    came_from,
    grid,
    width,
    height,
    check=(lambda x: x == 0),
):
    neighbors, tentative_gscores = grid_get_neighbors_taxi(
        current, gscore, close_set, open_queue, came_from, grid, width, height
    )
    current_gscore = gscore[current]

    for i, j in utils.CHESSBOARD_NEIGHBORHOOD_EXTRAS:
        neighbor = current[0] + i, current[1] + j
        neighbor_is_valid = (
            neighbor not in close_set
            and utils.is_in_matrix(neighbor, width, height)
            and check(grid[neighbor[0]][neighbor[1]])
            and check(grid[current[0]][neighbor[1]])
            and check(grid[neighbor[0]][current[1]])
        )
        if neighbor_is_valid:
            neighbors.append(neighbor)
            tentative_gscores.append(current_gscore + utils.SQRT_OF_2)
    return neighbors, tentative_gscores


def grid_search_a_star(
    *,
    start: t.Any,
    goal: t.Any,
    grid: npt.NDArray[t.Any],
    width: int,
    height: int,
    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
    check_diag_neighbors: bool = False,
):
    is_chess_neighborhood = neighborhood == utils.CHESSBOARD_NEIGHBORHOOD

    if is_chess_neighborhood:
        if check_diag_neighbors:

            def grid_get_neighbors_instance(
                current, gscore, close_set, open_queue, came_from
            ):
                return grid_get_neighbors_chessboard_check_diag_neighbors(
                    current,
                    gscore,
                    close_set,
                    open_queue,
                    came_from,
                    grid,
                    width,
                    height,
                )

        else:

            def grid_get_neighbors_instance(
                current, gscore, close_set, open_queue, came_from
            ):
                return grid_get_neighbors_chessboard_simple(
                    current,
                    gscore,
                    close_set,
                    open_queue,
                    came_from,
                    grid,
                    width,
                    height,
                )

        heuristic = utils.chebyshev_distance
    else:

        def grid_get_neighbors_instance(
            current, gscore, close_set, open_queue, came_from
        ):
            return grid_get_neighbors_taxi(
                current, gscore, close_set, open_queue, came_from, grid, width, height
            )

        heuristic = utils.manhattan_distance

    return new_generic_a_star(
        start, goal, basic_exit_condition, grid_get_neighbors_instance, heuristic
    )


def real_to_grid_search_a_star(start_pose, goal_pose, grid):
    start_cell = utils.real_to_grid(
        start_pose[0], start_pose[1], grid.cell_size, grid.grid_pose
    )
    goal_cell = utils.real_to_grid(
        goal_pose[0], goal_pose[1], grid.cell_size, grid.grid_pose
    )

    if start_cell == goal_cell:
        return [start_pose, goal_pose]

    if (
        grid.grid[start_cell[0]][start_cell[1]] > 0
        or grid.grid[goal_cell[0]][goal_cell[1]] > 0
    ):
        return []

    path_found, last_cell, came_from, _, _, _ = grid_search_a_star(
        start=start_cell,
        goal=goal_cell,
        grid=grid.grid,
        width=grid.d_width,
        height=grid.d_height,
        check_diag_neighbors=False,
    )

    if path_found:
        raw_path = reconstruct_path(came_from, last_cell)
        real_path = utils.grid_path_to_real_path(
            raw_path, start_pose, goal_pose, grid.cell_size, grid.grid_pose
        )
        raw_path.append(
            raw_path[-1]
        )  # Copy last cell element to have same number of cells and real poses
        return real_path

    return []


def new_generic_dijkstra(
    start,
    exit_condition,
    get_neighbors,
    came_from=None,
    open_queue=None,
    current=None,
    close_set=None,
):
    if came_from is None:
        came_from = dict()
    if open_queue is None:
        open_queue = PriorityQueue()
    if close_set is None:
        close_set = set()

    if isinstance(start, list) or isinstance(start, set):
        gscore = {element: 0.0 for element in start}
        for element in start:
            open_queue.push(0.0, element)
    elif isinstance(start, dict):
        gscore = {element: cost for element, cost in start.items()}
        for element, cost in start.items():
            open_queue.push(cost, element)
    else:
        gscore = {start: 0.0}
        open_queue.push(0.0, start)

    while open_queue:
        # The first node in open_queue
        current = open_queue.pop()

        # Exit early if goal is reached
        if exit_condition(current):
            return True, current, came_from, close_set, gscore, open_queue

        # Add current to the close set to prevent unneeded future re-evaluation
        if current in close_set:
            continue
        else:
            close_set.add(current)

        # For each neighbor of current node in the defined neighborhood
        neighbors, tentative_g_scores = get_neighbors(
            current, gscore, close_set, open_queue, came_from
        )
        for neighbor, tentative_g_score in zip(neighbors, tentative_g_scores):
            if neighbor not in gscore or (
                neighbor in gscore and tentative_g_score < gscore[neighbor]
            ):
                # This path is the best until now. Record it!
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                open_queue.push(tentative_g_score, neighbor)

    # If goal could not be reached despite exploring the full search space
    return False, current, came_from, close_set, gscore, open_queue


def grid_search_dijkstra(
    start,
    goal,
    grid,
    width,
    height,
    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
    check_diag_neighbors=False,
    max_visited: float = float("inf"),
):
    is_chess_neighborhood = neighborhood == utils.CHESSBOARD_NEIGHBORHOOD

    if grid[start[0]][start[1]] > 0:
        return False, start, dict(), set(), dict(), []

    if is_chess_neighborhood:
        if check_diag_neighbors:

            def grid_get_neighbors_instance(
                current, gscore, close_set, open_queue, came_from
            ):
                if len(close_set) >= max_visited:
                    return [], []

                return grid_get_neighbors_chessboard_check_diag_neighbors(
                    current,
                    gscore,
                    close_set,
                    open_queue,
                    came_from,
                    grid,
                    width,
                    height,
                )

        else:

            def grid_get_neighbors_instance(
                current, gscore, close_set, open_queue, came_from
            ):
                if len(close_set) >= max_visited:
                    return [], []
                return grid_get_neighbors_chessboard_simple(
                    current,
                    gscore,
                    close_set,
                    open_queue,
                    came_from,
                    grid,
                    width,
                    height,
                )

    else:

        def grid_get_neighbors_instance(
            current, gscore, close_set, open_queue, came_from
        ):
            if len(close_set) >= max_visited:
                return [], []
            return grid_get_neighbors_taxi(
                current, gscore, close_set, open_queue, came_from, grid, width, height
            )

    def goal_reached_exit_condition(current):
        return basic_exit_condition(current, goal)

    return new_generic_dijkstra(
        start, goal_reached_exit_condition, grid_get_neighbors_instance
    )


def grid_search_closest_free_cell(
    start,
    grid,
    width,
    height,
    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
    check_diag_neighbors=False,
):
    is_chess_neighborhood = neighborhood == utils.CHESSBOARD_NEIGHBORHOOD

    if is_chess_neighborhood:
        if check_diag_neighbors:

            def grid_get_neighbors_instance(
                current, gscore, close_set, open_queue, came_from
            ):
                return grid_get_neighbors_chessboard_check_diag_neighbors(
                    current,
                    gscore,
                    close_set,
                    open_queue,
                    came_from,
                    grid,
                    width,
                    height,
                    check=(lambda x: x > 0),
                )

        else:

            def grid_get_neighbors_instance(
                current, gscore, close_set, open_queue, came_from
            ):
                return grid_get_neighbors_chessboard_simple(
                    current,
                    gscore,
                    close_set,
                    open_queue,
                    came_from,
                    grid,
                    width,
                    height,
                    check=(lambda x: x > 0),
                )

    else:

        def grid_get_neighbors_instance(
            current, gscore, close_set, open_queue, came_from
        ):
            return grid_get_neighbors_taxi(
                current,
                gscore,
                close_set,
                open_queue,
                came_from,
                grid,
                width,
                height,
                check=(lambda x: x > 0),
            )

    def is_cell_free(current):
        return grid[current[0]][current[1]] == 0

    return new_generic_dijkstra(start, is_cell_free, grid_get_neighbors_instance)
