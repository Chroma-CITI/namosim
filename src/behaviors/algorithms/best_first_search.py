import heapq
from src.utils import utils


class CellHeapNode:
    def __init__(self, cell, cost, min_dist_cell):
        self.cell = cell
        self.cost = cost
        self.min_dist_cell = min_dist_cell

    def __cmp__(self, other):
        return cmp(self.cost, other.cost)

    def __lt__(self, other):
        return self.cost < other.cost


def squared_euclidean_distance(b, a):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def multi_heuristic_cost_estimate(cur_cell, goal_cells):
    min_dist = float("inf")
    min_dist_cell = None
    for goal_cell in goal_cells:
        dist = utils.euclidean_distance(cur_cell, goal_cell)
        if dist < min_dist:
            min_dist = dist
            min_dist_cell = goal_cell
    return min_dist, min_dist_cell


def multi_best_first_search(grid, start_cell, goal_cells, threshold_obstacle_value=0,
                            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD):
    # Create empty priority queue, visited cells set and visited goal cells set
    priority_queue = []
    visited_cells_set = set()
    visited_goal_cells = set()
    encountered_cell_values = set()

    # Insert all start cell in priority queue
    heapq.heappush(priority_queue, CellHeapNode(start_cell, *multi_heuristic_cost_estimate(start_cell, goal_cells)))

    # Until priority queue is not empty
    while priority_queue:
        # Pop the best cell in terms of heuristic cost
        current_cell = heapq.heappop(priority_queue).cell
        visited_cells_set.add(current_cell)

        # If the cell is a goal cell remove it from the list of goal cells to reach and update the heap accordingly
        if current_cell in goal_cells:
            goal_cells.remove(current_cell)
            visited_goal_cells.add(current_cell)
            encountered_cell_values.add(grid[current_cell[0]][current_cell[1]])
            # End exploration if all goal cells have been reached
            if not goal_cells:
                break

            heap_changed = False
            for cell_heap_node in priority_queue:
                if cell_heap_node.min_dist_cell == current_cell:
                    cell_heap_node.cost, cell_heap_node.min_dist_cell = multi_heuristic_cost_estimate(
                        cell_heap_node.cell, goal_cells)
                    heap_changed = True
            if heap_changed:
                heapq.heapify(priority_queue)

        # Iterate over current cell neighbors and add them to the priority queue if they have never been visited before
        for i, j in neighborhood:
            neighbor_cell = current_cell[0] + i, current_cell[1] + j

            if (utils.is_in_matrix(neighbor_cell, grid.shape[0], grid.shape[1])
                    and neighbor_cell not in visited_cells_set
                    and grid[neighbor_cell[0]][neighbor_cell[1]] > threshold_obstacle_value):
                visited_cells_set.add(neighbor_cell)
                heapq.heappush(priority_queue,
                               CellHeapNode(neighbor_cell, *multi_heuristic_cost_estimate(start_cell, goal_cells)))

    return visited_goal_cells, visited_cells_set, encountered_cell_values
