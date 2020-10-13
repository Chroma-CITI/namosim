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


class BeFS:
    def __init__(self):
        pass


def best_first_search(grid, width, height, root_cell, goal_cell, neighborhood=utils.TAXI_NEIGHBORHOOD):
    queue = [root_cell]
    visited = {root_cell}
    came_from = {}
    goes_to = {}

    while queue:
        current = queue.pop(0)
        for neighbor in utils.get_neighbors_no_coll(current, grid, width, height, neighborhood):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                came_from[neighbor] = current
                if current in goes_to:
                    goes_to[current].add(neighbor)
                else:
                    goes_to[current] = {neighbor}

    return BFS(visited, came_from, goes_to, root_cell)