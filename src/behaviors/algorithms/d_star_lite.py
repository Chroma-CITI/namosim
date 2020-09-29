from collections import deque
from functools import partial
import heapq
import numpy as np
from src.utils import utils


# region D STAR LITE PRIORITY QUEUE IMPLEMENTATION
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def pop(self):
        item = heapq.heappop(self.elements)
        return item[1]

    def first_key(self):
        try:
            key = heapq.nsmallest(1, self.elements)[0][0]
        except IndexError as e:
            print()
        return key

    def delete(self, cell):
        self.elements = [e for e in self.elements if e[1] != cell]
        heapq.heapify(self.elements)

    def __iter__(self):
        for key, cell in self.elements:
            yield cell
# endregion


class DStarLite(object):
    def __init__(self, grid, start, goal):
        # Init the graphs
        self.grid = grid

        self.back_pointers = {}
        self.G_VALS = {}
        self.RHS_VALS = {}
        self.Km = 0
        self.position = start
        self.last_position = self.position
        self.goal = goal
        self.frontier = PriorityQueue()
        self.frontier.put(self.goal, self.calculate_key(self.goal))
        self.back_pointers[self.goal] = None

        self.freed_cells = set()
        self.invaded_cells = set()
        self.compute_shortest_path()

    def calculate_rhs(self, node):
        lowest_cost_neighbour = self.lowest_cost_neighbour(node)
        self.back_pointers[node] = lowest_cost_neighbour
        return self.lookahead_cost(node, lowest_cost_neighbour)

    def lookahead_cost(self, node, neighbour):
        return self.g(neighbour) + (float("inf")
                                    if self.grid[node[0]][node[1]] > 0 or self.grid[neighbour[0]][neighbour[1]] > 0
                                    else 1)

    def lowest_cost_neighbour(self, node):
        cost = partial(self.lookahead_cost, node)
        return min(self.neighbors(node), key=cost)

    def g(self, node):
        return self.G_VALS.get(node, float('inf'))

    def rhs(self, node):
        return self.RHS_VALS.get(node, float('inf')) if node != self.goal else 0

    def heuristic(self, a, b):
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)

    def calculate_key(self, node):
        g_rhs = min([self.g(node), self.rhs(node)])

        return (
            g_rhs + self.heuristic(node, self.position) + self.Km,
            g_rhs
        )

    def update_node(self, node):
        if node != self.goal:
            self.RHS_VALS[node] = self.calculate_rhs(node)
        self.frontier.delete(node)
        if self.g(node) != self.rhs(node):
            self.frontier.put(node, self.calculate_key(node))

    def update_nodes(self, nodes):
        [self.update_node(n) for n in nodes]

    def compute_shortest_path(self):
        last_nodes = deque(maxlen=10)
        while self.frontier.first_key() < self.calculate_key(self.position) or self.rhs(self.position) != self.g(self.position):
            k_old = self.frontier.first_key()
            node = self.frontier.pop()
            last_nodes.append(node)
            if len(last_nodes) == 10 and len(set(last_nodes)) < 3:
                raise Exception("Fail! Stuck in a loop")
            k_new = self.calculate_key(node)
            if k_old < k_new:
                self.frontier.put(node, k_new)
            elif self.g(node) > self.rhs(node):
                self.G_VALS[node] = self.rhs(node)
                self.update_nodes(self.neighbors(node))
            else:
                self.G_VALS[node] = float('inf')
                self.update_nodes(self.neighbors(node) + [node])

    def build_path(self):
        current_position = self.position
        path = [current_position]
        while current_position != self.goal:
            current_position = self.lowest_cost_neighbour(current_position)
            path.append(current_position)
        return path

    def structures_need_update(self):
        """
        If D* Lite structures have been initialized and the grid has not changed, the best path is still the same
        :return:
        :rtype:
        """
        return self.position != self.last_position or self.freed_cells or self.invaded_cells

    def update_position_and_cells(self, position, freed_cells, invaded_cells):
        self.freed_cells.update(freed_cells)
        self.freed_cells.difference_update(invaded_cells)
        self.invaded_cells.update(invaded_cells)
        self.invaded_cells.difference_update(freed_cells)

    def neighbors(self, _id):
        (x, y) = _id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        return list(filter(self.in_bounds, results))

    def in_bounds(self, _id):
        (x, y) = _id
        return 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]

    def get_shortest_path(self):
        if self.g(self.position) == float('inf'):
            raise Exception("No path")

        if self.structures_need_update():
            self.Km += self.heuristic(self.last_position, self.position)
            self.last_position = self.position
            nodes_to_update = {node for wallnode in self.invaded_cells
                               for node in utils.get_neighbors(wallnode, *self.grid.shape)
                               if self.grid[node[0]][node[1]] == 0}
            self.update_nodes(nodes_to_update)
            self.compute_shortest_path()
        return self.build_path()


def main():
    test_array = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0]
    ])
    d_star_lite = DStarLite(test_array, (0, 0), (4, 3))
    path = d_star_lite.get_shortest_path()
    print(path)

    test_array[2][2] = 1
    d_star_lite.update_position_and_cells((0, 0), set(), {(2, 2)})
    path_02 = d_star_lite.get_shortest_path()
    print(path_02)


if __name__ == "__main__":
    main()
