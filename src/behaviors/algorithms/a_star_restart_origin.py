import heapq
from src.utils import utils
try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False

BY_START = 1
BY_END = 2


def find_path(_start, _end, _grid):
    """
    find a path from start to end node on grid using the A* algorithm
    :param start: start node
    :param end: end node
    :param grid: grid that stores all possible steps/tiles as 2D-list
    :return:
    """

    grid = Grid(matrix=(1 - _grid).transpose())
    start = grid.node(*_start)
    end = grid.node(*_end)

    open_list = [start]

    while len(open_list) > 0:
        # pop node with minimum 'f' value
        node = heapq.nsmallest(1, open_list)[0]
        open_list.remove(node)
        node.closed = True

        # if reached the end position, construct the path and return it
        # (ignored for bi-directional a*, there we look for a neighbor that is
        #  part of the oncoming path)
        if node == end:
            return backtrace(end)

        # get neighbors of the current node
        neighbors = find_neighbors(grid, node)
        for neighbor in neighbors:
            if neighbor.closed:
                # already visited last minimum f value
                continue

            # check if the neighbor has not been inspected yet, or
            # can be reached with smaller cost from the current node
            process_node(neighbor, node, end, open_list)

    # failed to find path
    return []


def backtrace(node):
    """
    Backtrace according to the parent records and return the path.
    (including both start and end nodes)
    """
    path = [(node.x, node.y)]
    while node.parent:
        node = node.parent
        path.append((node.x, node.y))
    path.reverse()
    return path


def find_neighbors(grid, node):
    return grid.neighbors(node)


def process_node(node, parent, end, open_list):
    """
    we check if the given node is path of the path by calculating its
    cost and add or remove it from our path
    :param node: the node we like to test
        (the neighbor in A* or jump-node in JumpPointSearch)
    :param parent: the parent node (the current node we like to test)
    :param end: the end point to calculate the cost of the path
    :param open_list: the list that keeps track of our current path
    :param open_value: needed if we like to set the open list to something
        else than True (used for bi-directional algorithms)

    """
    # calculate cost from current node (parent) to the next node (neighbor)
    ng = calc_cost(parent, node)

    if not node.opened or ng < node.g:
        node.g = ng
        if not node.h:
            node.h = chebyshev_heuristic(node, end)
        # f is the estimated total cost from start to goal
        node.f = node.g + node.h
        node.parent = parent

        if not node.opened:
            heapq.heappush(open_list, node)
            node.opened = True
        else:
            # the node can be reached with smaller cost.
            # Since its f value has been updated, we have to
            # update its position in the open list
            open_list.remove(node)
            heapq.heappush(open_list, node)


def calc_cost(node_a, node_b):
    """
    get the distance between current node and the neighbor (cost)
    """
    if node_b.x - node_a.x == 0 or node_b.y - node_a.y == 0:
        # direct neighbor - distance is 1
        ng = 1
    else:
        # not a direct neighbor - diagonal movement
        ng = utils.SQRT_OF_2

    return node_a.g + ng


def chebyshev_heuristic(node_a, node_b):
    dx = abs(node_a.x - node_b.x)
    dy = abs(node_a.y - node_b.y)
    return dx + dy + utils.SQRT_OF_2_MIN_2 * min(dx, dy)


class Grid(object):
    def __init__(self, width=0, height=0, matrix=None, inverse=False):
        """
        a grid represents the map (as 2d-list of nodes).
        """
        self.width = width
        self.height = height
        if isinstance(matrix, (tuple, list)) or (
                USE_NUMPY and isinstance(matrix, np.ndarray) and
                matrix.size > 0):
            self.height = len(matrix)
            self.width = self.width = len(matrix[0]) if self.height > 0 else 0
        if self.width > 0 and self.height > 0:
            self.nodes = self.build_nodes(self.width, self.height, matrix, inverse)
        else:
            self.nodes = [[]]

    def node(self, x, y):
        """
        get node at position
        :param x: x pos
        :param y: y pos
        :return:
        """
        return self.nodes[y][x]

    def inside(self, x, y):
        """
        check, if field position is inside map
        :param x: x pos
        :param y: y pos
        :return:
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def walkable(self, x, y):
        """
        check, if the tile is inside grid and if it is set as walkable
        """
        return self.inside(x, y) and self.nodes[y][x].walkable

    def neighbors(self, node):
        """
        get all neighbors of one node
        :param node: node
        """
        x = node.x
        y = node.y
        neighbors = []
        s0 = d0 = s1 = d1 = s2 = d2 = s3 = d3 = False

        # top
        if self.walkable(x, y - 1):
            neighbors.append(self.nodes[y - 1][x])
            s0 = True
        # right
        if self.walkable(x + 1, y):
            neighbors.append(self.nodes[y][x + 1])
            s1 = True
        # bottom
        if self.walkable(x, y + 1):
            neighbors.append(self.nodes[y + 1][x])
            s2 = True
        # left
        if self.walkable(x - 1, y):
            neighbors.append(self.nodes[y][x - 1])
            s3 = True

        d0 = s3 and s0
        d1 = s0 and s1
        d2 = s1 and s2
        d3 = s2 and s3

        # top-left
        if d0 and self.walkable(x - 1, y - 1):
            neighbors.append(self.nodes[y - 1][x - 1])

        # top-right
        if d1 and self.walkable(x + 1, y - 1):
            neighbors.append(self.nodes[y - 1][x + 1])

        # bottom-right
        if d2 and self.walkable(x + 1, y + 1):
            neighbors.append(self.nodes[y + 1][x + 1])

        # bottom-left
        if d3 and self.walkable(x - 1, y + 1):
            neighbors.append(self.nodes[y + 1][x - 1])

        return neighbors

    @staticmethod
    def build_nodes(width, height, matrix=None, inverse=False):
        """
        create nodes according to grid size. If a matrix is given it
        will be used to determine what nodes are walkable.
        :rtype : list
        """
        nodes = []
        use_matrix = (isinstance(matrix, (tuple, list))) or \
            (USE_NUMPY and isinstance(matrix, np.ndarray) and matrix.size > 0)

        for y in range(height):
            nodes.append([])
            for x in range(width):
                # 1, '1', True will be walkable
                # while others will be obstacles
                # if inverse is False, otherwise
                # it changes
                weight = int(matrix[y][x]) if use_matrix else 1
                walkable = weight <= 0 if inverse else weight >= 1

                nodes[y].append(Node(x=x, y=y, walkable=walkable, weight=weight))
        return nodes

class Node(object):
    """
    basic node, saves X and Y coordinates on some grid and determine if
    it is walkable.
    """
    def __init__(self, x=0, y=0, walkable=True, weight=1):
        # Coordinates
        self.x = x
        self.y = y

        # cost from this node to the goal
        self.h = 0.0

        # cost from the start node to this node
        self.g = 0.0

        # distance from start to this point (f = g + h )
        self.f = 0.0

        self.opened = 0
        self.closed = False

        # used for backtracking to the start point
        self.parent = None

        # used for recurion tracking of IDA*
        self.retain_count = 0
        # used for IDA* and Jump-Point-Search
        self.tested = False

        # Whether this node can be walked through.
        self.walkable = walkable

        # used for weighted algorithms
        self.weight = weight

    def __lt__(self, other):
        """
        nodes are sorted by f value (see a_star.py)

        :param other: compare Node
        :return:
        """
        return self.f < other.f
