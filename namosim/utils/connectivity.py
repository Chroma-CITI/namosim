import typing as t

import numpy as np

from namosim.data_models import GridCellModel
from namosim.utils import utils
from PIL import Image


class BFS:
    def __init__(self, visited, came_from, goes_to, root_cell: GridCellModel):
        # Set of cells in the connected components
        self.visited = visited
        # Dictionnaries that describe the parent-children relationships in the BFS search tree
        self.came_from = came_from
        self.goes_to = goes_to
        # Remember root cell of search tree to allow faster destruction of component if root cell is invaded
        self.root_cell = root_cell


class CCSData:
    def __init__(self, ccs: t.Dict[int, BFS], grid, current_uid: int):
        self.ccs = ccs
        self.grid = grid
        self.current_uid = current_uid

    def to_image(self):
        # Convert NumPy array to image
        image = Image.fromarray(self.grid.astype(np.uint8) * 255, mode="L")
        return image


def bfs_init(grid, width, height, root_cell, neighborhood=utils.TAXI_NEIGHBORHOOD):
    queue = [root_cell]
    visited = {root_cell}
    came_from = {}
    goes_to = {}

    while queue:
        current = queue.pop(0)
        for neighbor in utils.get_neighbors_no_coll(
            current, grid, width, height, neighborhood
        ):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                came_from[neighbor] = current
                if current in goes_to:
                    goes_to[current].add(neighbor)
                else:
                    goes_to[current] = {neighbor}

    return BFS(visited, came_from, goes_to, root_cell)


def bfs_update(
    grid, width, height, root_cell, ccs_grid, neighborhood=utils.TAXI_NEIGHBORHOOD
):
    queue = [root_cell]
    visited = {root_cell}
    came_from = {}
    goes_to = {}

    prev_component_uid = ccs_grid[root_cell[0]][root_cell[1]]
    affected_components_uids = set()

    while queue:
        current = queue.pop(0)
        for neighbor in utils.get_neighbors_no_coll(
            current, grid, width, height, neighborhood
        ):
            if neighbor not in visited:
                neighbor_prev_component = ccs_grid[neighbor[0]][neighbor[1]]
                if neighbor_prev_component != prev_component_uid:
                    affected_components_uids.add(neighbor_prev_component)
                queue.append(neighbor)
                visited.add(neighbor)
                came_from[neighbor] = current
                if current in goes_to:
                    goes_to[current].add(neighbor)
                else:
                    goes_to[current] = {neighbor}

    return BFS(visited, came_from, goes_to, root_cell), affected_components_uids


def init_ccs_for_grid(grid, width, height, neighborhood=utils.TAXI_NEIGHBORHOOD):
    init_free_cells = set(zip(*np.where(grid == 0)))

    ccs = {}
    ccs_grid = np.zeros(grid.shape, dtype=int)
    current_uid = 0

    while init_free_cells:
        root_cell = init_free_cells.pop()
        current_uid += 1
        new_cc = bfs_init(grid, width, height, root_cell, neighborhood)
        ccs[current_uid] = new_cc
        init_free_cells.difference_update(new_cc.visited)
        for cell in new_cc.visited:
            ccs_grid[cell[0]][cell[1]] = current_uid

    return CCSData(ccs, ccs_grid, current_uid)


def update_ccs_and_grid(
    current_ccs_data, grid, width, height, neighborhood=utils.TAXI_NEIGHBORHOOD
):
    ccs, current_uid = current_ccs_data.ccs, current_ccs_data.current_uid
    free_cells = set(zip(*np.where(grid == 0)))

    new_ccs = {}
    new_ccs_grid = np.zeros(grid.shape, dtype=int)

    while free_cells:
        root_cell = free_cells.pop()

        new_cc = bfs_init(grid, width, height, root_cell, neighborhood)
        new_cc_is_actually_new = True

        potentially_same_ccs = {
            cc_uid: cc
            for cc_uid, cc in ccs.items()
            if len(cc.visited) == len(new_cc.visited)
        }

        for cc_uid, cc in potentially_same_ccs.items():
            if new_cc.visited == cc.visited:
                free_cells.difference_update(cc.visited)
                new_ccs[cc_uid] = cc
                for cell in cc.visited:
                    new_ccs_grid[cell[0]][cell[1]] = cc_uid
                new_cc_is_actually_new = False
                break

        if new_cc_is_actually_new:
            current_uid += 1
            new_ccs[current_uid] = new_cc
            free_cells.difference_update(new_cc.visited)
            for cell in new_cc.visited:
                new_ccs_grid[cell[0]][cell[1]] = current_uid

    return CCSData(new_ccs, new_ccs_grid, current_uid)
