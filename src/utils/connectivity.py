import numpy as np
from src.utils import utils


class CCSData:
    def __init__(self, ccs, ccs_grid, current_uid):
        self.ccs = ccs
        self.ccs_grid = ccs_grid
        self.current_uid = current_uid


class BFS:
    def __init__(self, visited, came_from, goes_to, root_cell):
        # Set of cells in the connected components
        self.visited = visited
        # Dictionnaries that describe the parent-children relationships in the BFS search tree
        self.came_from = came_from
        self.goes_to = goes_to
        # Remember root cell of search tree to allow faster destruction of component if root cell is invaded
        self.root_cell = root_cell


def bfs_init(grid, width, height, root_cell, neighborhood=utils.TAXI_NEIGHBORHOOD):
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


def bfs_update(grid, width, height, root_cell, ccs_grid, neighborhood=utils.TAXI_NEIGHBORHOOD):
    queue = [root_cell]
    visited = {root_cell}
    came_from = {}
    goes_to = {}

    prev_component_uid = ccs_grid[root_cell[0]][root_cell[1]]
    affected_components_uids = set()

    while queue:
        current = queue.pop(0)
        for neighbor in utils.get_neighbors_no_coll(current, grid, width, height, neighborhood):
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

    return ccs, ccs_grid, current_uid


# def update_ccs_and_grid_tentive(ccs, ccs_grid, current_uid, grid, ccs_uids_to_update):
#     # TODO : Fix case where another component adjacent to the moved obstacle has been reduced but not fused with the
#     #  ones we considered at first
#     cells_to_study = set()
#     for uid in ccs_uids_to_update:
#         cells_to_study.union(ccs[uid].visited)
#         del ccs[uid]
#
#     while cells_to_study:
#         root_cell = cells_to_study.pop()
#         if grid[root_cell[0]][root_cell[1]] == 0:
#             current_uid += 1
#             new_cc, affected_components_uids = bfs_update(grid, width, height, root_cell, ccs_grid)
#             ccs[current_uid] = new_cc
#             for uid in affected_components_uids:
#                 cells_to_study.update(ccs[uid].visited)
#             cells_to_study.difference_update(new_cc.visited)
#             for cell in new_cc.visited:
#                 ccs_grid[cell[0]][cell[1]] = current_uid
#
#     return ccs, ccs_grid, current_uid


def update_ccs_and_grid(ccs, current_uid, grid, width, height, neighborhood=utils.TAXI_NEIGHBORHOOD):
    free_cells = set(zip(*np.where(grid == 0)))

    new_ccs = {}
    new_ccs_grid = np.zeros(grid.shape, dtype=int)

    while free_cells:
        root_cell = free_cells.pop()

        new_cc = bfs_init(grid, width, height, root_cell, neighborhood)
        new_cc_is_new = True

        for cc_uid, cc in ccs.items():
            if new_cc.visited == cc.visited:
                free_cells.difference_update(cc.visited)
                new_ccs[cc_uid] = cc
                for cell in cc.visited:
                    new_ccs_grid[cell[0]][cell[1]] = cc_uid
                new_cc_is_not_new = False
            break

        if new_cc_is_not_new:
            current_uid += 1
            new_ccs[current_uid] = new_cc
            free_cells.difference_update(new_cc.visited)
            for cell in new_cc.visited:
                new_ccs_grid[cell[0]][cell[1]] = current_uid

    return new_ccs, new_ccs_grid, current_uid



# def update_ccs(updated_grid, width, height, prev_ccs, prev_cc_grid, current_uid, invaded_cells, freed_cells):
#     # TODO ADD PROPER MANAGEMENT OF EDGE CASE OF ROOT CELL BEING INVADED
#
#
#     # First, invalidate invaded cells and their descendants in previous BFS trees and connected components grid
#     # and remember their parent cells to restart propagation from them
#     cc_to_repropagation_cells = {}
#     unattributed_cells = set()
#     for invaded_cell in invaded_cells:
#         invaded_cell_prev_cc_uid = prev_cc_grid[invaded_cell[0]][invaded_cell[1]]
#         prev_cc_grid[invaded_cell[0]][invaded_cell[1]] = 0
#         invaded_cell_prev_cc = prev_ccs[invaded_cell_prev_cc_uid]
#
#         # Remove invaded cell from parent's children and save siblings of invaded cell for future re-propagation
#         invaded_cell_parent = invaded_cell_prev_cc.came_from[invaded_cell]
#         invaded_cell_prev_cc.goes_to[invaded_cell_parent].remove(invaded_cell)
#         invaded_cell_siblings = invaded_cell_prev_cc.goes_to[invaded_cell_parent]
#
#         invaded_cell_siblings_and_their_descendants = copy.copy(invaded_cell_siblings)
#
#
#         if invaded_cell_parent not in invaded_cells:
#             if invaded_cell_prev_cc_uid in cc_to_repropagation_cells:
#                 cc_to_repropagation_cells[invaded_cell_prev_cc_uid].append(invaded_cell_parent)
#             else:
#                 cc_to_repropagation_cells[invaded_cell_prev_cc_uid] = [invaded_cell_parent]
#
#         # Compute cells that are invalidated in the bfs tree because of the invaded cell
#         # and remove computed cells from bfs data
#         cells_to_invalidate = {invaded_cell}
#         queue = [invaded_cell]
#         while cells_to_invalidate:
#             cell = queue.pop(0)
#             if cell in invaded_cell_prev_cc.goes_to:
#                 children = invaded_cell_prev_cc.goes_to[cell]
#                 queue += list(children)
#                 cells_to_invalidate.update(children)
#                 del invaded_cell_prev_cc.goes_to[cell]
#             del invaded_cell_prev_cc.came_from[cell]
#         unattributed_cells.update(cells_to_invalidate)
#         invaded_cell_prev_cc.visited.difference_update(cells_to_invalidate)
#
#         # Compute frontier
#         frontier = []
#         for cell in cells_to_invalidate:
#             for neighbor in get_neighbors_no_coll(cell, updated_grid, width, height):
#                 if neighbor in
#
#         # cells_to_invalidate = {invaded_cell}
#         # while cells_to_invalidate :
#         #     for cell in cells_to_invalidate:
#         #         cells_to_invalidate.remove(cell)
#         #         del invaded_cell_prev_cc.came_from[cell]
#         #         invaded_cell_prev_cc.visited.remove(cell)
#         #         if cell in invaded_cell_prev_cc.goes_to:
#         #             cells_to_invalidate.update(invaded_cell_prev_cc.goes_to[cell])
#         #             del invaded_cell_prev_cc.goes_to[cell]
#         #         if cell not in invaded_cells:
#         #             unattributed_cells.add(cell)
#
#     # Second, for each bfs tree that got cells invalidated, we repropagate the bfs trees
#     # from the parent cells of invaded cells
#     unattributed_cells.update(freed_cells)
#
#     for invaded_cell_prev_cc_uid, invaded_cells_parents in cc_to_repropagation_cells.items():
#         prev_cc = prev_ccs[invaded_cell_prev_cc_uid]
#         queue = invaded_cells_parents
#         while queue:
#             current = queue.pop(0)
#             for neighbor in get_neighbors_no_coll(current, updated_grid, width, height):
#                 if neighbor not in prev_cc.visited:
#                     queue.append(neighbor)
#                     prev_cc.visited.add(neighbor)
#                     prev_cc.came_from[neighbor] = current
#                     if current in prev_cc.goes_to:
#                         prev_cc.goes_to[current].add(neighbor)
#                     else:
#                         prev_cc.goes_to[current] = {neighbor}
#
#                     if neighbor in unattributed_cells:
#                         # Don't forget to remove newly attributed cell from unattributed set
#                         unattributed_cells.remove(neighbor)
#                     elif prev_cc_grid[neighbor[0]][neighbor[1]] != invaded_cell_prev_cc_uid:
#                         # If we reach another component, it means we are going to fuse with it
#                         # So basically, we just let the current bfs join with it and we'll have to remove it after


if __name__ == '__main__':
    grid_00 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0]
    ])

    width, height = grid_00.shape

    ccs_00, ccs_grid_00, current_uid_00 = init_ccs_for_grid(grid_00, width, height)

    grid_01 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0]
    ])

    invaded_cells = {(2,2), (2,4)}

    ccs_01, cc_grid_01, current_uid_01 = init_ccs_for_grid(grid_01, width, height)


    print('')
