from src.utils import utils
from src.behaviors.algorithms.best_first_search import multi_best_first_search
import numpy as np
import copy


class ConnectedComponentsGrid:
    def __init__(self, occupation_grid, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD):
        self._occupation_grid = occupation_grid
        self._grid = np.zeros(occupation_grid.shape, dtype=np.int16)
        self.neighborhood = neighborhood
        self._components = dict()

        self.freed_cells = set()
        self.invaded_cells = set()

        self.init_grid()

    def init_grid(self):
        """
        Initialize connected components grid from occupation grid. Iterates over the occupation grid's cells until a
        free cell that has not already been evaluated is found, then proceed to create the connected component by
        propagating over neighbors.
        """
        d_width, d_height = self._occupation_grid.shape

        closed_set = set()
        current_component_index = 1
        self.components[current_component_index] = set()

        for i in range(d_width):
            for j in range(d_height):
                current_cell = (i, j)
                if current_cell not in closed_set and self._occupation_grid[i][j] == 0:
                    self._grid[i][j] = current_component_index
                    if current_component_index in self.components:
                        self.components[current_component_index].add(current_cell)
                    else:
                        self.components[current_component_index] = {current_cell}
                    open_set = utils.get_neighbors(current_cell, d_width, d_height, self.neighborhood)
                    closed_set.add(current_cell)
                    while open_set:
                        neighbor_cell = open_set.pop()
                        if (neighbor_cell not in closed_set
                                and self._occupation_grid[neighbor_cell[0]][neighbor_cell[1]] == 0):
                            self._grid[neighbor_cell[0]][neighbor_cell[1]] = current_component_index
                            if current_component_index in self.components:
                                self.components[current_component_index].add(neighbor_cell)
                            else:
                                self.components[current_component_index] = {neighbor_cell}
                            open_set = open_set.union(
                                utils.get_neighbors(neighbor_cell, d_width, d_height, self.neighborhood))
                            closed_set.add(neighbor_cell)
                    current_component_index += 1

    def _update_grid(self):
        # 1. Apply invaded cells to grid and components
        self.update_invaded_cells()

        # 2. Fill in freed cells from neighbors ids
        propagated_ids = set()  # Keep track of the component ids that have already been used for updates
        self.update_from_freed_cells(propagated_ids)
        if not self.invaded_cells:
            # Stop here if there are no invaded cells
            return

        # 3. Determine the contour cells around invaded cells for each obstacle
        contour_of_invaded_cells = self.compute_contour_cells_of_invaded_cells()

        # 4. Try to link them together to deduce then whether connectivity is kept or lost
        visited_cells_sets = self.try_to_reach_contour_cells_from_one_another(contour_of_invaded_cells)
        # In the case we have several components with the same encountered cell values, it means that a new component
        # must be created
        # TODO Should be optimised so that components that get new ids are the smallest ones
        visited_components_ids = []
        for closed_set, encountered_cell_values in visited_cells_sets:
            if encountered_cell_values in visited_components_ids:
                new_component_id = self.get_new_component_id()
                encountered_cell_values.clear()
                encountered_cell_values.add(new_component_id)
                self.components[new_component_id] = set()
            visited_components_ids.append(encountered_cell_values)

        # 5. Update the last visited cells set through wave propagation,
        # since we are not sure that the set is the full connected component
        last_set, last_encountered_cell_values = visited_cells_sets.pop()
        unpropagated_encountered_cell_values = last_encountered_cell_values.difference(propagated_ids)
        if unpropagated_encountered_cell_values:
            id_to_propagate = next(iter(unpropagated_encountered_cell_values))
            propagated_ids.add(id_to_propagate)
            open_set = {last_set.pop()}
            closed_set = set()
            while open_set:
                current_cell = open_set.pop()
                self.actually_update_cell_and_component(current_cell, id_to_propagate)
                closed_set.add(current_cell)
                for i, j in self.neighborhood:
                    neighbor_cell = current_cell[0] + i, current_cell[1] + j
                    if (utils.is_in_matrix(neighbor_cell, *self.grid.shape)
                            and self.grid[neighbor_cell[0]][neighbor_cell[1]] != 0
                            and neighbor_cell not in closed_set):
                        open_set.add(neighbor_cell)

        # 7. For the other sets of cells, simply update the grid and components
        for closed_set, encountered_cell_values in visited_cells_sets:
            unpropagated_encountered_cell_values = encountered_cell_values.difference(propagated_ids)
            if unpropagated_encountered_cell_values:
                id_to_propagate = next(iter(unpropagated_encountered_cell_values))
                propagated_ids.add(id_to_propagate)

                self.components[id_to_propagate] = closed_set

                for cell in closed_set:
                    self.grid[cell[0]][cell[1]] = id_to_propagate

    def update_invaded_cells(self):
        for invaded_cell in self.invaded_cells:
            self.grid[invaded_cell[0]][invaded_cell[1]] = 0
            for component_id, component in self.components.items():
                if invaded_cell in component:
                    component.remove(invaded_cell)
                    if not component:
                        del self.components[component_id]

    def update_from_freed_cells(self, propagated_ids):
        cells_to_recheck = set()
        last_cells_to_recheck = set()
        while self.freed_cells:
            freed_cell = self.freed_cells.pop()

            if self.grid[freed_cell[0]][freed_cell[1]] != 0:
                # If a freed cell has already been updated because of expansion, drop it and get to a new loop
                continue

            # Find minimum cell id among freed cell neighbors
            min_neighbor_cell_component = float("inf")
            for i, j in self.neighborhood:
                neighbor_cell = freed_cell[0] + i, freed_cell[1] + j

                if utils.is_in_matrix(neighbor_cell, *self.grid.shape):
                    neighbor_cell_component = self.grid[neighbor_cell[0]][neighbor_cell[1]]
                    if (neighbor_cell_component != 0
                            and neighbor_cell not in self.freed_cells
                            and self.grid[neighbor_cell[0]][neighbor_cell[1]] < min_neighbor_cell_component):
                        min_neighbor_cell_component = self.grid[neighbor_cell[0]][neighbor_cell[1]]

            if min_neighbor_cell_component != float("inf"):
                # Update the freed cell value if it is not just surrounded by other
                # freed cells to update or occupied cells
                self.update_cell_and_propagate(freed_cell, min_neighbor_cell_component)
                propagated_ids.add(min_neighbor_cell_component)
            else:
                # If the cell is surrounded by occupied cells or freed cells to update, we postpone its update
                cells_to_recheck.add(freed_cell)
                if cells_to_recheck and not self.freed_cells:
                    # Try to update postponed cells
                    if cells_to_recheck != last_cells_to_recheck:
                        # Loop again after updating the cells of free cells by the one of the freed cells to recheck
                        self.freed_cells = copy.copy(cells_to_recheck)
                        last_cells_to_recheck = copy.copy(cells_to_recheck)
                    else:
                        # If the cells to recheck are twice the same, it means these cells
                        # form new free space components
                        new_component_id = self.get_new_component_id()
                        while last_cells_to_recheck:
                            cell = last_cells_to_recheck.pop()
                            if self.grid[freed_cell[0]][freed_cell[1]] != 0:
                                # Same as before : if a freed cell has already been updated because of expansion,
                                # drop it and get to a new loop
                                continue
                            self.update_cell_and_propagate(cell, new_component_id)
                            propagated_ids.add(new_component_id)
                        break

    def update_cell_and_propagate(self, cell, id_to_propagate):
        self.actually_update_cell_and_component(cell, id_to_propagate)
        open_set = {cell}
        closed_set = set()
        while open_set:
            current_cell = open_set.pop()
            closed_set.add(current_cell)
            for i, j in self.neighborhood:
                neighbor_cell = current_cell[0] + i, current_cell[1] + j
                if utils.is_in_matrix(neighbor_cell, *self.grid.shape):
                    neighbor_cell_component = self.grid[neighbor_cell[0]][neighbor_cell[1]]
                    if (neighbor_cell_component != 0
                            and neighbor_cell_component != id_to_propagate
                            and neighbor_cell not in closed_set):
                        self.actually_update_cell_and_component(neighbor_cell, id_to_propagate)
                        open_set.add(neighbor_cell)

    def actually_update_cell_and_component(self, cell, new_component_id):
        cell_component_before_update = self.grid[cell[0]][cell[1]]
        if cell_component_before_update != new_component_id:
            self.grid[cell[0]][cell[1]] = new_component_id
            if cell_component_before_update != 0:
                self.components[cell_component_before_update].remove(cell)
                if not self.components[cell_component_before_update]:
                    del self.components[cell_component_before_update]
            self.components[new_component_id].add(cell)

    def compute_contour_cells_of_invaded_cells(self):
        contour_of_invaded_cells = set()
        for invaded_cell in self.invaded_cells:
            for i, j in self.neighborhood:
                neighbor_cell = invaded_cell[0] + i, invaded_cell[1] + j
                if (utils.is_in_matrix(neighbor_cell, *self.grid.shape)
                        and self.grid[neighbor_cell[0]][neighbor_cell[1]] != 0
                        and neighbor_cell not in self.invaded_cells):
                    contour_of_invaded_cells.add(neighbor_cell)
        return contour_of_invaded_cells

    def try_to_reach_contour_cells_from_one_another(self, contour_of_invaded_cells):
        visited_cells_sets = []
        while contour_of_invaded_cells:
            start_cell = next(iter(contour_of_invaded_cells))
            visited_goal_cells, closed_set, encountered_cell_values = multi_best_first_search(
                self.grid, start_cell, contour_of_invaded_cells, neighborhood=self.neighborhood)
            contour_of_invaded_cells.difference_update(visited_goal_cells)
            visited_cells_sets.append((closed_set, encountered_cell_values))
        return visited_cells_sets

    def get_new_component_id(self):
        components_ids = set(self.components.keys())
        max_id_plus_1 = max(components_ids) + 1
        for i in range(1, max_id_plus_1):
            if i not in components_ids:
                return i
        return max_id_plus_1

    @property
    def grid(self):
        return self._grid

    @property
    def components(self):
        return self._components
