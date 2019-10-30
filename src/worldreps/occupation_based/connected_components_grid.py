from src.utils import utils
import numpy as np


class ConnectedComponentsGrid:
    def __init__(self, occupation_grid, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD):
        self._occupation_grid = occupation_grid
        self._grid = np.zeros(occupation_grid.shape, dtype=np.int16)
        self.neighborhood = neighborhood
        self.components = dict()

        self.init_grid()

    def init_grid(self):
        """
        Initialize connected components grid from occupation grid. Iterates over the occupation grid's cells until a
        free cell that has already been evaluated is found, then proceed to create the connected component by
        propagating over neighbors (ignoring those already evaluated).
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
                    self.components[current_component_index].add(current_cell)
                    open_set = utils.get_neighbors(current_cell, d_width, d_height, self.neighborhood)
                    closed_set.add(current_cell)
                    while open_set:
                        neighbor_cell = open_set.pop()
                        if (neighbor_cell not in closed_set
                                and self._occupation_grid[neighbor_cell[0]][neighbor_cell[1]] == 0):
                            self._grid[neighbor_cell[0]][neighbor_cell[1]] = current_component_index
                            self.components[current_component_index].add(current_cell)
                            open_set = open_set.union(
                                utils.get_neighbors(neighbor_cell, d_width, d_height, self.neighborhood))
                            closed_set.add(neighbor_cell)
                    current_component_index += 1
                    self.components[current_component_index] = set()

    @property
    def grid(self):
        return self._grid
