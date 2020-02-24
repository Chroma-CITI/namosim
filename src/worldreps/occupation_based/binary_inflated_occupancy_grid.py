from binary_occupancy_grid import BinaryOccupancyGrid


class BinaryInflatedOccupancyGrid(BinaryOccupancyGrid):
    def __init__(self, d_width, d_height, res, grid_pose, inflation_radius, entities, entities_to_ignore=None):
        BinaryOccupancyGrid.__init__(
            self, d_width, d_height, res, grid_pose, inflation_radius, entities, entities_to_ignore)

    def _update_grid(self):
        for new_entity in self._next_entities.values():
            if new_entity.uid not in self._entities_to_ignore:
                new_cells = new_entity.get_discrete_inflated_cells_set(
                    self.inflation_radius, self.res, self.grid_pose, self.d_width, self.d_height)
                for cell in new_cells:
                    self._grid[cell[0]][cell[1]] += 1
        for prev_entity in self._prev_entities.values():
            if prev_entity.uid not in self._entities_to_ignore:
                prev_cells = prev_entity.get_discrete_inflated_cells_set(
                    self.inflation_radius, self.res, self.grid_pose, self.d_width, self.d_height)
                for cell in prev_cells:
                    self._grid[cell[0]][cell[1]] -= 1

        self._prev_entities = dict()
        self._next_entities = dict()

        # plt.imshow(self.get_inflated_grid()); plt.show()

    def update_grid_and_return_freed_and_invaded_cells(self):
        invaded_cells = set()
        freed_cells = set()

        for new_entity in self._next_entities.values():
            if new_entity.uid not in self._entities_to_ignore:
                new_cells = new_entity.get_discrete_inflated_cells_set(
                    self.inflation_radius, self.res, self.grid_pose, self.d_width, self.d_height)
                for cell in new_cells:
                    if self._grid[cell[0]][cell[1]] == 0:
                        invaded_cells.add(cell)
                    self._grid[cell[0]][cell[1]] += 1
        for prev_entity in self._prev_entities.values():
            if prev_entity.uid not in self._entities_to_ignore:
                prev_cells = prev_entity.get_discrete_inflated_cells_set(
                    self.inflation_radius, self.res, self.grid_pose, self.d_width, self.d_height)
                for cell in prev_cells:
                    self._grid[cell[0]][cell[1]] -= 1
                    if self._grid[cell[0]][cell[1]] == 0:
                        freed_cells.add(cell)

        self._prev_entities = dict()
        self._next_entities = dict()

        return freed_cells, invaded_cells
