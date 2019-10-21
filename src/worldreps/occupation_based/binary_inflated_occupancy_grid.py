from binary_occupancy_grid import BinaryOccupancyGrid


class BinaryInflatedOccupancyGrid(BinaryOccupancyGrid):
    def __init__(self, dd, entities, entities_to_ignore=None):
        BinaryOccupancyGrid.__init__(self, dd, entities, entities_to_ignore)

    def _update_grid(self):

        for prev_entity in self._prev_entities.values():
            if prev_entity.uid not in self._entities_to_ignore:
                prev_cells = prev_entity.get_discrete_inflated_cells_set(self._dd)
                for cell in prev_cells:
                    self._grid[cell[0]][cell[1]] -= 1
        for new_entity in self._next_entities.values():
            if new_entity.uid not in self._entities_to_ignore:
                new_cells = new_entity.get_discrete_inflated_cells_set(self._dd)
                for cell in new_cells:
                    self._grid[cell[0]][cell[1]] += 1

        self._prev_entities = dict()
        self._next_entities = dict()

        # plt.imshow(self.get_inflated_grid()); plt.show()
