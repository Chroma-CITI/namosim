import numpy as np
import copy


class BinaryOccupancyGrid:
    def __init__(self):
        self._int_grid = None
        self._is_int_grid_valid = False

        self.prev_entities = dict()
        self.new_entities = dict()

    def _update_int_grid(self, world):
        world._update_dd_and_reset_grids()

        if self._int_grid is None:
            self._int_grid = np.zeros((world.dd.d_width, world.dd.d_height), dtype=np.int16)
            self.new_entities = copy.copy(world.entities)
            self.prev_entities = dict()

        # plt.imshow(self.get_inflated_grid()); plt.show()

        for prev_entity in self.prev_entities.values():
            if prev_entity.uid != world.robot_uid:
                prev_cells = prev_entity.get_discrete_cells(world.dd)
                for cell in prev_cells:
                    self._int_grid[cell[0]][cell[1]] -= 1
        for new_entity in self.new_entities.values():
            if new_entity.uid != world.robot_uid:
                new_cells = new_entity.get_discrete_cells_set(world.dd)
                for cell in new_cells:
                    self._int_grid[cell[0]][cell[1]] += 1

        self.prev_entities = dict()
        self.new_entities = dict()

        self._is_int_grid_valid = True
        # plt.imshow(self._int_grid); plt.show()

    def get_int_grid(self, world):
        if not self._is_int_grid_valid:
            self._update_int_grid(world)
        return self._int_grid
