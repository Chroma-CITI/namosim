import numpy as np
import copy
from src.utils import utils


class BinaryOccupancyGrid:
    def __init__(self, d_width, d_height, res, grid_pose, inflation_radius, entities, entities_to_ignore=None):
        self._entities_to_ignore = entities_to_ignore if entities_to_ignore is not None else dict()
        self._prev_entities = dict()
        self._next_entities = copy.copy(entities)
        self.d_width, self.d_height = d_width, d_height
        self.res = res
        self.grid_pose = grid_pose
        self.inflation_radius = inflation_radius
        self._grid = np.zeros((self.d_width, self.d_height), dtype=np.int16)
        self._update_grid()

    def _update_grid(self):
        # plt.imshow(self.get_inflated_grid()); plt.show()
        for new_entity in self._next_entities.values():
            if new_entity.uid not in self._entities_to_ignore:
                new_cells = new_entity.get_discrete_cells_set(
                    self.inflation_radius, self.res, self.grid_pose, self.d_width, self.d_height)
                for cell in new_cells:
                    self._grid[cell[0]][cell[1]] += 1
        for prev_entity in self._prev_entities.values():
            if prev_entity.uid not in self._entities_to_ignore:
                prev_cells = prev_entity.get_discrete_cells_set(
                    self.inflation_radius, self.res, self.grid_pose, self.d_width, self.d_height)
                for cell in prev_cells:
                    self._grid[cell[0]][cell[1]] -= 1

        self._prev_entities = dict()
        self._next_entities = dict()

        # plt.imshow(self._int_grid); plt.show()

    def update_buffered_entities(self, prev_entities, next_entities):
        for entity_uid, entity in prev_entities.items():
            # Only update the prev_entity if it is not already stored (otherwise, we would not have the original
            # state of the entity when the grid needs to be updated) and if it is not ignored
            if entity_uid not in self._prev_entities and entity_uid not in self._entities_to_ignore:
                # self._prev_entities[1] = "a"
                self._prev_entities[entity_uid] = entity
                # print(self._prev_entities)
                # del self._prev_entities[1]
                is_update_an_entity_removal = entity_uid not in next_entities
                if is_update_an_entity_removal and entity_uid in self._next_entities:
                    # Prevents artifacts if translation/rotation is applied to removed object before removal,
                    # which could in some cases lead to the obstacle be re-added to the grid after it has been removed
                    del self._next_entities[entity_uid]

        for entity_uid, entity in next_entities.items():
            # Always update the next_entity to reflect the latest state to be used when the grid is updated, except if
            # the entity is supposed to be ignored
            if entity_uid not in self._entities_to_ignore:
                self._next_entities[entity_uid] = entity

    def update_grid_and_return_freed_and_invaded_cells(self):
        invaded_cells = set()
        freed_cells = set()

        for new_entity in self._next_entities.values():
            if new_entity.uid not in self._entities_to_ignore:
                new_cells = new_entity.get_discrete_cells_set(
                    self.inflation_radius, self.res, self.grid_pose, self.d_width, self.d_height)
                for cell in new_cells:
                    if self._grid[cell[0]][cell[1]] == 0:
                        invaded_cells.add(cell)
                    self._grid[cell[0]][cell[1]] += 1
        for prev_entity in self._prev_entities.values():
            if prev_entity.uid not in self._entities_to_ignore:
                prev_cells = prev_entity.get_discrete_cells_set(
                    self.inflation_radius, self.res, self.grid_pose, self.d_width, self.d_height)
                for cell in prev_cells:
                    self._grid[cell[0]][cell[1]] -= 1
                    if self._grid[cell[0]][cell[1]] == 0:
                        freed_cells.add(cell)

        self._prev_entities = dict()
        self._next_entities = dict()

        return invaded_cells, freed_cells

    def get_grid(self):
        is_grid_valid = not self._prev_entities and not self._next_entities
        if not is_grid_valid:
            self._update_grid()
        return self._grid


class NewBinaryOccupancyGrid:
    def __init__(self, d_width, d_height, res, grid_pose, polygons, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD):
        self.d_width, self.d_height = d_width, d_height
        self.res = res
        self.grid_pose = grid_pose
        self.neighborhood=neighborhood

        self.cells_sets = dict()
        self.grid = np.zeros((self.d_width, self.d_height), dtype=np.int16)

        self.update(new_polygons=polygons)

    def update(self, new_polygons=None, removed_polygons=None):
        fill_polygons = self.neighborhood == utils.CHESSBOARD_NEIGHBORHOOD

        if new_polygons is not None:
            for uid, new_polygon in new_polygons.items():
                if uid in self._cells_sets:
                    prev_cells = self._cells_sets[uid]
                    for cell in prev_cells:
                        self._grid[cell[0]][cell[1]] -= 1

                    new_cells = utils.polygon_to_discrete_cells_set(
                        new_polygon, self.res, self.grid_pose, self.d_width, self.d_height, fill=fill_polygons)
                    self._cells_sets[uid] = new_cells
                    for cell in new_cells:
                        self._grid[cell[0]][cell[1]] += 1

        if removed_polygons is not None:
            for uid in removed_polygons:
                prev_cells = self._cells_sets[uid]
                for cell in prev_cells:
                    self._grid[cell[0]][cell[1]] -= 1


class NewBinaryInflatedOccupancyGrid(NewBinaryOccupancyGrid):
    def __init__(self, d_width, d_height, res, grid_pose, polygons, inflation_radius, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD):
        self.inflation_radius = inflation_radius

        NewBinaryOccupancyGrid.__init__(
            self, d_width, d_height, res, grid_pose, polygons, neighborhood)

    def update(self, new_polygons=None, removed_polygons=None):
        inflated_polygons = {
            uid: polygon.buffer(self.inflation_radius)
            for uid, polygon in new_polygons
        }
        NewBinaryOccupancyGrid.update(self, inflated_polygons, removed_polygons)
