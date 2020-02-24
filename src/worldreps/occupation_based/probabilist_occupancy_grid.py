import numpy as np


class ProbabilistOccupancyGrid:
    def __init__(self, dd, entities, entities_to_ignore=None):
        self._dd = dd
        self._entities_to_ignore = entities_to_ignore if entities_to_ignore is not None else dict()
        self._grid = None
        self._entities = entities
        self._update_grid()

    def _update_grid(self):
        grid = np.zeros((self._dd.d_width, self._dd.d_height), dtype=np.int16)

        for entity_uid, entity in self._entities.items():
            if entity_uid not in self._entities_to_ignore:
                e_min_x, e_min_y, e_max_x, e_max_y = entity.get_inflated_polygon(
                    self._dd.inflated_radius, self._dd.res).bounds

                min_cell_x = int(round((e_min_x - self._dd.grid_pose[0]) / self._dd.res))
                min_cell_y = int(round((e_min_y - self._dd.grid_pose[1]) / self._dd.res))
                discrete_inflated_polygon = entity.get_discrete_inflated_polygon(
                    self._dd.inflation_radius, self._dd.res, self._dd.cost_lethal, self._dd.cost_inscribed)
                max_cell_x = min_cell_x + discrete_inflated_polygon.shape[0]
                max_cell_y = min_cell_y + discrete_inflated_polygon.shape[1]

                i = 0
                for x in range(min_cell_x, max_cell_x):
                    j = 0
                    for y in range(min_cell_y, max_cell_y):
                        # VERY IMPORTANT CONDITION: OTHERWISE INDEX NEGATIVELY WILL START FROM END OF ARRAY !
                        if x >= 0 and y >= 0:
                            try:
                                if grid[x][y] < discrete_inflated_polygon[i][j]:
                                    grid[x][y] = discrete_inflated_polygon[i][j]
                            except IndexError:
                                pass  # Trim non-lethal obstacle cells around map
                        j = j + 1
                    i = i + 1
        # plt.imshow(grid); plt.show()
        self._grid = grid

    def get_grid(self):
        return self._grid
