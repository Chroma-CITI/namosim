class ProbabilistOccupancyGrid:
    def __init__(self):
        self._inflated_grid = None
        self._is_inflated_grid_valid = False

    def _update_inflated_grid(self, world):
        world._update_dd_and_reset_grids()

        grid = np.zeros((world.dd.d_width, world.dd.d_height), dtype=np.int16)

        for entity_uid, entity in world.entities.items():
            if entity_uid != world.robot_uid:
                e_min_x, e_min_y, e_max_x, e_max_y = entity.get_inflated_polygon(world.dd).bounds

                min_cell_x = int(round((e_min_x - world.dd.grid_pose[0]) / world.dd.res))
                min_cell_y = int(round((e_min_y - world.dd.grid_pose[1]) / world.dd.res))
                discrete_inflated_polygon = entity.get_discrete_inflated_polygon(world.dd)
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
        self._inflated_grid = grid
        self._is_inflated_grid_valid = True

    def get_inflated_grid(self, world):
        if not self._is_inflated_grid_valid:
            self._update_inflated_grid(world)
        return self._inflated_grid
