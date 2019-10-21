from src.utils import utils
import copy


class ConnectedComponentsGrid:
    def __init__(self):
        self.grid = None

    def compute_discrete_connected_components(self):
        connected_grid = copy.deepcopy(self.get_inflated_grid())
        neighborhood = utils.CHESSBOARD_NEIGHBORHOOD

        closed_set = set()
        current_component_index = -1

        for i in range(self.dd.d_width):
            for j in range(self.dd.d_height):
                current_cell = (i, j)
                if current_cell not in closed_set and connected_grid[i][j] == 0:
                    connected_grid[i][j] = current_component_index
                    open_set = utils.get_neighbors(current_cell, self.dd.d_width, self.dd.d_height, neighborhood)
                    closed_set.add(current_cell)
                    while open_set:
                        neighbor_cell = open_set.pop()
                        if connected_grid[neighbor_cell[0]][neighbor_cell[1]] == 0:
                            connected_grid[neighbor_cell[0]][neighbor_cell[1]] = current_component_index
                            open_set = open_set.union(
                                utils.get_neighbors(neighbor_cell, self.dd.d_width, self.dd.d_height, neighborhood))
                            closed_set.add(neighbor_cell)
                    current_component_index -= 1

        # plt.imshow(connected_grid); plt.show()
        nb_components = -current_component_index
        return connected_grid, nb_components