import numpy as np
import math
from shapely.geometry import Polygon

from src.utils import utils


class GridParams:
    def __init__(self, grid_pose, d_width, d_height, r_width, r_height, aabb_polygon):
        self.grid_pose, self.d_width, self.d_height, self.r_width, self.r_height, self.aabb_polygon = (
            grid_pose, d_width, d_height, r_width, r_height, aabb_polygon
        )

    def __eq__(self, other):
        return (self.grid_pose, self.d_width, self.d_height, self.r_width, self.r_height, self.aabb_polygon) == (
            other.grid_pose, other.d_width, other.d_height, other.r_width, other.r_height, other.aabb_polygon
        )

    def all(self):
        return self.grid_pose, self.d_width, self.d_height, self.r_width, self.r_height, self.aabb_polygon


def grid_parameters(polygons, res):
    r_min_x, r_min_y, r_max_x, r_max_y = utils.map_bounds(polygons)
    min_x, min_y = math.floor(r_min_x / res) * res, math.floor(r_min_y / res) * res
    max_x, max_y = math.ceil(r_max_x / res) * res, math.ceil(r_max_y / res) * res
    d_width = abs(int(math.floor(r_min_x / res))) + abs(int(math.ceil(r_max_x / res)))
    d_height = abs(int(math.floor(r_min_y / res))) + abs(int(math.ceil(r_max_y / res)))
    real_width, real_height = d_width * res, d_height * res
    real_pose = min_x, min_y, 0.0
    aabb_polygon = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])
    return GridParams(real_pose, d_width, d_height, real_width, real_height, aabb_polygon)


class BinaryOccupancyGrid:
    def __init__(self, polygons, res, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD, params=None):
        self.res = res

        self.params = params if params else grid_parameters(polygons, res)

        self.grid_pose, self.d_width, self.d_height, self.r_width, self.r_height, self.aabb_polygon = self.params.all()

        self.neighborhood = neighborhood

        self.cells_sets = dict()
        self.grid = np.zeros((self.d_width, self.d_height), dtype=np.int16)

        self.update(new_polygons=polygons)

    def update(self, new_polygons=None, removed_polygons=None):
        fill_polygons = self.neighborhood == utils.CHESSBOARD_NEIGHBORHOOD

        if new_polygons is not None:
            for uid, new_polygon in new_polygons.items():
                if uid in self.cells_sets:
                    prev_cells = self.cells_sets[uid]
                    for cell in prev_cells:
                        self.grid[cell[0]][cell[1]] -= 1

                new_cells = utils.polygon_to_discrete_cells_set(
                    new_polygon, self.res, self.grid_pose, self.d_width, self.d_height, self.r_width, self.r_height, fill=fill_polygons
                )
                self.cells_sets[uid] = new_cells
                for cell in new_cells:
                    self.grid[cell[0]][cell[1]] += 1

        if removed_polygons is not None:
            for uid in removed_polygons:
                prev_cells = self.cells_sets[uid]
                for cell in prev_cells:
                    self.grid[cell[0]][cell[1]] -= 1

    def only_obstacle_uid_in_cell(self, cell):
        """
        If cell is contained only by one obstacle o_i, returns o_i.
        If contained by no obstacle, returns 0. If contained by more than one, returns -1.
        :param cell: cell coordinates (x, y)
        :type cell: tuple(int, int)
        :return: obstacle uid or 0 or -1
        :rtype: int
        """
        if self.grid[cell[0]][cell[1]] == 0:
            return 0
        elif self.grid[cell[0]][cell[1]] > 1:
            return -1
        else:
            for uid, cell_set in self.cells_sets.items():
                if cell in cell_set:
                    return uid
            raise RuntimeError('It should be impossible for an occupied cell of the grid to not be in any cells set.')


class BinaryInflatedOccupancyGrid(BinaryOccupancyGrid):
    def __init__(self, polygons, res, inflation_radius, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD, params=None):
        self.inflation_radius = inflation_radius

        BinaryOccupancyGrid.__init__(self, polygons, res, neighborhood, params)

    def update(self, new_polygons=None, removed_polygons=None):
        if new_polygons:
            inflated_polygons = {
                uid: polygon.buffer(self.inflation_radius)
                for uid, polygon in new_polygons.items()
            }
            BinaryOccupancyGrid.update(self, inflated_polygons, removed_polygons)
        else:
            BinaryOccupancyGrid.update(self, new_polygons, removed_polygons)
