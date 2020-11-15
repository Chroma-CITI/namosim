import unittest
from src.utils import utils
from shapely.geometry import Polygon
from src.worldreps.occupation_based import binary_occupancy_grid
import matplotlib.pyplot as plt
import numpy as np

class GraphSearchTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_polygon_to_grid(self):
        bounds_polygon= Polygon([(-12.9, -11.9), (-12.9, 11.9), (12.9, 11.9), (12.9, -11.9)])
        test_polygon = Polygon([(6.1, 5.1), (6.1, 6.1), (8.1, 5.1)])
        inflation_radius = 2.
        inflated_test_polygon = test_polygon.buffer(inflation_radius)
        res = 1.
        grid = binary_occupancy_grid.BinaryInflatedOccupancyGrid({1: bounds_polygon, 2: test_polygon}, res, inflation_radius=inflation_radius, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD)

        subgrid, min_d_x, min_d_y = utils.polygon_to_subgrid(inflated_test_polygon, grid.res, grid.grid_pose, grid.r_width, grid.r_height, fill=True)
        cells_set = utils.subgrid_to_cells_set(subgrid, min_d_x, min_d_y, grid.d_width, grid.d_height)

        dep_subgrid, dep_subgrid_pose = utils.polygon_to_grid(inflated_test_polygon, res, True)
        dep_cells_set = utils.subgrid_to_discrete_cells_set(dep_subgrid, dep_subgrid_pose, res, grid.grid_pose, grid.d_width, grid.d_height)

        fig, ax = plt.subplots()
        ax.plot(*bounds_polygon.exterior.xy)
        ax.plot(*test_polygon.exterior.xy)
        ax.plot(*inflated_test_polygon.exterior.xy)
        ax.set_xticks(np.arange(-grid.d_width/2 * res, grid.d_width/2 * res + res, res))
        ax.set_yticks(np.arange(-grid.d_height/2 * res, grid.d_height/2 * res + res, res))
        ax.set_xticklabels(np.arange(0, grid.d_width, 1))
        ax.set_yticklabels(np.arange(0, grid.d_height, 1))
        ax.grid(which='major', alpha=0.5)
        ax.axis('equal')
        fig.show()


        print()


if __name__ == '__main__':
    unittest.main()
