import unittest
from src.utils import utils
from shapely.geometry import Polygon


class GraphSearchTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_polygon_to_grid(self):
        base_polygon = Polygon([(1, 1), (1, 2), (2, 1)])
        res = 0.05
        subgrid, (min_x, min_y, theta) = utils.polygon_to_grid(base_polygon, res, fill=True)
        print()


if __name__ == '__main__':
    unittest.main()
