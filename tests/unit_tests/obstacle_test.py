import unittest

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from namosim.worldreps.entity_based.entity import Style
from namosim.worldreps.entity_based.obstacle import Obstacle


class ObstacleTest(unittest.TestCase):
    def setUp(self):
        self.simple_square = Obstacle(
            name="simple_square",
            polygon=Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            pose=(0.0, 0.0, 0.0),
            full_geometry_acquired=True,
            type_="box",
            style=Style(),
        )

    def test_polygon_by_visualization(self):
        plt.plot(*self.simple_square.polygon.exterior.xy)
        # plt.show()


if __name__ == "__main__":
    unittest.main()
