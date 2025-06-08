import unittest

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from namosim.world.entity import Movability, Style
from namosim.world.obstacle import Obstacle


class TestObstacle:
    def setup_method(self):
        self.simple_square = Obstacle(
            uid="simple_square",
            polygon=Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            pose=(0.0, 0.0, 0.0),
            type_="box",
            style=Style(),
            movability=Movability.MOVABLE,
        )

    def test_polygon_by_visualization(self):
        plt.plot(*self.simple_square.polygon.exterior.xy)
        # plt.show()


if __name__ == "__main__":
    unittest.main()
