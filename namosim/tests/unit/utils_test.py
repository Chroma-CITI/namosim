import unittest

from shapely.geometry import Polygon

from namosim.utils import utils
from namosim.utils.conversion import (
    shapely_geometry_to_svg_pathd,
    svg_pathd_to_shapely_geometry,
)


class TestGraphSearch:
    def setup_method(self):
        pass

    def test_to_from_svg(self):
        poly = Polygon([(0, 0), (0, 1), (1, 1), (0, 0)])
        pathd = shapely_geometry_to_svg_pathd(shape=poly, ymax_meters=0)
        poly2 = svg_pathd_to_shapely_geometry(svg_path=pathd, ymax_meters=0)
        assert poly == poly2

    def test_set_polygon_pose(self):
        poly = Polygon([(0, 0), (0, 1), (1, 1), (0, 0)])
        poly2 = utils.set_polygon_pose(
            polygon=poly,
            init_polygon_pose=(0, 0, 0),
            end_polygon_pose=(1, 1, 180),
            rotation_center=(0, 0),
        )
        assert poly2 == Polygon([(1, 1), (1, 0), (0, 0), (1, 1)])

        poly3 = utils.set_polygon_pose(
            polygon=poly2,
            end_polygon_pose=(0, 0, 0),
            init_polygon_pose=(1, 1, 180),
            rotation_center=(1, 1),
        )
        assert poly == poly3


if __name__ == "__main__":
    unittest.main()
