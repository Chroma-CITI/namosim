import unittest

from shapely.geometry import Polygon

from namosim.mapgen import svg_utils


class Test:
    def test_perturb_polygon(self):
        a = Polygon([(0, 0), (0, 1), (1, 1), (0, 0)])
        b = svg_utils.perturb_polygon(a, bounds=(0, 0, 1, 1))
        ax, ay = a.exterior.coords.xy
        bx, by = b.exterior.coords.xy
        assert ax.tolist() == bx.tolist()
        assert ay.tolist() == by.tolist()
