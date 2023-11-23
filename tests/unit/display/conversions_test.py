import unittest

import numpy as np
from std_msgs.msg import ColorRGBA

import namosim.display.conversions as conversions


class ConversionsTest(unittest.TestCase):
    def test_path_to_polygon(self):
        points = [np.array(x) for x in ((0, 0), (1, 0))]
        polygon = conversions.path_to_polygon(points=points, line_width=1)
        assert polygon.area == 1

    def test_polygon_to_triangle_list(self):
        points = [np.array(x) for x in ((0, 0), (1, 0))]
        polygon = conversions.path_to_polygon(points=points, line_width=1)
        triangle_list = conversions.polygon_to_triangle_list(
            polygon=polygon,
            namespace="ns",
            p_id=0,
            frame_id="frame",
            color=ColorRGBA(r=0, g=0, b=0, a=1),
            z_index=0,
        )
        # Should be two triangles, or six points, in the triangulated square
        assert len(triangle_list.points) == 6

    def test_real_path_to_triangle_list(self):
        triangle_list = conversions.real_path_to_triangle_list(
            real_path=[(0, 0), (1, 0)],
            namespace="ns",
            p_id=0,
            frame_id="frame",
            color=ColorRGBA(r=0, g=0, b=0, a=1),
            z_index=0,
            line_width=1,
        )
        # Should be two triangles, or six points, in the triangulated square
        assert len(triangle_list.points) == 6
