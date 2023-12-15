import unittest

from shapely.geometry import Polygon

from namosim.navigation.navigation_path import TransitPath
from namosim.world.entity import Style
from namosim.world.obstacle import Obstacle


class NavigationPathTest(unittest.TestCase):
    def setUp(self):
        self.simple_square = Obstacle(
            name="simple_square",
            polygon=Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            pose=(0.0, 0.0, 0.0),
            full_geometry_acquired=True,
            type_="box",
            style=Style(),
        )

    # Bug in TransitPath.from_poses()!
    def test_from_poses(self):
        path = TransitPath.from_poses(
            robot_pose=(401.81603000000007, -184.19798500000007, 90.0),
            robot_polygon=Polygon([(0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0)]),
            poses=[
                (401.81603000000007, -184.19798500000007, 90.0),
                (413.3967763298543, -184.4751680398556, -129.75446123154683),
            ],
        )
        assert len(path.actions) == 2

    def test_from_poses_v2(self):
        path = TransitPath.from_poses_v2(
            robot_pose=(401.81603000000007, -184.19798500000007, 90.0),
            robot_polygon=Polygon([(0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0)]),
            poses=[
                (401.81603000000007, -184.19798500000007, 90.0),
                (413.3967763298543, -184.4751680398556, -129.75446123154683),
                (401.81603000000007, -184.19798500000007, 90.0),
            ],
        )
        assert len(path.actions) == 3
