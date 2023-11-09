import unittest
from snamosim.utils.collision import *
import math
import matplotlib.pyplot as plt


class ArcBoundingBoxParams:
    def __init__(self, point_a, rot_angle, center, bb_type):
        self.point_a = point_a
        self.rot_angle = rot_angle
        self.center = center
        self.bb_type = bb_type

    def __hash__(self):
        return hash((self.point_a, self.rot_angle, self.center, self.bb_type))


class CollisionTest(unittest.TestCase):
    def setUp(self):
        self.nb_places = 7
        self.display = False

    def test_arc_bounding_box(self):
        params_to_expected_results = {
            # No rotation
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0), rot_angle=0.0, center=(0.0, 0.0), bb_type="aabbox"
            ): [(1.0, 0.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0), rot_angle=-0.0, center=(0.0, 0.0), bb_type="aabbox"
            ): [(1.0, 0.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=0.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(1.0, 0.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-0.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(1.0, 0.0)],
            # Right-angle arc
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0), rot_angle=90.0, center=(0.0, 0.0), bb_type="aabbox"
            ): [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0), rot_angle=-90.0, center=(0.0, 0.0), bb_type="aabbox"
            ): [(1.0, 0.0), (0.0, 0.0), (0.0, -1.0), (1.0, -1)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=90.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [
                (1.0, 0.0),
                (0.0, 1.0),
                (1.2071067811865475, 0.20710678118654746),
                (0.20710678118654768, 1.2071067811865475),
            ],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-90.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [
                (1.0, 0.0),
                (0.0, -1.0),
                (1.2071067811865475, -0.20710678118654746),
                (0.20710678118654768, -1.2071067811865475),
            ],
            # Horizontal arc
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0), rot_angle=180.0, center=(0.0, 0.0), bb_type="aabbox"
            ): [(1.0, 0.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-180.0,
                center=(0.0, 0.0),
                bb_type="aabbox",
            ): [(1.0, 0.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 0.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=180.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(1.0, 0.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-180.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(1.0, 0.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 0.0)],
            # Vertical arc
            ArcBoundingBoxParams(
                point_a=(0.0, 1.0), rot_angle=180.0, center=(0.0, 0.0), bb_type="aabbox"
            ): [(0.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (0.0, -1.0)],
            ArcBoundingBoxParams(
                point_a=(0.0, 1.0),
                rot_angle=-180.0,
                center=(0.0, 0.0),
                bb_type="aabbox",
            ): [(0.0, 1.0), (1.0, 1.0), (1.0, -1.0), (0.0, -1.0)],
            ArcBoundingBoxParams(
                point_a=(0.0, 1.0),
                rot_angle=180.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(0.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (0.0, -1.0)],
            ArcBoundingBoxParams(
                point_a=(0.0, 1.0),
                rot_angle=-180.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(0.0, 1.0), (1.0, 1.0), (1.0, -1.0), (0.0, -1.0)],
            # 3/4 arc
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0), rot_angle=270.0, center=(0.0, 0.0), bb_type="aabbox"
            ): [
                (-1.414213562373095, -1.2071067811865475),
                (-1.414213562373095, 1.414213562373095),
                (1.2071067811865475, 1.414213562373095),
                (1.2071067811865475, -1.2071067811865475),
            ],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-270.0,
                center=(0.0, 0.0),
                bb_type="aabbox",
            ): [
                (-1.414213562373095, -1.414213562373095),
                (-1.414213562373095, 1.2071067811865475),
                (1.2071067811865475, 1.2071067811865475),
                (1.2071067811865475, -1.414213562373095),
            ],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=270.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [
                (-1.414213562373095, 2.220446049250313e-16),
                (1.1102230246251565e-16, 1.414213562373095),
                (1.2071067811865475, 0.20710678118654746),
                (-0.20710678118654768, -1.2071067811865475),
            ],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-270.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [
                (-1.414213562373095, -2.220446049250313e-16),
                (1.1102230246251565e-16, -1.414213562373095),
                (1.2071067811865475, -0.20710678118654746),
                (-0.20710678118654768, 1.2071067811865475),
            ],
            # 3/4 arc but the ray passing through C is horizontal
            ArcBoundingBoxParams(
                point_a=(math.cos(math.pi * 7.0 / 4.0), math.sin(math.pi * 7.0 / 4.0)),
                rot_angle=270.0,
                center=(0.0, 0.0),
                bb_type="aabbox",
            ): [
                (-1.0, -0.7071067811865477),
                (-1.0, 1.0),
                (1.0, 1.0),
                (1.0, -0.7071067811865477),
            ],
            ArcBoundingBoxParams(
                point_a=(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)),
                rot_angle=-270.0,
                center=(0.0, 0.0),
                bb_type="aabbox",
            ): [
                (-1.0, -1.0),
                (-1.0, 0.7071067811865476),
                (1.0, 0.7071067811865476),
                (1.0, -1.0),
            ],
            ArcBoundingBoxParams(
                point_a=(math.cos(math.pi * 7.0 / 4.0), math.sin(math.pi * 7.0 / 4.0)),
                rot_angle=270.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [
                (-1.0, -0.7071067811865477),
                (-1.0, 1.0),
                (1.0, 1.0),
                (1.0, -0.7071067811865477),
            ],
            ArcBoundingBoxParams(
                point_a=(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)),
                rot_angle=-270.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [
                (-1.0, -1.0),
                (-1.0, 0.7071067811865476),
                (1.0, 0.7071067811865476),
                (1.0, -1.0),
            ],
            # 3/4 arc but the ray passing through C is vertical
            ArcBoundingBoxParams(
                point_a=(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)),
                rot_angle=270.0,
                center=(0.0, 0.0),
                bb_type="aabbox",
            ): [
                (-1.0, -1.0),
                (-1.0, 1.0),
                (0.7071067811865476, 1.0),
                (0.7071067811865476, -1.0),
            ],
            ArcBoundingBoxParams(
                point_a=(math.cos(math.pi * 7.0 / 4.0), math.sin(math.pi * 7.0 / 4.0)),
                rot_angle=-270.0,
                center=(0.0, 0.0),
                bb_type="aabbox",
            ): [
                (-1.0, -1.0),
                (-1.0, 1.0),
                (0.7071067811865476, 1.0),
                (0.7071067811865476, -1.0),
            ],
            ArcBoundingBoxParams(
                point_a=(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)),
                rot_angle=270.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [
                (-1.0, -1.0),
                (-1.0, 1.0),
                (0.7071067811865476, 1.0),
                (0.7071067811865476, -1.0),
            ],
            ArcBoundingBoxParams(
                point_a=(math.cos(math.pi * 7.0 / 4.0), math.sin(math.pi * 7.0 / 4.0)),
                rot_angle=-270.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [
                (-1.0, -1.0),
                (-1.0, 1.0),
                (0.7071067811865476, 1.0),
                (0.7071067811865476, -1.0),
            ],
            # Full circle
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0), rot_angle=360.0, center=(0.0, 0.0), bb_type="aabbox"
            ): [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-360.0,
                center=(0.0, 0.0),
                bb_type="aabbox",
            ): [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=360.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-360.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
            # Beyond full circle
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0), rot_angle=400.0, center=(0.0, 0.0), bb_type="aabbox"
            ): [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-400.0,
                center=(0.0, 0.0),
                bb_type="aabbox",
            ): [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=400.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-400.0,
                center=(0.0, 0.0),
                bb_type="minimum_rotated_rectangle",
            ): [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        }

        for params, expected_result in params_to_expected_results.items():
            bb = arc_bounding_box(
                point_a=params.point_a,
                rot_angle=params.rot_angle,
                center=params.center,
                bb_type=params.bb_type,
            )
            if self.display:
                fig, ax = plt.subplots()
                bb_x, bb_y = zip(*bb)
                ax.scatter(bb_x, bb_y, marker="x", color="blue")
                if expected_result:
                    bb_ex_x, bb_ex_y = zip(*expected_result)
                else:
                    bb_ex_x, bb_ex_y = [], []
                ax.scatter(bb_ex_x, bb_ex_y, marker="x", color="green")

                r = math.sqrt(
                    (params.point_a[0] - params.center[0]) ** 2
                    + (params.point_a[1] - params.center[1]) ** 2
                )
                plt_circle = plt.Circle(
                    (params.center[0], params.center[1]), r, fill=False
                )
                ax.add_artist(plt_circle)

                ax.axis("equal")
                fig.show()

            done_expected_bb_point = set()
            for bb_point in bb:
                for expected_bb_point in expected_result:
                    if expected_bb_point in done_expected_bb_point:
                        continue
                    try:
                        self.assertAlmostEqual(
                            bb_point[0], expected_bb_point[0], places=self.nb_places
                        )
                        self.assertAlmostEqual(
                            bb_point[1], expected_bb_point[1], places=self.nb_places
                        )
                    except AssertionError:
                        continue
                    done_expected_bb_point.add(expected_bb_point)
            self.assertEqual(len(done_expected_bb_point), len(expected_result))


if __name__ == "__main__":
    unittest.main()
