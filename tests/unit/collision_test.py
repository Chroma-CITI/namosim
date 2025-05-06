import math
import typing as t
import matplotlib.pyplot as plt
from matplotlib import patches
from namosim.utils.collision import arc_bounding_box


class ArcBoundingBoxParams:
    def __init__(
        self,
        point_a: t.Tuple[float, float],
        rot_angle: float,
        center: t.Tuple[float, float],
    ):
        self.point_a = point_a
        self.rot_angle = rot_angle
        self.center = center

    def __hash__(self):
        return hash((self.point_a, self.rot_angle, self.center))

    def __str__(self):
        return f"{self.point_a} - {self.rot_angle} - {self.center}"


class TestCollision:
    def setup_method(self):
        self.nb_places = 7
        self.display = False

    def test_arc_bounding_box_no_rot(self):
        # No rotation
        # bb = arc_bounding_box(degrees=0, point=(1, 0), center=(0, 0))
        # assert bb == [(1, 0), (1, 0), (1, 0), (1, 0)]

        # bb = arc_bounding_box(degrees=-0, point=(1, 0), center=(0, 0))
        # assert bb == [(1, 0), (1, 0), (1, 0), (1, 0)]

        bb = arc_bounding_box(degrees=90.0, point=(1, 0), center=(0, 0))
        bb = list(tuple(round(x, 6) for x in p) for p in bb)
        assert bb == [
            (1.0, 0.0),
            (0.0, 1.0),
            (1.207107, 0.207107),
            (0.207107, 1.207107),
        ]

        bb = arc_bounding_box(degrees=-90.0, point=(1, 0), center=(0, 0))
        bb = list(tuple(round(x, 6) for x in p) for p in bb)
        assert set(bb) == set(
            [
                (1.0, 0.0),
                (0.0, -1.0),
                (1.207107, -0.207107),
                (0.207107, -1.207107),
            ]
        )

        # 180
        bb = arc_bounding_box(degrees=180.0, point=(1, 0), center=(0, 0))
        bb = list(tuple(round(x, 6) for x in p) for p in bb)
        assert set(bb) == set(
            [
                (1.0, 0.0),
                (1, 1),
                (-1, 0),
                (-1, 1),
            ]
        )

        # -180
        bb = arc_bounding_box(degrees=-180.0, point=(1, 0), center=(0, 0))
        bb = list(tuple(round(x, 6) for x in p) for p in bb)
        assert set(bb) == set(
            [
                (1.0, 0.0),
                (1, -1),
                (-1, 0),
                (-1, -1),
            ]
        )

    def test_arc_bounding_box_270(self):
        bb = arc_bounding_box(degrees=270.0, point=(1, 0), center=(0, 0))
        bb = list(tuple(round(x, 6) for x in p) for p in bb)
        assert set(bb) == set(
            tuple(round(x, 6) for x in p)
            for p in [
                (-1.414213562373095, 2.220446049250313e-16),
                (1.1102230246251565e-16, 1.414213562373095),
                (1.2071067811865475, 0.20710678118654746),
                (-0.20710678118654768, -1.2071067811865475),
            ]
        )

    def test_arc_bounding_box(self):
        params_to_expected_results = {
            # Horizontal arc
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=180.0,
                center=(0.0, 0.0),
            ): [(1.0, 0.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)],
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=-180.0,
                center=(0.0, 0.0),
            ): [(1.0, 0.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 0.0)],
            # Vertical arc
            ArcBoundingBoxParams(
                point_a=(0.0, 1.0),
                rot_angle=180.0,
                center=(0.0, 0.0),
            ): [(0.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (0.0, -1.0)],
            ArcBoundingBoxParams(
                point_a=(0.0, 1.0),
                rot_angle=-180.0,
                center=(0.0, 0.0),
            ): [(0.0, 1.0), (1.0, 1.0), (1.0, -1.0), (0.0, -1.0)],
            # 3/4 arc
            ArcBoundingBoxParams(
                point_a=(1.0, 0.0),
                rot_angle=270.0,
                center=(0.0, 0.0),
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
            ): [
                (-1.0, -1.0),
                (-1.0, 1.0),
                (0.7071067811865476, 1.0),
                (0.7071067811865476, -1.0),
            ],
        }

        for params, expected_result in params_to_expected_results.items():
            bb = arc_bounding_box(
                point=params.point_a,
                degrees=params.rot_angle,
                center=params.center,
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
                plt_circle = patches.Circle(
                    (params.center[0], params.center[1]), radius=r, fill=False
                )  # type: ignore
                ax.add_artist(plt_circle)

                ax.axis("equal")
                fig.show()

            expected_points_rounded = set(
                tuple(round(x, 6) for x in p) for p in expected_result
            )
            actual_points_rounded = set(tuple(round(x, 6) for x in p) for p in bb)

            if len(expected_points_rounded.difference(actual_points_rounded)) != 0:
                print("expected", expected_points_rounded)
                print("actual", actual_points_rounded)

            assert len(expected_points_rounded.difference(actual_points_rounded)) == 0
