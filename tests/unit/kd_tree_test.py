import unittest
from typing import List
from namosim.algorithms.kd_tree import (
    KDTree,
    default_distance_func,
)  # Assuming the KDTree code is in kdtree.py
import typing as t


class Point2D:
    def __init__(self, x: float, y: float, label: str):
        self.x = x
        self.y = y
        self.label = label

    def __repr__(self):
        return f"Point2D({self.x}, {self.y}, '{self.label}')"


def point_getter(p: Point2D) -> List[float]:
    return [p.x, p.y]


class TestKDTree(unittest.TestCase):
    def setUp(self):
        self.tree = KDTree[Point2D](dimensions=2, point_getter=point_getter)

    def test_empty_tree(self):
        self.assertIsNone(self.tree.root)
        self.assertEqual(self.tree.query([0, 0], k=1), [])
        self.assertEqual(self.tree.query_radius([0, 0], 1.0), [])

    def test_add_single_point(self):
        point = Point2D(1, 2, "A")
        self.tree.add(point)
        self.assertIsNotNone(self.tree.root)
        self.assertEqual(self.tree.root.object, point)
        self.assertEqual(self.tree.root.point, [1, 2])
        self.assertEqual(self.tree.query([1, 2], k=1), [point])

    def test_add_multiple_points(self):
        points = [Point2D(1, 1, "A"), Point2D(2, 2, "B"), Point2D(0, 3, "C")]
        for p in points:
            self.tree.add(p)

        # Test structure
        self.assertEqual(self.tree.root.object.label, "A")
        self.assertEqual(self.tree.root.right.object.label, "B")
        self.assertEqual(self.tree.root.left.object.label, "C")

    def test_query_k_nearest(self):
        points = [
            Point2D(0, 0, "A"),
            Point2D(1, 1, "B"),
            Point2D(2, 2, "C"),
            Point2D(-1, -0.5, "D"),
        ]
        for p in points:
            self.tree.add(p)

        # Query for 2 nearest to (0, 0)
        result = self.tree.query([0, 0], k=2)
        labels = [p.label for p in result]
        self.assertEqual(set(labels), {"A", "D"})  # A and D are closest to (0,0)

        # Query for 1 nearest
        result = self.tree.query([0, 0], k=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "A")  # A is at (0,0)

    def test_query_radius(self):
        points = [
            Point2D(0, 0, "A"),
            Point2D(1, 1, "B"),
            Point2D(2, 2, "C"),
            Point2D(-1, -1, "D"),
        ]
        for p in points:
            self.tree.add(p)

        # Query points within radius 1.5 from (0,0)
        result = self.tree.query_radius([0, 0], 2)
        labels = [p.label for p in result]
        self.assertEqual(
            set(labels), {"A", "B", "D"}
        )  # A(0,0) and D(-1,-1) are within sqrt(2) ≈ 1.41

        # Query with zero radius
        result = self.tree.query_radius([0, 0], 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "A")

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            self.tree.query([0, 0], k=0)
        with self.assertRaises(ValueError):
            self.tree.query([0, 0], k=-1)

    def test_custom_distance_function(self):
        # Use Manhattan distance instead of Euclidean
        def manhattan_distance(a: t.Iterable[float], b: t.Iterable[float]) -> float:
            return sum(abs(x - y) for x, y in zip(a, b))

        tree = KDTree[Point2D](
            dimensions=2, point_getter=point_getter, distance_func=manhattan_distance
        )
        points = [Point2D(0, 0, "A"), Point2D(1, 2, "B"), Point2D(2, 0, "C")]
        for p in points:
            tree.add(p)

        result = tree.query([0, 0], k=2)
        labels = [p.label for p in result]
        self.assertEqual(labels, ["A", "C"])

    def test_large_k(self):
        points = [Point2D(0, 0, "A"), Point2D(1, 1, "B")]
        for p in points:
            self.tree.add(p)

        # Request more neighbors than points in tree
        result = self.tree.query([0, 0], k=5)
        self.assertEqual(len(result), 2)  # Should return all points
        labels = [p.label for p in result]
        self.assertEqual(set(labels), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
