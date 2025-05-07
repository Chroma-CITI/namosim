from typing import Iterable, List, Optional


class KDNode:
    def __init__(self, point: List[float], axis: int):
        self.point = list(point)  # Convert to list for storage
        self.axis = axis
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, dimensions: int):
        self.root = None
        self.dimensions = dimensions

    def add(self, point: Iterable[float]) -> None:
        """Add a point to the KDTree."""
        point_list = list(point)
        if len(point_list) != self.dimensions:
            raise ValueError("Point dimension does not match tree dimensions")

        def _add_recursive(
            node: Optional[KDNode], point: List[float], depth: int
        ) -> KDNode:
            if node is None:
                return KDNode(point, depth % self.dimensions)

            if point[node.axis] < node.point[node.axis]:
                node.left = _add_recursive(node.left, point, depth + 1)
            else:
                node.right = _add_recursive(node.right, point, depth + 1)
            return node

        self.root = _add_recursive(self.root, point_list, 0)

    def query(self, point: Iterable[float], k: int = 1) -> List[List[float]]:
        """Find k nearest neighbor points to the query point."""
        point_list = list(point)
        if len(point_list) != self.dimensions:
            raise ValueError("Query point dimension does not match tree dimensions")
        if k < 1:
            raise ValueError("k must be positive")

        def squared_distance(p1: Iterable[float], p2: Iterable[float]) -> float:
            return sum((a - b) ** 2 for a, b in zip(p1, p2))

        # Priority queue for k nearest neighbors (distance, point)
        nearest = []

        def _query_recursive(node: Optional[KDNode], depth: int) -> None:
            if node is None:
                return

            dist = squared_distance(point_list, node.point)
            if len(nearest) < k:
                nearest.append((dist, node.point))
                nearest.sort()  # Keep smallest distances first
            elif dist < nearest[-1][0]:
                nearest[-1] = (dist, node.point)
                nearest.sort()

            axis = depth % self.dimensions
            diff = point_list[axis] - node.point[axis]

            # Search closer subtree first
            near_subtree = node.left if diff < 0 else node.right
            far_subtree = node.right if diff < 0 else node.left

            _query_recursive(near_subtree, depth + 1)

            # Check if we need to search the far subtree
            if len(nearest) < k or (diff**2) < nearest[-1][0]:
                _query_recursive(far_subtree, depth + 1)

        _query_recursive(self.root, 0)
        return [pt for _, pt in nearest]
