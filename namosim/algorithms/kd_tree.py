from typing import Iterable, List, Optional, Callable, TypeVar, Generic

T = TypeVar("T")  # Type variable for generic objects


class KDNode(Generic[T]):
    def __init__(self, object: T, point: List[float], axis: int):
        self.object = object  # Store the original object
        self.point = point  # Store the point for efficient comparisons
        self.axis = axis
        self.left = None
        self.right = None


class KDTree(Generic[T]):
    def __init__(self, dimensions: int, point_getter: Callable[[T], Iterable[float]]):
        self.root = None
        self.dimensions = dimensions
        self.point_getter = point_getter

    def add(self, object: T) -> None:
        """Add an object to the KDTree."""
        point_list = list(self.point_getter(object))
        if len(point_list) != self.dimensions:
            raise ValueError("Point dimension does not match tree dimensions")

        def _add_recursive(
            node: Optional[KDNode[T]], object: T, point: List[float], depth: int
        ) -> KDNode[T]:
            if node is None:
                return KDNode(object, point, depth % self.dimensions)

            if point[node.axis] < node.point[node.axis]:
                node.left = _add_recursive(node.left, object, point, depth + 1)
            else:
                node.right = _add_recursive(node.right, object, point, depth + 1)
            return node

        self.root = _add_recursive(self.root, object, point_list, 0)

    def query(self, point: Iterable[float], k: int = 1) -> List[T]:
        """Find k nearest neighbor objects to the query point."""
        point_list = list(point)
        if len(point_list) != self.dimensions:
            raise ValueError("Query point dimension does not match tree dimensions")
        if k < 1:
            raise ValueError("k must be positive")

        def squared_distance(p1: Iterable[float], p2: Iterable[float]) -> float:
            return sum((a - b) ** 2 for a, b in zip(p1, p2))

        # Priority queue for k nearest neighbors (distance, object)
        nearest = []

        def _query_recursive(node: Optional[KDNode[T]], depth: int) -> None:
            if node is None:
                return

            dist = squared_distance(point_list, node.point)
            if len(nearest) < k:
                nearest.append((dist, node.object))
                nearest.sort()  # Keep smallest distances first
            elif dist < nearest[-1][0]:
                nearest[-1] = (dist, node.object)
                nearest.sort()

            axis = depth % self.dimensions
            diff = point_list[axis] - node.point[axis]

            # Search closer subtree first
            near_subtree = node.left if diff < 0 else node.right
            far_subtree = node.right if diff < 0 else node.left

            _query_recursive(near_subtree, depth + 1)

            # Check if we need to search the far subtree
            if len(nearest) < k or (diff ** 2) < nearest[-1][0]:
                _query_recursive(far_subtree, depth + 1)

        _query_recursive(self.root, 0)
        return [obj for _, obj in nearest]
