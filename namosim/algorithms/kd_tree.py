from typing import Iterable, List, Optional, Callable, TypeVar, Generic
import heapq
import math

T = TypeVar("T")  # Type variable for generic objects


class KDNode(Generic[T]):
    def __init__(self, object: T, point: List[float], axis: int):
        self.object = object  # Store the original object
        self.point = point  # Store the point for efficient comparisons
        self.axis = axis
        self.left = None
        self.right = None


def default_distance_func(A: Iterable[float], B: Iterable[float]):
    x = sum((a - b) ** 2 for (a, b) in zip(A, B))
    return math.sqrt(x)


class KDTree(Generic[T]):
    def __init__(
        self,
        dimensions: int,
        point_getter: Callable[[T], Iterable[float]],
        distance_func: Callable[
            [Iterable[float], Iterable[float]], float
        ] = default_distance_func,
    ):
        self.root = None
        self.dimensions = dimensions
        self.point_getter = point_getter
        self.distance_func = distance_func

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

        # Use a max-heap for k nearest neighbors (negative distance, object)
        nearest = []

        def _query_recursive(node: Optional[KDNode[T]], depth: int) -> None:
            if node is None:
                return

            dist = self.distance_func(point_list, node.point)
            if len(nearest) < k:
                heapq.heappush(nearest, (-dist, node.object))
            else:
                if dist < -nearest[0][0]:
                    heapq.heappushpop(nearest, (-dist, node.object))

            axis = depth % self.dimensions
            diff = point_list[axis] - node.point[axis]

            # Search closer subtree first
            near_subtree = node.left if diff < 0 else node.right
            far_subtree = node.right if diff < 0 else node.left

            _query_recursive(near_subtree, depth + 1)

            # Only check the far subtree if it could contain closer points
            if len(nearest) < k or (diff**2) < -nearest[0][0]:
                if far_subtree is not near_subtree and far_subtree is not None:
                    _query_recursive(far_subtree, depth + 1)

        _query_recursive(self.root, 0)
        # Return objects sorted by distance
        return [obj for _, obj in sorted(nearest, key=lambda x: -x[0])]

    def query_radius(self, point: Iterable[float], radius: float) -> List[T]:
        """Renvoie tous les objets dont la distance euclidienne au 'point' est ≤ radius."""
        target = list(point)
        result: List[T] = []

        def _search(node: Optional[KDNode[T]], depth: int):
            if node is None:
                return
            axis = depth % self.dimensions
            diff = target[axis] - node.point[axis]
            # Pruning: skip subtree if axis difference alone is greater than radius
            if abs(diff) > radius:
                if diff < 0:
                    _search(node.left, depth + 1)
                else:
                    _search(node.right, depth + 1)
                return
            # Test real distance
            dist = self.distance_func(target, node.point)
            if dist <= radius:
                result.append(node.object)
            _search(node.left, depth + 1)
            _search(node.right, depth + 1)

        _search(self.root, 0)
        return result
