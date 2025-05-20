import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
import random
import math
import time

from namosim.data_models import PoseModel
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from shapely import affinity
from shapely.geometry import Polygon

from namosim.algorithms.kd_tree import KDTree as CustomKDTree
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point


def default_cost_calc(p1: PoseModel, p2: PoseModel) -> float:
    return utils.distance_between_poses(p1, p2)


@dataclass
class Node:
    pose: PoseModel
    parent: Optional["Node"] = None
    cost: float = 0.0


def default_exit_condition(pose: Node, iteration: int) -> bool:
    return False


class DiffDriveRRTStar:
    def __init__(
        self,
        polygon: Polygon,
        start: PoseModel,
        goal: PoseModel | None,
        map: BinaryOccupancyGrid,
        cost_calc=default_cost_calc,
        early_exit_condition=default_exit_condition,
        max_iter: int = 5000,
        goal_tolerance=0.1,
        use_kdtree: bool = True,
        informed: bool = True,
        exit_check_interval: int = 10,
    ):
        self.polygon = polygon
        self.start = Node(start)
        self.goal = Node(goal) if goal is not None else None
        self.map = map
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance
        self.tree: List[Node] = [self.start]
        self.rejected = []
        self.accepted = []
        self.use_kdtree = use_kdtree
        self._kdtree = None
        self.cost_calc = cost_calc
        self.early_exit_condition = early_exit_condition
        self.exit_interval = exit_check_interval

        self.max_vel = self.map.cell_size
        self.search_radius = self.map.cell_size * 5
        self.informed = informed
        # Precompute reduced control inputs
        linear_vels = [-self.max_vel * 0.5, 0, self.max_vel]
        angular_vels = np.linspace(-np.pi / 8, np.pi / 8, 3)
        self.control_inputs = [
            (v, w)
            for v in linear_vels
            for w in angular_vels
            if not (abs(v) < 1e-6 and abs(w) < 1e-6)
        ]

        # Collision cache
        self._collision_cache = {}

        if goal is not None:
            self.best_cost = float("inf")
            self.c_best = None
            self.c_min = self.cost_calc(self.start.pose, self.goal.pose)
            self.x_center = np.array(
                [
                    (self.start.pose[0] + self.goal.pose[0]) / 2,
                    (self.start.pose[1] + self.goal.pose[1]) / 2,
                ]
            )
            dx = (
                (self.goal.pose[0] - self.start.pose[0]) / self.c_min
                if self.c_min > 0
                else 1
            )
            dy = (
                (self.goal.pose[1] - self.start.pose[1]) / self.c_min
                if self.c_min > 0
                else 0
            )
            self.C = np.array([[dx, -dy], [dy, dx]])

        if self.use_kdtree:
            self._kdtree = CustomKDTree(
                dimensions=2, point_getter=lambda node: node.pose[:2]
            )
            self._kdtree.add(self.start)

        self.elapsed_time: Optional[float] = None

    def _is_collision_free(self, pose: PoseModel) -> bool:
        key = (round(pose[0], 4), round(pose[1], 4), round(pose[2], 2))
        if key in self._collision_cache:
            return self._collision_cache[key]
        node = Node(pose)
        free = self.collision_free(node)
        self._collision_cache[key] = free
        return free

    def random_pose(self) -> PoseModel:
        if self.goal and self.informed and self.c_best not in (None, float("inf")):
            a = self.c_best / 2.0
            b = math.sqrt(max(self.c_best**2 - self.c_min**2, 1e-6)) / 2.0
            while True:
                x_ball, y_ball = self._sample_unit_ball()
                # appliquer la matrice de rotation C
                x, y = (self.C @ np.array([a * x_ball, b * y_ball])) + self.x_center
                if 0 <= x <= self.map.width and 0 <= y <= self.map.height:
                    theta = random.uniform(-180, 180)
                    return (float(x), float(y), theta)
        # tirage uniforme global (cas non informé ou avant 1er chemin)
        return (
            random.uniform(0, self.map.width),
            random.uniform(0, self.map.height),
            random.uniform(-180, 180),
        )

    def nearest_node(self, pose: PoseModel) -> Node:
        if self.use_kdtree and self._kdtree:
            res = self._kdtree.query(pose[:2], k=1)
            if res:
                return res[0]
        dists = [self.cost_calc(pose, n.pose) for n in self.tree]
        return self.tree[int(np.argmin(dists))]

    def steer(self, from_node: Node, target: PoseModel, step_size=0.02) -> Node:
        x0, y0, th0 = from_node.pose
        th0_rad = utils.normalize_angle_radians(math.radians(th0))
        best_node = from_node
        best_d = float("inf")

        for v, w in self.control_inputs:
            if abs(w) < 1e-6:
                x1 = x0 + v * math.cos(th0_rad)
                y1 = y0 + v * math.sin(th0_rad)
                th1_rad = th0_rad
            else:
                x1 = x0 + (v / w) * (math.sin(th0_rad + w) - math.sin(th0_rad))
                y1 = y0 - (v / w) * (math.cos(th0_rad + w) - math.cos(th0_rad))
                th1_rad = th0_rad + w

            new_pose = (x1, y1, math.degrees(utils.normalize_angle_radians(th1_rad)))
            dx, dy = x1 - x0, y1 - y0
            dth = utils.normalize_angle_radians(th1_rad - th0_rad)
            dist = np.hypot(dx, dy)
            n_steps = max(1, int(dist / step_size))
            free_path = True

            for i in range(n_steps + 1):
                t = i / n_steps
                xi = x0 + t * dx
                yi = y0 + t * dy
                thi = utils.normalize_angle_radians(th0_rad + t * dth)
                if not self._is_collision_free((xi, yi, math.degrees(thi))):
                    free_path = False
                    break

            if free_path:
                d = self.cost_calc(new_pose, target)
                if d < best_d:
                    best_d = d
                    best_node = Node(new_pose, from_node)
                    best_node.cost = from_node.cost + self.cost_calc(
                        from_node.pose, new_pose
                    )
        if best_node.pose == from_node.pose:
            pass  # self.rejected.append(target)
        else:
            pass  # self.accepted.append(target)
        return best_node

    def collision_free(self, node: Node) -> bool:
        dx = node.pose[0] - self.start.pose[0]
        dy = node.pose[1] - self.start.pose[1]
        dth = node.pose[2] - self.start.pose[2]
        poly = affinity.rotate(self.polygon, dth, origin=self.start.pose[:2])
        poly = affinity.translate(poly, xoff=dx, yoff=dy)
        return not self.map.polygon_has_collisions(poly)

    def near_goal(self, node: Node) -> bool:
        return self.cost_calc(node.pose, self.goal.pose) <= self.goal_tolerance

    def get_near_nodes(self, node: Node) -> List[Node]:
        if self.use_kdtree and self._kdtree:
            cands = self._kdtree.query_radius(node.pose[:2], self.search_radius)
            return [n for n in cands if n is not node]
        poses = np.array([n.pose for n in self.tree])
        d = np.linalg.norm(poses[:, :2] - np.array(node.pose)[:2], axis=1)
        return [
            self.tree[i]
            for i in np.where(d < self.search_radius)[0]
            if self.tree[i] is not node
        ]

    def _sample_unit_ball(self) -> np.ndarray:
        r = math.sqrt(random.random())
        theta = 2 * math.pi * random.random()
        return np.array([r * math.cos(theta), r * math.sin(theta)])

    def plan(self) -> Optional[List[Node]]:
        t0 = time.time()
        best_path = None
        for i in range(self.max_iter):
            cfg = self.random_pose()
            if self.goal and random.random() < 0.1:
                cfg = self.goal.pose
            n0 = self.nearest_node(cfg)
            n1 = self.steer(n0, cfg)
            if not self._is_collision_free(n1.pose):
                continue
            near = self.get_near_nodes(n1)
            parent, cost = n0, n0.cost + self.cost_calc(n0.pose, n1.pose)
            for nbr in near:
                c2 = nbr.cost + self.cost_calc(nbr.pose, n1.pose)
                if c2 < cost and self._is_collision_free(n1.pose):
                    parent, cost = nbr, c2
            n1.parent, n1.cost = parent, cost
            self.tree.append(n1)
            if self.use_kdtree:
                self._kdtree.add(n1)
            for nbr in near:
                c2 = n1.cost + self.cost_calc(n1.pose, nbr.pose)
                if c2 < nbr.cost and self._is_collision_free(nbr.pose):
                    nbr.parent, nbr.cost = n1, c2
            if self.goal:
                if self.near_goal(n1):
                    path = self._get_path(n1)
                    total = path[-1].cost
                    if self.informed:
                        if total < self.best_cost:
                            self.best_cost = self.c_best = total
                            best_path = path
                    else:
                        self.elapsed_time = time.time() - t0
                        return path
            elif i % self.exit_interval == 0 and self.early_exit_condition(n1, i):
                return self.tree
        self.elapsed_time = time.time() - t0
        return best_path if self.informed else None

    def smooth_path(self, path: List[Node], max_trials: int = 100) -> List[Node]:
        if len(path) < 3:
            return path
        for _ in range(max_trials):
            if len(path) < 3:
                break
            i, j = random.randint(0, len(path) - 3), random.randint(2, len(path) - 1)
            if self._shortcut_collision_free(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path

    def _get_path(self, node: Node) -> List[Node]:
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]

    def _shortcut_collision_free(self, a: Node, b: Node, steps: int = 10) -> bool:
        x0, y0, t0 = a.pose
        x1, y1, t1 = b.pose
        for k in range(1, steps):
            alpha = k / steps
            if not self.collision_free(
                Node(
                    (
                        x0 + alpha * (x1 - x0),
                        y0 + alpha * (y1 - y0),
                        t0 + alpha * (t1 - t0),
                    )
                )
            ):
                return False
        return True

    def plot(self, path: Optional[List[Node]] = None):
        fig = plt.figure(figsize=(8, 8))
        for accepted in self.rejected:
            plt.plot(accepted[0], accepted[1], "go", markersize=10)
        for accepted in self.accepted:
            plt.plot(accepted[0], accepted[1], "ro", markersize=10)
        for n in self.tree:
            if n.parent:
                plt.plot(
                    [n.pose[0], n.parent.pose[0]],
                    [n.pose[1], n.parent.pose[1]],
                    "b-",
                    alpha=0.2,
                )
        if path:
            xs, ys = zip(*[(n.pose[0], n.pose[1]) for n in path])
            plt.plot(xs, ys, "g-", linewidth=2)
        plt.plot(self.start.pose[0], self.start.pose[1], "ro")

        if self.goal:
            plt.plot(self.goal.pose[0], self.goal.pose[1], "go", markersize=10)
            title = (
                f"RRT* Path Planning (informed={self.informed})\n"
                f"c_best={self.c_best}, c_min={self.c_min:.2f}\n"
                f"Time: {self.elapsed_time:.2f}s"
            )
        else:
            title = (
                f"RRT* Path Planning (no goal)\n"
                f"Time: {self.elapsed_time if self.elapsed_time is not None else 1.0:.2f}s\n"
                f"tree length : {len(self.tree)}"
            )

        plt.axis("equal")
        plt.grid(True)
        plt.title(title)
        plt.show()
        plt.close(fig)

    def get_tree_marker(
        self, frame_id="map", ns="rrt_star_tree", color=(0.0, 0.0, 1.0, 0.2)
    ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # line width
        marker.color = ColorRGBA(*color)
        marker.points = []
        for node in self.tree:
            if node.parent:
                p1 = Point(x=node.pose[0], y=node.pose[1], z=0)
                p2 = Point(x=node.parent.pose[0], y=node.parent.pose[1], z=0)
                marker.points.extend([p1, p2])
        return marker
