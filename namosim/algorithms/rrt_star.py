import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import random
import math
import time

from namosim.data_models import Pose2D
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from shapely import affinity
from shapely.geometry import Polygon, box, Point

from namosim.algorithms.kd_tree import KDTree as CustomKDTree
from namosim.algorithms.rrt_node import RRTNode
import typing as t


def default_cost_calc(p1: Pose2D, p2: Pose2D) -> float:
    return utils.distance_between_poses(p1, p2)


def default_exit_condition(
    tree: t.List[RRTNode], node: RRTNode, iteration: int
) -> bool:
    return False


class DiffDriveRRTStar:
    def __init__(
        self,
        polygon: Polygon,
        start: Pose2D,
        goal: Pose2D | None,
        map: BinaryOccupancyGrid,
        early_exit_condition: t.Callable[
            [t.List[RRTNode], RRTNode, int], bool
        ] = default_exit_condition,
        max_iter: int = 10000,
        goal_tolerance=0.1,
        use_kdtree: bool = True,
        informed: bool = True,
        exit_check_interval: int = 3,
    ):
        self.polygon = polygon
        self.start = RRTNode(start)
        self.goal = RRTNode(goal) if goal is not None else None
        self.map = map
        self.map_box = box(0, 0, self.map.width, self.map.height)
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance
        self.tree: List[RRTNode] = [self.start]
        self.rejected = []
        self.accepted = []
        self.use_kdtree = use_kdtree
        self._kdtree = None
        self.early_exit_condition = early_exit_condition
        self.exit_interval = exit_check_interval

        self.max_vel = self.map.cell_size
        self.max_angular_vel = np.pi / 8

        self.search_radius = self.map.cell_size * 5
        self.informed = informed
        # Precompute reduced control inputs
        linear_vels = [-self.max_vel, 0, self.max_vel]
        angular_vels = np.linspace(-self.max_angular_vel, self.max_angular_vel, 3)

        # Arc actions
        self.control_inputs = [
            (v, w)
            for v in linear_vels
            for w in angular_vels
            if not (abs(v) < 1e-6 and abs(w) < 1e-6)
        ]

        # Translations and in-place rotations.
        # self.control_inputs = [
        #     (-self.max_vel, 0),
        #     (self.max_vel, 0),
        #     (0, -self.max_angular_vel),
        #     (0, self.max_angular_vel),
        # ]

        # Collision cache
        self._collision_cache = {}

        self._visited_poses: t.Set[t.Tuple[int, int, int]] = set()

        if goal is not None:
            self.best_cost = float("inf")
            self.c_best = None
            self.c_min = self.cost(self.start.pose, self.goal.pose)
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

            def distance_func(a: t.Iterable[float], b: t.Iterable[float]):
                return self.cost(a, b)  # type: ignore

            self._kdtree = CustomKDTree[RRTNode](
                dimensions=3,
                point_getter=lambda node: node.pose,
                distance_func=distance_func,
            )
            self._kdtree.add(self.start)

        self.elapsed_time: Optional[float] = None

    def cost(self, a: Pose2D, b: Pose2D) -> float:
        # Extract coordinates and angles
        x1, y1, theta1 = a
        x2, y2, theta2 = b

        # Euclidean distance between positions
        position_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate smallest angle difference (in radians)
        theta1_rad = math.radians(theta1)
        theta2_rad = math.radians(theta2)
        angle_diff = abs((theta1_rad - theta2_rad + math.pi) % (2 * math.pi) - math.pi)

        angle_cost = angle_diff / self.max_angular_vel
        linear_cost = position_distance / self.max_vel

        cost = linear_cost + 2 * angle_cost

        return cost

    def _is_collision_free(self, pose: Pose2D) -> bool:
        key = (round(pose[0], 4), round(pose[1], 4), round(pose[2], 2))
        if key in self._collision_cache:
            return self._collision_cache[key]
        node = RRTNode(pose)
        free = self.collision_free(node)
        self._collision_cache[key] = free
        return free

    def random_pose(self) -> Pose2D:
        if self.goal and self.informed and self.c_best not in (None, float("inf")):
            a = self.c_best / 2.0
            b = math.sqrt(max(self.c_best**2 - self.c_min**2, 1e-6)) / 2.0
            while True:
                x_ball, y_ball = self._sample_unit_ball()
                # appliquer la matrice de rotation C
                x, y = (self.C @ np.array([a * x_ball, b * y_ball])) + self.x_center
                if 0 <= x <= self.map.width and 0 <= y <= self.map.height:
                    theta = random.uniform(-180, 180)
                    return Pose2D(float(x), float(y), theta)
        # tirage uniforme global (cas non informé ou avant 1er chemin)
        return Pose2D(
            random.uniform(0, self.map.width),
            random.uniform(0, self.map.height),
            random.uniform(-180, 180),
        )

    def nearest_node(self, pose: Pose2D) -> RRTNode:
        if self.use_kdtree and self._kdtree:
            res = self._kdtree.query(pose, k=1)
            if res:
                return res[0]
        dists = [self.cost(pose, n.pose) for n in self.tree]
        return self.tree[int(np.argmin(dists))]

    def steer(
        self, from_node: RRTNode, target: Pose2D, step_size=0.01
    ) -> RRTNode | None:
        x0, y0, th0 = from_node.pose
        th0_rad = utils.normalize_angle_radians(math.radians(th0))
        best_node: RRTNode | None = None
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

            new_pose = Pose2D(
                x1, y1, math.degrees(utils.normalize_angle_radians(th1_rad))
            )
            dx, dy = x1 - x0, y1 - y0
            dth = utils.normalize_angle_radians(th1_rad - th0_rad)
            dist = np.hypot(dx, dy)
            n_steps = max(1, int(dist / step_size))
            # free_path = True

            # for i in range(n_steps + 1):
            #     t = i / n_steps
            #     xi = x0 + t * dx
            #     yi = y0 + t * dy
            #     thi = utils.normalize_angle_radians(th0_rad + t * dth)
            #     if not self._is_collision_free((xi, yi, math.degrees(thi))):
            #         free_path = False
            #         break

            free_path = self._is_collision_free(new_pose)
            if free_path:
                d = self.cost(new_pose, target)
                if d < best_d:
                    best_d = d
                    best_node = RRTNode(new_pose, from_node)
                    best_node.cost = from_node.cost + self.cost(
                        from_node.pose, new_pose
                    )

        return best_node

    def collision_free(self, node: RRTNode) -> bool:
        dx, dy, dtheta = (
            node.pose[0] - self.start.pose[0],
            node.pose[1] - self.start.pose[1],
            node.pose[2] - self.start.pose[2],
        )
        new_polygon = affinity.rotate(
            self.polygon, origin=(self.start.pose[0], self.start.pose[1]), angle=dtheta  # type: ignore
        )
        new_polygon = affinity.translate(new_polygon, xoff=dx, yoff=dy)
        if not self.map_box.contains(new_polygon):
            return False

        collision_free = not self.map.polygon_has_collisions(new_polygon)
        # if collision_free:
        #     debug_img = self.map.draw_polygon_on_map(polygon=new_polygon)
        #     debug_img.save('debug_img.png')
        return collision_free

    def predict_polygon_for_node(self, node: RRTNode, polygon: Polygon) -> Polygon:
        """
        For a given polygon with a fixed pose relative to the robot, this function computes an updated polygon
        corresponding to the robot's new pose at the given node.
        """
        dx = node.pose[0] - self.start.pose[0]
        dy = node.pose[1] - self.start.pose[1]
        dth = node.pose[2] - self.start.pose[2]
        polygon = affinity.rotate(polygon, dth, origin=self.start.pose[:2])  # type: ignore
        polygon = affinity.translate(polygon, xoff=dx, yoff=dy)
        return polygon

    def predict_pose_for_node(self, node: RRTNode, pose: Pose2D) -> Pose2D:
        """
        For a given pose that is fixed relative to the robot's start pose, this function computes where
        that pose should be given robot's pose at the given node.
        """
        dx = node.pose[0] - self.start.pose[0]
        dy = node.pose[1] - self.start.pose[1]
        dth = node.pose[2] - self.start.pose[2]

        next_position = affinity.translate(
            geom=Point((pose[0], pose[1])),
            xoff=dx,
            yoff=dy,
            zoff=0.0,
        ).coords[0]
        next_position = affinity.rotate(
            geom=Point((next_position[0], next_position[1])),
            angle=dth,
            origin=(node.pose[0], node.pose[1]),  # type: ignore
            use_radians=False,
        ).coords[0]
        orientation = utils.normalize_angle_degrees(pose[2] + dth)
        return Pose2D(next_position[0], next_position[1], orientation)

    def get_polygon_at_node(self, node: RRTNode) -> Polygon:
        dx = node.pose[0] - self.start.pose[0]
        dy = node.pose[1] - self.start.pose[1]
        dth = node.pose[2] - self.start.pose[2]
        polygon = affinity.rotate(self.polygon, dth, origin=self.start.pose[:2])  # type: ignore
        polygon = affinity.translate(polygon, xoff=dx, yoff=dy)
        return polygon

    def near_goal(self, node: RRTNode) -> bool:
        return (
            utils.distance_between_poses(node.pose, self.goal.pose)
            <= self.goal_tolerance
        )

    def get_near_nodes(self, node: RRTNode) -> List[RRTNode]:
        if self.use_kdtree and self._kdtree:
            cands = self._kdtree.query_radius(node.pose, self.search_radius)
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

    def plan(self) -> Optional[List[RRTNode]]:
        t0 = time.time()
        best_path = None
        for i in range(self.max_iter):
            cfg = self.random_pose()
            if self.goal and random.random() < 0.1:
                cfg = self.goal.pose
            n0 = self.nearest_node(cfg)
            n1 = self.steer(n0, cfg)

            if n1 is None or not self.collision_free(n1):
                continue

            n1_discretized_pose = self._pose_to_fixed_precision(n1.pose)
            if n1_discretized_pose in self._visited_poses:
                continue

            near = self.get_near_nodes(n1)
            cost = n0.cost + self.cost(n0.pose, n1.pose)
            for neighbor in near:
                dth = abs(neighbor.pose[2] - n1.pose[2])
                if dth > self.max_angular_vel:
                    continue
                c2 = neighbor.cost + self.cost(neighbor.pose, n1.pose)
                if c2 < cost:
                    n1.parent = neighbor
                    n1.cost = c2

            self.add_new_node(n1)

            for neighbor in near:
                if neighbor == n1.parent:
                    continue
                c2 = n1.cost + self.cost(n1.pose, neighbor.pose)
                if c2 < neighbor.cost:
                    neighbor.parent = n1
                    neighbor.cost = c2

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
            elif i % self.exit_interval == 0 and self.early_exit_condition(
                self.tree, n1, i
            ):
                return self.tree
        self.elapsed_time = time.time() - t0
        return best_path if self.informed else None

    def add_new_node(self, node: RRTNode):
        discretized_pose = self._pose_to_fixed_precision(node.pose)
        self._visited_poses.add(discretized_pose)
        self.tree.append(node)
        if self.use_kdtree:
            self._kdtree.add(node)

    def debug_plan(self, path: List[RRTNode]):
        for i, node in enumerate(path):
            polygon = self.get_polygon_at_node(node)
            debug_img = self.map.draw_polygon_on_map(polygon=polygon)
            utils.save_image(debug_img, f"rrt_plan/robot_in_map_{i}.png")

    def smooth_path(self, path: List[RRTNode], max_trials: int = 100) -> List[RRTNode]:
        if len(path) < 3:
            return path
        for _ in range(max_trials):
            if len(path) < 3:
                break
            i, j = random.randint(0, len(path) - 3), random.randint(2, len(path) - 1)
            if self._shortcut_collision_free(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path

    def _get_path(self, node: RRTNode | None) -> List[RRTNode]:
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]

    def _shortcut_collision_free(self, a: RRTNode, b: RRTNode, steps: int = 10) -> bool:
        x0, y0, t0 = a.pose
        x1, y1, t1 = b.pose
        for k in range(1, steps):
            alpha = k / steps
            if not self.collision_free(
                RRTNode(
                    Pose2D(
                        x0 + alpha * (x1 - x0),
                        y0 + alpha * (y1 - y0),
                        t0 + alpha * (t1 - t0),
                    )
                )
            ):
                return False
        return True

    def plot(self, path: Optional[List[RRTNode]] = None):
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

    def _pose_to_fixed_precision(self, pose: Pose2D) -> t.Tuple[int, int, int]:
        return utils.real_pose_to_fixed_precision_pose(
            pose,
            2 / self.max_vel,
            1 / math.degrees(self.max_angular_vel),
        )
