from dataclasses import dataclass
import random
from typing import List, Optional

from namosim.data_models import PoseModel
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from shapely.geometry import Polygon
from namosim.algorithms.kd_tree import KDTree
import math
from namosim.utils import utils
import matplotlib.pyplot as plt
from shapely import affinity
import numpy as np


@dataclass
class Node:
    pose: PoseModel
    parent: Optional["Node"] = None
    cost: float = 0.0


class DiffDriveRRT:
    def __init__(
        self,
        polygon: Polygon,
        start: PoseModel,
        goal: PoseModel,
        map: BinaryOccupancyGrid,
        max_iter: int = 10000,
        goal_tolerance=0.2,
        use_kd_tree: bool = True,
    ):
        """
        Initialize RRT planner for differential drive robot
        start: (x, y, theta) initial pose
        goal: (x, y, theta) goal pose
        bounds: (x_min, x_max, y_min, y_max) workspace boundaries
        """
        self.polygon = polygon
        self.start = Node(start)
        self.goal = Node(goal)
        self.map = map
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance
        self.tree: List[Node] = [self.start]
        self.use_kd_tree = use_kd_tree

        def point_getter(node: Node):
            return (node.pose[0], node.pose[1])

        self.kd_tree = KDTree(dimensions=2, point_getter=point_getter)
        self.kd_tree.add(self.start)

        # Robot parameters
        self.max_vel = self.map.cell_size * 2

    def random_pose(self) -> PoseModel:
        """Generate random configuration in workspace"""
        x = random.uniform(0, self.map.width)
        y = random.uniform(0, self.map.height)
        theta = random.uniform(-180, 180)
        return (x, y, theta)

    def nearest_node(self, pose: PoseModel) -> Node:
        """Find nearest node in tree to given pose"""

        if self.use_kd_tree:
            nearest_node = self.kd_tree.query((pose[0], pose[1]))[0]
            return nearest_node

        distances = [
            utils.distance_between_poses(pose, node.pose) for node in self.tree
        ]
        return self.tree[np.argmin(distances)]

    def steer(self, from_node: Node, target: PoseModel) -> Node:
        """Steer by testing ranges of linear and angular velocities towards target"""
        x0, y0, theta0 = from_node.pose
        theta0_rad = utils.normalize_angle_radians(math.radians(theta0))

        # Define ranges of linear and angular velocities
        linear_vels = np.linspace(-self.max_vel, self.max_vel, 3)
        angular_vels = np.linspace(-np.pi / 8, np.pi / 8, 5)

        # Create combinations of control inputs
        control_inputs = [(v, w) for v in linear_vels for w in angular_vels]

        best_node = from_node
        best_distance = float("inf")

        # Simulate each control input
        for v, w in control_inputs:
            if v == 0 and w == 0:
                continue

            # Calculate new pose based on velocity inputs
            if abs(w) < 1e-6:  # Straight line motion
                x_new = x0 + v * math.cos(theta0_rad)
                y_new = y0 + v * math.sin(theta0_rad)
                theta_new_rad = theta0_rad
            else:  # Arc motion
                x_new = x0 + (v / w) * (math.sin(theta0_rad + w) - math.sin(theta0_rad))
                y_new = y0 - (v / w) * (math.cos(theta0_rad + w) - math.cos(theta0_rad))
                theta_new_rad = theta0_rad + w  # Don't normalize here yet

            # Normalize the new angle relative to the target to avoid 180-degree flips
            # theta_new_rad = utils.normalize_angle_radians(theta_new_rad)
            theta_new_degrees = utils.normalize_angle_degrees(
                math.degrees(theta_new_rad)
            )
            new_pose = (x_new, y_new, theta_new_degrees)

            # Calculate distance to target with proper angle difference
            distance_to_target = utils.distance_between_poses(new_pose, target)

            # Create temporary node for collision checking
            temp_node = Node(new_pose)

            if distance_to_target < best_distance and self.collision_free(temp_node):
                best_distance = distance_to_target
                best_node = Node(new_pose, from_node)
                best_node.cost = from_node.cost + utils.distance_between_poses(
                    from_node.pose, new_pose
                )

        return best_node

    def collision_free(self, node: Node) -> bool:
        dx, dy, dtheta = (
            node.pose[0] - self.start.pose[0],
            node.pose[1] - self.start.pose[1],
            node.pose[2] - self.start.pose[2],
        )
        new_polygon = affinity.rotate(
            self.polygon, origin=(self.start.pose[0], self.start.pose[1]), angle=dtheta
        )
        new_polygon = affinity.translate(new_polygon, xoff=dx, yoff=dy)

        occupied = self.map.polygon_has_collisions(new_polygon)

        # debug_img = self.map.draw_polygon_on_map(polygon=new_polygon)
        # debug_img.save('debug_img.png')

        return occupied == False

    def near_goal(self, node: Node) -> bool:
        """Check if node is near goal"""
        return (
            utils.distance_between_poses(node.pose, self.goal.pose)
            <= self.goal_tolerance
        )

    def plan(self) -> Optional[List[Node]]:
        """Main RRT planning algorithm"""
        for n in range(self.max_iter):
            rand_config = self.random_pose()

            # Occasionally sample goal directly
            if random.random() < 0.1:
                rand_config = self.goal.pose

            nearest = self.nearest_node(rand_config)
            new_node = self.steer(nearest, rand_config)

            if self.use_kd_tree:
                self.kd_tree.add(new_node)

            if self.collision_free(new_node):
                self.tree.append(new_node)

                if self.near_goal(new_node):
                    path = self._get_path(new_node)
                    return path

        return None  # No path found

    def _get_path(self, node: Node) -> List[Node]:
        """Extract path from goal to start"""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]

    def plot(self, path: Optional[List[Node]] = None):
        """Visualize the RRT and path"""
        fig = plt.figure(figsize=(10, 10))

        # Plot tree
        for node in self.tree:
            if node.parent:
                plt.plot(
                    [node.pose[0], node.parent.pose[0]],
                    [node.pose[1], node.parent.pose[1]],
                    "b-",
                    alpha=0.2,
                )

        # Plot path
        if path:
            path_x = [node.pose[0] for node in path]
            path_y = [node.pose[1] for node in path]
            plt.plot(path_x, path_y, "g-", linewidth=2)

        # Plot start and goal
        # plt.plot(self.start.pose[0], self.start.pose[1], "bo", markersize=10)
        # plt.plot(self.goal.pose[0], self.goal.pose[1], "go", markersize=10)

        plt.xlim(0, self.map.width)
        plt.ylim(0, self.map.height)
        plt.grid(True)
        plt.axis("equal")
        plt.title("RRT Path Planning for Differential Drive Robot")
        plt.show()
        plt.close(fig)
