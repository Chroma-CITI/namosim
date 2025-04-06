import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random
import math

from namosim.data_models import PoseModel
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid

@dataclass
class Node:
    x: float
    y: float
    theta: float
    parent: Optional['Node'] = None
    cost: float = 0.0

class DiffDriveRRT:
    def __init__(self, start: PoseModel, 
                 goal: PoseModel,
                 map: BinaryOccupancyGrid,
                 max_iter: int = 1000,
                 step_size: float = 0.1,
                 goal_tolerance: float = 0.2):
        """
        Initialize RRT planner for differential drive robot
        start: (x, y, theta) initial pose
        goal: (x, y, theta) goal pose
        bounds: (x_min, x_max, y_min, y_max) workspace boundaries
        """
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.map = map
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.tree: List[Node] = [self.start]
        
        # Robot parameters
        self.wheel_radius = 0.05  # meters
        self.wheel_base = 0.2    # meters
        

    def random_pose(self) -> PoseModel:
        """Generate random configuration in workspace"""
        x = random.uniform(0, self.map.width)
        y = random.uniform(0, self.map.height)
        theta = random.uniform(-180, 180)
        return (x, y, theta)

    def nearest_node(self, pose: PoseModel) -> Node:
        """Find nearest node in tree to given pose"""
        distances = [math.sqrt((node.x - pose[0])**2 + 
                             (node.y - pose[1])**2) 
                    for node in self.tree]
        return self.tree[np.argmin(distances)]

    def steer(self, from_node: Node, to_pose: PoseModel) -> Node:
        """Steer from nearest node towards random config with diff-drive constraints"""
        # Calculate desired heading
        dx = to_pose[0] - from_node.x
        dy = to_pose[1] - from_node.y
        desired_theta = math.degrees(math.atan2(dy, dx))
        
        # Simple differential drive kinematics
        theta_diff = utils.angle_to_360_interval(desired_theta - from_node.theta)
        distance = min(self.step_size, math.sqrt(dx**2 + dy**2))
        
        # Update position and orientation
        new_theta = from_node.theta + theta_diff
        new_x = from_node.x + distance * math.cos(math.radians(new_theta))
        new_y = from_node.y + distance * math.sin(math.radians(new_theta))
        
        new_node = Node(new_x, new_y, new_theta, from_node)
        new_node.cost = from_node.cost + distance
        
        return new_node

    def collision_free(self, node: Node) -> bool:
        """Check if node is in collision with obstacles"""
        for obs_x, obs_y, obs_r in self.obstacles:
            dist = math.sqrt((node.x - obs_x)**2 + (node.y - obs_y)**2)
            if dist < obs_r + 0.1:  # Adding robot radius buffer
                return False
        # Check boundaries
        return (self.x_min <= node.x <= self.x_max and 
                self.y_min <= node.y <= self.y_max)

    def near_goal(self, node: Node) -> bool:
        """Check if node is near goal"""
        dist = math.sqrt((node.x - self.goal.x)**2 + (node.y - self.goal.y)**2)
        return dist < self.goal_tolerance

    def plan(self) -> Optional[List[Node]]:
        """Main RRT planning algorithm"""
        for _ in range(self.max_iter):
            rand_config = self.random_pose()
            
            # Occasionally sample goal directly
            if random.random() < 0.1:
                rand_config = (self.goal.x, self.goal.y, self.goal.theta)
                
            nearest = self.nearest_node(rand_config)
            new_node = self.steer(nearest, rand_config)
            
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
        plt.figure(figsize=(10, 10))
        
        # Plot obstacles
        for obs_x, obs_y, obs_r in self.obstacles:
            circle = patches.Circle((obs_x, obs_y), obs_r, color='r', alpha=0.5)
            plt.gca().add_artist(circle)
        
        # Plot tree
        for node in self.tree:
            if node.parent:
                plt.plot([node.x, node.parent.x], 
                        [node.y, node.parent.y], 'b-', alpha=0.2)
        
        # Plot path
        if path:
            path_x = [node.x for node in path]
            path_y = [node.y for node in path]
            plt.plot(path_x, path_y, 'g-', linewidth=2)
        
        # Plot start and goal
        plt.plot(self.start.x, self.start.y, 'bo', markersize=10)
        plt.plot(self.goal.x, self.goal.y, 'go', markersize=10)
        
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.grid(True)
        plt.axis('equal')
        plt.title('RRT Path Planning for Differential Drive Robot')
        plt.show()

def main():
    # Define start and goal poses (x, y, theta)
    start = (0.0, 0.0, 0.0)
    goal = (3.0, 3.0, 0.0)
    bounds = (-0.5, 3.5, -0.5, 3.5)
    
    # Create and run RRT planner
    planner = DiffDriveRRT(start, goal, bounds)
    path = planner.plan()
    
    # Visualize results
    planner.plot(path)
    
    if path:
        print(f"Path found with {len(path)} nodes")
        for i, node in enumerate(path):
            print(f"Node {i}: x={node.x:.2f}, y={node.y:.2f}, theta={node.theta:.2f}")
    else:
        print("No path found")

if __name__ == "__main__":
    main()