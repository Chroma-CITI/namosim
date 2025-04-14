import time
import py_trees

from namoros.behavior_node import NamoBehaviorNode
from namoros.config import Config


class WaitForFullObstacleDetection(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode, max_seconds: float = 8):
        super().__init__(name="WaitForFullObstacleDetection")
        self.node = node
        self.max_seconds = max_seconds
        self.start_time: float = 0

    def initialise(self):
        self.start_time = time.time()

    def update(self):
        marker_id = self.node.movable_obstacle_tracker.newly_detected_obstacle_ids[0]
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            self.node.get_logger().info(
                f"Timed out waiting for full obstacle detection. Continuing anyways."
            )
            self.status = py_trees.common.Status.SUCCESS
            self.node.movable_obstacle_tracker.update_obstacle_polygons()
        elif (
            self.node.movable_obstacle_tracker.is_obstacle_fully_detected(marker_id)
            is True
        ):
            self.node.get_logger().info(
                f"Movable obstacle {marker_id} is fully detected"
            )
            self.status = py_trees.common.Status.SUCCESS
            self.node.movable_obstacle_tracker.update_obstacle_polygons()
        else:
            self.node.get_logger().info(
                f"Waiting for movable obstacle {marker_id} to be fully detected"
            )
            self.status = py_trees.common.Status.RUNNING

        return self.status
