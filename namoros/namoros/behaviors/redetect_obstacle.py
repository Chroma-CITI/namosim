from namoros.config import Config
import py_trees
import time
from namoros.behavior_node import NamoBehaviorNode


class RedetectObstacle(py_trees.behaviour.Behaviour):
    def __init__(
        self, node: NamoBehaviorNode, obstacle_id: str, max_seconds: float = 5
    ):
        super().__init__(name="RedetectObstacle")
        self.node = node
        self.obtacle_id = obstacle_id
        self.start_time: float = 0
        self.max_seconds = max_seconds

    def initialise(self):
        self.node.movable_obstacle_tracker.reset_obstacle(self.obtacle_id)
        self.start_time = time.time()

    def update(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            self.status = py_trees.common.Status.SUCCESS
            self.node.get_logger().info(
                f"Failed to redetect obstacle. Continuing current plan."
            )
            return self.status

        if self.node.movable_obstacle_tracker.is_obstacle_fully_detected(
            self.obtacle_id
        ):
            self.status = py_trees.common.Status.SUCCESS
            self.node.get_logger().info(
                f"Successfully redetected obstacle {self.obtacle_id}"
            )
            self.node.movable_obstacle_tracker.update_obstacle_polygons()
            self.node.trigger_a_replan()
        else:
            n = len(
                self.node.movable_obstacle_tracker.detected_movables.get(
                    self.obtacle_id, []
                )
            )
            self.node.get_logger().info(
                f"Waiting to redetect obstacle {self.obtacle_id}. ({n}/{Config.MOVABLE_OBSTACLE_DETECTION_AVG_N})"
            )
            self.status = py_trees.common.Status.RUNNING

        return self.status
