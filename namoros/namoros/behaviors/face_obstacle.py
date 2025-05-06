import math
import py_trees
from py_trees.common import Status
from namoros.behavior_node import NamoBehaviorNode
from rclpy import Future
from nav2_msgs.action._spin import Spin_GetResult_Response
from action_msgs.msg import GoalStatus


class FaceObstacle(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode, obstacle_id: str):
        super().__init__(name="FaceObstacle")
        self.node = node
        self.obstacle_id = obstacle_id

    def callback(self, future: Future):
        try:
            result: Spin_GetResult_Response = future.result()  # type: ignore
            if (
                result.status == GoalStatus.STATUS_SUCCEEDED
                or result.status == GoalStatus.STATUS_CANCELED
            ):
                self._status = Status.SUCCESS
            else:
                self._status = Status.SUCCESS
                self.node.get_logger().info(f"Spin failed with result: {result}")
        except Exception as e:
            self.node.get_logger().error(f"Spin failed with error: {e}")
            self._status = Status.FAILURE

    def initialise(self):
        self._status = Status.RUNNING
        obstacle_pose = self.node.movable_obstacle_tracker.get_averaged_marker_pose(
            marker_id=self.obstacle_id
        )
        if obstacle_pose is None:
            self._status = Status.SUCCESS
            return

        obstacle_pose = self.node.transform_pose(obstacle_pose, f"base_link")
        if obstacle_pose is None:
            raise Exception("Failed to transform obstacle pose")
        angle = math.atan2(obstacle_pose.pose.position.y, obstacle_pose.pose.position.x)
        future = self.node.spin(angle)
        future.add_done_callback(self.callback)

    def update(self):
        if self._status == Status.RUNNING:
            self.node.get_logger().info("Turning towards obstacle")
        self.status = self._status
        return self.status
