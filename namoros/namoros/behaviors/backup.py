import py_trees
from py_trees.common import Status
from rclpy import Future
from nav2_msgs.action._back_up import BackUp_GetResult_Response
from namoros.behavior_node import NamoBehaviorNode
from action_msgs.msg import GoalStatus


class BackUp(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode, distance: float):
        super().__init__(name="BackUp")
        self.node = node
        self.distance = distance
        self._status: Status = Status.INVALID

    def callback(self, future: Future):
        try:
            result: BackUp_GetResult_Response = future.result()  # type: ignore
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self._status = Status.SUCCESS
            else:
                self._status = Status.FAILURE
                self.node.get_logger().info(f"BackUp failed with result: {result}")
        except Exception as e:
            self.node.get_logger().error(f"BackUp failed with error: {e}")
            self._status = Status.FAILURE

    def initialise(self):
        self._status = Status.RUNNING
        future = self.node.backup(self.distance)
        future.add_done_callback(self.callback)

    def update(self):
        if self._status == Status.RUNNING:
            self.node.get_logger().info("Backing up")
        self.status = self._status
        return self.status
