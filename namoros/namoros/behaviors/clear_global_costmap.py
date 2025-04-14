import py_trees
from py_trees.common import Status
from rclpy import Future
from nav2_msgs.srv._clear_costmap_around_robot import ClearCostmapAroundRobot_Response
from namoros.behavior_node import NamoBehaviorNode


class ClearGlobalCostmap(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="ClearGlobalCostmap")
        self.node = node
        self._status: Status = Status.INVALID

    def callback(self, future: Future):
        try:
            result: ClearCostmapAroundRobot_Response = future.result()  # type: ignore
            self._status = Status.SUCCESS
        except Exception as e:
            self.node.get_logger().error(f"ClearGlobalCostmap failed with error: {e}")
            self._status = Status.FAILURE

    def initialise(self):
        self._status = Status.RUNNING
        future = self.node.clear_global_costmap()
        future.add_done_callback(self.callback)

    def update(self):
        if self._status == Status.RUNNING:
            self.node.get_logger().info("Clearing global costap")
        self.status = self._status
        return self.status
