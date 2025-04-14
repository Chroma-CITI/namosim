import py_trees
from py_trees.common import Status
from rclpy import Future

from namoros.behavior_node import NamoBehaviorNode
from nav2_msgs.action._follow_path import FollowPath_GetResult_Response
from action_msgs.msg import GoalStatus
from namoros_msgs.msg import NamoPath


class FollowPath(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        node: NamoBehaviorNode,
        namo_path: NamoPath,
        synchronize_planner: bool = False,
        is_evasion: bool = False,
    ):
        super().__init__(name="follow_path")
        self.node = node
        self.namo_path = namo_path
        self._status: Status = Status.INVALID
        self.synchronize_planner = synchronize_planner
        self.is_evasion = is_evasion

    def callback(self, future: Future):
        try:
            result: FollowPath_GetResult_Response = future.result()  # type: ignore
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self._status = Status.SUCCESS
            else:
                self._status = Status.FAILURE
                self.node.get_logger().info(f"FollowPath failed with result: {result}")
        except Exception as e:
            self.logger.error("FollowPath failed with error: {e}")
            self._status = Status.FAILURE

    def initialise(self):
        self._status = Status.RUNNING

        # advance simulation
        future = self.node.follow_path(
            self.namo_path.path,
            controller_id=(
                "TransferPath" if self.namo_path.is_transfer else "TransitPath"
            ),
        )
        future.add_done_callback(self.callback)

    def update(self):
        if self._status == Status.RUNNING:
            self.node.get_logger().info(
                "Following path." if not self.is_evasion else "Evading."
            )
            if self.synchronize_planner:
                self.node.synchronize_planner()
        self.status = self._status
        return self.status
