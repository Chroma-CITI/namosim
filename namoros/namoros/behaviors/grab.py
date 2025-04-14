import py_trees
from py_trees.common import Status
from rclpy import Future
from nav2_msgs.action._compute_path_to_pose import ComputePathToPose_GetResult_Response
from nav2_msgs.action._smooth_path import SmoothPath_GetResult_Response
from nav2_msgs.action._follow_path import FollowPath_GetResult_Response
from namoros.behavior_node import NamoBehaviorNode
from action_msgs.msg import GoalStatus
from namoros_msgs.msg import NamoPath
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class Grab(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode, path: NamoPath):
        super().__init__(name="Grab")
        self.node = node
        self.path = path
        self._status: Status = Status.INVALID

    def follow_path_callback(self, future: Future):
        try:
            result: FollowPath_GetResult_Response = future.result()  # type: ignore
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self._status = Status.SUCCESS
                self.node.grab(self.path.obstacle_id)
            else:
                self._status = Status.FAILURE
                self.node.get_logger().error(f"FollowPath failed with result: {result}")
        except Exception as e:
            self.node.get_logger().error(f"FollowPath failed with error: {e}")
            self._status = Status.FAILURE

    def get_path_callback(self, future: Future):
        try:
            result: ComputePathToPose_GetResult_Response = future.result()  # type: ignore
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                result.result.path

                follow_path_future = self.node.follow_path(
                    path=result.result.path, controller_id="TransferPath"
                )
                follow_path_future.add_done_callback(self.follow_path_callback)

            else:
                self._status = Status.FAILURE
                self.node.get_logger().error(
                    f"ComputePath failed with result: {result}"
                )
        except Exception as e:
            self.node.get_logger().error(f"ComputePath failed with error: {e}")
            self._status = Status.FAILURE

    def smooth_path_callback(self, future: Future):
        try:
            result: SmoothPath_GetResult_Response = future.result()  # type: ignore
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                result.result.path
                follow_path_future = self.node.follow_path(
                    path=result.result.path, controller_id="TransferPath"
                )
                follow_path_future.add_done_callback(self.follow_path_callback)

            else:
                self._status = Status.FAILURE
                self.node.get_logger().error(f"SmoothPath failed with result: {result}")
        except Exception as e:
            self.node.get_logger().error(f"SmoothPath failed with error: {e}")
            self._status = Status.FAILURE

    def follow_path(self, path: Path):
        follow_path_future = self.node.follow_path(
            path=path, controller_id="TransferPath"
        )
        follow_path_future.add_done_callback(self.follow_path_callback)

    def initialise(self):
        if not self.path.obstacle_id:
            raise Exception("Path has no obstacle id")
        self._status = Status.RUNNING
        robot_pose = self.node.lookup_robot_pose()
        if robot_pose is None:
            raise Exception("No robot pose")
        grab_path = self.get_grab_path(robot_pose)
        self.follow_path(grab_path)

    def get_grab_path(self, robot_pose: PoseStamped):
        path_msg = Path()
        path_msg.header = self.path.path.header
        path_msg.poses = []
        path_msg.poses.append(robot_pose)
        path_msg.poses.append(self.path.path.poses[1])  # type: ignore
        return path_msg

    def update(self):
        self.status = self._status
        return self.status
