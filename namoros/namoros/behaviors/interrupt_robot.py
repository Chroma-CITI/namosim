import typing as t
from concurrent.futures import Future, ThreadPoolExecutor

import py_trees
from namoros.namo_planner import NamoRosPath
from namoros.utils import Pose2D
from py_trees.common import Status
from std_msgs.msg import Header

from namoros.behavior_node import NamoBehaviorNode


class InterruptRobot(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="InterruptRobot")
        self.node = node
        self._status: Status = Status.INVALID

    def callback(self, future: Future):
        self._status = Status.SUCCESS

    def initialise(self):
        self._status: Status = Status.INVALID
        if not self.node.goal_pose:
            raise Exception("No goal pose")
        self._status = Status.RUNNING
        future = self.node.cancel_nav_task()
        future.add_done_callback(self.callback)

    def update(self):
        self.status = self._status
        return self.status
