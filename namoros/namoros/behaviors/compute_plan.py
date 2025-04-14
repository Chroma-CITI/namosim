import typing as t
from concurrent.futures import Future, ThreadPoolExecutor

from namoros_msgs.msg._namo_plan import NamoPlan
import py_trees
from namoros.namo_planner import NamoRosPath
from py_trees.common import Status

from namoros.behavior_node import NamoBehaviorNode


class ComputeNamoPlan(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="compute_namo_plan")
        self.node = node
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.plan: NamoPlan | None = None
        self._status: Status = Status.INVALID

    def compute_plan(self):
        try:
            plan = self.node.get_plan()
            return plan
        except Exception as e:
            self.node.get_logger().error(f"Error while computing plan: {e}")
            raise e

    def callback(self, future: Future):
        try:
            plan: NamoPlan = future.result()
            if plan is None:
                self._status = Status.FAILURE
                self.node.get_logger().info("Failed to compute NAMO plan")
            else:
                self.node.plan = plan
                self._status = Status.SUCCESS
        except Exception as e:
            self.node.get_logger().error(
                f"An error occured while computing namo plan: {e}"
            )
            self._status = Status.FAILURE

    def initialise(self):
        if not self.node.goal_pose:
            raise Exception("No goal pose")
        self._status = Status.RUNNING
        future = self.executor.submit(self.compute_plan)
        future.add_done_callback(self.callback)

    def update(self):
        if self._status == Status.RUNNING:
            self.node.get_logger().info("Computing NAMO plan")
        self.status = self._status
        return self.status
