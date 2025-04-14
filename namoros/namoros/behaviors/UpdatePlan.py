from concurrent.futures import Future, ThreadPoolExecutor

import py_trees
from namoros_msgs.msg._namo_plan import NamoPlan
from py_trees.common import Status

from namoros.behavior_node import NamoBehaviorNode


class UpdatePlan(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="UpdatePlan")
        self.node = node
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.updated_plan: NamoPlan | None = None
        self._status: Status = Status.INVALID

    def think(self):
        try:
            plan = self.node.update_plan()
            return plan
        except Exception as e:
            self.node.get_logger().error(f"Error while updating plan: {e}")
            raise e

    def callback(self, future: Future):
        try:
            plan: NamoPlan = future.result()
            if plan is None or len(plan.paths) == 0:
                self._status = Status.RUNNING
                self.node.get_logger().info("Failed to update the plan")
                self.node.trigger_a_replan()
            else:
                self.node.plan = plan
                self._status = Status.SUCCESS

        except Exception as e:
            self.node.get_logger().error(f"An error occured while updating plan: {e}")
            self._status = Status.FAILURE

    def initialise(self):
        self.node.get_logger().info("initialise UpdatePlan")
        if not self.node.goal_pose:
            raise Exception("No goal pose")
        self._status = Status.RUNNING
        future = self.executor.submit(self.think)
        future.add_done_callback(self.callback)

    def update(self):
        if self._status == Status.RUNNING:
            self.node.get_logger().info("Updating plan")
        return self._status
