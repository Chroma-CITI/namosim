import py_trees

from namoros.behavior_node import NamoBehaviorNode


class ReplanGuard(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="ReplanGuard")
        self.node = node

    def update(self):
        if self.node.replan_flag:
            self.status = py_trees.common.Status.RUNNING
            self.node.get_logger().info("REPLANNING...")
            self.node.replan_flag = False
        else:
            self.status = py_trees.common.Status.SUCCESS
        return self.status
