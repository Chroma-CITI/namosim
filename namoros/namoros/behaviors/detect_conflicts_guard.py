import py_trees
from namoros.behavior_node import NamoBehaviorNode
from py_trees.common import Status


class DetectConflictsGuard(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="DetectConflictsGuard")
        self.node = node

    def initialise(self):
        self.node.conflicts = []
        self.status = Status.RUNNING

    def update(self):
        conflicts = self.node.detect_conflicts()
        if len(conflicts) > 0:
            self.status = py_trees.common.Status.FAILURE
            self.node.get_logger().info("Conflict detected")
        else:
            self.status = py_trees.common.Status.SUCCESS
        return self.status
