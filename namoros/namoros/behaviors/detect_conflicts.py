import py_trees
from namoros.behavior_node import NamoBehaviorNode
from py_trees.common import Status


class DetectConflicts(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="DetectConflicts")
        self.node = node

    def initialise(self):
        self.node.conflicts = []
        self.status = Status.RUNNING

    def update(self):
        self.node.detect_conflicts()
        return self.status
