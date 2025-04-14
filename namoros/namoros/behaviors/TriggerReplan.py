import py_trees

from namoros.behavior_node import NamoBehaviorNode
from py_trees.common import Status


class TriggerReplan(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode, update_plan: bool = False):
        super().__init__(name="TriggerReplan")
        self.node = node
        self.update_plan = update_plan
        self.triggered = False

    def initialise(self):
        self.triggered = False

    def update(self):
        if not self.triggered:
            if self.update_plan:
                self.node.trigger_update_plan()
            else:
                self.node.trigger_a_replan()
            self.triggered = True
            return Status.RUNNING
        return Status.SUCCESS
