import py_trees
from py_trees.common import Status

from namoros.behavior_node import NamoBehaviorNode


class ClearNewMovables(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="ClearNewMovables")
        self.node = node

    def update(self):
        self.node.movable_obstacle_tracker.newly_detected_obstacle_ids.popleft()
        self.status = Status.SUCCESS
        return self.status
