import py_trees

from namoros.behavior_node import NamoBehaviorNode


class NewObstacleGuard(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="NewObstacleGuard")
        self.node = node

    def update(self):
        if len(self.node.movable_obstacle_tracker.newly_detected_obstacle_ids) > 0:
            self.status = py_trees.common.Status.FAILURE
        else:
            self.status = py_trees.common.Status.SUCCESS
        return self.status
