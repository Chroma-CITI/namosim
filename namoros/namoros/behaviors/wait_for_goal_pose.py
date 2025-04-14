import py_trees

from namoros.behavior_node import NamoBehaviorNode


class WaitForGoalPose(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="wait_for_goal_pose")
        self.node = node

    def update(self):
        if self.node.goal_pose is None:
            self.status = py_trees.common.Status.RUNNING
            self.node.get_logger().info("Waiting for goal pose")
        else:
            self.status = py_trees.common.Status.SUCCESS
        return self.status
