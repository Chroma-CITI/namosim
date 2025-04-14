import py_trees

from namoros.behavior_node import NamoBehaviorNode


class WaitForInitPose(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="wait_for_init_pose")
        self.node = node

    def update(self):
        robot_pose = self.node.lookup_robot_pose()
        if robot_pose is None:
            self.status = py_trees.common.Status.RUNNING
            self.node.get_logger().info("Waiting for robot pose")
        else:
            self.status = py_trees.common.Status.SUCCESS
        return self.status
