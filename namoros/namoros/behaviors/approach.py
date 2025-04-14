import py_trees
from geometry_msgs.msg import Twist
from py_trees.common import Status
from namoros.behavior_node import NamoBehaviorNode


class Approach(py_trees.behaviour.Behaviour):
    def __init__(
        self, node: NamoBehaviorNode, obstacle_id: str, min_distance: float = 0.23
    ):
        super().__init__(name="Approach")
        self.node = node
        self.min_distance = min_distance
        self.obstacle_id = obstacle_id

    def update(self):
        if self.node.forward_dist_to_obstacle > self.min_distance:
            # advance
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.05
            cmd_vel.angular.z = 0.0
            self.node.publish_cmd_vel(cmd_vel)
            self.status = Status.RUNNING
            self.node.get_logger().info(
                f"Advancing towards obstacle. Distance = {self.node.forward_dist_to_obstacle}"
            )
        else:
            # stop
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.node.publish_cmd_vel(cmd_vel)
            self.status = Status.SUCCESS
            self.node.get_logger().info("Finished advancing towards obstacle")
            self.node.grab(obs_marker_id=self.obstacle_id)
        return self.status
