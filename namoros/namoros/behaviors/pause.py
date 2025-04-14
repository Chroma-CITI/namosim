import time

from namoros.behavior_node import NamoBehaviorNode
import py_trees


class Pause(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode, seconds: float):
        super().__init__(name="pause")
        self.init_time = 0
        self.node = node
        self.seconds = seconds

    def initialise(self):
        self.init_time = time.time()

    def update(self):
        elapsed = time.time() - self.init_time
        self.node.get_logger().info(f"Pausing ({elapsed:.2f}/{self.seconds}).")
        if elapsed > self.seconds:
            self.status = py_trees.common.Status.SUCCESS
        else:
            self.status = py_trees.common.Status.RUNNING
        return self.status
