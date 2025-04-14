import py_trees
from kobuki_ros_interfaces.msg import Sound

from namoros.behavior_node import NamoBehaviorNode


class PlayRandomSound(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="play_random_sound")
        self.node = node

    def initialise(self):
        self.node.play_random_sound()

    def update(self):
        self.status = py_trees.common.Status.SUCCESS
        return self.status
