import py_trees
from kobuki_ros_interfaces.msg import Sound

from namoros.behavior_node import NamoBehaviorNode


class WaitForBumperPressed(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="wait_for_bumper")
        self.node = node

    def initialise(self) -> None:
        self.node.clear_bumper()
        self.node.request_bumper()

    def update(self):
        if self.node.bumper_pressed is True:
            self.status = py_trees.common.Status.SUCCESS
            self.node.play_sound(Sound.BUTTON)
        else:
            self.status = py_trees.common.Status.RUNNING
        return self.status
