import py_trees

from namoros.behavior_node import NamoBehaviorNode
from py_trees.common import Status


class PlaySound(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode, sound: int):
        super().__init__(name="play_sound")
        self.node = node
        self.sound = sound
        self._status = Status.INVALID

    def initialise(self):
        self._status = Status.RUNNING

    def update(self):
        if self._status == Status.RUNNING:
            self.node.play_sound(sound=self.sound)
            self._status = Status.SUCCESS
        return self._status
