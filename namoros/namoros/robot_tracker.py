import typing as t
from namoros_msgs.msg import NamoEntity


class RobotTracker:
    def __init__(self):
        self.robots: t.Dict[str, NamoEntity] = {}

    def update(self, robot: NamoEntity):
        self.robots[robot.entity_id] = robot
