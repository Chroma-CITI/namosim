import typing as t

from namosim.data_models import Pose2D
from namosim.utils import utils
from enum import Enum, auto


class ConflictType(Enum):
    ROBOT_ROBOT = auto()
    SSA = auto()
    ROBOT_OBSTACLE = auto()
    STOLEN_OBSTACLE = auto()
    SIMULTAEOUS_GRAB = auto()


class BaseConflict:
    def __repr__(self):
        return str(self)


class RobotRobotConflict(BaseConflict):
    conflict_type = ConflictType.ROBOT_OBSTACLE

    def __init__(
        self,
        agent_id: str,
        robot_pose: Pose2D,
        other_agent_id: str,
        other_robot_pose: Pose2D,
    ):
        if agent_id == other_agent_id:
            raise Exception("Invalid robot-robot conflict - robot ids are identical")
        self.agent_id = agent_id
        self.robot_pose = utils.real_pose_to_fixed_precision_pose(
            robot_pose, 20.0, 1 / 15
        )

        self.other_agent_id: str = other_agent_id
        self.other_robot_pose = utils.real_pose_to_fixed_precision_pose(
            other_robot_pose, 20.0, 1 / 15
        )

    def __eq__(self, other: object):
        return (
            isinstance(other, RobotRobotConflict)
            and self.agent_id == other.agent_id
            and self.robot_pose == other.robot_pose
            and self.other_agent_id == other.other_agent_id
            and self.other_robot_pose == other.other_robot_pose
        )

    def __hash__(self):
        return hash(
            (
                self.agent_id,
                self.robot_pose,
                self.other_agent_id,
                self.other_robot_pose,
            )
        )

    def __str__(self):
        return f"RobotRobotConflict({self.other_agent_id})"


class SimultaneousSpaceAccess(RobotRobotConflict):
    conflict_type = ConflictType.SSA

    def __init__(
        self,
        agent_id: str,
        robot_pose: Pose2D,
        other_agent_id: str,
        other_robot_pose: Pose2D,
    ):
        RobotRobotConflict.__init__(
            self,
            agent_id,
            robot_pose,
            other_agent_id,
            other_robot_pose,
        )

    def __str__(self):
        return f"SimultaneousSpaceAccess({self.other_agent_id})"


class RobotObstacleConflict(BaseConflict):
    conflict_type = ConflictType.ROBOT_OBSTACLE

    def __init__(self, obstacle_uid: str):
        self.obstacle_uid = obstacle_uid

    def __str__(self):
        return f"RobotObstacleConflict({self.obstacle_uid})"

    def __repr__(self):
        return self.__str__()


class StolenMovableConflict(BaseConflict):
    # If Movable is in grabbed state, postpone, else immediate replan
    conflict_type = ConflictType.STOLEN_OBSTACLE

    def __init__(self, obstacle_uid: str, expected_pose: Pose2D, actual_pose: Pose2D):
        self.obstacle_uid = obstacle_uid
        self.expected_pose = expected_pose
        self.actual_pose = actual_pose

    def __str__(self):
        return f"StolenMovableConflict({self.obstacle_uid}, expected_pose={self.expected_pose}, actual_pose={self.actual_pose})."


class StealingMovableConflict(BaseConflict):
    conflict_type = ConflictType.STOLEN_OBSTACLE

    def __init__(self, obstacle_uid: str, thief_uid: str):
        self.obstacle_uid = obstacle_uid
        self.thief_uid = thief_uid

    def __str__(self):
        return f"StealingMovableConflict({self.obstacle_uid}, {self.thief_uid})"


class ConcurrentGrabConflict(StealingMovableConflict):  # Systematic postpone
    conflict_type = ConflictType.SIMULTAEOUS_GRAB

    def __init__(self, obstacle_uid: str, other_agent_id: str):
        self.obstacle_uid = obstacle_uid
        self.other_agent_id = other_agent_id

    def __str__(self):
        return f"ConcurrentGrabConflict({self.obstacle_uid}, {self.other_agent_id})"


Conflict = t.Union[
    RobotRobotConflict,
    RobotObstacleConflict,
    SimultaneousSpaceAccess,
    StolenMovableConflict,
    StealingMovableConflict,
    ConcurrentGrabConflict,
]
