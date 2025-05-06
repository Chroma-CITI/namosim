import typing as t

from namosim.data_models import PoseModel
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
        robot_pose: PoseModel,
        other_agent_id: str,
        other_robot_pose: PoseModel,
        colliding_uids: t.Tuple[str, str],
        robot_transfered_obstacle_uid: str | None = None,
        robot_transfered_obstacle_pose: PoseModel | None = None,
        other_robot_transfered_obstacle_uid: str | None = None,
        other_robot_transfered_obstacle_pose: PoseModel | None = None,
        at_grab: bool = False,
        at_release: bool = False,
    ):
        if agent_id == other_agent_id:
            raise Exception("Invalid robot-robot conflict - robot ids are identical")
        self.agent_id = agent_id
        self.robot_pose = utils.real_pose_to_fixed_precision_pose(
            robot_pose, 100.0, 1.0
        )

        self.other_agent_id: str = other_agent_id
        self.other_robot_pose = utils.real_pose_to_fixed_precision_pose(
            other_robot_pose, 100.0, 1.0
        )

        self.colliding_uids = colliding_uids

        self.robot_transfered_obstacle_uid = robot_transfered_obstacle_uid
        self.robot_transfered_obstacle_pose = (
            None
            if robot_transfered_obstacle_pose is None
            else utils.real_pose_to_fixed_precision_pose(
                robot_transfered_obstacle_pose, 100.0, 1.0
            )
        )

        self.other_robot_transfered_obstacle_uid = other_robot_transfered_obstacle_uid
        self.other_robot_transfered_obstacle_pose = (
            None
            if other_robot_transfered_obstacle_pose is None
            else utils.real_pose_to_fixed_precision_pose(
                other_robot_transfered_obstacle_pose, 100.0, 1.0
            )
        )

        self.at_grab = at_grab
        self.at_release = at_release

    def __eq__(self, other: object):
        return (
            isinstance(other, RobotRobotConflict)
            and self.agent_id == other.agent_id
            and self.robot_pose == other.robot_pose
            and self.other_agent_id == other.other_agent_id
            and self.other_robot_pose == other.other_robot_pose
            and self.colliding_uids == other.colliding_uids
            and self.robot_transfered_obstacle_uid
            == other.robot_transfered_obstacle_uid
            and self.robot_transfered_obstacle_pose
            == other.robot_transfered_obstacle_pose
            and self.other_robot_transfered_obstacle_uid
            == other.other_robot_transfered_obstacle_uid
            and self.other_robot_transfered_obstacle_pose
            == other.other_robot_transfered_obstacle_pose
        )

    def __hash__(self):
        return hash(
            (
                self.agent_id,
                self.robot_pose,
                self.other_agent_id,
                self.other_robot_pose,
                self.colliding_uids,
                self.robot_transfered_obstacle_uid,
                self.robot_transfered_obstacle_pose,
                self.other_robot_transfered_obstacle_uid,
                self.other_robot_transfered_obstacle_pose,
            )
        )

    def __str__(self):
        return f"RobotRobotConflict({self.other_agent_id})"


class SimultaneousSpaceAccess(RobotRobotConflict):
    conflict_type = ConflictType.SSA

    def __init__(
        self,
        agent_id: str,
        robot_pose: PoseModel,
        other_agent_id: str,
        other_robot_pose: PoseModel,
        colliding_uids: t.Tuple[str, str],
        robot_transfered_obstacle_uid: str | None = None,
        robot_transfered_obstacle_pose: PoseModel | None = None,
        other_robot_transfered_obstacle_uid: str | None = None,
        other_robot_transfered_obstacle_pose: PoseModel | None = None,
        at_grab: bool = False,
        at_release: bool = False,
    ):
        RobotRobotConflict.__init__(
            self,
            agent_id,
            robot_pose,
            other_agent_id,
            other_robot_pose,
            colliding_uids,
            robot_transfered_obstacle_uid,
            robot_transfered_obstacle_pose,
            other_robot_transfered_obstacle_uid,
            other_robot_transfered_obstacle_pose,
            at_grab,
            at_release,
        )

    def __str__(self):
        return f"SimultaneousSpaceAccess({self.other_agent_id}, {self.colliding_uids})"


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

    def __init__(self, obstacle_uid: str):
        self.obstacle_uid = obstacle_uid

    def __str__(self):
        return f"StolenMovableConflict({self.obstacle_uid})."


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
