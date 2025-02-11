import typing as t

from namosim.data_models import PoseModel
from namosim.utils import utils


class Conflict:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__str__()


class RobotRobotConflict(Conflict):
    CONFLICT_STRING = "Robot - Robot"

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

        self.other_agent_id = other_agent_id
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
        robot_state = (
            "in transit"
            if self.robot_transfered_obstacle_uid is None
            else "transfering obstacle uid {}".format(
                self.robot_transfered_obstacle_uid
            )
        )

        s = "{} conflict between robot uid {} () and other robot uid {} - {}.".format(
            self.CONFLICT_STRING,
            self.agent_id,
            self.other_agent_id,
            robot_state,
        )

        robot_transfered_obstacle_pose_text = (
            ""
            if self.robot_transfered_obstacle_pose is None
            else "robot's transfered obstacle: {}, ".format(
                self.robot_transfered_obstacle_pose
            )
        )

        s += " Collision detected between entities {} at configuration: robot: {}, {}other robot {}.".format(
            self.colliding_uids,
            self.robot_pose,
            robot_transfered_obstacle_pose_text,
            self.other_robot_pose,
        )
        return s


class SimultaneousSpaceAccess(RobotRobotConflict):
    CONFLICT_STRING = "Simultaneous Space Access"

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


class RobotObstacleConflict(Conflict):
    def __init__(self, obstacle_uid: str):
        self.obstacle_uid = obstacle_uid

    def __str__(self):
        return "Robot-Obstacle conflict with obstacle uid {}.".format(self.obstacle_uid)

    def __repr__(self):
        return self.__str__()


class StolenMovableConflict(
    Conflict
):  # If Movable is in grabbed state, postpone, else immediate replan
    def __init__(self, obstacle_uid: str):
        self.obstacle_uid = obstacle_uid

    def __str__(self):
        return "Stolen Movable conflict concerning obstacle uid {}.".format(
            self.obstacle_uid
        )


class StealingMovableConflict(Conflict):
    def __init__(self, obstacle_uid: str, thief_uid: str):
        self.obstacle_uid = obstacle_uid
        self.thief_uid = thief_uid

    def __str__(self):
        return "Stealing Movable conflict concerning obstacle uid {} by robot uid {}.".format(
            self.obstacle_uid, self.thief_uid
        )


class ConcurrentGrabConflict(StealingMovableConflict):  # Systematic postpone
    def __init__(self, obstacle_uid: str, other_agent_id: str):
        self.obstacle_uid = obstacle_uid
        self.other_agent_id = other_agent_id

    def __str__(self):
        return "Concurrent Grab conflict concerning obstacle uid {}, with concurrent agents uid {}.".format(
            self.obstacle_uid, self.other_agent_id
        )
