import typing as t

from namosim.data_models import Pose2D
from namosim.navigation.basic_actions import Action


class ActionResult:
    def __init__(self, action: Action | None = None):
        self.action = action


class ActionSuccess(ActionResult):
    def __init__(
        self,
        action: Action | None = None,
        robot_pose: Pose2D | None = None,
        is_transfer: bool = False,
        obstacle_uid: t.Optional[str] = None,
    ):
        super().__init__(action)
        self.robot_pose = robot_pose
        self.is_transfer = is_transfer
        self.obstacle_uid = obstacle_uid

    def __str__(self):
        return "Action was a success"


class ActionFailure(ActionResult):
    def __init__(self, action: Action):
        super().__init__(action)

    def __str__(self) -> str:
        return "Action was a failure"


class ManipulationFailure(ActionFailure):
    def __init__(self, action: Action, manipulated_obstacle_uid: str):
        super().__init__(action)
        self.manipulated_obstacle_uid = manipulated_obstacle_uid

    def __str__(self) -> str:
        return "Manipulation of obstacle {uid} failed.".format(
            uid=self.manipulated_obstacle_uid
        )


class UnmanipulableFailure(ManipulationFailure):
    def __init__(self, action: Action, manipulated_obstacle_uid: str):
        super().__init__(action, manipulated_obstacle_uid)

    def __str__(self):
        return "Manipulation of unmovable obstacle {uid} failed.".format(
            uid=self.manipulated_obstacle_uid
        )


class AlreadyGrabbedFailure(ActionFailure):
    def __init__(self, action: Action, other_agent_uid: str):
        super().__init__(action)
        self.other_agent_uid = other_agent_uid

    def __str__(self) -> str:
        return (
            "Action failed because agent {} is already grabbing this obstacle.".format(
                self.other_agent_uid
            )
        )


class NotGrabbedFailure(ActionFailure):
    def __init__(self, action: Action):
        super().__init__(action)

    def __str__(self) -> str:
        return "Action failed because the obstacle was not in a grabbed state."


class GrabbedByOtherFailure(ActionFailure):
    def __init__(self, action: Action, other_agent_uid: str):
        super().__init__(action)
        self.other_agent_uid = other_agent_uid

    def __str__(self) -> str:
        return (
            "Action failed because agent {} is already grabbing this obstacle.".format(
                self.other_agent_uid
            )
        )


class GrabMoreThanOneFailure(ActionFailure):
    def __init__(self, action: Action):
        super().__init__(action)

    def __str__(self):
        return "Action failed because the agent is already grabbing something."


class SimultaneousGrabFailure(ActionFailure):
    def __init__(self, action: Action, other_agents_uids: t.Iterable[str]):
        super().__init__(action)
        self.other_agents_uids = other_agents_uids

    def __str__(self) -> str:
        return "Action failed because several agents {} tried to grab the same entity in the same time step.".format(
            self.other_agents_uids
        )


class DynamicCollisionFailure(ActionFailure):
    def __init__(self, action: Action, colliding_entities_uids: t.Iterable[str]):
        super().__init__(action)
        self.colliding_entities_uids = colliding_entities_uids

    def __str__(self) -> str:
        return "Action failed, because of collision between {}.".format(
            self.colliding_entities_uids
        )
