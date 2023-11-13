class ActionSuccess:
    def __init__(self, action, robot_pose, is_transfer=False, obstacle_uid=None):
        self.action = action
        # TODO remove these temporary attributes
        self.robot_pose = robot_pose
        self.is_transfer = is_transfer
        self.obstacle_uid = obstacle_uid

    def __str__(self):
        return "Action was a success"


class ActionFailure:
    def __init__(self, action):
        self.action = action

    def __str__(self):
        return "Action was a failure"


class ManipulationFailure(ActionFailure):
    def __init__(self, action, manipulated_obstacle_uid):
        ActionFailure.__init__(self, action)
        self.manipulated_obstacle_uid = manipulated_obstacle_uid

    def __str__(self):
        return "Manipulation of obstacle {uid} failed.".format(
            uid=self.manipulated_obstacle_uid
        )


class UnmanipulableFailure(ManipulationFailure):
    def __init__(self, action, manipulated_obstacle_uid):
        ManipulationFailure.__init__(self, action, manipulated_obstacle_uid)

    def __str__(self):
        return "Manipulation of unmovable obstacle {uid} failed.".format(
            uid=self.manipulated_obstacle_uid
        )


class AlreadyGrabbedFailure(ActionFailure):
    def __init__(self, action, other_agent_uid):
        ActionFailure.__init__(self, action)
        self.other_agent_uid = other_agent_uid

    def __str__(self):
        return (
            "Action failed because agent {} is already grabbing this obstacle.".format(
                self.other_agent_uid
            )
        )


class NotGrabbedFailure(ActionFailure):
    def __init__(self, action):
        ActionFailure.__init__(self, action)

    def __str__(self):
        return "Action failed because the obstacle was not in a grabbed state."


class GrabbedByOtherFailure(ActionFailure):
    def __init__(self, action, other_agent_uid):
        ActionFailure.__init__(self, action)
        self.other_agent_uid = other_agent_uid

    def __str__(self):
        return (
            "Action failed because agent {} is already grabbing this obstacle.".format(
                self.other_agent_uid
            )
        )


class GrabMoreThanOneFailure(ActionFailure):
    def __init__(self, action):
        ActionFailure.__init__(self, action)

    def __str__(self):
        return "Action failed because the agent is already grabbing something."


class SimultaneousGrabFailure(ActionFailure):
    def __init__(self, action, other_agents_uids):
        ActionFailure.__init__(self, action)
        self.other_agents_uids = other_agents_uids

    def __str__(self):
        return "Action failed because several agents {} tried to grab the same entity in the same time step.".format(
            self.other_agents_uids
        )


class DynamicCollisionFailure(ActionFailure):
    def __init__(self, action, colliding_entities_uids):
        ActionFailure.__init__(self, action)
        self.colliding_entities_uids = colliding_entities_uids

    def __str__(self):
        return "Action failed, because of collision between {}.".format(
            self.colliding_entities_uids
        )
