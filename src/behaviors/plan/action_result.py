class ActionSuccess:
    def __init__(self, action):
        self.action = action

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
        return "Manipulation of obstacle {uid} failed.".format(uid=self.manipulated_obstacle_uid)


class UnmanipulableFailure(ManipulationFailure):
    def __init__(self, action, manipulated_obstacle_uid):
        ManipulationFailure.__init__(self, action, manipulated_obstacle_uid)

    def __str__(self):
        return "Manipulation of unmovable obstacle {uid} failed.".format(uid=self.manipulated_obstacle_uid)


class IntersectionFailure(ActionFailure):
    def __init__(self, action, obstacle_in_collision_1, obstacle_in_collision_2):
        ActionFailure.__init__(self, action)
        self.obstacle_in_collision_1 = obstacle_in_collision_1
        self.obstacle_in_collision_2 = obstacle_in_collision_2

    def __str__(self):
        return "Action failed," \
               "because of collision between {coll_obs_1} and {coll_obs_2}".format(
                    coll_obs_1=self.obstacle_in_collision_1,
                    coll_obs_2=self.obstacle_in_collision_2)
