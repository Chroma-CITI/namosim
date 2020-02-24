class ActionGoalResult:
    def __init__(self, goal):
        self.goal = goal


class ActionGoalSuccess(ActionGoalResult):
    def __init__(self, goal):
        ActionGoalResult.__init__(self, goal)

    def __str__(self):
        return "success"


class ActionGoalFailure(ActionGoalResult):
    def __init__(self, goal):
        ActionGoalResult.__init__(self, goal)

    def __str__(self):
        return "failure"


class ActionGoalsFinished:
    def __init__(self):
        pass
