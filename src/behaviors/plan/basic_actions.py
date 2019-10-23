class ActionGoalResult:
    def __init__(self):
        pass


class ActionGoalSuccess(ActionGoalResult):
    def __init__(self):
        ActionGoalResult.__init__(self)


class ActionGoalFailure(ActionGoalResult):
    def __init__(self):
        ActionGoalResult.__init__(self)


class ActionGoalsFinished:
    def __init__(self, report):
        self.report = report
