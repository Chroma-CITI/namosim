class PlanStep:
    def __init__(self, target_pose, goal, is_transfer=False, obstacle_uid=None):
        self.target_pose = target_pose
        self.goal = goal
        self.is_transfer = is_transfer
        self.obstacle_uid = obstacle_uid
