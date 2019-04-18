from plan_step import PlanStep


class Plan:
    def __init__(self, path_components):
        self.path_components = path_components
        self.cost = 0.0
        if path_components:
            for path in path_components:
                self.cost = self.cost + path.cost
        else:
            self.cost = float("inf")

    def has_infinite_cost(self):
        return True if self.cost == float("inf") else False

    def is_not_empty(self):
        return bool(self.path_components)

    def is_valid(self, world):
        if self.has_infinite_cost():
            return False
        if not self.path_components:
            return False

        for i in range(len(self.path_components)):
            path = self.path_components[i]
            # Check collisions between robot (+ manipulated obstacle if transfer path) + movability
            if not path.is_valid(world):
                return False
            if not path.is_transfer:
                for j in range(i + 1, len(self.path_components)):
                    try:
                        next_path = self.path_components[i + 1]
                        if (next_path.is_transfer and tuple(next_path.path[0]) !=
                                world.entities[next_path.obstacle_uid].actions[next_path.translation]):
                            return False
                    except (IndexError, KeyError):
                        continue
        return True

    def pop_next_step(self):
        # If the currently executed path component still has steps to execute, pop the first
        if self.path_components[0].path:
            return PlanStep(target_pose=self.path_components[0].pop_next_step(),
                            is_transfer=self.path_components[0].is_transfer,
                            obstacle_uid=self.path_components[0].obstacle_uid)
        else:
            # If the plan still has components to execute after the one we just finished,
            if self.path_components:
                # pop the finished one,
                self.path_components.pop(0)
                # and send back the second element of the next path, since the last element of
                # the previous path is the same.
                if self.path_components:
                    self.path_components[0].pop_next_step()
                    return PlanStep(target_pose=self.path_components[0].pop_next_step(),
                                    is_transfer=self.path_components[0].is_transfer,
                                    obstacle_uid=self.path_components[0].obstacle_uid)
            else:
                return None
