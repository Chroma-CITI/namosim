from plan_step import PlanStep


class Plan:
    def __init__(self, path_components, goal):
        self.path_components = path_components
        self.goal = goal
        self.phys_cost = 0.0
        self.social_cost = 0.0
        self.total_cost = 0.0
        if path_components:
            for path in path_components:
                self.phys_cost = self.phys_cost + path.phys_cost
                self.social_cost = self.social_cost + path.social_cost
                self.total_cost = self.total_cost + path.total_cost
        else:
            self.phys_cost = float("inf")
            self.social_cost = float("inf")
            self.total_cost = float("inf")

    def append(self, future_plan):
        self.path_components += future_plan.path_components
        self.phys_cost += future_plan.phys_cost
        self.social_cost += future_plan.social_cost
        self.total_cost += future_plan.total_cost
        return self

    def has_infinite_cost(self):
        return True if self.total_cost == float("inf") else False

    def is_empty(self):
        if self.path_components:
            for path_component in self.path_components:
                if path_component.path:
                    return False
            return True
        else:
            return True

    def is_valid(self, world, robot_uid, _blocked_obstacles=None):
        blocked_obstacles = _blocked_obstacles if _blocked_obstacles is not None else set()

        if self.has_infinite_cost():
            return False
        if not self.path_components:
            return False

        for i in range(len(self.path_components)):
            path = self.path_components[i]
            # Check collisions between robot (+ manipulated obstacle if transfer path) + movability
            if not path.is_valid(world, robot_uid, blocked_obstacles):
                return False
            if not path.is_transfer:
                for j in range(i + 1, len(self.path_components)):
                    try:
                        next_path = self.path_components[i + 1]
                        if next_path.is_transfer:
                            obstacle = world.entities[next_path.obstacle_uid]
                            robot = world.entities[robot_uid]
                            if tuple(next_path.path[0]) != obstacle.get_actions(
                                    world.dd.inflation_radius, world.dd.res,
                                    robot.deduce_push_only(obstacle.type))[next_path.translation]:
                                return False
                    except (IndexError, KeyError):
                        continue
        return True

    def pop_next_step(self):
        # If the currently executed path component still has steps to execute, pop the first
        if self.path_components[0].path:
            return PlanStep(target_pose=self.path_components[0].pop_next_step(),
                            goal=self.goal,
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
                                    goal=self.goal,
                                    is_transfer=self.path_components[0].is_transfer,
                                    obstacle_uid=self.path_components[0].obstacle_uid)
            else:
                return None

    def get_all_transferred_obstacles_set(self):
        """
        Computes and returns the set of to-be transferred obstacles uids in the plan.
        :return: set of to-be transferred obstacles uids in the plan.
        :rtype: set(int)
        """
        all_transferred_obstacles = set()
        for path_component in self.path_components:
            if path_component.is_transfer:
                all_transferred_obstacles.update(path_component.obstacle_uid)
        return all_transferred_obstacles

    def get_all_transferred_obstacles_sequence(self):
        """
        Computes and returns the sequence (list) of to-be transferred obstacles uids in the plan, keeping the order
        in which they are moved.
        :return: list of to-be transferred obstacles uids in the plan.
        :rtype: list(int)
        """
        all_transferred_obstacles = []
        for path_component in self.path_components:
            if path_component.is_transfer:
                all_transferred_obstacles += path_component.obstacle_uid
        return all_transferred_obstacles

    def get_nb_all_transferred_obstacles(self):
        """
        Computes and returns the number of transferred obstacles.
        :return: number of transferred obstacles.
        :rtype: int
        """
        return len(self.get_all_transferred_obstacles_set())
