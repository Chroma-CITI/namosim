# DEPRECATED CODE !


class BehaviorReport:
    def __init__(self, ref_world, behavior_world, robot_uid):
        self._ref_world = ref_world
        self._behavior_world = behavior_world
        self._robot_uid = robot_uid
        self.goal_reports = dict()

    def get_total_planning_duration(self):
        return sum(goal_report.planning_duration for goal_report in self.goal_reports.values())

    def get_all_transferred_obstacles_set(self):
        all_transferred_obstacles = set()
        for goal in self.goal_reports.values():
            all_transferred_obstacles.update(self.goal_reports[goal].get_transferred_obstacles_set())
        return all_transferred_obstacles

    def get_all_transferred_obstacles_sequence(self):
        all_transferred_obstacles = []
        for goal in self.goal_reports.values():
            all_transferred_obstacles += self.goal_reports[goal].get_transferred_obstacles_sequence()
        return all_transferred_obstacles

    def get_nb_all_transferred_obstacles(self):
        return len(self.get_all_transferred_obstacles_set())

    def get_total_transit_path_length(self):
        return sum(goal_report.get_total_transit_path_length() for goal_report in self.goal_reports.values())

    def get_total_transfer_path_length(self):
        return sum(goal_report.get_total_transfer_path_length() for goal_report in self.goal_reports.values())

    def get_total_transit_transfer_ratio(self):
        try:
            return self.get_total_transit_path_length() / self.get_total_transfer_path_length()
        except ZeroDivisionError:
            return float("inf")

    def get_absolute_social_placement_cost(self):
        """
        Computes the aggregation of the costs of all cells occupied by transferred obstacles during behavior execution,
        ignoring the order in which they were moved.
        :return: the aggregation of social costs
        """
        transferred_obstacles = tuple(self.get_all_transferred_obstacles_set())
        social_grid = self._ref_world.get_social_topological_occupation_cost_grid((self._robot_uid,) + transferred_obstacles)
        return self._ref_world.agg_grid_cost_for_entities(transferred_obstacles, social_grid)

    def get_relative_social_placement_cost(self):
        """
        Computes the aggregation of the costs of all cells occupied by transferred obstacles during behavior execution,
        taking into account the order in which it was decided they were to be moved (from last to first).
        :return: the aggregation of social costs
        """
        transferred_obstacles_sequence = self.get_all_transferred_obstacles_sequence()
        evaluated_obstacles = []
        social_cost = 0.
        for obstacle in reversed(transferred_obstacles_sequence):
            social_grid = self._ref_world.get_social_topological_occupation_cost_grid((self._robot_uid,) + tuple(evaluated_obstacles))
            evaluated_obstacles.append(obstacle)
            social_cost += self._ref_world.agg_grid_cost_for_entities(obstacle, social_grid)
        return social_cost

