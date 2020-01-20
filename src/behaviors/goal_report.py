from src.behaviors.plan.action_result import ActionSuccess
import numpy as np


class GoalReport:
    """
    Object meant to retain the following information about the goal execution:
        - Total planning duration [ ]
        - Number of replans [ ]
        -
        - Number of moved obstacles
        -
    """

    def __init__(self):
        self.plans = []
        self.actions_results = []

        self.planning_duration = 0.

    def get_transferred_obstacles_set(self):
        transferred_obstacles = set()
        for action_result in self.actions_results:
            if isinstance(action_result, ActionSuccess) and action_result.action.is_transfer:
                transferred_obstacles.add(action_result.action.obstacle_uid)
        return transferred_obstacles

    def get_transferred_obstacles_sequence(self):
        transferred_obstacles = []
        for action_result in self.actions_results:
            action = action_result.action
            if isinstance(action_result, ActionSuccess) and action.is_transfer:
                if len(transferred_obstacles) >= 1:
                    if action.obstacle_uid != transferred_obstacles[-1]:
                        transferred_obstacles.append(action.obstacle_uid)
                else:
                    transferred_obstacles.append(action.obstacle_uid)
        return transferred_obstacles

    def get_nb_transferred_obstacles(self):
        return len(self.get_transferred_obstacles_set())

    def get_total_path_lengths(self):
        transit_path_length = 0.
        transfer_path_length = 0.

        if len(self.actions_results) >= 2:
            action_result_iter = iter(self.actions_results)
            prev_action_result = next(action_result_iter)
            for action_result in action_result_iter:
                if isinstance(action_result, ActionSuccess):
                    cur_pose = action_result.action.target_pose
                    prev_pose = prev_action_result.action.target_pose
                    if action_result.action.is_transfer:
                        transfer_path_length += np.linalg.norm([cur_pose[0] - prev_pose[0], cur_pose[1] - prev_pose[1]])
                    else:
                        transit_path_length += np.linalg.norm([cur_pose[0] - prev_pose[0], cur_pose[1] - prev_pose[1]])
                    prev_action_result = action_result

        return transit_path_length, transfer_path_length

    def get_total_transit_path_length(self):
        return self.get_total_path_lengths()[0]

    def get_total_transfer_path_length(self):
        return self.get_total_path_lengths()[1]

    def get_transit_transfer_ratio(self):
        transit_path_length, transfer_path_length = self.get_total_path_lengths()
        try:
            return transit_path_length / transfer_path_length
        except ZeroDivisionError:
            return float("inf")
