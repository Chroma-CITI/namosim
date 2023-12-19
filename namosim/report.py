import typing as t

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba


class AgentStats(BaseModel):
    agent_id: str
    """The svg id attribute of the agent
    """

    n_goals_failed: int = 0
    """The number of goals the agent failed to complete.
    """

    n_goals_completed: int = 0
    """The number of goals the agent completed successfully.
    """

    n_actions_failed: int = 0
    """The number of actions the agent failed to complete.
    """

    n_actions_completed: int = 0
    """The number of actions the agent completed successfully.
    """

    distance_traveled: float = 0.0
    """Total amount the traveled, under any circumstance.
    """

    degrees_rotated: float = 0.0
    """Total amount the robot rotated, in degrees.
    """

    transfer_distance_traveled: float = 0.0
    """Total distance the agent traveled while carrying an obstacle
    """

    transfer_degrees_rotated: float = 0.0
    """Total distance the agent rotated while carrying an obstacle
    """

    def update(self, action_result: ar.ActionResult):
        if not isinstance(action_result, ar.ActionSuccess):
            self.n_actions_failed += 1
            return

        self.n_actions_completed += 1

        action = action_result.action

        if isinstance(action, ba.GoalFailed):
            self.n_goals_failed += 1
        elif isinstance(action, ba.GoalSuccess):
            self.n_goals_completed += 1
        elif isinstance(action, ba.Translation):
            self.distance_traveled += float(action.translation_length)
            if action_result.is_transfer:
                self.transfer_distance_traveled += float(action.translation_length)
        elif isinstance(action, ba.Rotation):
            self.degrees_rotated += abs(float(action.angle))
            if action_result.is_transfer:
                self.transfer_degrees_rotated += abs(float(action.angle))


class SimulationReport(BaseModel):
    agent_stats: t.Dict[str, AgentStats] = {}

    def update(self, agent_id: str, action_result: ar.ActionResult):
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = AgentStats(agent_id=agent_id)

        self.agent_stats[agent_id].update(action_result=action_result)

    def to_json_data(self):
        return self.model_dump()

    def plot(self):
        goal_attributes = (
            "Goals Completed",
            "Goals Failed",
        )
        action_attributes = (
            "Actions Completed",
            "Actions Failed",
        )
        rotation_attributes = (
            "Total Rotation",
            "Transfer Rotation",
        )
        distance_attributes = (
            "Total Distance",
            "Transfer Distance",
        )
        agent_goals = {agent_id: [] for agent_id in self.agent_stats.keys()}
        agent_actions = {agent_id: [] for agent_id in self.agent_stats.keys()}
        agent_rotations = {agent_id: [] for agent_id in self.agent_stats.keys()}
        agent_distance = {agent_id: [] for agent_id in self.agent_stats.keys()}

        for agent_id, stats in self.agent_stats.items():
            agent_goals[agent_id].append(stats.n_goals_completed)
            agent_goals[agent_id].append(stats.n_goals_failed)
            agent_actions[agent_id].append(stats.n_actions_completed)
            agent_actions[agent_id].append(stats.n_actions_failed)
            agent_rotations[agent_id].append(stats.degrees_rotated)
            agent_rotations[agent_id].append(stats.transfer_degrees_rotated)
            agent_distance[agent_id].append(stats.distance_traveled)
            agent_distance[agent_id].append(stats.transfer_distance_traveled)
        x = np.arange(len(goal_attributes))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0

        fig, ((ax_goals, ax_actions), (ax_rotations, ax_distance)) = plt.subplots(
            2, 2, layout="constrained"
        )

        for agent_id, measurement in agent_goals.items():
            offset = (width) * multiplier
            rects = ax_goals.bar(
                x + offset, measurement, width, label=agent_id, align="edge"
            )
            ax_goals.bar_label(rects, padding=3)
            multiplier += 1

        multiplier = 0
        for agent_id, measurement in agent_actions.items():
            offset = (width) * multiplier
            rects = ax_actions.bar(
                x + offset, measurement, width, label=agent_id, align="edge"
            )
            ax_actions.bar_label(rects, padding=3)
            multiplier += 1

        multiplier = 0
        for agent_id, measurement in agent_rotations.items():
            offset = (width) * multiplier
            rects = ax_rotations.bar(
                x + offset, measurement, width, label=agent_id, align="edge"
            )
            ax_rotations.bar_label(rects, padding=3)
            multiplier += 1

        multiplier = 0
        for agent_id, measurement in agent_distance.items():
            offset = (width) * multiplier
            rects = ax_distance.bar(
                x + offset, measurement, width, label=agent_id, align="edge"
            )
            ax_distance.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax_goals.set_title("Goals")
        ax_goals.set_xticks(x + width, goal_attributes)
        ax_goals.legend(loc="upper center", ncols=3)

        ax_actions.set_title("Actions")
        ax_actions.set_xticks(x + width, action_attributes)
        ax_actions.legend(loc="upper center", ncols=3)

        ax_rotations.set_ylabel("Degrees")
        ax_rotations.set_title("Rotations")
        ax_rotations.set_xticks(x + width, rotation_attributes)
        ax_rotations.legend(loc="upper center", ncols=3)

        ax_distance.set_title("Distances")
        ax_distance.set_xticks(x + width, distance_attributes)
        ax_distance.legend(loc="upper center", ncols=3)

        ax_goals.margins(y=1)
        ax_actions.margins(y=1)
        ax_rotations.margins(y=1)
        ax_distance.margins(y=1)

        plt.show()
        plt.close("all")
