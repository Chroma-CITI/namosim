import typing as t

import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba


class AgentStats:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        """The svg id attribute of the agent
        """

        self.n_goals_failed: int = 0
        """The number of goals the agent failed to complete.
        """

        self.n_goals_completed: int = 0
        """The number of goals the agent completed successfully.
        """

        self.n_actions_failed: int = 0
        """The number of actions the agent failed to complete.
        """

        self.n_actions_completed: int = 0
        """The number of actions the agent completed successfully.
        """

        self.distance_traveled: float = 0.0
        """Total amount the traveled, under any circumstance.
        """

        self.degrees_rotated: float = 0.0
        """Total amount the robot rotated, in degrees.
        """

        self.transfer_distance_traveled: float = 0.0
        """Total distance the agent traveled while carrying an obstacle
        """

        self.transfer_degrees_rotated: float = 0.0
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
            self.distance_traveled += action.translation_length
            if action_result.is_transfer:
                self.transfer_distance_traveled += action.translation_length
        elif isinstance(action, ba.Rotation):
            self.degrees_rotated += abs(action.angle)
            if action_result.is_transfer:
                self.transfer_degrees_rotated += abs(action.angle)

    def to_json_data(self):
        return {
            "n_goals_failed": self.n_goals_failed,
            "n_goals_completed": self.n_goals_completed,
            "n_actions_failed": self.n_actions_failed,
            "n_actions_completed": self.n_actions_completed,
        }


class SimulationReport:
    def __init__(self):
        self.agent_stats: t.Dict[str, AgentStats] = {}

    def update(self, agent_id: str, action_result: ar.ActionResult):
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = AgentStats(agent_id=agent_id)

        self.agent_stats[agent_id].update(action_result=action_result)

    def to_json_data(self):
        return {
            agent_id: agent.to_json_data()
            for agent_id, agent in self.agent_stats.items()
        }
