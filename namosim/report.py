import typing as t

import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba


class AgentStats:
    def __init__(self, agent_id: int | str):
        self.agent_id = agent_id
        self.n_goals_failed: int = 0
        self.n_goals_completed: int = 0
        self.n_actions_failed: int = 0
        self.n_actions_completed: int = 0

    def update(self, action_result: ar.ActionResult):
        if isinstance(action_result, ar.ActionFailure):
            self.n_actions_failed += 1
            return

        self.n_actions_completed += 1

        if isinstance(action_result.action, ba.GoalFailed):
            self.n_goals_failed += 1
        elif isinstance(action_result.action, ba.GoalSuccess):
            self.n_goals_completed += 1

    def to_json_data(self):
        return {
            "n_goals_failed": self.n_goals_failed,
            "n_goals_completed": self.n_goals_completed,
            "n_actions_failed": self.n_actions_failed,
            "n_actions_completed": self.n_actions_completed,
        }


class SimulationReport:
    def __init__(self):
        self.agent_stats: t.Dict[int | str, AgentStats] = {}

    def update(self, agent_id: int | str, action_result: ar.ActionResult):
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = AgentStats(agent_id=agent_id)

        self.agent_stats[agent_id].update(action_result=action_result)

    def to_json_data(self):
        return {
            agent_id: agent.to_json_data()
            for agent_id, agent in self.agent_stats.items()
        }
