import typing as t

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
        pass
