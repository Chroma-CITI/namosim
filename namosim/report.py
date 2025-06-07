import typing as t

import numpy as np
from pydantic import BaseModel

import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
from namosim.agents.agent import ThinkResult
from namosim.data_models import Pose2D
from namosim.navigation.conflict import RobotRobotConflict


class WorldStepReport(BaseModel):
    nb_components: float = 0
    biggest_component_size: float = 0
    free_space_size: float = 0
    fragmentation: float = 0
    absolute_social_cost: float = 0


class GoalStats(BaseModel):
    goal_pose: Pose2D
    """The goal pose"""

    succeeded: bool | None = None
    """Whether or not the agent attained this goal"""

    n_actions_failed: float = 0
    """The number of actions the agent failed to complete.
    """

    n_actions_completed: float = 0
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

    postponements: float = 0.0
    """The number of times the robot postponed its current plan
    """

    replans: float = 0.0
    """The number of times the robot computed a plan
    """

    initial_planning_time: float = 0.0
    """The amount of time the robot spent planning on the first step.
    """

    planning_time: float = 0.0
    """The total amount of time the robot spent in planning
    """

    n_transfers: float = 0.0
    """The total number of obstacle transfers
    """

    n_planning_timeouts: float = 0.0
    """The total number of times the agent timed out
    """

    n_conflicts: float = 0.0
    """The total number of conflicts of any type
    """

    n_rr_conflicts: float = 0.0
    """The total number of robot-robot conflicts
    """

    n_steps: float = 0.0
    """The total number of simulation steps until the goal was failed or succeeded.
    """

    def update(
        self,
        *,
        think_result: ThinkResult,
        action_result: ar.ActionResult,
        planning_time: float,
    ):
        self.planning_time += planning_time
        if self.n_steps == 0:
            self.initial_planning_time = planning_time

        self.n_steps += 1

        if think_result.did_replan:
            self.replans += 1
        if think_result.did_postpone:
            self.postponements += 1

        if isinstance(action_result, ar.ActionSuccess):
            self.n_actions_completed += 1
            action = action_result.action

            if isinstance(action, ba.GoalFailed):
                self.succeeded = False
                if action.is_timeout:
                    self.n_planning_timeouts += 1
            if isinstance(action, ba.GoalSuccess):
                self.succeeded = True
            if isinstance(action, ba.Advance):
                self.distance_traveled += np.abs(action.distance)
                if action_result.is_transfer:
                    self.transfer_distance_traveled += np.abs(action.distance)
            if isinstance(action, ba.Translation):
                self.distance_traveled += np.abs(action.length)
                if action_result.is_transfer:
                    self.transfer_distance_traveled += np.abs(action.length)
            if isinstance(action, ba.Rotation):
                self.degrees_rotated += abs(float(action.angle))
                if action_result.is_transfer:
                    self.transfer_degrees_rotated += abs(float(action.angle))
            if isinstance(action, ba.Release):
                self.n_transfers += 1
        else:
            self.n_actions_failed += 1

        conflicts = set(think_result.conflicts)
        if len(conflicts) > 0:
            self.n_conflicts += len(conflicts)

            for conflict in conflicts:
                if isinstance(conflict, RobotRobotConflict):
                    self.n_rr_conflicts += 1


class AgentStats(BaseModel):
    agent_id: str
    """The svg id attribute of the agent
    """

    goal_stats: t.Dict[str, GoalStats] = {}
    """A dictionary of stats for each of the agent's goals"""

    def update(
        self,
        *,
        think_result: ThinkResult,
        action_result: ar.ActionResult,
        planning_time: float,
    ):
        goal_pose = think_result.goal_pose

        if goal_pose is None:
            return

        if str(goal_pose) not in self.goal_stats:
            self.goal_stats[str(goal_pose)] = GoalStats(goal_pose=goal_pose)

        self.goal_stats[str(goal_pose)].update(
            think_result=think_result,
            action_result=action_result,
            planning_time=planning_time,
        )


class SimulationReport(BaseModel):
    agent_stats: t.Dict[str, AgentStats] = {}

    world_steps: t.List[WorldStepReport] = []
    """A list of world statistics for each step of the simulation
    """

    def update_for_step(
        self,
        *,
        agent_id: str,
        think_result: ThinkResult,
        action_result: ar.ActionResult,
        planning_time: float,
    ):
        if agent_id not in self.agent_stats:
            raise Exception(f"Agent ${agent_id} not found in report")

        self.agent_stats[agent_id].update(
            think_result=think_result,
            action_result=action_result,
            planning_time=planning_time,
        )

    def to_json_data(self):
        return self.model_dump()

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))
