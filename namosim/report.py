import copy
import typing as t

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pydantic import BaseModel

import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba


class WorldStepReport(BaseModel):
    nb_components: float = 0
    biggest_component_size: float = 0
    free_space_size: float = 0
    fragmentation: float = 0
    absolute_social_cost: float = 0


class AgentStats(BaseModel):
    agent_id: str
    """The svg id attribute of the agent
    """

    n_goals: float
    """The total number of navigation goals for the agent.
    """

    n_goals_failed: float = 0
    """The number of goals the agent failed to complete.
    """

    n_goals_completed: float = 0
    """The number of goals the agent completed successfully.
    """

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

    planning_time: float = 0.0
    """The total amount of time the robot spent in planning
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

    world_steps: t.List[WorldStepReport] = []
    """A list of world statistics for each step of the simulation
    """

    def update_for_step(self, agent_id: str, action_result: ar.ActionResult):
        if agent_id not in self.agent_stats:
            raise Exception(f"Agent ${agent_id} not found in report")

        self.agent_stats[agent_id].update(action_result=action_result)

    def to_json_data(self):
        return self.model_dump()

    def get_avg_over_agents(self) -> t.Optional["SimulationReport"]:
        avg = AgentStats(agent_id="avg", n_goals=0)
        N = len(self.agent_stats)
        if N == 0:
            return

        for stats in self.agent_stats.values():
            avg.n_goals_failed += stats.n_goals_failed / N
            avg.n_goals_completed += stats.n_goals_completed / N
            avg.n_actions_failed += stats.n_actions_failed / N
            avg.n_actions_completed += stats.n_actions_completed / N
            avg.distance_traveled += stats.distance_traveled / N
            avg.degrees_rotated += stats.degrees_rotated / N
            avg.transfer_distance_traveled += stats.transfer_distance_traveled / N
            avg.transfer_degrees_rotated += stats.transfer_degrees_rotated / N
            avg.n_goals += stats.n_goals / N
            avg.postponements += stats.postponements / N
            avg.replans += stats.replans / N
            avg.planning_time += stats.planning_time / N

        return SimulationReport(agent_stats={"avg": avg})

    def sum(self, other: "SimulationReport") -> "SimulationReport":
        result = copy.deepcopy(self)

        for agent_id, stats in other.agent_stats.items():
            if agent_id not in result.agent_stats:
                result.agent_stats[agent_id] = stats
            else:
                result.agent_stats[agent_id].n_goals_failed += stats.n_goals_failed
                result.agent_stats[
                    agent_id
                ].n_goals_completed += stats.n_goals_completed
                result.agent_stats[agent_id].n_actions_failed += stats.n_actions_failed
                result.agent_stats[
                    agent_id
                ].n_actions_completed += stats.n_actions_completed
                result.agent_stats[
                    agent_id
                ].distance_traveled += stats.distance_traveled
                result.agent_stats[agent_id].degrees_rotated += stats.degrees_rotated
                result.agent_stats[
                    agent_id
                ].transfer_distance_traveled += stats.transfer_distance_traveled
                result.agent_stats[
                    agent_id
                ].transfer_degrees_rotated += stats.transfer_degrees_rotated

        return result

    def divide_by(self, divisor: float) -> "SimulationReport":
        result = copy.deepcopy(self)

        for stats in result.agent_stats.values():
            stats.n_goals_failed /= divisor
            stats.n_goals_completed /= divisor
            stats.n_actions_failed /= divisor
            stats.n_actions_completed /= divisor
            stats.distance_traveled /= divisor
            stats.degrees_rotated /= divisor
            stats.transfer_distance_traveled /= divisor
            stats.transfer_degrees_rotated /= divisor

        return result

    def plot(self):
        groups = list(self.agent_stats.keys())
        agent_goals = {"Goals Completed": []}
        agent_rotations = {"Total": [], "Transfer": []}
        agent_translations = {"Total": [], "Transfer": []}

        agent_stats_ls = list(self.agent_stats.values())
        total_goals = (
            agent_stats_ls[0].n_goals_completed + agent_stats_ls[0].n_goals_failed
        )

        for stats in self.agent_stats.values():
            agent_goals["Goals Completed"].append(stats.n_goals_completed)
            agent_rotations["Total"].append(
                stats.degrees_rotated / stats.n_goals_completed
            )
            agent_rotations["Transfer"].append(
                stats.transfer_degrees_rotated / stats.n_goals_completed
            )
            agent_translations["Total"].append(
                stats.distance_traveled / stats.n_goals_completed
            )
            agent_translations["Transfer"].append(
                stats.transfer_distance_traveled / stats.n_goals_completed
            )

        width = 0.2  # the width of the bars

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 2, figure=fig)

        # create sub plots as grid
        ax_goals = fig.add_subplot(gs[0, :])
        ax_distance = fig.add_subplot(gs[1, 0])
        ax_transfer_distance = fig.add_subplot(gs[1, 1])
        ax_rotations = fig.add_subplot(gs[2, 0])
        ax_transfer_rotations = fig.add_subplot(gs[2, 1])

        ax_goals.set_title(f"Avg Goals Completed Out of {int(total_goals)}")
        ax_goals.grid()
        ax_goals.bar(
            x=groups,
            height=agent_goals["Goals Completed"],
            width=width,
            align="center",
        )
        ax_goals.margins(y=1)
        ax_goals.tick_params(axis="x", rotation=45)

        ax_rotations.set_title("Avg Total Rotation / Avg Goals Completed")
        ax_rotations.grid()
        ax_rotations.bar(groups, agent_rotations["Total"], width=width, align="center")
        ax_rotations.set_ylabel("Degrees")
        ax_rotations.legend(loc="upper center", ncols=3)
        ax_rotations.margins(y=1)

        ax_transfer_rotations.set_title("Avg Transfer Rotation / Avg Goals Completed")
        ax_transfer_rotations.grid()
        ax_rotations.set_ylabel("Degrees")
        ax_transfer_rotations.bar(
            groups, agent_rotations["Transfer"], width=width, align="center"
        )

        ax_distance.set_title("Avg Total Distance / Avg Goals Completed")
        ax_distance.grid()
        ax_distance.bar(
            groups, agent_translations["Total"], width=width, align="center"
        )
        ax_distance.legend(loc="upper center", ncols=3)
        ax_distance.margins(y=1)

        ax_transfer_distance.set_title("Avg Transfer Distance / Avg Goals Completed")
        ax_transfer_distance.grid()
        ax_transfer_distance.bar(
            groups,
            agent_translations["Transfer"],
            width=width,
            align="center",
        )

        plt.show()
        plt.close("all")

    def plot_agent_avg(self):
        avg = self.get_avg_over_agents()
        if avg:
            avg.plot()
