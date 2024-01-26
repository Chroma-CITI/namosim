"""
This is an ad-hoc script to compare and visualize namo simulation results
"""
import glob
import json
import os
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from namosim.report import SimulationReport


def main():
    goal_success_rates: t.Dict[str, t.Dict[int, float]] = {}
    distance_traveled: t.Dict[str, t.Dict[int, float]] = {}
    replans: t.Dict[str, t.Dict[int, float]] = {}
    planning_time: t.Dict[str, t.Dict[int, float]] = {}
    n_transfers: t.Dict[str, t.Dict[int, float]] = {}
    n_conflicts: t.Dict[str, t.Dict[int, float]] = {}

    max_robots = 7
    algs = {
        "namo": "NAMO",
        "namo_ndr": "NAMO w/o Deadlock Resolution",
        "namo_ncr": "NAMO w/o Conflict Resolution",
        "snamo": "SNAMO",
        "snamo_ndr": "SNAMO w/o Deadlock Resolution",
        "snamo_ncr": "SNAMO w/o Conflict Resolution",
    }
    for alg in algs.keys():
        goal_success_rates[alg] = {}
        distance_traveled[alg] = {}
        replans[alg] = {}
        planning_time[alg] = {}
        n_transfers[alg] = {}
        n_conflicts[alg] = {}

        for n_robots in range(1, max_robots + 1):
            goal_success_rates[alg][n_robots] = 0
            distance_traveled[alg][n_robots] = 0
            replans[alg][n_robots] = 0
            planning_time[alg][n_robots] = 0
            n_transfers[alg][n_robots] = 0
            n_conflicts[alg][n_robots] = 0

            dir = f"namo_logs/intersections/{n_robots}_robots_50_goals_{alg}"

            report = SimulationReport()
            result_files = glob.glob(os.path.join(dir, "**/report.json"))

            n_skipped = 0
            for result_file in result_files:
                result_file_dir = os.path.dirname(result_file)
                print(result_file_dir)
                exceptions = glob.glob(
                    os.path.join(result_file_dir, "./exceptions.json")
                )
                if len(exceptions) != 0:
                    print(
                        f"Skipping file {result_file} because exceptions where raised during the simulation"
                    )
                    n_skipped += 1
                    continue
                with open(result_file) as f:
                    data = json.load(f)

                report = report.sum(SimulationReport.model_validate(data))
                if not report:
                    raise Exception("Failed to load results")

            report = report.get_sum_over_agents()
            print("i", n_robots)

            if len(result_files) > 0:
                n_sims = (len(result_files) - n_skipped) * n_robots
                report = report.divide_by(n_sims)
                assert np.isclose(report.agent_stats["sum"].n_goals, 50)

                goal_success_rates[alg][n_robots] = report.agent_stats[
                    "sum"
                ].n_goals_completed / (report.agent_stats["sum"].n_goals)
                distance_traveled[alg][n_robots] = report.agent_stats[
                    "sum"
                ].distance_traveled
                replans[alg][n_robots] = report.agent_stats["sum"].replans
                planning_time[alg][n_robots] = report.agent_stats["sum"].planning_time
                n_transfers[alg][n_robots] = report.agent_stats["sum"].n_transfers
                n_conflicts[alg][n_robots] = report.agent_stats["sum"].n_conflicts
            else:
                goal_success_rates[alg][n_robots] = 0

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)

    # create sub plots as grid
    ax_goals = fig.add_subplot(gs[0, 0])
    for alg, title in algs.items():
        ax_goals.plot(
            range(1, max_robots + 1),
            [goal_success_rates[alg][i] for i in range(1, max_robots + 1)],
            label=title,
        )
    # ax_goals.legend(prop={"size": 6})
    ax_goals.set_xlabel("Number of Robots")
    ax_goals.set_ylabel("Goal Success Rate")
    ax_goals.set_title("Goal Success Rates")

    ax_dist = fig.add_subplot(gs[0, 1])
    for alg, title in algs.items():
        ax_dist.plot(
            range(1, max_robots + 1),
            [distance_traveled[alg][i] for i in range(1, max_robots + 1)],
            label=title,
        )
    # ax_dist.legend()
    ax_dist.set_xlabel("Number of Robots")
    ax_dist.set_ylabel("Total Distance")
    ax_dist.set_title("Total Distance")

    ax_replans = fig.add_subplot(gs[1, 0])
    for alg, title in algs.items():
        ax_replans.plot(
            range(1, max_robots + 1),
            [replans[alg][i] for i in range(1, max_robots + 1)],
            label=title,
        )
    ax_replans.legend()
    ax_replans.set_xlabel("Number of Robots")
    ax_replans.set_ylabel("Replans")
    ax_replans.set_title("Replans")

    ax_planning_time = fig.add_subplot(gs[1, 1])
    for alg, title in algs.items():
        ax_planning_time.plot(
            range(1, max_robots + 1),
            [planning_time[alg][i] for i in range(1, max_robots + 1)],
            label=title,
        )
    # ax_planning_time.legend()
    ax_planning_time.set_xlabel("Number of Robots")
    ax_planning_time.set_ylabel("Planning Time")
    ax_planning_time.set_title("Planning Time")

    ax_transfers = fig.add_subplot(gs[2, 0])
    for alg, title in algs.items():
        ax_transfers.plot(
            range(1, max_robots + 1),
            [n_transfers[alg][i] for i in range(1, max_robots + 1)],
            label=title,
        )
    # ax_transfers.legend()
    ax_transfers.set_xlabel("Number of Robots")
    ax_transfers.set_ylabel("Transfers")
    ax_transfers.set_title("Transfers")

    ax_conflicts = fig.add_subplot(gs[2, 1])
    for alg, title in algs.items():
        ax_conflicts.plot(
            range(1, max_robots + 1),
            [n_conflicts[alg][i] for i in range(1, max_robots + 1)],
            label=title,
        )
    # ax_conflicts.legend()
    ax_conflicts.set_xlabel("Number of Robots")
    ax_conflicts.set_ylabel("Conflicts")
    ax_conflicts.set_title("Conflicts")

    # fig.legend(loc="lower center")
    plt.show()


if __name__ == "__main__":
    main()
