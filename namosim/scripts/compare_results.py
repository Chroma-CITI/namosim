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

    max_robots = 9
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
        for i in range(1, max_robots + 1):
            goal_success_rates[alg][i] = 0

            dir = f"namo_logs/intersections/{i}_robots_50_goals_{alg}"

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

                print(data["report"]["agent_stats"])
                report = report.sum(SimulationReport.model_validate(data["report"]))
                if not report:
                    raise Exception("Failed to load results")

            report = report.divide_by(len(result_files) - n_skipped)

            avg = report.get_avg_over_agents()

            if avg:
                print(avg.agent_stats["avg"].n_goals)
                assert np.isclose(avg.agent_stats["avg"].n_goals, 50)
                goal_success_rates[alg][i] = avg.agent_stats[
                    "avg"
                ].n_goals_completed / (avg.agent_stats["avg"].n_goals)

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(1, 1, figure=fig)

    # create sub plots as grid
    ax_goals = fig.add_subplot(gs[0, :])

    for alg, title in algs.items():
        ax_goals.plot(
            range(1, max_robots + 1),
            [goal_success_rates[alg][i] for i in range(1, max_robots + 1)],
            label=title,
        )
    ax_goals.legend()
    ax_goals.set_xlabel("Number of Robots")
    ax_goals.set_ylabel("Goal Success Rate")
    ax_goals.set_title("Goal Success Rates")
    plt.show()


if __name__ == "__main__":
    main()
