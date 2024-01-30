"""
This is an ad-hoc script to compare and visualize namo simulation results
"""
import glob
import json
import os
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from namosim.report import SimulationReport


def main():
    goal_success_rates: t.Dict[str, t.Dict[int, t.List[float]]] = {}
    distance_traveled: t.Dict[str, t.Dict[int, t.List[float]]] = {}
    replans: t.Dict[str, t.Dict[int, t.List[float]]] = {}
    planning_time: t.Dict[str, t.Dict[int, t.List[float]]] = {}
    n_transfers: t.Dict[str, t.Dict[int, t.List[float]]] = {}
    n_conflicts: t.Dict[str, t.Dict[int, t.List[float]]] = {}

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
            goal_success_rates[alg][n_robots] = []
            distance_traveled[alg][n_robots] = []
            replans[alg][n_robots] = []
            planning_time[alg][n_robots] = []
            n_transfers[alg][n_robots] = []
            n_conflicts[alg][n_robots] = []

            dir = f"namo_logs/intersections/{n_robots}_robots_50_goals_{alg}"

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

                report = SimulationReport.model_validate(data)
                if not report:
                    raise Exception("Failed to load results")

                report = report.get_avg_over_agents()

                # n_sims = (len(result_files) - n_skipped) * n_robots
                # report = report.divide_by(n_sims)
                assert np.isclose(report.agent_stats["avg"].n_goals, 50)

                goal_success_rates[alg][n_robots].append(
                    report.agent_stats["avg"].n_goals_completed
                    / (report.agent_stats["avg"].n_goals)
                )
                distance_traveled[alg][n_robots].append(
                    report.agent_stats["avg"].distance_traveled
                )
                replans[alg][n_robots].append(report.agent_stats["avg"].replans)
                planning_time[alg][n_robots].append(
                    report.agent_stats["avg"].planning_time
                )
                n_transfers[alg][n_robots].append(report.agent_stats["avg"].n_transfers)
                n_conflicts[alg][n_robots].append(report.agent_stats["avg"].n_conflicts)

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)

    plot_metric_by_num_robots(
        ax=fig.add_subplot(gs[0, 0]),
        algs=algs,
        max_robots=max_robots,
        metric=goal_success_rates,
        ylabel="Goal Success Rate",
        title="Goal Success Rate",
    )

    plot_metric_by_num_robots(
        ax=fig.add_subplot(gs[0, 1]),
        algs=algs,
        max_robots=max_robots,
        metric=distance_traveled,
        ylabel="Distance",
        title="Total Distance",
    )

    plot_metric_by_num_robots(
        ax=fig.add_subplot(gs[1, 0]),
        algs=algs,
        max_robots=max_robots,
        metric=replans,
        ylabel="Replans",
        title="Replans",
        show_legend=True,
    )

    plot_metric_by_num_robots(
        ax=fig.add_subplot(gs[1, 1]),
        algs=algs,
        max_robots=max_robots,
        metric=planning_time,
        ylabel="Planning Time",
        title="Planning Time",
    )

    plot_metric_by_num_robots(
        ax=fig.add_subplot(gs[2, 0]),
        algs=algs,
        max_robots=max_robots,
        metric=n_transfers,
        ylabel="Transfers",
        title="Transfers",
    )

    plot_metric_by_num_robots(
        ax=fig.add_subplot(gs[2, 1]),
        algs=algs,
        max_robots=max_robots,
        metric=n_conflicts,
        ylabel="Conflicts",
        title="Conflicts",
    )

    # fig.legend(loc="lower center")
    plt.show()


def plot_metric_by_num_robots(
    *,
    ax: Axes,
    algs: t.Dict[str, str],
    max_robots: int,
    metric: t.Dict[str, t.Dict[int, t.List[float]]],
    ylabel: str,
    title: str,
    show_legend: bool = False,
):
    for alg, title in algs.items():
        means = np.array([np.mean(metric[alg][i]) for i in range(1, max_robots + 1)])
        stds = np.array([np.std(metric[alg][i]) for i in range(1, max_robots + 1)])
        ax.plot(
            range(1, max_robots + 1),
            means,
            label=title,
        )
        ax.fill_between(
            x=range(1, max_robots + 1),
            y1=means - stds,
            y2=means + stds,
            alpha=0.2,
        )
    if show_legend:
        ax.legend()
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


if __name__ == "__main__":
    main()
