"""
This is an ad-hoc script to compare and visualize namo simulation results
"""

import csv
import glob
import json
import os
import typing as t

import typer
from pydantic import BaseModel

from namosim.report import GoalStats, SimulationReport

app = typer.Typer()


class CsvRow(BaseModel):
    sim_id: str
    agent_id: str
    n_robots: int
    algorithm: str
    succeeded: bool | None
    distance_traveled: float
    n_transfers: float
    planning_time: float
    n_planning_timeouts: float
    postponements: float
    replans: float
    transfer_distance_traveled: float
    n_conflicts: float
    n_rr_conflicts: float
    n_steps: float


def get_csv_row(
    sim_id: str, n_robots: int, alg: str, agent_id: str, stats: GoalStats
) -> CsvRow:
    return CsvRow(
        sim_id=sim_id,
        agent_id=agent_id,
        n_robots=n_robots,
        algorithm=alg,
        succeeded=stats.succeeded,
        distance_traveled=stats.distance_traveled,
        n_transfers=stats.n_transfers,
        planning_time=stats.planning_time,
        n_planning_timeouts=stats.n_planning_timeouts,
        postponements=stats.postponements,
        replans=stats.replans,
        transfer_distance_traveled=stats.transfer_distance_traveled,
        n_conflicts=stats.n_conflicts,
        n_rr_conflicts=stats.n_rr_conflicts,
        n_steps=stats.n_steps,
    )


@app.command()
def run(
    *,
    results_dir: t.Annotated[str, typer.Option("--results-dir")],
    out: t.Annotated[str, typer.Option("--out")] = "report.csv",
    max_robots: t.Annotated[int, typer.Option("--max-robots")] = 10,
):
    algs = {
        "namo",
        "namo_ndr",
        "namo_ncr",
        "snamo",
        "snamo_ndr",
        "snamo_ncr",
        "snamo_distance_dr",
    }

    with open(out, "w") as fp:
        fieldnames = list(CsvRow.model_json_schema()["properties"].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for alg in algs:
            for n_robots in range(1, max_robots + 1):
                dir = os.path.join(results_dir, f"{n_robots}_robots_50_goals_{alg}")
                result_files = glob.glob(os.path.join(dir, "**/report.json"))

                for i, result_file in enumerate(result_files):
                    result_file_dir = os.path.dirname(result_file)
                    print(result_file_dir)
                    exceptions = glob.glob(
                        os.path.join(result_file_dir, "./exceptions.json")
                    )
                    if len(exceptions) != 0:
                        print(
                            f"Skipping file {result_file} because exceptions where raised during the simulation"
                        )
                        continue
                    with open(result_file) as f:
                        data = json.load(f)

                    report = SimulationReport.model_validate(data)
                    for agent in report.agent_stats.values():
                        assert len(agent.goal_stats) == 50
                        for stats in agent.goal_stats.values():
                            row = get_csv_row(
                                sim_id=f"{n_robots}_{alg}_{i}",
                                agent_id=agent.agent_id,
                                n_robots=n_robots,
                                alg=alg,
                                stats=stats,
                            )
                            writer.writerow(row.model_dump())


if __name__ == "__main__":
    app()
