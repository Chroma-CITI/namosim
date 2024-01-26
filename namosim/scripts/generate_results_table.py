"""
This is an ad-hoc script to compare and visualize namo simulation results
"""
import csv
import glob
import json
import os

import numpy as np
from pydantic import BaseModel

from namosim.report import AgentStats, SimulationReport


def main():
    max_robots = 7
    algs = {
        "namo": "NAMO",
        "namo_ndr": "NAMO w/o Deadlock Resolution",
        "namo_ncr": "NAMO w/o Conflict Resolution",
        "snamo": "SNAMO",
        "snamo_ndr": "SNAMO w/o Deadlock Resolution",
        "snamo_ncr": "SNAMO w/o Conflict Resolution",
    }

    with open("report.csv", "w") as fp:
        fieldnames = list(CsvRow.model_json_schema()["properties"].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for alg in algs.keys():
            for n_robots in range(1, max_robots + 1):
                dir = f"namo_logs/intersections/{n_robots}_robots_50_goals_{alg}"
                result_files = glob.glob(os.path.join(dir, "**/report.json"))

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
                        continue
                    with open(result_file) as f:
                        data = json.load(f)

                    report = SimulationReport.model_validate(data)
                    for stats in report.agent_stats.values():
                        assert np.isclose(stats.n_goals, 50)
                        row = get_csv_row(n_robots=n_robots, alg=alg, stats=stats)
                        writer.writerow(row.model_dump())


class CsvRow(BaseModel):
    n_robots: int
    algorithm: str
    n_goals: float
    n_goals_completed: float
    n_goals_failed: float
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


def get_csv_row(n_robots: int, alg: str, stats: AgentStats) -> CsvRow:
    return CsvRow(
        n_robots=n_robots,
        algorithm=alg,
        n_goals=stats.n_goals,
        n_goals_completed=stats.n_goals_completed,
        n_goals_failed=stats.n_goals_failed,
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


if __name__ == "__main__":
    main()
