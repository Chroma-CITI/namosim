"""
This is an ad-hoc script to compare and visualize namo simulation results
"""
import csv
import glob
import json
import os

from pydantic import BaseModel

from namosim.report import GoalStats, SimulationReport


def main():
    max_robots = 10
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
                    for agent in report.agent_stats.values():
                        assert len(agent.goal_stats) == 5
                        for stats in agent.goal_stats.values():
                            row = get_csv_row(
                                agent_id=agent.agent_id,
                                n_robots=n_robots,
                                alg=alg,
                                stats=stats,
                            )
                            writer.writerow(row.model_dump())


class CsvRow(BaseModel):
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


def get_csv_row(n_robots: int, alg: str, agent_id: str, stats: GoalStats) -> CsvRow:
    return CsvRow(
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


if __name__ == "__main__":
    main()
