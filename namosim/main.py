import glob
import json
import os
import typing as t

import typer

from namosim.report import SimulationReport
from namosim.scenario_generation import generate_alternative_scenarios
from namosim.simulator import Simulator

app = typer.Typer()


@app.command()
def run(
    scenario: str,
    logs_dir: t.Annotated[t.Optional[str], typer.Option("--logs-dir")] = None,
):
    sim = Simulator(simulation_file_path=scenario, logs_dir=logs_dir)
    sim.run()


@app.command()
def visualize_results(results_file: str, avg: bool = False):
    with open(results_file) as f:
        data = json.load(f)
        report = SimulationReport.model_validate(data["report"])

        if avg:
            report.plot_agent_avg()
        else:
            report.plot()


@app.command()
def compare_results(
    *,
    result_dirs_str: t.Annotated[
        str, typer.Option("--result-dirs", help="A comma-separated list of result dirs")
    ],
    titles_str: t.Annotated[
        str,
        typer.Option(
            "--titles", help="A comma-separated list of titles for each result dir"
        ),
    ],
):
    combined = SimulationReport()

    result_dirs: t.List[str] = [x.strip() for x in result_dirs_str.split(",")]
    titles: t.List[str] = [x.strip() for x in titles_str.split(",")]

    for dir, title in zip(result_dirs, titles):
        report = SimulationReport()
        result_files = glob.glob(os.path.join(dir, "**/stats.json"))

        n_skipped = 0
        for result_file in result_files:
            result_file_dir = os.path.dirname(result_file)
            print(result_file_dir)
            exceptions = glob.glob(os.path.join(result_file_dir, "./exceptions.json"))
            if len(exceptions) != 0:
                print(
                    f"Skipping file {result_file} because exceptions where raised during the simulation"
                )
                n_skipped += 1
                continue
            with open(result_file) as f:
                data = json.load(f)
            report = report.sum(SimulationReport.model_validate(data["report"]))
            if not report:
                raise Exception("Failed to load results")

        report = report.divide_by(len(result_files) - n_skipped)

        avg = report.get_avg_over_agents()
        if avg:
            combined.agent_stats[title] = avg.agent_stats["avg"]

    combined.plot()


@app.command()
def gen_alt_scenarios(
    *,
    scenario: t.Annotated[str, typer.Option("--base-scenario")],
    out_dir: t.Annotated[str, typer.Option("--out-dir")],
    n_robots: t.Annotated[int, typer.Option("--n-robots")] = 1,
    goals_per_robot: t.Annotated[int, typer.Option("--goals-per-robot")] = 50,
    n_scenarios: t.Annotated[int, typer.Option("--n-scenarios")] = 1,
    use_social_cost: t.Annotated[bool, typer.Option("--use-social-cost")] = False,
    no_resolve_deadlocks: t.Annotated[
        bool, typer.Option("--no-resolve-deadlocks")
    ] = False,
    no_resolve_conflicts: t.Annotated[
        bool, typer.Option("--no-resolve-conflicts")
    ] = False,
):
    generate_alternative_scenarios(
        base_svg_filepath=scenario,
        nb_robots=n_robots,
        nb_goals_per_robot=goals_per_robot,
        nb_scenarios=n_scenarios,
        use_social_cost=use_social_cost,
        resolve_deadlocks=not no_resolve_deadlocks,
        resolve_conflicts=not no_resolve_conflicts,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    app()
