import json
import typing as t

import typer

from namosim.report import SimulationReport
from namosim.scenario_generation import generate_alternative_scenarios
from namosim.simulator import Simulator

app = typer.Typer()


@app.command()
def run(scenario: str):
    sim = Simulator(simulation_file_path=scenario)
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
    results_a: t.Annotated[str, typer.Option("--results-a")],
    results_b: t.Annotated[str, typer.Option("--results-b")],
    title_a: t.Annotated[str, typer.Option("--title-a")],
    title_b: t.Annotated[str, typer.Option("--title-b")],
):
    with open(results_a) as f:
        data = json.load(f)
        report_a = SimulationReport.model_validate(data["report"]).get_agent_average()

    with open(results_b) as f:
        data = json.load(f)
        report_b = SimulationReport.model_validate(data["report"]).get_agent_average()

    if not report_a or not report_b:
        raise Exception("Failed to load results")

    combined = SimulationReport()
    combined.agent_stats[title_a] = report_a.agent_stats["avg"]
    combined.agent_stats[title_b] = report_b.agent_stats["avg"]
    combined.plot()


@app.command()
def gen_alt_scenarios(
    *,
    scenario: t.Annotated[str, typer.Option("--base-scenario")],
    out_dir: t.Annotated[str, typer.Option("--out-dir")],
    n_robots: t.Annotated[int, typer.Option("--n-robots")] = 4,
    goals_per_robot: t.Annotated[int, typer.Option("--goals-per-robot")] = 25,
    n_scenarios: t.Annotated[int, typer.Option("--n-scenarios")] = 1,
    use_social_cost: t.Annotated[bool, typer.Option("--use-social-cost")] = False,
):
    generate_alternative_scenarios(
        base_svg_filepath=scenario,
        nb_robots=n_robots,
        nb_goals_per_robot=goals_per_robot,
        nb_scenarios=n_scenarios,
        use_social_cost=use_social_cost,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    app()
