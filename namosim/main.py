import json
import typing as t

import typer

from namosim.report import SimulationReport
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


if __name__ == "__main__":
    app()
