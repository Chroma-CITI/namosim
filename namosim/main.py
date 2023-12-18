import json

import typer

from namosim.report import SimulationReport
from namosim.simulator import Simulator

app = typer.Typer()


@app.command()
def run(scenario: str):
    sim = Simulator(simulation_file_path=scenario)
    sim.run()


@app.command()
def visualize_results(results_file: str):
    with open(results_file) as f:
        data = json.load(f)
        report = SimulationReport.model_validate(data["report"])
        report.plot()


if __name__ == "__main__":
    app()
