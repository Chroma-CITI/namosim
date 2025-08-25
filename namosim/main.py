import typing as t
from enum import Enum

import typer

from namosim.mapgen.mapgen import MapGen
from namosim.scenario_generation import generate_alternative_scenarios
from namosim.simulator import create_sim_from_file

app = typer.Typer()


@app.command()
def run(
    scenario: str,
    logs_dir: t.Annotated[t.Optional[str], typer.Option("--logs-dir")] = None,
):
    sim = create_sim_from_file(simulation_file_path=scenario, logs_dir=logs_dir)
    sim.run()


class DeadlockStrategy(str, Enum):
    SOCIAL = "SOCIAL"
    DISTANCE = "DISTANCE"


@app.command()
def gen_alt_scenarios(
    *,
    scenario: t.Annotated[str, typer.Option("--base-scenario")],
    out_dir: t.Annotated[str, typer.Option("--out-dir")],
    deadlock_strategy: t.Annotated[
        t.Optional[DeadlockStrategy], typer.Option("--deadlock-strategy")
    ] = None,
    n_robots: t.Annotated[int, typer.Option("--n-robots")] = 1,
    goals_per_robot: t.Annotated[int, typer.Option("--goals-per-robot")] = 50,
    n_scenarios: t.Annotated[int, typer.Option("--n-scenarios")] = 1,
    cell_size_cm: t.Annotated[float, typer.Option("--cell-size")] = 15.0,
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
        cell_size_cm=cell_size_cm,
        deadlock_strategy=deadlock_strategy.value if deadlock_strategy else None,
        use_social_cost=use_social_cost,
        resolve_deadlocks=not no_resolve_deadlocks,
        resolve_conflicts=not no_resolve_conflicts,
        out_dir=out_dir,
    )


@app.command()
def gen_map(
    width: int = 20,
    height: int = 20,
    init_open: float = 0.42,
    robot_radius_cm: float = 60,
):
    ng = MapGen(height=height, width=width, init_open=init_open)
    ng.gen_map()
    ng.print_grid()
    doc = ng.to_svg(robot_radius_cm=robot_radius_cm)
    with open("out.svg", "w") as f:
        f.write(doc.toprettyxml())


if __name__ == "__main__":
    app()
