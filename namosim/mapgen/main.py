import typer

from namosim.mapgen.mapgen import MapGen

app = typer.Typer()


@app.command()
def gen_map(width: int = 40, height: int = 40, init_open: float = 0.42):
    ng = MapGen(height=height, width=width, init_open=init_open)
    ng.gen_map()
    ng.print_grid()
    doc = ng.to_svg(robot_radius_cm=60)
    with open("out.svg", "w") as f:
        f.write(doc.toprettyxml())


if __name__ == "__main__":
    app()
