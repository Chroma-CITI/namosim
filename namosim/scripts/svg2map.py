"""Converts an svg scenario file to a ros map"""

import re
import typing as t
from xml.dom import minidom

import typer
import yaml

from namosim.scripts.svg2stl import svg_to_mesh
from namosim.utils.utils import NamosimLogger
from namosim.world.world import World

app = typer.Typer()


def svg2map(svg_file: str, width: int):
    logger = NamosimLogger()
    w = World.load_from_svg(svg_file, logs_dir=".", logger=logger)

    non_static_entities = []
    for eid, ent in w.dynamic_entities.items():
        if ent.type_ != "wall":
            non_static_entities.append(eid)

    for eid in non_static_entities:
        w.remove_entity(eid)

    img = w.to_image(width=width, grayscale=True, draw_grid_lines=False)
    return img


def write_yaml(data: t.Any, yaml_file: str):
    with open(yaml_file, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def strip_non_numeric(x: str):
    return re.sub(r"\D", "", x)


@app.command()
def run(
    *,
    svg_file: t.Annotated[str, typer.Option("--svg-file")],
    res: t.Annotated[float, typer.Option("--res", help="meters per pixel")] = 0.1,
    name: t.Annotated[str, typer.Option("--name")] = "map",
):
    doc = minidom.parse(svg_file)
    if not doc.documentElement.hasAttribute("width"):
        raise Exception("svg has no width attribute")
    if not doc.documentElement.hasAttribute("height"):
        raise Exception("svg has no height attribute")

    svg = doc.getElementsByTagName("svg")[0]
    svg_width = int(strip_non_numeric(svg.getAttribute("width")))
    width_in_meters = svg_width / 100
    width_in_pixels = int(width_in_meters / res)

    img = svg2map(svg_file, width=width_in_pixels)
    mesh = svg_to_mesh(svg_file, wall_height_meters=1.0)

    img_name = f"{name}.png"
    img.save(img_name)
    mesh.save(f"{name}_walls.stl")  # type: ignore

    data = {
        "image": img_name,
        "resolution": width_in_meters / width_in_pixels,
        "origin": [(-img.width * res) / 2, (-img.height * res) / 2, 0.0],
        "occupied_thresh": 0.5,
        "free_thresh": 0.1,
        "negate": 0,
    }
    write_yaml(data, "map.yaml")


if __name__ == "__main__":
    app()
