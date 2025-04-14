"""Converts an svg scenario file to a ros map"""

import os
from pathlib import Path
import re
import typing as t
from xml.dom import minidom

import typer
import yaml
from jinja2 import Template
from namosim.scripts.svg2stl import svg_to_mesh
from namosim.world.world import World
import xml.etree.ElementTree as ET
import cairosvg

from ament_index_python import get_package_share_directory

app = typer.Typer()


def filter_svg(input_svg: str) -> bytes:
    # Parse the SVG file
    tree = ET.parse(input_svg)
    root = tree.getroot()

    # Define the SVG namespace
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Find all 'path' or 'svg:path' elements that do not have type='wall'
    for element in root.findall(".//svg:path", namespaces=ns) + root.findall(".//path"):
        if element.get("type") != "wall":
            parent = element.find("..")
            if parent is not None:
                parent.remove(element)

    # Convert the modified SVG tree to a string
    svg_data = ET.tostring(root, encoding="utf-8").decode("utf-8")
    return svg_data.encode("utf-8")


def svg_to_png(svg_data: bytes, output_png: str, width: int):
    # Convert SVG data to PNG
    cairosvg.svg2png(bytestring=svg_data, write_to=output_png, output_width=width)


def process_template(template_file: str, context: t.Any):
    # Load the template from file
    with open(template_file, "r") as f:
        template_content = f.read()

    # Create a Jinja2 template object
    template = Template(template_content)

    # Render the template with the provided context
    rendered_content = template.render(context)

    return rendered_content


def svg2map(svg_file: str, width: int):
    w = World.load_from_svg(svg_file, logs_dir=".")

    non_static_entities: t.List[str] = []
    for eid, ent in w.dynamic_entities.items():
        non_static_entities.append(eid)

    for eid in non_static_entities:
        w.remove_entity(eid)

    img = w.to_image(width=width, grayscale=True, draw_grid_lines=False)
    return img


def write_yaml(data: t.Any, yaml_file: str):
    with open(yaml_file, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def strip_non_numeric(x: str):
    result = re.sub(r"[^0-9.]", "", x)
    return result


@app.command()
def run(
    *,
    svg_file: t.Annotated[str, typer.Option("--svg-file")],
    out_dir: t.Annotated[str, typer.Option("--out-dir")],
):
    doc = minidom.parse(svg_file)
    if not doc.documentElement.hasAttribute("width"):
        raise Exception("svg has no width attribute")
    if not doc.documentElement.hasAttribute("height"):
        raise Exception("svg has no height attribute")

    basename = Path(svg_file).stem
    world = World.load_from_svg(svg_file)
    svg = doc.getElementsByTagName("svg")[0]
    svg_width = float(strip_non_numeric(svg.getAttribute("width")))
    svg_height = float(strip_non_numeric(svg.getAttribute("height")))
    width_in_meters = svg_width / 100
    width_in_pixels = int(width_in_meters / world.map.cell_size)

    mesh = svg_to_mesh(svg_file, wall_height_meters=1.0)
    mesh.save(os.path.join(out_dir, "map_walls.stl"))

    img_name = f"{basename}.png"
    img_path = os.path.join(out_dir, img_name)
    img = svg2map(svg_file, width=width_in_pixels)
    img.save(img_path)

    data = {
        "image": img_name,
        "resolution": world.map.cell_size,
        "origin": [0.0, 0.0, 0.0],
        "occupied_thresh": 0.5,
        "free_thresh": 0.1,
        "negate": 0,
    }
    write_yaml(data, os.path.join(out_dir, f"{basename}.yaml"))

    # generate namo world from template
    w = World.load_from_svg(svg_file, logs_dir=".")
    movable_boxes: t.List[t.Any] = []
    robots: t.List[t.Any] = []
    for entity in w.dynamic_entities.values():
        x, y, theta = entity.pose
        if entity.type_ == "movable":
            minx, miny, maxx, maxy = entity.polygon.minimum_rotated_rectangle.bounds
            width = maxx - minx
            height = maxy - miny
            size = max(width, height)
            scale = size
            movable_boxes.append(
                {"name": entity.uid, "pose": f"{x} {y} 1 0 0 {theta}", "scale": scale}
            )
        if entity.type_ == "robot":
            robots.append({"name": entity.uid, "pose": f"{x} {y} 1 0 0 {theta}"})
    context = {"movable_boxes": movable_boxes, "robots": robots}
    print(context)

    with open(os.path.join(out_dir, "namo_world.sdf"), "w") as f:
        pkg_share = get_package_share_directory("namoros")
        result = process_template(
            os.path.join(pkg_share, "namo_world_template.sdf"), context
        )
        f.write(result)


if __name__ == "__main__":
    app()
