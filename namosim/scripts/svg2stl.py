import typing as t

import numpy as np
import typer
from namosim.utils.utils import NamosimLogger
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from namosim.world.world import World
from shapely import affinity
from shapely.ops import triangulate
from stl import mesh
import cv2
from shapely.geometry import Polygon

app = typer.Typer()

from PIL import Image
import numpy as np
from stl import mesh


# Find contours of the black regions
def find_contours(map: BinaryOccupancyGrid):
    grid = (
        np.where(map.grid != 0, 1, 0).astype(np.uint8).transpose()
    )  # transpose because map.grid has dimensions (x,y) whereas image has (y,x)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Convert contours to Shapely Polygons
def contours_to_polygons(contours, scale: float):
    polygons = []
    for contour in contours:
        # Flatten the contour array and convert it to a Polygon
        contour = contour.squeeze()  # Remove redundant dimensions
        if len(contour) > 2:  # We need at least 3 points to make a polygon
            polygon = Polygon(contour)
            polygon = affinity.scale(polygon, scale, scale, origin=(0, 0))  # type: ignore
            polygons.append(polygon)
    return polygons


def map_to_polygons(map: BinaryOccupancyGrid) -> t.List[Polygon]:
    contours = find_contours(map)
    return contours_to_polygons(contours, scale=map.cell_size)


def svg_to_mesh(svg_file: str, wall_height_meters: float):
    faces = []
    walls = World.get_wall_polygons_from_svg(svg_file)

    # If no wall geometries were found, the scenario must use an image for the map, so get walls from the occupancy grid.
    if len(walls) == 0:
        w = World.load_from_svg(svg_file)
        walls = map_to_polygons(w.map)

    for wall in walls:
        triangles = triangulate(wall)  # type: ignore
        for tri in triangles:
            if tri.within(wall):
                face_down = []
                face_up = []
                for point in tri.exterior.coords[:-1]:
                    (x, y) = point
                    face_down.append((x, y, 0))
                    face_up.append((x, y, wall_height_meters))

                faces.append(face_down)
                faces.append(face_up)

        # we want to iterate over coords in counter-clockwise direction so that mesh face normals point outwards.
        coords = wall.exterior.coords
        if wall.exterior.is_ccw is False:
            coords = coords[::-1]

        for a, b in zip(coords, coords[1:]):
            (xa, ya) = a
            (xb, yb) = b

            face1 = [(xa, ya, 0), (xb, yb, 0), (xb, yb, wall_height_meters)]
            face2 = [
                (xa, ya, 0),
                (xb, yb, wall_height_meters),
                (xa, ya, wall_height_meters),
            ]

            faces.append(face1)
            faces.append(face2)

    faces = np.array(faces)
    # Create mesh
    stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = face[j]  # type: ignore

    return stl_mesh


def export_stl(mesh: mesh.Mesh, output_file: str):
    mesh.save(output_file)


@app.command()
def run(
    *,
    svg_file: t.Annotated[
        str, typer.Option("--svg-file", help="path to the scenario file")
    ],
    out: t.Annotated[str, typer.Option("--out")] = "mesh.stl",
    height: t.Annotated[
        float, typer.Option("--height", help="wall height in meters")
    ] = 2,
):
    stl_mesh = svg_to_mesh(svg_file, height)
    export_stl(stl_mesh, out)  # type: ignore


if __name__ == "__main__":
    app()
