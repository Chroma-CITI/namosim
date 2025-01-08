import typing as t
from xml.dom import minidom

import numpy as np
from shapely.geometry import Polygon


def polygon_to_svg_path(polygon: Polygon) -> minidom.Element:
    return minidom.parseString(
        polygon.svg(fill_color="#000000", opacity=1)  # type: ignore
    ).documentElement  # type: ignore


def perturb_polygon(
    polygon: Polygon,
    bounds: t.Tuple[float, float, float, float],
    sigma: float = 2,
) -> Polygon:
    """
    Perturb the vertices of a Shapely polygon randomly.

    Parameters:
    - polygon: Shapely Polygon object
    - sigma: Standard deviation of the random perturbations

    Returns:
    - perturbed_polygon: Perturbed Shapely Polygon object
    """
    # Extract the coordinates of the polygon
    X, Y = polygon.exterior.coords.xy  # type: ignore

    x_min, y_min, x_max, y_max = bounds

    points = []
    for x, y in zip(X.tolist(), Y.tolist()):
        if x_min < x < x_max and y_min < y < y_max:
            # Generate random perturbations for each coordinate
            dx = np.random.normal(loc=0.0, scale=sigma)
            dy = np.random.normal(loc=0.0, scale=sigma)
            points.append((x + dx, y + dy))
        else:
            points.append((x, y))

    # pertub holes
    holes = []
    for hole in polygon.interiors:
        X, Y = hole.coords.xy
        hole_points = []
        for x, y in zip(X.tolist(), Y.tolist()):
            # Generate random perturbations for each coordinate
            dx = np.random.normal(loc=0.0, scale=sigma)
            dy = np.random.normal(loc=0.0, scale=sigma)
            hole_points.append((x + dx, y + dy))

        holes.append(hole_points)
    return Polygon(points, holes=holes)
