import re
import typing as t
from xml.dom import minidom

import numpy as np
import numpy.typing as npt
from shapely import affinity
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
)
from shapely.ops import polygonize, unary_union
from svgpath2mpl import parse_path
import triangle

from namosim.display.conversions import polygon_to_triangle_vertices

SVG_PATH_ATTRIBUTES_WHITELIST = ["id", "d", "style", "type"]


def add_group(svg_data, group_id, parent=None, is_layer=True):
    new_group = svg_data.createElement("svg:g")
    new_group.setAttribute("id", group_id)
    if is_layer:
        new_group.setAttribute("inkscape:groupmode", "layer")
    # new_group.setAttribute('inkscape:label', group_id)
    if parent:
        parent.appendChild(new_group)
    else:
        svg_data.childNodes[0].appendChild(new_group)
    return new_group


def add_shapely_geometry_to_svg(
    *,
    shape: Polygon | LineString | Point,
    uid: str,
    style: str,
    svg_data: minidom.Document,
    ymax_meters: float,
    svg_group: minidom.Element | None = None,
    namo_type: str | None = None,
):
    pathd = shapely_geometry_to_svg_pathd(shape=shape, ymax_meters=ymax_meters)
    new_path = svg_data.createElement("svg:path")
    new_path.setAttribute("id", uid)
    new_path.setAttribute("d", pathd)
    new_path.setAttribute("style", style)
    if namo_type:
        new_path.setAttribute("type", namo_type)

    if svg_group:
        svg_group.appendChild(new_path)
    else:
        svg_data.childNodes[0].appendChild(new_path)


def svg_pathd_to_shapely_geometry(
    *,
    svg_path: str,
    ymax_meters: float,
    precision: float = 1e9,
    scale: float = 1 / 100,  # cm to meters
) -> t.Union[Polygon, LineString, Point]:
    parse_result = parse_path(svg_path)
    coords: npt.NDArray[np.float32] = (
        t.cast(npt.NDArray[np.float32], parse_result.vertices) * scale
    )
    coords[:, 1] = ymax_meters - coords[:, 1]

    # Remove duplicates
    pts_set: t.Set[t.Tuple[float, float]] = set()
    dedup_geom_pts: t.List[t.Tuple[float, float]] = []
    for pt in coords:
        pt_tuple = (round(pt[0] * precision), round(pt[1] * precision))
        if pt_tuple not in pts_set:
            pts_set.add(pt_tuple)
            dedup_geom_pts.append(pt)

    # or on y-axis
    if len(dedup_geom_pts) >= 3:
        return Polygon(dedup_geom_pts)
    if len(dedup_geom_pts) == 2:
        return LineString(dedup_geom_pts)
    if len(dedup_geom_pts) == 1:
        return Point(dedup_geom_pts)
    raise RuntimeError("SVG path could not be converted to Shapely geometry.")


def shapely_geometry_to_svg_pathd(
    *, shape: Polygon | LineString | Point, ymax_meters: float
):
    coords: npt.NDArray[t.Any]

    # Extract polygon coordinates
    if isinstance(shape, Polygon):
        coords = np.array(shape.exterior.coords)
    elif isinstance(shape, Point) or isinstance(shape, LineString):
        coords = np.array(shape.coords)
    else:
        raise TypeError(
            "Only shapely Point, LineString and Polygon objects can be turned into svg."
        )
    coords[:, 1] = ymax_meters - coords[:, 1]
    coords *= 100  # meters to cm

    # Rebuild polygon
    if isinstance(shape, Polygon):
        new_geometry = Polygon(coords)
        return minidom.parseString(new_geometry.svg()).documentElement.getAttribute("d")
    if isinstance(shape, LineString):
        new_geometry = LineString(coords)
        return polyline2pathd(
            dom2dict(minidom.parseString(new_geometry.svg()).firstChild)
        )
    if isinstance(shape, Point):
        new_geometry = Point(coords)
        return ellipse2pathd(
            dom2dict(minidom.parseString(new_geometry.svg()).firstChild)
        )


# region SVG elements to SVG paths conversion functions, extracted from svgpathtools library, available at :
# https://github.com/mathandy/svgpathtools/
COORD_PAIR_TMPLT = re.compile(
    r"([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)"
    + r"(?:\s*,\s*|\s+|(?=-))"
    + r"([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)"
)


def dom2dict(element):
    """Converts DOM elements to dictionaries of attributes."""
    keys = list(element.attributes.keys())
    values = [val.value for val in list(element.attributes.values())]
    return dict(list(zip(keys, values)))


def ellipse2pathd(ellipse):
    """converts the parameters from an ellipse or a circle to a string for a
    Path object d-attribute"""

    cx = ellipse.get("cx", 0)
    cy = ellipse.get("cy", 0)
    rx = ellipse.get("rx", None)
    ry = ellipse.get("ry", None)
    r = ellipse.get("r", None)

    if r is not None:
        rx = ry = float(r)
    else:
        rx = float(rx)
        ry = float(ry)

    cx = float(cx)
    cy = float(cy)

    d = ""
    d += "M" + str(cx - rx) + "," + str(cy)
    d += "a" + str(rx) + "," + str(ry) + " 0 1,0 " + str(2 * rx) + ",0"
    d += "a" + str(rx) + "," + str(ry) + " 0 1,0 " + str(-2 * rx) + ",0"

    return d


def polyline2pathd(polyline, is_polygon=False):
    """converts the string from a polyline points-attribute to a string for a
    Path object d-attribute"""
    points = COORD_PAIR_TMPLT.findall(polyline.get("points", ""))
    closed = float(points[0][0]) == float(points[-1][0]) and float(
        points[0][1]
    ) == float(points[-1][1])

    # The `parse_path` call ignores redundant 'z' (closure) commands
    # e.g. `parse_path('M0 0L100 100Z') == parse_path('M0 0L100 100L0 0Z')`
    # This check ensures that an n-point polygon is converted to an n-Line path.
    if is_polygon and closed:
        points.append(points[0])

    d = "M" + "L".join("{0} {1}".format(x, y) for x, y in points)
    if is_polygon or closed:
        d += "z"
    return d


def polygon2pathd(polyline):
    """converts the string from a polygon points-attribute to a string
    for a Path object d-attribute.
    Note:  For a polygon made from n points, the resulting path will be
    composed of n lines (even if some of these lines have length zero).
    """
    return polyline2pathd(polyline, True)


def rect2pathd(rect):
    """Converts an SVG-rect element to a Path d-string.

    The rectangle will start at the (x,y) coordinate specified by the
    rectangle object and proceed counter-clockwise."""
    x0, y0 = float(rect.get("x", 0)), float(rect.get("y", 0))
    w, h = float(rect.get("width", 0)), float(rect.get("height", 0))
    x1, y1 = x0 + w, y0
    x2, y2 = x0 + w, y0 + h
    x3, y3 = x0, y0 + h

    d = "M{} {} L {} {} L {} {} L {} {} z" "".format(x0, y0, x1, y1, x2, y2, x3, y3)
    return d


def line2pathd(l):
    return (
        "M"
        + l.attrib.get("x1", "0")
        + " "
        + l.attrib.get("y1", "0")
        + "L"
        + l.attrib.get("x2", "0")
        + " "
        + l.attrib.get("y2", "0")
    )


# endregion


def set_all_id_attributes_as_ids(xml_doc):
    queue = [xml_doc]
    while queue:
        current = queue.pop()
        queue += current.childNodes
        if getattr(current, "hasAttribute", None) and current.hasAttribute("id"):
            current.setIdAttribute("id")


def clean_attributes(xml_doc):
    path_elements = xml_doc.getElementsByTagName("path")

    for path_element in path_elements:
        attributes_to_remove = []

        for attribute in path_element.attributes.keys():
            if attribute not in SVG_PATH_ATTRIBUTES_WHITELIST:
                attributes_to_remove.append(attribute)

        for attribute in attributes_to_remove:
            path_element.removeAttribute(attribute)


def color_clamp(x):
    return max(0, min(x, 255))


def rgb_tuple_to_hex(rgb):
    return "#{0:02x}{1:02x}{2:02x}".format(
        color_clamp(rgb[0]), color_clamp(rgb[1]), color_clamp(rgb[2])
    )


def concave_hull_polygon(
    polygon: Polygon, alpha: float
) -> t.Union[Polygon, MultiPolygon]:
    # Get triangulation from your function (shape: n, 3, 2)
    triangles = np.array(polygon_to_triangle_vertices(polygon))

    # Check if triangulation is empty or insufficient
    if triangles.shape[0] == 0:
        return polygon.convex_hull  # type: ignore

    # Extract unique edges from triangles
    edges = set()
    for tri in triangles:
        # Each triangle has 3 edges: (0-1), (1-2), (2-0)
        for i in range(3):
            pt1 = tuple(tri[i])  # (x, y)
            pt2 = tuple(tri[(i + 1) % 3])  # (x, y)
            # Sort to ensure (pt1, pt2) and (pt2, pt1) are treated as the same edge
            edge = tuple(sorted([pt1, pt2]))
            edges.add(edge)

    # Filter edges by length <= alpha
    edge_points = []
    for edge in edges:
        pt1, pt2 = np.array(edge[0]), np.array(edge[1])
        if np.linalg.norm(pt1 - pt2) <= alpha:
            edge_points.append([pt1, pt2])

    # Create Shapely geometry
    if not edge_points:
        return polygon.convex_hull  # type: ignore # Fallback if no valid edges

    lines = [LineString([pt[0], pt[1]]) for pt in edge_points]
    merged = unary_union(lines)
    hulls = list(polygonize(merged))

    result = unary_union(hulls) if hulls else polygon.convex_hull

    # Ensure output is a Polygon or MultiPolygon
    if result.is_empty:
        return polygon.convex_hull  # type: ignore
    return result  # type: ignore
