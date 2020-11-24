from svgpath2mpl import parse_path
from shapely.geometry import Polygon, Point, LineString
from xml.dom import minidom
import numpy as np
import re
from shapely import affinity


SVG_PATH_ATTRIBUTES_WHITELIST = ["id", "d", "style"]

OBSTACE_TRACE_STYLE = 'fill:#000000;fill-opacity:0.05231688;fill-rule:evenodd;stroke:#f1c232;stroke-width:1;stroke-linecap:square;stroke-miterlimit:10;stroke-opacity:1'
MOVABLE_ENTITY_STYLE = 'fill:#f1c232;fill-rule:evenodd'
FIXED_ENTITY_STYLE = 'fill:#000000;fill-rule:evenodd'
ROBOT_ENTITY_STYLE = 'fill:#6d9eeb;fill-opacity:1;stroke:none;stroke-opacity:1'
GOAL_STYLE = 'fill:none;fill-rule:evenodd;stroke:#1155cc;stroke-width:3.5999999;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1'
POSE_STYLE = 'fill:none;stroke:#1155cc;stroke-width:3.5999999;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1'


def add_shapely_geometry_to_svg(shapely_geometry, scaling_value, map_width, map_height, uname, style, svg_data):
    projected_geometry = affinity.translate(
        shapely_geometry, map_width / 2., -map_height / 2.
    ) # TODO Take rotation into account
    pathd = shapely_geometry_to_svg_pathd(projected_geometry, scaling_value)
    new_path = svg_data.createElement('svg:path')
    new_path.setAttribute('id', uname)
    new_path.setAttribute('d', pathd)
    new_path.setAttribute('style', style)
    svg_data.childNodes[0].appendChild(new_path)


def svg_pathd_to_shapely_geometry(svg_path, scaling_value):
    parse_result = parse_path(svg_path)
    geom_pts = parse_result.vertices * scaling_value
    geom_pts[:, 1] = -geom_pts[:, 1]  # Mirror on y-axis
    if len(geom_pts) >= 3:
        return Polygon(geom_pts)
    elif len(geom_pts) == 2:
        return LineString(geom_pts)
    elif len(geom_pts) == 1:
        return Point(geom_pts)
    else:
        raise RuntimeError("SVG path could not be converted to Shapely geometry.")


def shapely_geometry_to_svg_pathd(shapely_geometry, scaling_value):
    # Extract polygon coordinates
    if isinstance(shapely_geometry, Polygon):
        coords = np.array(shapely_geometry.exterior.coords)
    elif isinstance(shapely_geometry, Point) or isinstance(shapely_geometry, LineString):
        coords = np.array(shapely_geometry.coords)
    else:
        raise TypeError("Only shapely Point, LineString and Polygon objects can be turned into svg.")
    coords /= scaling_value  # Scale them back to appropriate SVG measurements
    coords[:, 1] = -coords[:, 1]  # Mirror back on y-axis
    # Rebuild polygon
    if isinstance(shapely_geometry, Polygon):
        new_geometry = Polygon(coords)
        return minidom.parseString(new_geometry.svg()).documentElement.getAttribute('d')
    elif isinstance(shapely_geometry, LineString):
        new_geometry = LineString(coords)
        return polyline2pathd(dom2dict(minidom.parseString(new_geometry.svg()).firstChild))
    elif isinstance(shapely_geometry, Point):
        new_geometry = Point(coords)
        return ellipse2pathd(dom2dict(minidom.parseString(new_geometry.svg()).firstChild))


# region SVG elements to SVG paths conversion functions, extracted from svgpathtools library, available at :
# https://github.com/mathandy/svgpathtools/
COORD_PAIR_TMPLT = re.compile(
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)' +
    r'(?:\s*,\s*|\s+|(?=-))' +
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)'
)


def dom2dict(element):
    """Converts DOM elements to dictionaries of attributes."""
    keys = list(element.attributes.keys())
    values = [val.value for val in list(element.attributes.values())]
    return dict(list(zip(keys, values)))


def ellipse2pathd(ellipse):
    """converts the parameters from an ellipse or a circle to a string for a
    Path object d-attribute"""

    cx = ellipse.get('cx', 0)
    cy = ellipse.get('cy', 0)
    rx = ellipse.get('rx', None)
    ry = ellipse.get('ry', None)
    r = ellipse.get('r', None)

    if r is not None:
        rx = ry = float(r)
    else:
        rx = float(rx)
        ry = float(ry)

    cx = float(cx)
    cy = float(cy)

    d = ''
    d += 'M' + str(cx - rx) + ',' + str(cy)
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(2 * rx) + ',0'
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(-2 * rx) + ',0'

    return d


def polyline2pathd(polyline, is_polygon=False):
    """converts the string from a polyline points-attribute to a string for a
    Path object d-attribute"""
    points = COORD_PAIR_TMPLT.findall(polyline.get('points', ''))
    closed = (float(points[0][0]) == float(points[-1][0]) and
              float(points[0][1]) == float(points[-1][1]))

    # The `parse_path` call ignores redundant 'z' (closure) commands
    # e.g. `parse_path('M0 0L100 100Z') == parse_path('M0 0L100 100L0 0Z')`
    # This check ensures that an n-point polygon is converted to an n-Line path.
    if is_polygon and closed:
        points.append(points[0])

    d = 'M' + 'L'.join('{0} {1}'.format(x, y) for x, y in points)
    if is_polygon or closed:
        d += 'z'
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
    x0, y0 = float(rect.get('x', 0)), float(rect.get('y', 0))
    w, h = float(rect.get('width', 0)), float(rect.get('height', 0))
    x1, y1 = x0 + w, y0
    x2, y2 = x0 + w, y0 + h
    x3, y3 = x0, y0 + h

    d = ("M{} {} L {} {} L {} {} L {} {} z"
         "".format(x0, y0, x1, y1, x2, y2, x3, y3))
    return d


def line2pathd(l):
    return (
            'M' + l.attrib.get('x1', '0') + ' ' + l.attrib.get('y1', '0')
            + 'L' + l.attrib.get('x2', '0') + ' ' + l.attrib.get('y2', '0')
    )
# endregion

def set_all_id_attributes_as_ids(xml_doc):
    cur = [xml_doc.firstChild]
    cur_child = cur[0]
    while cur_child.hasChildNodes and cur:
        cur_child = cur.pop(0)
        cur += cur_child.childNodes
        if cur_child.nodeType != minidom.Node.TEXT_NODE and cur_child.hasAttribute('id'):
            cur_child.setIdAttribute('id')

def clean_attributes(xml_doc):
    path_elements = xml_doc.getElementsByTagName('path')
    for path_element in path_elements:
        for attribute in path_element.attributes.keys():
            if attribute not in SVG_PATH_ATTRIBUTES_WHITELIST:
                path_element.removeAttribute(attribute)