from __future__ import annotations

import math
import typing as t

import mapbox_earcut as earcut
import numpy as np
import numpy.typing as npt
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Vector3
from grid_map_msgs.msg import GridMap
from shapely.geometry import Polygon
from std_msgs.msg import (
    ColorRGBA,
    Float32MultiArray,
    Header,
    MultiArrayDimension,
    MultiArrayLayout,
)
from visualization_msgs.msg import Marker, MarkerArray

import namosim.display.colors as colors
import namosim.display.ros_publisher_config as cfg
import namosim.world.robot as namosim_robot
from namosim.data_models import PoseModel
from namosim.display import tf_replacement


def init_header(stamp: Time = Time()):
    return Header(stamp=stamp, frame_id=cfg.main_frame_id)


def plan_to_markerarray(
    plan: t.Any, robot: namosim_robot.Robot, frame_id: str, stamp: Time = Time()
):
    markerarray = MarkerArray()
    markers = []
    p_id = 0
    for component in plan.path_components:
        current_color = ColorRGBA(**colors.hex_to_rgba(robot.style.fill))
        if component.is_transfer:
            current_color = ColorRGBA(
                **colors.hex_to_rgba(colors.darken(robot.style.fill))
            )
            obstacle_end_polygon_marker = polygon_to_line_strip(
                component.obstacle_path.polygons[-1],
                "/end_obstacles",
                p_id,
                frame_id,
                current_color,
                cfg.path_line_z_index,
                cfg.border_width,
            )
            markers.append(obstacle_end_polygon_marker)
        path_marker = real_path_to_triangle_list(
            component.robot_path.poses,
            "/plan",
            p_id,
            frame_id,
            current_color,
            robot.min_inflation_radius / 4,
            cfg.path_line_z_index,
            stamp=stamp,
        )
        markers.append(path_marker)
        p_id += 1
    markerarray.markers = markers
    return markerarray


# Basic conversion functions


def polygon_to_triangle_list(
    polygon: Polygon,
    namespace: str,
    p_id: int,
    frame_id: str,
    color: ColorRGBA,
    z_index: float,
    stamp: Time = Time(),
):
    """Takes a polygon and converts it to a TRIANGLE_LIST marker for RVIZ

    :param polygon
    :type polygon: Polygon
    :param namespace: rviz namespace
    :type namespace: str
    :param p_id: marker id
    :type p_id: int
    :param frame_id: rviz frame
    :type frame_id: str
    :param color: color of the rendered marker
    :type color: ColorRGBA
    :param z_index: _description_
    :type z_index: a z-axis offset
    :param stamp: timestamp, defaults to Time()
    :type stamp: Time, optional
    :return: a TRIANGLE_LIST marker
    :rtype: Marker
    """
    marker = Marker(
        type=Marker.TRIANGLE_LIST,
        ns=namespace,
        id=p_id,
        header=Header(frame_id=frame_id, stamp=stamp),
        color=color,
        scale=Vector3(x=1.0, y=1.0, z=1.0),
        points=[],
    )
    if isinstance(polygon, Polygon):
        verts = list(zip(*polygon.exterior.coords.xy))[:-1]
        verts = np.array(verts)
        rings = np.array([verts.shape[0]])
        triangles_vertices = verts[earcut.triangulate_float64(verts, rings)]
        triangles = [
            triangles_vertices[n : n + 3] for n in range(0, len(triangles_vertices), 3)
        ]
        marker.points = [
            Point(x=point[0], y=point[1], z=z_index)
            for triangle in triangles
            for point in triangle
        ]
    return marker


def polygon_to_line_strip(
    polygon: Polygon,
    namespace: str,
    p_id: int,
    frame_id: str,
    color: ColorRGBA,
    z_index: float,
    line_width: float,
    stamp: Time = Time(),
):
    marker = Marker(
        type=Marker.LINE_STRIP,
        ns=namespace,
        id=p_id,
        header=Header(frame_id=frame_id, stamp=stamp),
        color=color,
        scale=Vector3(x=line_width, y=0.0, z=0.0),
        points=[],
    )
    for i in range(len(polygon.exterior.coords) - 1):
        point = polygon.exterior.coords[i]
        next_point = polygon.exterior.coords[i + 1]
        marker.points.append(Point(x=point[0], y=point[1], z=z_index))  # type: ignore
        marker.points.append(Point(x=next_point[0], y=next_point[1], z=z_index))  # type: ignore
    marker.points.append(  # type: ignore
        Point(
            x=polygon.exterior.coords[0][0],
            y=polygon.exterior.coords[0][1],
            z=z_index,
        )
    )
    marker.points.append(  # type: ignore
        Point(
            x=polygon.exterior.coords[1][0],
            y=polygon.exterior.coords[1][1],
            z=z_index,
        )
    )
    return marker


def string_to_text(
    string: str,
    coordinates: t.Tuple[float | int, float | int],
    namespace: str,
    p_id: int,
    frame_id: str,
    color: ColorRGBA,
    z_index: float,
    text_height: float,
    stamp: Time = Time(),
):
    x, y, z = coordinates[0], coordinates[1], z_index
    marker = Marker(
        type=Marker.TEXT_VIEW_FACING,
        ns=namespace,
        id=p_id,
        pose=Pose(
            position=(Point(x=x, y=y, z=z)),
            orientation=Quaternion(),
        ),
        scale=Vector3(x=0.0, y=0.0, z=text_height),
        header=Header(frame_id=frame_id, stamp=stamp),
        color=color,
        text=string,
    )
    return marker


def costmap_to_grid_map(
    costmap: npt.NDArray[t.Any],
    resolution: float,
    frame_id: str = cfg.social_gridmap_frame_id,
    stamp: Time = Time(),
):
    grid_map = GridMap()
    if hasattr(grid_map.info, "header"):
        grid_map.info.header = Header(stamp=stamp, frame_id=frame_id)  # type: ignore
    elif hasattr(grid_map, "header"):
        grid_map.header = Header(stamp=stamp, frame_id=frame_id)

    grid_map.info.resolution = resolution
    grid_map.info.length_x = costmap.shape[0] * resolution
    grid_map.info.length_y = costmap.shape[1] * resolution
    # grid_map.info.pose.position.z = 0. # The lib does not take this parameter into account...
    grid_map.layers = ["elevation"]
    inflated_costmap_data = Float32MultiArray(
        layout=MultiArrayLayout(
            dim=[
                MultiArrayDimension(
                    label="column_index",
                    size=costmap.shape[1],
                    stride=costmap.shape[1] * costmap.shape[0],
                ),
                MultiArrayDimension(
                    label="row_index", size=costmap.shape[0], stride=costmap.shape[0]
                ),
            ],
            data_offset=0,
        ),
        data=(costmap.flatten("F")).astype(np.float32).tolist(),
    )
    grid_map.data = [inflated_costmap_data]

    return grid_map


def geom_quat_from_yaw(yaw: float):
    explicit_quat = tf_replacement.quaternion_from_euler(0.0, 0.0, math.radians(yaw))
    return Quaternion(
        x=explicit_quat[0], y=explicit_quat[1], z=explicit_quat[2], w=explicit_quat[3]
    )


def pose_to_ros_pose(pose: PoseModel) -> Pose:
    x, y, z = pose[0], pose[1], 0.0
    return Pose(
        position=(Point(x=x, y=y, z=z)),
        orientation=geom_quat_from_yaw(pose[2]),
    )


def poses_to_poses_array(poses: t.List[PoseModel], stamp: Time = Time()):
    pose_array = PoseArray(header=init_header(stamp), poses=[])
    for pose in poses:
        pose_array.poses.append(pose_to_ros_pose(pose))  # type: ignore
    return pose_array


def real_path_to_triangle_list(
    real_path: t.List[t.Tuple[float, float, float] | t.Tuple[float, float]],
    namespace: str,
    p_id: int,
    frame_id: str,
    color: ColorRGBA,
    line_width: float,
    z_index: float,
    stamp: Time = Time(),
):
    """Takes a robot path as a sequence of points and converts them to a TRIANGLE_LIST marker for RVIZ.

    :param real_path: A nagivation path as a sequency of points
    :type real_path: t.List[t.Tuple[float, float, float]  |  t.Tuple[float, float]]
    :param namespace: the rviz namespace
    :type namespace: str
    :param p_id: _description_
    :type p_id: int
    :param frame_id: _description_
    :type frame_id: str
    :param color: _description_
    :type color: ColorRGBA
    :param line_width: _description_
    :type line_width: float
    :param z_index: _description_
    :type z_index: float
    :param stamp: _description_, defaults to Time()
    :type stamp: Time, optional
    :return: _description_
    :rtype: _type_
    """
    points = [np.array(x) for x in real_path]
    polygon = path_to_polygon(points=points, line_width=line_width)
    return polygon_to_triangle_list(
        polygon=polygon,
        namespace=namespace,
        p_id=p_id,
        frame_id=frame_id,
        color=color,
        z_index=z_index,
        stamp=stamp,
    )


def make_delete_marker(namespace: str, p_id: int, frame_id: str, stamp: Time = Time()):
    return Marker(
        ns=namespace,
        id=p_id,
        header=Header(frame_id=frame_id, stamp=stamp),
        action=Marker.DELETE,
    )


def make_delete_all_marker(frame_id: str, ns: str = "", stamp: Time = Time()):
    return MarkerArray(
        markers=[
            Marker(
                ns=ns,
                header=Header(frame_id=frame_id, stamp=stamp),
                action=Marker.DELETEALL,
            )
        ]
    )


def path_to_polygon(
    points: t.List[npt.NDArray[np.float_]], line_width: float
) -> Polygon:
    """Converts a sequence of points representing a navigation path into a polygonal "line strip".

    :param points: A sequence of points
    :type points: t.List[npt.NDArray[np.float_]
    :param line_width: width to use for the polygonal line strip
    :type line_width: float
    :raises Exception: if less than two points are in the path
    :return: a polygonal line strip
    :rtype: Polygon
    """

    # remove z-coord, if any
    points = [x[:2] for x in points]

    # remove duplicate points
    seen = set()
    dedup_points = []
    for p in points:
        hp = p[0], p[1]
        if hp not in seen:
            seen.add(hp)
            dedup_points.append(p)
    points = dedup_points

    if len(points) < 2:
        raise Exception("Less than two points")

    def get_z_ortho(x: npt.NDArray[t.Any]) -> npt.NDArray[t.Any]:
        """Return a unit-length vector orthogonal to both x and the z-axis"""
        z = np.array((0.0, 0.0, 1.0))
        ortho = np.cross((x[0], x[1], 0.0), z)
        return (ortho / np.linalg.norm(ortho))[:2]

    forward_coords = []
    backward_coords = []

    if len(points) == 2:
        a, b = points
        o = get_z_ortho(b - a) * line_width / 2.0
        forward_coords.extend([a + o, b + o])
        backward_coords.extend([a - o, b - o])
    else:
        for i in range(len(points) - 2):
            a = points[i]
            b = points[i + 1]
            c = points[i + 2]

            a_to_b = b - a
            b_to_c = c - b
            o1 = get_z_ortho(a_to_b)
            o2 = get_z_ortho(b_to_c)
            o = (o1 + o2) / 2
            o *= line_width / 2.0

            # Don't forget to add the first point!
            if i == 0:
                forward_coords.append(a + o)
                backward_coords.append(a - o)

            forward_coords.append(b + o)
            backward_coords.append(b - o)

            # Don't forget to add the last point!
            if i == len(points) - 3:
                forward_coords.append(c + o)
                backward_coords.append(c - o)

    backward_coords.reverse()
    backward_coords.append(forward_coords[0])

    return Polygon(forward_coords + backward_coords)
