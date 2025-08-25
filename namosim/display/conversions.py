from __future__ import annotations

import math
import typing as t

import numpy as np
import numpy.typing as npt
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from grid_map_msgs.msg import GridMap
from shapely.geometry import Polygon, MultiPolygon
from namosim.utils.conversion import polygon_to_triangle_vertices
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from namosim.world.entity import Style
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
import namosim.navigation.navigation_plan as nav_plan
from namosim.agents import agent
from namosim.data_models import Pose2D
from namosim.display import tf_replacement
from namosim.navigation.path_type import PathType
from namosim.utils import utils
import triangle


def plan_to_markerarray(
    plan: "nav_plan.Plan",
    map: BinaryOccupancyGrid,
    robot: agent.Agent,
    frame_id: str,
    stamp: Time = Time(),
):
    markerarray = MarkerArray()
    markers = []
    p_id = 0
    for component in plan.paths:
        current_color = ColorRGBA(
            **colors.hex_to_rgba(Style.from_string(robot.agent_style.shape).fill)
        )

        if component.path_type == PathType.TRANSFER:
            current_color = ColorRGBA(
                **colors.hex_to_rgba(
                    colors.darken(Style.from_string(robot.agent_style.shape).fill)
                )
            )

            polygon = component.obstacle_path.polygons[-1]

            obstacle_end_polygon_marker = polygon_to_line_strip(
                polygon=polygon,
                namespace="/end_obstacles",
                p_id=p_id,
                frame_id=frame_id,
                color=current_color,
                z_index=cfg.path_line_z_index,
                line_width=robot.min_inflation_radius / 4,
            )
            markers.append(obstacle_end_polygon_marker)
        path_marker = real_path_to_triangle_list(
            real_path=component.robot_path.poses,
            map=map,
            p_id=p_id,
            frame_id=frame_id,
            color=current_color,
            line_width=robot.min_inflation_radius / 5,
            z_index=cfg.path_line_z_index,
            stamp=stamp,
        )
        markers.append(path_marker)
        p_id += 1
    markerarray.markers = markers
    return markerarray


# Basic conversion functions


def real_path_to_linestrip(
    real_path: t.List[Pose2D],
    namespace: str,
    p_id: int,
    frame_id: str,
    color: ColorRGBA,
    line_width: float,
    z_index: float,
    link_point: Point | None = None,
    stamp: Time = Time(),
):
    marker = Marker(
        type=Marker.LINE_STRIP,
        ns=namespace,
        id=p_id,
        header=Header(frame_id=frame_id, stamp=stamp),
        color=color,
        scale=Vector3(x=line_width, y=line_width, z=0.0),
        points=[],
    )
    for i in range(len(real_path) - 1):
        point = real_path[i]
        next_point = real_path[i + 1]
        marker.points.append(Point(x=point[0], y=point[1], z=z_index))  # type: ignore
        marker.points.append(Point(x=next_point[0], y=next_point[1], z=z_index))  # type: ignore
    if link_point:
        marker.points.append(Point(x=real_path[-1][0], y=real_path[-1][1], z=z_index))  # type: ignore
        marker.points.append(Point(x=link_point[0], y=link_point[1], z=z_index))  # type: ignore
    return marker


def polygon_to_triangle_list(
    *,
    polygon: Polygon,
    p_id: int,
    frame_id: str,
    color: ColorRGBA,
    z_index: float,
    stamp: Time = Time(),
    namespace: str = "",
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
        id=p_id,
        header=Header(frame_id=frame_id, stamp=stamp),
        color=color,
        scale=Vector3(x=1.0, y=1.0, z=1.0),
        points=[],
        ns=namespace,
    )
    if isinstance(polygon, Polygon):
        triangles = polygon_to_triangle_vertices(polygon)
        marker.points = [
            Point(x=point[0], y=point[1], z=z_index)
            for triangle in triangles
            for point in triangle
        ]
    return marker


def polygon_to_line_strip(
    *,
    polygon: Polygon,
    p_id: int,
    frame_id: str,
    color: ColorRGBA,
    z_index: float,
    line_width: float,
    stamp: Time = Time(),
    namespace: str = "",
):
    marker = Marker(
        type=Marker.LINE_STRIP,
        ns=namespace,
        id=p_id,
        header=Header(frame_id=frame_id, stamp=stamp),
        color=color,
        scale=Vector3(x=line_width, y=line_width, z=0.0),
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


def polygon_to_rim_points(
    polygon: Polygon,
):
    points: t.List[npt.NDArray[t.Any]] = []
    for i in range(len(polygon.exterior.coords)):
        point = polygon.exterior.coords[i]
        points.append(np.array((point[0], point[1])))

    if len(points) > 0:
        points.append(points[0])

    return points


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
    grid_map.info.pose.position.x = -grid_map.info.length_x / 2
    grid_map.info.pose.position.y = -grid_map.info.length_y / 2
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


def pose_to_ros_pose(pose: Pose2D) -> Pose:
    x, y, z = pose[0], pose[1], 0.0
    return Pose(
        position=(Point(x=x, y=y, z=z)),
        orientation=geom_quat_from_yaw(pose[2]),
    )


def real_path_to_triangle_list(
    real_path: t.Sequence[t.Tuple[float, float, float] | t.Tuple[float, float]],
    map: BinaryOccupancyGrid,
    p_id: int,
    frame_id: str,
    color: ColorRGBA,
    line_width: float,
    z_index: float,
    stamp: Time = Time(),
):
    """Takes a robot path as a sequence of points and converts them to a TRIANGLE_LIST marker for RVIZ."""
    points = [np.array(x) for x in real_path]
    polygon = utils.path_to_polygon(points=points, line_width=line_width)

    return polygon_to_triangle_list(
        polygon=polygon,
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
