from __future__ import annotations

import math
import typing as t

import mapbox_earcut as earcut
import numpy as np
import numpy.typing as npt
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseArray,
    Quaternion,
    Vector3,
)
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
from namosim.display import tf_replacement
from namosim.models import PoseModel
from namosim.worldreps.entity_based.robot import Robot


def init_header(stamp: Time = Time()):
    return Header(stamp=stamp, frame_id=cfg.main_frame_id)


def plan_to_markerarray(plan: t.Any, robot: Robot, frame_id: str, stamp: Time = Time()):
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
        path_marker = real_path_to_linestrip(
            component.robot_path.poses,
            "/plan",
            p_id,
            frame_id,
            current_color,
            cfg.path_line_width,
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
        verts = np.array(list(polygon.exterior.coords)).reshape(-1, 2)
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


def real_path_to_linestrip(
    real_path: t.List[t.Tuple[float, float]],
    namespace: str,
    p_id: int,
    frame_id: str,
    color: ColorRGBA,
    line_width: float,
    z_index: float,
    link_point: t.Optional[t.Tuple[float, float]] = None,
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
    for i in range(len(real_path) - 1):
        point = real_path[i]
        next_point = real_path[i + 1]
        marker.points.append(Point(x=point[0], y=point[1], z=z_index))  # type: ignore
        marker.points.append(Point(x=next_point[0], y=next_point[1], z=z_index))  # type: ignore
    if link_point:
        marker.points.append(Point(x=real_path[-1][0], y=real_path[-1][1], z=z_index))  # type: ignore
        marker.points.append(Point(x=link_point[0], y=link_point[1], z=z_index))  # type: ignore
    return marker


def make_delete_marker(namespace, p_id, frame_id, stamp=Time()):
    return Marker(
        ns=namespace,
        id=p_id,
        header=Header(frame_id=frame_id, stamp=stamp),
        action=Marker.DELETE,
    )


def make_delete_all_marker(frame_id, ns="", stamp=Time()):
    return MarkerArray(
        markers=[
            Marker(
                ns=ns,
                header=Header(frame_id=frame_id, stamp=stamp),
                action=Marker.DELETEALL,
            )
        ]
    )
