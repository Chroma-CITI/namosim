import time
import math
import numpy as np

import mapbox_earcut as earcut

import rospy
from shapely.geometry import Polygon

from snamosim.display import tf_replacement
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Quaternion, Point, Vector3, PoseArray, PoseStamped
from std_msgs.msg import Header, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from nav_msgs.msg import Path, GridCells, OccupancyGrid
from grid_map_msgs.msg import GridMap

from snamosim.utils import utils
import ros_publisher_config as cfg
from snamosim.worldreps.entity_based.robot import Robot
from snamosim.worldreps.entity_based.obstacle import Obstacle
from snamosim.worldreps.entity_based.sensors.g_fov_sensor import GFOVSensor
from snamosim.worldreps.entity_based.sensors.s_fov_sensor import SFOVSensor
from snamosim.worldreps.occupation_based.binary_occupancy_grid import BinaryInflatedOccupancyGrid

def init_header():
    return Header(stamp=rospy.Time.now(), frame_id="map")


def init_grid_cells(resolution):
    return GridCells(header=init_header(), cell_width=resolution, cell_height=resolution,
                     cells=[Point(x=10000, y=10000, z=10000)])


def init_ros_path():
    return Path(header=init_header(), poses=[])


def world_to_costmap(world, robot_uid):
    robot = world.entities[robot_uid]
    polygons = {
        uid: entity.polygon for uid, entity in world.entities.items()
        if not isinstance(entity, Robot)
    }
    robot_max_inflation_radius = utils.get_circumscribed_radius(robot.polygon)
    static_obs_inf_grid = BinaryInflatedOccupancyGrid(
        polygons, world.dd.res, robot_max_inflation_radius, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD
    )

    costmap = OccupancyGrid(header=init_header())
    costmap.info.map_load_time = costmap.header.stamp
    costmap.info.resolution = static_obs_inf_grid.res
    costmap.info.width = static_obs_inf_grid.d_width
    costmap.info.height = static_obs_inf_grid.d_height
    costmap.info.origin.position.x = static_obs_inf_grid.grid_pose[0]
    costmap.info.origin.position.y = static_obs_inf_grid.grid_pose[1]
    costmap.info.origin.position.z = -0.1
    costmap.data = np.fliplr(np.rot90(static_obs_inf_grid.grid, 3)).flatten().astype(np.int8).tolist()

    return costmap


def world_to_marker_array(world, robot_uid, entities_to_ignore=tuple()):
    marker_array = MarkerArray()
    markers = []
    robot = world.entities[robot_uid]
    for entity in world.entities.values():
        if entity.uid not in entities_to_ignore:
            if isinstance(entity, Robot):
                markers = markers + entity_to_markers(
                    entity, "/robot", entity.uid, cfg.main_frame_id, cfg.robot_color, cfg.robot_border_color,
                    cfg.text_color_on_filling, cfg.text_color_on_empty, cfg.entities_z_index,
                    cfg.border_width, cfg.text_height, add_border=False, add_text=False)

                for sensor in entity.sensors:
                    if isinstance(sensor, SFOVSensor):
                        markers.append(polygon_to_line_strip(sensor.fov_polygon, "/robot/s_fov", 0,
                                                             cfg.main_frame_id, cfg.s_fov_border_color, cfg.fov_z_index,
                                                             cfg.fov_line_width))
                    elif isinstance(sensor, GFOVSensor):
                        markers.append(polygon_to_line_strip(sensor.fov_polygon, "/robot/g_fov", 0,
                                                             cfg.main_frame_id, cfg.g_fov_border_color, cfg.fov_z_index,
                                                             cfg.fov_line_width))

            if isinstance(entity, Obstacle):
                entity_movability = robot.deduce_movability(entity.type)
                if entity_movability == "movable":
                    markers = markers + entity_to_markers(
                        entity, "/obstacles", entity.uid, cfg.main_frame_id, cfg.movable_obstacle_color,
                        cfg.movable_obstacle_border_color, cfg.text_color_on_filling, cfg.text_color_on_empty,
                        cfg.entities_z_index, cfg.border_width, cfg.text_height, add_border=False, add_text=False)
                if entity_movability == "unmovable":
                    markers = markers + entity_to_markers(
                        entity, "/obstacles", entity.uid, cfg.main_frame_id, cfg.unmovable_obstacle_color,
                        cfg.unmovable_obstacle_border_color, cfg.text_color_on_filling, cfg.text_color_on_empty,
                        cfg.entities_z_index, cfg.border_width, cfg.text_height, add_border=False, add_text=False)
                if entity_movability == "unknown":
                    markers = markers + entity_to_markers(
                        entity, "/obstacles", entity.uid, cfg.main_frame_id, cfg.unknown_obstacle_color,
                        cfg.unknown_obstacle_border_color, cfg.text_color_on_filling, cfg.text_color_on_empty,
                        cfg.entities_z_index, cfg.border_width, cfg.text_height, add_border=False, add_text=False)
    for taboo in world.taboo_zones.values():
        markers = markers + entity_to_markers(
            taboo, "/taboos", taboo.uid, cfg.main_frame_id, cfg.taboo_color, cfg.taboo_border_color,
            cfg.text_color_on_filling, cfg.text_color_on_empty, cfg.taboos_z_index,
            cfg.border_width, cfg.text_height, add_border=False, add_text=False)
    marker_array.markers = markers
    return marker_array


def init_grid_map():
    grid_map = GridMap()
    grid_map.info.header = Header(stamp=rospy.Time.now(), frame_id="gridmap")
    grid_map.layers = []
    inflated_costmap_data = Float32MultiArray(
        layout=MultiArrayLayout(
            dim=[MultiArrayDimension(label="column_index",
                                     size=0,
                                     stride=0),
                 MultiArrayDimension(label="row_index",
                                     size=0,
                                     stride=0)],
            data_offset=0),
        data=[]
    )
    grid_map.data = [inflated_costmap_data]

    return grid_map


def costmap_to_grid_map(costmap, res, frame_id=cfg.social_gridmap_frame_id):
    grid_map = GridMap()
    grid_map.info.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    grid_map.info.resolution = res
    grid_map.info.length_x = costmap.shape[0] * res
    grid_map.info.length_y = costmap.shape[1] * res
    # grid_map.info.pose.position.z = 0. # The lib does not take this parameter into account...
    grid_map.layers = ["elevation"]
    inflated_costmap_data = Float32MultiArray(
        layout=MultiArrayLayout(
            dim=[MultiArrayDimension(label="column_index",
                                     size=costmap.shape[1],
                                     stride=costmap.shape[1]*costmap.shape[0]),
                 MultiArrayDimension(label="row_index",
                                     size=costmap.shape[0],
                                     stride=costmap.shape[0])],
            data_offset=0),
        # data=(costmap.flatten('F') / float(dd.cost_lethal)).astype(np.float32).tolist()
        data=(costmap.flatten('F')).astype(np.float32).tolist()
    )
    grid_map.data = [inflated_costmap_data]

    return grid_map


def grid_cells_to_cube_list_markers(grid_cells, res, grid_pose, color, cube_list=None):
    if cube_list is None:
        cube_list = Marker(
            type=Marker.CUBE_LIST,
            ns="",
            id=0,
            header=Header(frame_id=cfg.main_frame_id, stamp=rospy.Time.now()),
            color=color,
            scale=Vector3(res, res, res),
            points=[])
    for cell in grid_cells:
        point = Point()
        point.x, point.y = utils.grid_to_real(cell[0], cell[1], res, grid_pose)
        point.z = -0.5
        cube_list.points.append(point)
    return cube_list


def grid_cells_to_cube_markerarray(grid_cells, res, grid_pose, color, start_id=0):
    marker_array = MarkerArray()
    markers = []
    cur_id = start_id
    for cell in grid_cells:
        cur_id += 1
        x, y = utils.grid_to_real(cell[0], cell[1], res, grid_pose)
        z = -0.5
        cube = Marker(type=Marker.CUBE, ns="", id=cur_id,
            header=Header(frame_id=cfg.main_frame_id, stamp=rospy.Time.now()),
            color=color, scale=Vector3(res, res, res), pose=Pose(position=Vector3(x, y, z)))
        markers.append(cube)
    marker_array.markers = markers
    return marker_array, cur_id


def geom_quat_from_yaw(yaw):
    explicit_quat = tf_replacement.quaternion_from_euler(0.0, 0.0, math.radians(yaw))
    return Quaternion(x=explicit_quat[0], y=explicit_quat[1], z=explicit_quat[2], w=explicit_quat[3])


def real_path_to_ros_path(real_path):
    ros_path = Path(header=init_header(), poses=[])
    for pose in real_path:
        ros_path.poses.append(PoseStamped(header=ros_path.header, pose=pose_to_ros_pose(pose)))
    return ros_path


def plan_to_markerarray(plan, frame_id):
    markerarray = MarkerArray()
    markers = []
    p_id = 0
    for component in plan.path_components:
        current_color = cfg.transit_path_color
        if component.is_transfer:
            current_color = cfg.transfer_path_color
        marker = real_path_to_linestrip(
            component.robot_path.poses, '/plan', p_id, frame_id, current_color, cfg.path_line_width, cfg.path_line_z_index)
        markers.append(marker)
        p_id += 1
    markerarray.markers = markers
    return markerarray

# def real_path_to_pose_markers(real_path, )


def real_path_to_linestrip(real_path, namespace, p_id, frame_id, color, line_width, z_index, link_point=None):
    marker = Marker(type=Marker.LINE_STRIP,
                    ns=namespace,
                    id=p_id,
                    header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                    color=color,
                    scale=Vector3(line_width, 0.0, 0.0),
                    points=[])
    for i in range(len(real_path) - 1):
        point = real_path[i]
        next_point = real_path[i + 1]
        marker.points.append(Point(point[0], point[1], z_index))
        marker.points.append(Point(next_point[0], next_point[1], z_index))
    if link_point:
        marker.points.append(Point(real_path[-1][0], real_path[-1][1], z_index))
        marker.points.append(Point(link_point[0], link_point[1], z_index))
    return marker


def poses_to_poses_array(poses):
    pose_array = PoseArray(header=init_header(), poses=[])
    for pose in poses:
        pose_array.poses.append(pose_to_ros_pose(pose))
    return pose_array


def pose_to_ros_pose(pose):
    return Pose(position=Point(pose[0], pose[1], 0.0), orientation=geom_quat_from_yaw(pose[2]))


def pose_to_ros_pose_stamped(pose):
    return PoseStamped(header=init_header(), pose=pose_to_ros_pose(pose))


def polygon_to_triangle_list(polygon, namespace, p_id, frame_id, color, z_index):
    marker = Marker(type=Marker.TRIANGLE_LIST,
                    ns=namespace,
                    id=p_id,
                    header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                    color=color,
                    scale=Vector3(1.0, 1.0, 1.0),
                    points=[])
    if isinstance(polygon, Polygon):
        verts = np.array(list(polygon.exterior.coords)).reshape(-1, 2)
        rings = np.array([verts.shape[0]])
        triangles_vertices = verts[earcut.triangulate_float64(verts, rings)]
        triangles = [triangles_vertices[n:n + 3] for n in range(0, len(triangles_vertices), 3)]
        marker.points = [Point(point[0], point[1], z_index) for triangle in triangles for point in triangle]
    return marker


def polygon_to_line_strip(polygon, namespace, p_id, frame_id, color, z_index, line_width):
    marker = Marker(type=Marker.LINE_STRIP,
                    ns=namespace,
                    id=p_id,
                    header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                    color=color,
                    scale=Vector3(line_width, 0.0, 0.0),
                    points=[])
    if polygon is not None:
        for i in range(len(polygon.exterior.coords) - 1):
            point = polygon.exterior.coords[i]
            next_point = polygon.exterior.coords[i+1]
            marker.points.append(Point(point[0], point[1], z_index))
            marker.points.append(Point(next_point[0], next_point[1], z_index))
        marker.points.append(Point(polygon.exterior.coords[0][0], polygon.exterior.coords[0][1], z_index))
        marker.points.append(Point(polygon.exterior.coords[1][0], polygon.exterior.coords[1][1], z_index))
    return marker


def polygons_to_line_strips_marker_array(polygons, namespace, frame_id, color, z_index, line_width):
    marker_array = MarkerArray()
    markers = []
    p_id = 0
    for polygon in polygons:
        markers.append(
            polygon_to_line_strip(
                polygon, namespace, p_id, frame_id, color, z_index, line_width))
        p_id += 1
    marker_array.markers = markers
    return marker_array


def pose_to_arrow(pose, namespace, p_id, frame_id, color, z_index, shaft_diameter, head_diameter, head_length):
    marker = Marker(type=Marker.ARROW,
                    ns=namespace,
                    id=p_id,
                    pose=Pose(Point(pose[0], pose[1], z_index), geom_quat_from_yaw(pose[2])),
                    scale=Vector3(shaft_diameter, head_diameter, head_length),
                    header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                    color=color)
    return marker


def string_to_text(string, coordinates, namespace, p_id, frame_id, color, z_index, text_height):
    marker = Marker(type=Marker.TEXT_VIEW_FACING,
                    ns=namespace,
                    id=p_id,
                    pose=Pose(Point(coordinates[0], coordinates[1], z_index), Quaternion()),
                    scale=Vector3(0.0, 0.0, text_height),
                    header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                    color=color,
                    text=string)
    return marker


def make_delete_marker(namespace, p_id, frame_id):
    return Marker(ns=namespace, id=p_id, header=Header(frame_id=frame_id, stamp=rospy.Time.now()), action=Marker.DELETE)


def make_delete_all_marker(frame_id, ns=''):
    return MarkerArray(
        markers=[Marker(ns=ns, header=Header(frame_id=frame_id, stamp=rospy.Time.now()), action=Marker.DELETEALL)])


def entity_to_markers(entity, namespace, p_id, frame_id, color, border_color, text_color_filling, text_color_empty,
                      z_index, line_width, text_height,
                      add_filling=True, add_border=True, add_text=True,
                      add_uid=True, add_name=True):
    markers = []
    if add_filling:
        markers.append(
            polygon_to_triangle_list(entity.polygon, namespace + "/polygon", p_id, frame_id, color, z_index))
    if add_border:
        markers.append(
            polygon_to_line_strip(entity.polygon, namespace + "/border", p_id, frame_id,
                                  border_color, z_index, line_width))
    if add_text:
        string = ((("UID: " + str(entity.uid) + "\n") if add_uid else "") +
                  (("Name: " + entity.name + "\n") if add_name else ""))
        text_coordinates = entity.polygon.centroid.coords[0]
        markers.append(
            string_to_text(string, text_coordinates, namespace + "/text", p_id, frame_id,
                           text_color_filling if add_filling else text_color_empty, z_index, text_height))
    return markers


def make_entity_delete_markers(namespace, p_id, frame_id):
    return [make_delete_marker(namespace + "/polygon", p_id, frame_id),
            make_delete_marker(namespace + "/border", p_id, frame_id),
            make_delete_marker(namespace + "/text", p_id, frame_id)]


def wait_publisher_is_ready(publisher):
    while True:
        connections = publisher.get_num_connections()
        if connections > 0:
            return
        else:
            time.sleep(0.2)


def publish_once(publisher, msg):
    last_time = rospy.Time.now()
    while True:
        connections = publisher.get_num_connections()
        if connections > 0:
            publisher.publish(msg)
            break
        else:
            if rospy.Time.now() - last_time > rospy.Duration.from_sec(1.0):
                rospy.logwarn(
                    "Publishing data on " + publisher.name + ", but no one is listening, waiting...")
                last_time = rospy.Time.now()
