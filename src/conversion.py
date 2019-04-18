import rospy
import tf
from geometry_msgs.msg import Polygon as RosPolygon, Point32, PolygonStamped, PoseStamped, Point, Quaternion, PoseArray,\
    Pose
from nav_msgs.msg import Path, GridCells, OccupancyGrid
from std_msgs.msg import Header

from jsk_recognition_msgs.msg import PolygonArray
import numpy as np

from robot import Robot
from obstacle import Obstacle
from taboo import Taboo

import math

# Conversions between Continuous Coordinates to Grid Coordinates #


def real_to_grid(real_x, real_y, dd):
    return int((real_x - dd.grid_pose[0]) / dd.res), int((real_y - dd.grid_pose[1]) / dd.res)


def grid_to_real(cell_x, cell_y, dd):
    return dd.res * float(cell_x) + dd.grid_pose[0] + dd.res * 0.5, dd.res * float(cell_y) + dd.grid_pose[1] + dd.res * 0.5


def yaw_from_direction(direction_vector):
    if direction_vector[1] < 0:
        yaw = 2 * math.pi - math.acos(
            direction_vector[0] / math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
    else:
        yaw = math.acos(
            direction_vector[0] / math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
    return math.degrees(yaw)


def grid_path_to_real_path(grid_path, start_pose, goal_pose, dd):
    if not grid_path:
        return []
    real_path = [start_pose]
    previous_pose = start_pose
    for cell in grid_path[1:len(grid_path) - 1]:
        real_x, real_y = grid_to_real(cell[0], cell[1], dd)
        direction_vector = (real_x - previous_pose[0], real_y - previous_pose[1])
        real_yaw = yaw_from_direction(direction_vector)
        new_pose = (real_x, real_y, real_yaw)
        real_path.append(new_pose)
        previous_pose = new_pose
    real_path.append(goal_pose)
    return real_path

# Conversion between custom types (world, entity, ...) to ROS msgs #
def init_header():
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"
    return header

def init_jsk_polygon_array():
    jsk_polygon_array = PolygonArray()
    jsk_polygon_array.header = init_header()
    jsk_polygon_array.polygons = []
    jsk_polygon_array.labels = []
    jsk_polygon_array.likelihood = []
    return jsk_polygon_array


def init_grid_cells(resolution):
    grid_cells = GridCells()
    grid_cells.header = init_header()
    grid_cells.cell_width = resolution
    grid_cells.cell_height = resolution
    grid_cells.cells = []
    return grid_cells


def init_ros_path():
    ros_path = Path()
    ros_path.header = init_header()
    ros_path.poses = []
    return ros_path


def append_to_polygon_array(jsk_polygon_array, polygon, label, likelyhood, layer_value):
    jsk_polygon_array.polygons.append(shapely_polygon_to_ros_polygon(polygon, layer_value))
    jsk_polygon_array.labels.append(label)
    jsk_polygon_array.likelihood.append(likelyhood)

# See https://jsk-visualization.readthedocs.io/en/latest/jsk_rviz_plugins/plugins/polygon_array.html#properties
# And https://github.com/jsk-ros-pkg/jsk_visualization
# and http://wiki.ros.org/rviz/DisplayTypes/Marker
# and https://shapely.readthedocs.io/en/stable/manual.html
def world_to_multiple_jsk_polygon_array(world):
    movables, unmovables, unknowns = init_jsk_polygon_array(), init_jsk_polygon_array(), init_jsk_polygon_array()
    robot = init_jsk_polygon_array()
    taboos = init_jsk_polygon_array()

    # Add objects
    for entity_uuid, entity in world.entities.items():
        if isinstance(entity, Robot):
            append_to_polygon_array(robot, entity.polygon, entity.uid, 0, layer_value=-0.01)
        elif isinstance(entity, Obstacle):
            if entity.movability == "movable":
                append_to_polygon_array(movables, entity.polygon, entity.uid, choose_color(entity),
                                        layer_value=-0.05)
            elif entity.movability == "unmovable":
                append_to_polygon_array(unmovables, entity.polygon, entity.uid, choose_color(entity),
                                        layer_value=-0.05)
            elif entity.movability == "unknown":
                append_to_polygon_array(unknowns, entity.polygon, entity.uid, choose_color(entity),
                                        layer_value=-0.05)

    # Add taboos
    for taboo_uid, taboo in world.taboos.items():
        append_to_polygon_array(taboos, taboo.polygon, taboo.uid, 0.9, layer_value=-0.06)

    return movables, unmovables, unknowns, robot, taboos


def world_to_costmap(world):
    costmap = OccupancyGrid()
    costmap.header = init_header()
    costmap.info.map_load_time = costmap.header.stamp
    costmap.info.resolution = world.dd.res
    costmap.info.width = world.get_grid().shape[0]
    costmap.info.height = world.get_grid().shape[1]
    costmap.info.origin.position.x = world.dd.grid_pose[0]
    costmap.info.origin.position.y = world.dd.grid_pose[1]
    costmap.info.origin.position.z = -0.1
    costmap.data = np.fliplr(np.rot90(world.get_grid(), 3)).flatten().astype(np.int8).tolist()

    return costmap


def entities_to_jsk_polygon_array(entities, layer_value=0.0):
    jsk_polygon_array = init_jsk_polygon_array()

    # Add objects
    for entity_uuid, entity in entities.items():
        append_to_polygon_array(jsk_polygon_array, entity.polygon, entity.uid, choose_color(entity),
                                layer_value)

    return jsk_polygon_array

def polygons_to_jsk_polygon_array(polygons, label=1, color_value=0.0, layer_value=0.0):
    jsk_polygon_array = init_jsk_polygon_array()

    # Add objects
    for polygon in polygons:
        append_to_polygon_array(jsk_polygon_array, polygon, label, color_value, layer_value)

    return jsk_polygon_array


def choose_color(entity):
    if isinstance(entity, Robot):
        return 0.0
    elif isinstance(entity, Obstacle):
        if entity.movability == "unknown":
            return 0.22
        elif entity.movability == "unmovable":
            return 0.80
        elif entity.movability == "movable":
            return 0.35
    elif isinstance(entity, Taboo):
        return 0.9
    return 1.0  # Should not happen


def shapely_linestring_to_ros_path(shapely_linestring):
    path = Path()
    path.header = init_header()
    path.poses = []
    coords = list(shapely_linestring.coords)
    for coord in coords:
        pose = PoseStamped()
        pose.header.stamp = path.header.stamp
        pose.header.frame_id = path.header.frame_id
        pose.pose.position.x = coord[0]
        pose.pose.position.y = coord[1]
        path.poses.append(pose)
    return path


def shapely_polygon_to_ros_polygon(shapely_polygon, layer_value):
    polygon_stamped = PolygonStamped()
    polygon_stamped.header = init_header()
    ros_polygon = RosPolygon()
    coords = list(shapely_polygon.exterior.coords)
    for coord in coords:
        point = Point32()
        point.x = coord[0]
        point.y = coord[1]
        point.z = layer_value
        ros_polygon.points.append(point)
    polygon_stamped.polygon = ros_polygon
    return polygon_stamped


def grid_cells_to_ros_cells(grid_cells, dd):
    ros_cells = init_grid_cells(dd.res)
    for cell in grid_cells:
        point = Point()
        point.x, point.y = grid_to_real(cell[0], cell[1], dd)
        ros_cells.cells.append(point)
    return ros_cells


def geom_quat_from_yaw(yaw):
    explicit_quat = tf.transformations.quaternion_from_euler(0.0, 0.0, math.radians(yaw))
    geom_quat = Quaternion()
    geom_quat.x = explicit_quat[0]
    geom_quat.y = explicit_quat[1]
    geom_quat.z = explicit_quat[2]
    geom_quat.w = explicit_quat[3]
    return geom_quat


def real_path_to_ros_path(real_path):
    ros_path = Path()
    ros_path.header = init_header()
    ros_path.poses = []
    for pose in real_path:
        new_ros_pose = PoseStamped()
        new_ros_pose.header.stamp = ros_path.header.stamp
        new_ros_pose.header.frame_id = ros_path.header.frame_id
        new_ros_pose.pose.position.x = pose[0]
        new_ros_pose.pose.position.y = pose[1]
        new_ros_pose.pose.orientation = geom_quat_from_yaw(pose[2])
        ros_path.poses.append(new_ros_pose)
    return ros_path


def poses_to_poses_array(poses):
    pose_array = PoseArray()
    pose_array.header = init_header()
    pose_array.poses = []
    for pose in poses:
        pose_array.poses.append(pose_to_ros_pose(pose))
    return pose_array


def pose_to_ros_pose(pose):
    ros_pose = Pose()
    ros_pose.position.x = pose[0]
    ros_pose.position.y = pose[1]
    ros_pose.orientation = geom_quat_from_yaw(pose[2])
    return ros_pose