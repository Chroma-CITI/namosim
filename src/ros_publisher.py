import yaml
import time
import math
import numpy as np

from shapely.ops import triangulate
from shapely import affinity

import rospy
import tf
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Quaternion, Point, Vector3, PoseArray, PoseStamped, Polygon as RosPolygon, Point32,\
    PolygonStamped
from std_msgs.msg import Header, ColorRGBA
from nav_msgs.msg import Path, GridCells, OccupancyGrid, MapMetaData

import utils
from robot import Robot
from obstacle import Obstacle

# Conversion methods


def init_header():
    return Header(stamp=rospy.Time.now(), frame_id="map")


def init_grid_cells(resolution):
    return GridCells(header=init_header(), cell_width=resolution, cell_height=resolution, cells=[])


def init_ros_path():
    return Path(header=init_header(), poses=[])


def world_to_costmap(world):
    costmap = OccupancyGrid(header=init_header())
    costmap.info.map_load_time = costmap.header.stamp
    costmap.info.resolution = world.dd.res
    costmap.info.width = world.get_grid().shape[0]
    costmap.info.height = world.get_grid().shape[1]
    costmap.info.origin.position.x = world.dd.grid_pose[0]
    costmap.info.origin.position.y = world.dd.grid_pose[1]
    costmap.info.origin.position.z = -0.1
    costmap.data = np.fliplr(np.rot90(world.get_grid(), 3)).flatten().astype(np.int8).tolist()

    return costmap


def grid_cells_to_ros_cells(grid_cells, dd):
    ros_cells = init_grid_cells(dd.res)
    for cell in grid_cells:
        point = Point()
        point.x, point.y = utils.grid_to_real(cell[0], cell[1], dd)
        ros_cells.cells.append(point)
    return ros_cells


def geom_quat_from_yaw(yaw):
    explicit_quat = tf.transformations.quaternion_from_euler(0.0, 0.0, math.radians(yaw))
    return Quaternion(x=explicit_quat[0], y=explicit_quat[1], z=explicit_quat[2], w=explicit_quat[3])


def real_path_to_ros_path(real_path):
    ros_path = Path(header=init_header(), poses=[])
    for pose in real_path:
        ros_path.poses.append(PoseStamped(header=ros_path.header, pose=pose_to_ros_pose(pose)))
    return ros_path


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
    triangles = triangulate(polygon)
    for triangle in triangles:
        for point in triangle.exterior.coords[:len(triangle.exterior.coords) - 1]:
            marker.points.append(Point(point[0], point[1], z_index))
    return marker


def polygon_to_line_strip(polygon, namespace, p_id, frame_id, color, z_index, line_width):
    marker = Marker(type=Marker.LINE_STRIP,
                    ns=namespace,
                    id=p_id,
                    header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                    color=color,
                    scale=Vector3(line_width, 0.0, 0.0),
                    points=[])
    for i in range(len(polygon.exterior.coords) - 1):
        point = polygon.exterior.coords[i]
        next_point = polygon.exterior.coords[i+1]
        marker.points.append(Point(point[0], point[1], z_index))
        marker.points.append(Point(next_point[0], next_point[1], z_index))
    marker.points.append(Point(polygon.exterior.coords[0][0], polygon.exterior.coords[0][1], z_index))
    marker.points.append(Point(polygon.exterior.coords[1][0], polygon.exterior.coords[1][1], z_index))
    return marker


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


def make_delete_all_marker(frame_id):
    return MarkerArray(
        markers=[Marker(header=Header(frame_id=frame_id, stamp=rospy.Time.now()), action=Marker.DELETEALL)])


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


class RosPublisher:
    def __init__(self, config_path):
        config = yaml.load(open(config_path))

        # Actual topics
        self.movables_topic = config["movables_topic"]
        self.unmovables_topic = config["unmovables_topic"]
        self.unknowns_topic = config["unknowns_topic"]
        self.robot_topic = config["robot_topic"]
        self.taboos_topic = config["taboos_topic"]

        self.robot_g_fov_topic = config["robot_g_fov_topic"]
        self.robot_s_fov_topic = config["robot_s_fov_topic"]

        self.entities_in_g_fov_topic = config["entities_in_g_fov_topic"]

        self.entities_in_s_fov_topic = config["entities_in_s_fov_topic"]
        self.costmap_topic = config["costmap_topic"]
        self.path_grid_cells_topic = config["path_grid_cells_topic"]

        self.a_star_open_heap_topic = config["a_star_open_heap_topic"]
        self.a_star_close_set_topic = config["a_star_close_set_topic"]

        # Dictionary of Publishers
        self.publishers = dict()

        # Dictionary of Activation variables
        self.is_activated = dict()

        self.is_activated[self.movables_topic] = config["movables_topic_activated"]
        self.is_activated[self.unmovables_topic] = config["unmovables_topic_activated"]
        self.is_activated[self.unknowns_topic] = config["unknowns_topic_activated"]
        self.is_activated[self.robot_topic] = config["robot_topic_activated"]
        self.is_activated[self.taboos_topic] = config["taboos_topic_activated"]

        self.is_activated[self.robot_g_fov_topic] = config["robot_g_fov_topic_activated"]
        self.is_activated[self.robot_s_fov_topic] = config["robot_s_fov_topic_activated"]

        self.is_activated[self.entities_in_g_fov_topic] = config["entities_in_g_fov_topic_activated"]

        self.is_activated[self.entities_in_s_fov_topic] = config["entities_in_s_fov_topic_activated"]
        self.is_activated[self.costmap_topic] = config["costmap_topic_activated"]
        self.is_activated[self.path_grid_cells_topic] = config["path_grid_cells_topic_activated"]

        self.is_activated[self.a_star_open_heap_topic] = config["a_star_open_heap_topic_activated"]
        self.is_activated[self.a_star_close_set_topic] = config["a_star_close_set_topic_activated"]

        self.rate = rospy.Rate(20)

        self.frame_id = "/map"

        self.robot_color = ColorRGBA(0.427, 0.62, 0.922, 1.0)  # HEX 6d9eebff
        self.movable_obstacle_color = ColorRGBA(0.945, 0.761, 0.196, 1.0)  # HEX f1c232ff
        # self.unmovable_obstacle_color = ColorRGBA(0.471, 0.247, 0.016, 1.0)  # HEX 783f04ff
        self.unmovable_obstacle_color = ColorRGBA(0.0, 0.0, 0.0, 1.0)  # HEX ffffffff
        self.unknown_obstacle_color = ColorRGBA(0.557, 0.486, 0.765, 1.0)  # HEX 8e7cc3ff
        self.taboo_color = ColorRGBA(0.918, 0.6, 0.6, 0.7)  # HEX ea9999ff

        self.robot_border_color = ColorRGBA(0.067, 0.333, 0.8, 1.0)  # HEX 1155ccff
        self.movable_obstacle_border_color = ColorRGBA(0.498, 0.376, 0.0, 1.0)  # HEX 7f6000
        self.unmovable_obstacle_border_color = ColorRGBA(0.302, 0.157, 0.008, 1.0)  # HEX 4d2802
        self.unknown_obstacle_border_color = ColorRGBA(0.208, 0.11, 0.459, 1.0)  # HEX 351c75ff
        self.taboo_border_color = ColorRGBA(0.8, 0.0, 0.0, 0.7)  # HEX cc0000ff
        # self.g_fov_border_color = ColorRGBA(0.067, 0.333, 0.8, 1.0)  # HEX 1155ccff
        self.g_fov_border_color = ColorRGBA(0.427, 0.62, 0.922, 1.0)  # HEX 6d9eebff
        # self.s_fov_border_color = self.g_fov_border_color
        self.s_fov_border_color = ColorRGBA(0.416, 0.659, 0.31, 1.0)  # HEX 6aa84fff

        self.min_inflated_polygon_border_color = ColorRGBA(0.4, 0.4, 0.4, 1.0)  # HEX 666666ff
        self.max_inflated_polygon_border_color = ColorRGBA(0.4, 0.4, 0.4, 1.0)  # HEX 666666ff

        self.text_color_on_filling = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # HEX ffffffff
        self.text_color_on_empty = ColorRGBA(0.0, 0.0, 0.0, 1.0)  # HEX 000000ff

        self.fov_z_index = -0.04
        self.entities_z_index = -0.05
        self.taboos_z_index = -0.06

        self.fov_line_width = 0.05
        self.border_width = 0.075
        self.text_height = 0.2

    def publish(self, topic, msg, force_publish_once=False):
        self.rate.sleep()
        try:
            publisher = self.publishers[topic]
        except KeyError:
            publisher = rospy.Publisher(topic, type(msg), queue_size=10)
            self.publishers[topic] = publisher
            time.sleep(0.5)  # wait_publisher_is_ready(publisher)
        if force_publish_once:
            publish_once(publisher, msg)
        else:
            publisher.publish(msg)

    def world_to_marker_array(self, world, namespace):
        marker_array = MarkerArray()
        markers = []
        for entity in world.entities.values():
            if isinstance(entity, Robot):
                markers = markers + entity_to_markers(
                    entity, namespace + "/robot", entity.uid, self.frame_id, self.robot_color, self.robot_border_color,
                    self.text_color_on_filling, self.text_color_on_empty, self.entities_z_index,
                    self.border_width, self.text_height, add_border=False, add_text=False)

                markers.append(polygon_to_line_strip(entity.s_fov_polygon, namespace + "/robot/s_fov", 0,
                                                     self.frame_id, self.s_fov_border_color, self.fov_z_index,
                                                     self.fov_line_width))
                markers.append(polygon_to_line_strip(entity.g_fov_polygon, namespace + "/robot/g_fov", 0,
                                                     self.frame_id, self.g_fov_border_color, self.fov_z_index,
                                                     self.fov_line_width))
            if isinstance(entity, Obstacle):
                if entity.movability == "movable":
                    markers = markers + entity_to_markers(
                        entity, namespace + "/obstacles", entity.uid, self.frame_id, self.movable_obstacle_color,
                        self.movable_obstacle_border_color, self.text_color_on_filling, self.text_color_on_empty,
                        self.entities_z_index, self.border_width, self.text_height, add_border=False, add_text=False)
                if entity.movability == "unmovable":
                    markers = markers + entity_to_markers(
                        entity, namespace + "/obstacles", entity.uid, self.frame_id, self.unmovable_obstacle_color,
                        self.unmovable_obstacle_border_color, self.text_color_on_filling, self.text_color_on_empty,
                        self.entities_z_index, self.border_width, self.text_height, add_border=False, add_text=False)
                if entity.movability == "unknown":
                    markers = markers + entity_to_markers(
                        entity, namespace + "/obstacles", entity.uid, self.frame_id, self.unknown_obstacle_color,
                        self.unknown_obstacle_border_color, self.text_color_on_filling, self.text_color_on_empty,
                        self.entities_z_index, self.border_width, self.text_height, add_border=False, add_text=False)
        for taboo in world.taboos.values():
            markers = markers + entity_to_markers(
                taboo, namespace + "/taboos", taboo.uid, self.frame_id, self.taboo_color, self.taboo_border_color,
                self.text_color_on_filling, self.text_color_on_empty, self.taboos_z_index,
                self.border_width, self.text_height, add_border=False, add_text=False)
        marker_array.markers = markers
        return marker_array

    def publish_world(self, world, namespace="/sim"):
        self.publish(namespace + "/knowledge", self.world_to_marker_array(world, namespace))
        self.publish_costmap(world, namespace)

    def publish_costmap(self, world, namespace):
        costmap = world_to_costmap(world)
        self.publish(namespace + "/costmap", costmap)

    def publish_a_star_open_heap(self, open_heap, dd):
        if self.is_activated[self.a_star_open_heap_topic]:
            open_heap_data = []
            for element in open_heap:
                open_heap_data.append(element[1])
            open_heap_cells = grid_cells_to_ros_cells(open_heap_data, dd)
            self.publish(self.a_star_open_heap_topic, open_heap_cells)

    def publish_a_star_close_set(self, close_set, dd):
        if self.is_activated[self.a_star_close_set_topic]:
            close_set_cells = grid_cells_to_ros_cells(list(close_set), dd)
            self.publish(self.a_star_close_set_topic, close_set_cells)

    def publish_multigoal_a_star_open_heap(self, open_heap, dd):
        open_heap_data = []
        for element in open_heap:
            open_heap_data.append(element[1])
        open_heap_cells = grid_cells_to_ros_cells(open_heap_data, dd)
        self.publish("/test/multigoal_a_star_open_heap", open_heap_cells)

    def publish_multigoal_a_star_close_set(self, close_set, dd):
        close_set_cells = grid_cells_to_ros_cells(list(close_set), dd)
        self.publish("/test/multigoal_a_star_close_set", close_set_cells)

    def publish_grid_path(self, grid_path, dd):
        if self.is_activated[self.path_grid_cells_topic]:
            path_grid_cells = grid_cells_to_ros_cells(grid_path, dd)
            self.publish(self.path_grid_cells_topic, path_grid_cells)

    def publish_real_path(self, topic, real_path):
        path = real_path_to_ros_path(real_path)
        self.publish(topic, path)

    def publish_q_manips_for_obs(self, poses):
        pose_array = poses_to_poses_array(poses)
        self.publish("/test/obs_manip_poses", pose_array)

    def publish_c_1(self, c1):
        self.publish_real_path("/robot/eval_c_1", c1.path)

    def publish_c_2(self, c2):
        self.publish_real_path("/robot/eval_c_2", c2.path)

    def publish_c_3(self, c3):
        self.publish_real_path("/robot/eval_c_3", c3.path)

    def publish_plan(self, plan_prefix, plan):
        if plan.path_components:
            self.publish_real_path("/robot" + plan_prefix + "/c_1", plan.path_components[0].path)
        if len(plan.path_components) == 3:
            self.publish_real_path("/robot" + plan_prefix + "/c_2", plan.path_components[1].path)
            self.publish_real_path("/robot" + plan_prefix + "/c_3", plan.path_components[2].path)

    def publish_p_opt(self, p_opt):
        self.publish_plan("/p_opt", p_opt)

    def publish_sim(self, robot_polygon, obs_polygon, namespace="/init"):
        robot_color = self.robot_border_color if namespace == "/target" else self.robot_color
        obs_color = self.movable_obstacle_border_color if namespace == "/target" else self.movable_obstacle_color
        marker_array = MarkerArray(markers=[
            polygon_to_line_strip(robot_polygon, namespace + "/robot/polygon", 0, self.frame_id, robot_color,
                                  self.entities_z_index, self.border_width),
            polygon_to_line_strip(obs_polygon, namespace + "/obstacle/polygon", 0, self.frame_id, obs_color,
                                  self.entities_z_index, self.border_width)])
        self.publish("/robot/sim", marker_array)

    def publish_min_max_inflated(self, min_inflated_polygon, max_inflated_polygon):
        marker_array = MarkerArray(markers=[
            polygon_to_line_strip(min_inflated_polygon, "/min_inflated_polygon", 0, self.frame_id,
                                  self.min_inflated_polygon_border_color,
                                  self.entities_z_index, self.border_width),
            polygon_to_line_strip(max_inflated_polygon, "/max_inflated_polygon", 0, self.frame_id,
                                  self.max_inflated_polygon_border_color,
                                  self.entities_z_index, self.border_width)])
        self.publish("/robot/compute_c_0_c1/min_max_inflated_polygons", marker_array)

    def publish_q_l_cells(self, cells, dd):
        close_set_cells = grid_cells_to_ros_cells(list(cells), dd)
        self.publish("/robot/compute_c_0_c1/q_l_cells", close_set_cells)

    def publish_q_l_poses(self, poses):
        pose_array = poses_to_poses_array(poses)
        self.publish("/robot/compute_c_0_c1/q_l_poses", pose_array)

    def publish_goal(self, q_init, q_goal, polygon):
        polygon_at_goal_pose = affinity.translate(polygon, q_goal[0] - q_init[0], q_goal[1] - q_init[1])
        # ros_pose = pose_to_ros_pose_stamped(q_goal)
        marker_array = MarkerArray(markers=[
            polygon_to_line_strip(polygon_at_goal_pose, "/polygon", 0, self.frame_id, self.robot_border_color,
                                  self.fov_z_index, self.border_width)])
            # pose_to_arrow(q_goal, "/pose", 0, self.frame_id, self.robot_border_color,
            #               self.entities_z_index, 0.5, 0.2, 0.0)])
        self.publish("/robot/goal", marker_array)

    def cleanup_world(self, namespace="/sim"):
        self.publish(namespace + "/knowledge", make_delete_all_marker(self.frame_id))
        self.publish(namespace + "/costmap", OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    def cleanup_eval_c1_c2_c3_sim_init_target(self):
        self.publish("/robot/eval_c_1", init_ros_path())
        self.publish("/robot/eval_c_2", init_ros_path())
        self.publish("/robot/eval_c_3", init_ros_path())
        self.publish("/robot/sim", make_delete_all_marker(self.frame_id))

    def cleanup_p_opt(self):
        self.publish("/robot/p_opt/c_1", init_ros_path())
        self.publish("/robot/p_opt/c_2", init_ros_path())
        self.publish("/robot/p_opt/c_3", init_ros_path())
        self.publish("/robot_sim/costmap", OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    def cleanup_q_manips_for_obs(self):
        pose_array = PoseArray(header=Header(frame_id=self.frame_id, stamp=rospy.Time.now()), poses=[])
        self.publish("/test/obs_manip_poses", pose_array)

    def cleanup_goal(self):
        self.publish("/robot/goal", make_delete_all_marker(self.frame_id))

    def cleanup_q_l_cells_poses(self):
        self.publish("/robot/compute_c_0_c1/q_l_cells", init_grid_cells(0.1))
        self.publish("/robot/compute_c_0_c1/q_l_poses", PoseArray(header=init_header()))

    def cleanup_min_max_inflated(self):
        self.publish("/robot/compute_c_0_c1/min_max_inflated_polygons", make_delete_all_marker(self.frame_id))

    def cleanup_a_star_open_heap(self):
        self.publish(self.a_star_open_heap_topic, init_grid_cells(0.1))

    def cleanup_a_star_close_set(self):
        self.publish(self.a_star_close_set_topic, init_grid_cells(0.1))

    def cleanup_multigoal_a_star_open_heap(self):
        self.publish("/test/multigoal_a_star_open_heap", init_grid_cells(0.1))

    def cleanup_multigoal_a_star_close_set(self):
        self.publish("/test/multigoal_a_star_close_set", init_grid_cells(0.1))

    def cleanup_grid_path(self):
        self.publish(self.path_grid_cells_topic, init_grid_cells(0.1))

    def cleanup_all(self):
        self.cleanup_world(namespace="/sim")
        self.cleanup_world(namespace="/robot")
        self.cleanup_eval_c1_c2_c3_sim_init_target()
        self.cleanup_p_opt()
        self.cleanup_q_manips_for_obs()
        self.cleanup_goal()
        self.cleanup_q_l_cells_poses()
        self.cleanup_min_max_inflated()
        self.cleanup_a_star_open_heap()
        self.cleanup_a_star_close_set()
        self.cleanup_multigoal_a_star_open_heap()
        self.cleanup_multigoal_a_star_close_set()
        self.cleanup_grid_path()
