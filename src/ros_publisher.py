import rospy
import conversion
import yaml
import time

from shapely.ops import triangulate

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Quaternion, Point, Vector3, PoseArray
from std_msgs.msg import Header, ColorRGBA

from robot import Robot
from obstacle import Obstacle

# Conversion methods


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


def make_entity_delete_markers(namespace, p_id):
    return [make_delete_marker(namespace + "/polygon", p_id),
            make_delete_marker(namespace + "/border", p_id),
            make_delete_marker(namespace + "/text", p_id)]


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
        self.real_path_topic = config["real_path_topic"]

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
        self.is_activated[self.real_path_topic] = config["real_path_topic_activated"]

        self.is_activated[self.a_star_open_heap_topic] = config["a_star_open_heap_topic_activated"]
        self.is_activated[self.a_star_close_set_topic] = config["a_star_close_set_topic_activated"]

        self.rate = rospy.Rate(2000)

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
        self.s_fov_border_color = ColorRGBA(0.416, 0.659, 0.31, 1.0)  # HEX 6aa84fff

        self.text_color_on_filling = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # HEX ffffffff
        self.text_color_on_empty = ColorRGBA(0.0, 0.0, 0.0, 1.0)  # HEX 000000ff

        self.fov_z_index = -0.04
        self.entities_z_index = -0.05
        self.taboos_z_index = -0.06

        self.fov_line_width = 0.05
        self.border_width = 0.05
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
                                                     self.frame_id, self.g_fov_border_color, self.fov_z_index,
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
        costmap = conversion.world_to_costmap(world)
        self.publish(namespace + "/costmap", costmap)

    def publish_a_star_open_heap(self, open_heap, dd):
        if self.is_activated[self.a_star_open_heap_topic]:
            open_heap_data = []
            for element in open_heap:
                open_heap_data.append(element[1])
            open_heap_cells = conversion.grid_cells_to_ros_cells(open_heap_data, dd)
            self.publish(self.a_star_open_heap_topic, open_heap_cells)

    def publish_a_star_close_set(self, close_set, dd):
        if self.is_activated[self.a_star_close_set_topic]:
            close_set_cells = conversion.grid_cells_to_ros_cells(list(close_set), dd)
            self.publish(self.a_star_close_set_topic, close_set_cells)

    def publish_grid_path(self, grid_path, dd):
        if self.is_activated[self.path_grid_cells_topic]:
            path_grid_cells = conversion.grid_cells_to_ros_cells(grid_path, dd)
            self.publish(self.path_grid_cells_topic, path_grid_cells)

    def publish_real_path(self, topic, real_path):
        path = conversion.real_path_to_ros_path(real_path)
        self.publish(topic, path)

    def publish_q_manips_for_obs(self, poses):
        pose_array = conversion.poses_to_poses_array(poses)
        self.publish("/test/obs_manip_poses", pose_array)

    def publish_entity(self, topic, entity):
        jsk = conversion.entities_to_jsk_polygon_array({entity.uid: entity})
        self.publish(topic, jsk)

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

    def publish_p(self, p):
        self.publish_plan("/p", p)

    def publish_p_best(self, p_best):
        self.publish_plan("/p_best", p_best)

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

    def publish_goal(self, pose):
        ros_pose = conversion.pose_to_ros_pose(pose)
        self.publish("/robot/goal", ros_pose)

    def publish_evaluated_entity(self, entity):
        self.publish_entity("/robot/evaluated_entity", entity)

    def cleanup_world(self, namespace="/sim"):
        self.publish(namespace + "/knowledge", make_delete_all_marker(self.frame_id))

    def cleanup_eval_c1_c2_c3_sim_init_target(self):
        self.publish("/robot/eval_c_1", conversion.init_ros_path())
        self.publish("/robot/eval_c_2", conversion.init_ros_path())
        self.publish("/robot/eval_c_3", conversion.init_ros_path())
        self.publish("/robot/sim", make_delete_all_marker(self.frame_id))

    def cleanup_p_opt(self):
        self.publish("/robot/p_opt/c_1", conversion.init_ros_path())
        self.publish("/robot/p_opt/c_2", conversion.init_ros_path())
        self.publish("/robot/p_opt/c_3", conversion.init_ros_path())

    def cleanup_q_manips_for_obs(self):
        pose_array = PoseArray(header=Header(frame_id=self.frame_id, stamp=rospy.Time.now()), poses=[])
        self.publish("/test/obs_manip_poses", pose_array)
