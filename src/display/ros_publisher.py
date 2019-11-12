import time
import math
import numpy as np
from future.utils import with_metaclass

from src.display import triangulate
from shapely import affinity

import rospy

from src.display import tf_replacement
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Quaternion, Point, Vector3, PoseArray, PoseStamped, TransformStamped
from std_msgs.msg import Header, ColorRGBA, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from nav_msgs.msg import Path, GridCells, OccupancyGrid, MapMetaData
from grid_map_msgs.msg import GridMap

from src.utils import utils
from src.utils.singleton import Singleton
from src.worldreps.entity_based.robot import Robot
from src.worldreps.entity_based.obstacle import Obstacle

# Conversion methods


def hex_to_rgba(hex_string):
    hex_string = hex_string.lstrip('#')
    argb_tuple = tuple(int(hex_string[i:i + 2], 16) / 255. for i in (0, 2, 4, 6))
    rgba_tuple = (argb_tuple[1], argb_tuple[2], argb_tuple[3], argb_tuple[0])
    return rgba_tuple


def init_header():
    return Header(stamp=rospy.Time.now(), frame_id="map")


def init_grid_cells(resolution):
    return GridCells(header=init_header(), cell_width=resolution, cell_height=resolution, cells=[])


def init_ros_path():
    return Path(header=init_header(), poses=[])


def world_to_costmap(world, robot_uid):
    world_grid = world.get_binary_inflated_occupancy_grid((robot_uid,))
    costmap = OccupancyGrid(header=init_header())
    costmap.info.map_load_time = costmap.header.stamp
    costmap.info.resolution = world.dd.res
    costmap.info.width = world.dd.d_width
    costmap.info.height = world.dd.d_height
    costmap.info.origin.position.x = world.dd.grid_pose[0]
    costmap.info.origin.position.y = world.dd.grid_pose[1]
    costmap.info.origin.position.z = -0.1
    costmap.data = np.fliplr(np.rot90(world_grid, 3)).flatten().astype(np.int8).tolist()

    return costmap


def costmap_to_grid_map(costmap, dd):
    grid_map = GridMap()
    grid_map.info.header = Header(stamp=rospy.Time.now(), frame_id="gridmap")
    grid_map.info.resolution = dd.res
    grid_map.info.length_x = costmap.shape[0] * dd.res
    grid_map.info.length_y = costmap.shape[1] * dd.res
    grid_map.info.pose.position.x = 0.0
    grid_map.info.pose.position.y = 0.0
    grid_map.info.pose.position.z = -10.0
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
        data=(costmap.flatten('F') / float(dd.cost_lethal)).astype(np.float32).tolist()
    )
    grid_map.data = [inflated_costmap_data]

    return grid_map

def grid_cells_to_ros_cells(grid_cells, res, grid_pose):
    ros_cells = init_grid_cells(res)
    for cell in grid_cells:
        point = Point()
        point.x, point.y = utils.grid_to_real(cell[0], cell[1], res, grid_pose)
        ros_cells.cells.append(point)
    return ros_cells


def geom_quat_from_yaw(yaw):
    explicit_quat = tf_replacement.quaternion_from_euler(0.0, 0.0, math.radians(yaw))
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
    triangles = triangulate.triangulate(list(polygon.exterior.coords))
    for triangle in triangles:
        for point in triangle:
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


class RosPublisher(with_metaclass(Singleton)):
    def __init__(self):

        rospy.init_node('world_gui_test_node', log_level=rospy.INFO)

        # Actual topics
        self.min_max_inflated_polygons_topic = "/robot/compute_c_0_c1/min_max_inflated_polygons"
        self.path_grid_cells_topic = "/test/path_grid_cells"
        self.a_star_open_heap_topic = "/test/open_heap_cells"
        self.a_star_close_set_topic = "/test/close_set_cells"
        self.multi_a_star_open_heap_topic = "/test/multigoal_a_star_open_heap"
        self.multi_a_star_close_set_topic = "/test/multigoal_a_star_close_set"
        self.q_l_cells_topic = "/robot/compute_c_0_c1/q_l_cells"
        self.q_l_poses_topic = "/robot/compute_c_0_c1/q_l_poses"
        self.robot_goal_topic = "/robot/goal"
        self.obs_manip_poses_topic = "/test/obs_manip_poses"
        self.c_1_topic = "/robot/p_opt/c_1"
        self.c_2_topic = "/robot/p_opt/c_2"
        self.c_3_topic = "/robot/p_opt/c_3"
        self.robot_costmap = "/robot_sim/costmap"
        self.eval_c_1_topic = "/robot/eval_c_1"
        self.eval_c_2_topic = "/robot/eval_c_2"
        self.eval_c_3_topic = "/robot/eval_c_3"
        self.robot_sim_topic = "/robot/sim"
        self.robot_knowledge_topic = "/robot/knowledge"
        self.sim_knowledge_topic = "/sim/knowledge"
        self.robot_costmap_topic = "/robot/costmap"
        self.sim_costmap_topic = "/sim/costmap"
        self.robot_sim_costmap_topic = "/robot_sim/costmap"
        self.test_gridmap_topic = "/test/gridmap"

        self.social_cells_topic = "/test/social_cells"

        # Dictionary of Publishers
        self.default_queue_size = 10
        self.publishers = {
            self.min_max_inflated_polygons_topic: rospy.Publisher(self.min_max_inflated_polygons_topic, MarkerArray, queue_size=self.default_queue_size),
            self.path_grid_cells_topic: rospy.Publisher(self.path_grid_cells_topic, GridCells, queue_size=self.default_queue_size),
            self.a_star_open_heap_topic: rospy.Publisher(self.a_star_open_heap_topic, GridCells, queue_size=self.default_queue_size),
            self.a_star_close_set_topic: rospy.Publisher(self.a_star_close_set_topic, GridCells, queue_size=self.default_queue_size),
            self.multi_a_star_open_heap_topic: rospy.Publisher(self.multi_a_star_open_heap_topic, GridCells, queue_size=self.default_queue_size),
            self.multi_a_star_close_set_topic: rospy.Publisher(self.multi_a_star_close_set_topic, GridCells, queue_size=self.default_queue_size),
            self.q_l_cells_topic: rospy.Publisher(self.q_l_cells_topic, GridCells, queue_size=self.default_queue_size),
            self.q_l_poses_topic: rospy.Publisher(self.q_l_poses_topic, PoseArray, queue_size=self.default_queue_size),
            self.robot_goal_topic: rospy.Publisher(self.robot_goal_topic, MarkerArray, queue_size=self.default_queue_size),
            self.obs_manip_poses_topic: rospy.Publisher(self.obs_manip_poses_topic, PoseArray, queue_size=self.default_queue_size),
            self.c_1_topic: rospy.Publisher(self.c_1_topic, Path, queue_size=self.default_queue_size),
            self.c_2_topic: rospy.Publisher(self.c_2_topic, Path, queue_size=self.default_queue_size),
            self.c_3_topic: rospy.Publisher(self.c_3_topic, Path, queue_size=self.default_queue_size),
            self.robot_costmap: rospy.Publisher(self.robot_costmap, OccupancyGrid, queue_size=self.default_queue_size),
            self.eval_c_1_topic: rospy.Publisher(self.eval_c_1_topic, Path, queue_size=self.default_queue_size),
            self.eval_c_2_topic: rospy.Publisher(self.eval_c_2_topic, Path, queue_size=self.default_queue_size),
            self.eval_c_3_topic: rospy.Publisher(self.eval_c_3_topic, Path, queue_size=self.default_queue_size),
            self.robot_sim_topic: rospy.Publisher(self.robot_sim_topic, MarkerArray, queue_size=self.default_queue_size),
            self.robot_knowledge_topic: rospy.Publisher(self.robot_knowledge_topic, MarkerArray, queue_size=self.default_queue_size),
            self.sim_knowledge_topic: rospy.Publisher(self.sim_knowledge_topic, MarkerArray, queue_size=self.default_queue_size),
            self.robot_costmap_topic: rospy.Publisher(self.robot_costmap_topic, OccupancyGrid, queue_size=self.default_queue_size),
            self.sim_costmap_topic: rospy.Publisher(self.sim_costmap_topic, OccupancyGrid, queue_size=self.default_queue_size),
            self.robot_sim_costmap_topic: rospy.Publisher(self.robot_sim_costmap_topic, OccupancyGrid, queue_size=self.default_queue_size),
            self.test_gridmap_topic: rospy.Publisher(self.test_gridmap_topic, GridMap, queue_size= self.default_queue_size),
            self.social_cells_topic: rospy.Publisher(self.social_cells_topic, GridCells, queue_size= self.default_queue_size)
            }
        time.sleep(0.5)  # wait_publisher_is_ready(publisher)

        # Dictionary of Activation variables
        self.is_activated = dict()

        self.is_activated[self.path_grid_cells_topic] = False
        self.is_activated[self.a_star_open_heap_topic] = False
        self.is_activated[self.a_star_close_set_topic] = False

        self.rate = rospy.Rate(20000000)

        self.frame_id = "/map"

        self.robot_color = ColorRGBA(*hex_to_rgba("#ff6d9eeb"))
        self.movable_obstacle_color = ColorRGBA(*hex_to_rgba("#fff1c232"))
        self.unmovable_obstacle_color = ColorRGBA(*hex_to_rgba("#ff000000"))
        self.unknown_obstacle_color = ColorRGBA(*hex_to_rgba("#ff8e7cc3"))
        self.taboo_color = ColorRGBA(*hex_to_rgba("#ffea9999"))

        self.robot_border_color = ColorRGBA(*hex_to_rgba("#ff1155cc"))
        self.movable_obstacle_border_color = ColorRGBA(*hex_to_rgba("#ff7f6000"))
        self.unmovable_obstacle_border_color = ColorRGBA(*hex_to_rgba("#ff4d2802"))
        self.unknown_obstacle_border_color = ColorRGBA(*hex_to_rgba("#ff351c75"))
        self.taboo_border_color = ColorRGBA(*hex_to_rgba("#ffcc0000"))
        self.g_fov_border_color = ColorRGBA(*hex_to_rgba("#ff6d9eeb"))
        self.s_fov_border_color = ColorRGBA(*hex_to_rgba("#ff6aa84f"))

        self.min_inflated_polygon_border_color = ColorRGBA(*hex_to_rgba("#ff666666"))
        self.max_inflated_polygon_border_color = ColorRGBA(*hex_to_rgba("#ff666666"))

        self.text_color_on_filling = ColorRGBA(*hex_to_rgba("#ffffffff"))
        self.text_color_on_empty = ColorRGBA(*hex_to_rgba("#ff000000"))

        self.init_blocking_areas_color = ColorRGBA(*hex_to_rgba("#aafd5454"))
        self.target_blocking_areas_color = ColorRGBA(*hex_to_rgba("#aac85ab7"))
        self.init_diameter_inflated_polygon_color = ColorRGBA(*hex_to_rgba("#aa88dc7a"))
        self.target_diameter_inflated_polygon_color = ColorRGBA(*hex_to_rgba("#aa24641a"))

        self.fov_z_index = -0.04
        self.entities_z_index = -0.05
        self.taboos_z_index = -0.06

        self.fov_line_width = 0.05
        self.border_width = 0.075
        self.text_height = 0.2

        # Setup Static Transform for grid map (Hack so that it is properly placed in view)
        # TODO Parameterize this with the rest so that it can be changed from GUI
        broadcaster = StaticTransformBroadcaster()

        world_to_gridmap_transform = TransformStamped()
        world_to_gridmap_transform.header.stamp = rospy.Time.now()
        world_to_gridmap_transform.header.frame_id = "map"
        world_to_gridmap_transform.child_frame_id = "gridmap"
        world_to_gridmap_transform.transform.translation.z = -1.5
        world_to_gridmap_transform.transform.rotation.x = 0.0
        world_to_gridmap_transform.transform.rotation.y = 0.0
        world_to_gridmap_transform.transform.rotation.z = 1.0
        world_to_gridmap_transform.transform.rotation.w = 0.0

        broadcaster.sendTransform(world_to_gridmap_transform)

    def publish(self, topic, msg):
        self.rate.sleep()
        publisher = self.publishers[topic]
        connections = publisher.get_num_connections()
        if connections > 0:
            publisher.publish(msg)

    def world_to_marker_array(self, world, robot_uid):
        marker_array = MarkerArray()
        markers = []
        robot = world.entities[robot_uid]
        for entity in world.entities.values():
            if isinstance(entity, Robot):
                markers = markers + entity_to_markers(
                    entity, "/robot", entity.uid, self.frame_id, self.robot_color, self.robot_border_color,
                    self.text_color_on_filling, self.text_color_on_empty, self.entities_z_index,
                    self.border_width, self.text_height, add_border=False, add_text=False)

                markers.append(polygon_to_line_strip(entity.s_fov_sensor.fov_polygon, "/robot/s_fov", 0,
                                                     self.frame_id, self.s_fov_border_color, self.fov_z_index,
                                                     self.fov_line_width))
                markers.append(polygon_to_line_strip(entity.g_fov_sensor.fov_polygon, "/robot/g_fov", 0,
                                                     self.frame_id, self.g_fov_border_color, self.fov_z_index,
                                                     self.fov_line_width))
            if isinstance(entity, Obstacle):
                entity_movability = robot.deduce_movability(entity.type)
                if entity_movability == "movable":
                    markers = markers + entity_to_markers(
                        entity, "/obstacles", entity.uid, self.frame_id, self.movable_obstacle_color,
                        self.movable_obstacle_border_color, self.text_color_on_filling, self.text_color_on_empty,
                        self.entities_z_index, self.border_width, self.text_height, add_border=False, add_text=False)
                if entity_movability == "unmovable":
                    markers = markers + entity_to_markers(
                        entity, "/obstacles", entity.uid, self.frame_id, self.unmovable_obstacle_color,
                        self.unmovable_obstacle_border_color, self.text_color_on_filling, self.text_color_on_empty,
                        self.entities_z_index, self.border_width, self.text_height, add_border=False, add_text=False)
                if entity_movability == "unknown":
                    markers = markers + entity_to_markers(
                        entity, "/obstacles", entity.uid, self.frame_id, self.unknown_obstacle_color,
                        self.unknown_obstacle_border_color, self.text_color_on_filling, self.text_color_on_empty,
                        self.entities_z_index, self.border_width, self.text_height, add_border=False, add_text=False)
        for taboo in world.taboo_zones.values():
            markers = markers + entity_to_markers(
                taboo, "/taboos", taboo.uid, self.frame_id, self.taboo_color, self.taboo_border_color,
                self.text_color_on_filling, self.text_color_on_empty, self.taboos_z_index,
                self.border_width, self.text_height, add_border=False, add_text=False)
        marker_array.markers = markers
        return marker_array

    def publish_sim_world(self, world, robot_uid):
        self.publish(self.sim_knowledge_topic, self.world_to_marker_array(world, robot_uid))
        self.publish(self.sim_costmap_topic,  world_to_costmap(world, robot_uid))

    def publish_robot_world(self, world, robot_uid):
        self.publish(self.robot_knowledge_topic, self.world_to_marker_array(world, robot_uid))
        self.publish(self.robot_costmap_topic,  world_to_costmap(world, robot_uid))

    def publish_robot_sim_costmap(self, world, robot_uid):
        self.publish(self.robot_sim_costmap_topic, world_to_costmap(world, robot_uid))

    def publish_grid_map(self, costmap, dd):
        grid_map = costmap_to_grid_map(costmap, dd)
        self.publish(self.test_gridmap_topic, grid_map)

    def publish_a_star_open_heap(self, open_heap, res, grid_pose):
        if self.is_activated[self.a_star_open_heap_topic]:
            open_heap_data = []
            for element in open_heap:
                open_heap_data.append(element.cell)
            open_heap_cells = grid_cells_to_ros_cells(open_heap_data, res, grid_pose)
            self.publish(self.a_star_open_heap_topic, open_heap_cells)

    def publish_a_star_close_set(self, close_set, res, grid_pose):
        if self.is_activated[self.a_star_close_set_topic]:
            close_set_cells = grid_cells_to_ros_cells(list(close_set), res, grid_pose)
            self.publish(self.a_star_close_set_topic, close_set_cells)

    def publish_social_cells(self, social_cells_set, res, grid_pose):
        ros_cells = grid_cells_to_ros_cells(list(social_cells_set), res, grid_pose)
        self.publish(self.social_cells_topic, ros_cells)

    def publish_multigoal_a_star_open_heap(self, open_heap, res, grid_pose):
        open_heap_data = []
        for element in open_heap:
            open_heap_data.append(element.cell)
        open_heap_cells = grid_cells_to_ros_cells(open_heap_data, res, grid_pose)
        self.publish(self.a_star_open_heap_topic, open_heap_cells)

    def publish_multigoal_a_star_close_set(self, close_set, res, grid_pose):
        close_set_cells = grid_cells_to_ros_cells(list(close_set), res, grid_pose)
        self.publish(self.a_star_close_set_topic, close_set_cells)

    def publish_grid_path(self, grid_path, res, grid_pose):
        if self.is_activated[self.path_grid_cells_topic]:
            path_grid_cells = grid_cells_to_ros_cells(grid_path, res, grid_pose)
            self.publish(self.path_grid_cells_topic, path_grid_cells)

    def publish_q_manips_for_obs(self, poses):
        pose_array = poses_to_poses_array(poses)
        self.publish(self.obs_manip_poses_topic, pose_array)

    def publish_c_1(self, c1):
        self.publish(self.eval_c_1_topic, real_path_to_ros_path(c1.path))

    def publish_c_2(self, c2):
        self.publish(self.eval_c_2_topic, real_path_to_ros_path(c2.path))

    def publish_c_3(self, c3):
        self.publish(self.eval_c_3_topic, real_path_to_ros_path(c3.path))

    def publish_p_opt(self, plan):
        if plan.path_components:
            self.publish(self.c_1_topic, real_path_to_ros_path(plan.path_components[0].path))
        if len(plan.path_components) == 3:
            self.publish(self.c_2_topic, real_path_to_ros_path(plan.path_components[1].path))
            self.publish(self.c_3_topic, real_path_to_ros_path(plan.path_components[2].path))

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
        self.publish(self.min_max_inflated_polygons_topic, marker_array)

    def publish_q_l_cells(self, cells, res, grid_pose):
        close_set_cells = grid_cells_to_ros_cells(list(cells), res, grid_pose)
        self.publish(self.q_l_cells_topic, close_set_cells)

    def publish_q_l_poses(self, poses):
        pose_array = poses_to_poses_array(poses)
        self.publish(self.q_l_poses_topic, pose_array)

    def publish_goal(self, q_init, q_goal, polygon):
        if q_goal is not None:
            polygon_at_goal_pose = affinity.translate(polygon, q_goal[0] - q_init[0], q_goal[1] - q_init[1])
            # ros_pose = pose_to_ros_pose_stamped(q_goal)
            marker_array = MarkerArray(markers=[
                polygon_to_line_strip(polygon_at_goal_pose, "/polygon", 0, self.frame_id, self.robot_border_color,
                                      self.fov_z_index, self.border_width)])
                # pose_to_arrow(q_goal, "/pose", 0, self.frame_id, self.robot_border_color,
                #               self.entities_z_index, 0.5, 0.2, 0.0)])
            self.publish(self.robot_goal_topic, marker_array)

    def cleanup_sim_world(self):
        self.publish(self.sim_knowledge_topic, make_delete_all_marker(self.frame_id))
        self.publish(self.sim_costmap_topic, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    def cleanup_robot_world(self):
        self.publish(self.robot_knowledge_topic, make_delete_all_marker(self.frame_id))
        self.publish(self.robot_costmap_topic, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    def cleanup_eval_c1_c2_c3_sim_init_target(self):
        self.publish(self.eval_c_1_topic, init_ros_path())
        self.publish(self.eval_c_2_topic, init_ros_path())
        self.publish(self.eval_c_3_topic, init_ros_path())
        self.publish(self.robot_sim_topic, make_delete_all_marker(self.frame_id))

    def cleanup_p_opt(self):
        self.publish(self.c_1_topic, init_ros_path())
        self.publish(self.c_2_topic, init_ros_path())
        self.publish(self.c_3_topic, init_ros_path())
        self.publish(self.robot_costmap, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    def cleanup_q_manips_for_obs(self):
        pose_array = PoseArray(header=Header(frame_id=self.frame_id, stamp=rospy.Time.now()), poses=[])
        self.publish(self.obs_manip_poses_topic, pose_array)

    def cleanup_goal(self):
        self.publish(self.robot_goal_topic, make_delete_all_marker(self.frame_id))

    def cleanup_q_l_cells_poses(self):
        self.publish(self.q_l_cells_topic, init_grid_cells(0.1))
        self.publish(self.q_l_poses_topic, PoseArray(header=init_header()))

    def cleanup_min_max_inflated(self):
        self.publish(self.min_max_inflated_polygons_topic, make_delete_all_marker(self.frame_id))

    def cleanup_a_star_open_heap(self):
        self.publish(self.a_star_open_heap_topic, init_grid_cells(0.1))

    def cleanup_a_star_close_set(self):
        self.publish(self.a_star_close_set_topic, init_grid_cells(0.1))

    def cleanup_multigoal_a_star_open_heap(self):
        self.publish(self.a_star_open_heap_topic, init_grid_cells(0.1))

    def cleanup_multigoal_a_star_close_set(self):
        self.publish(self.a_star_close_set_topic, init_grid_cells(0.1))

    def cleanup_grid_path(self):
        self.publish(self.path_grid_cells_topic, init_grid_cells(0.1))

    def cleanup_all(self):
        self.cleanup_sim_world()
        self.cleanup_robot_world()
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
