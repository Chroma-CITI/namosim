from future.utils import with_metaclass
import time
import math
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
import mapbox_earcut as earcut

import copy
import subprocess

import snamosim.display.ros_publisher_config as cfg

if not cfg.deactivate_gui:
    import snamosim.display.colors as colors

    try:
        # Try to import rospy for ROS1
        import rospy
        ROS2 = False
        from std_msgs.msg import Time

    except ImportError:
        # Else try to import rclpy for ROS2
        import rclpy
        from rclpy.node import Node
        import threading
        ROS2 = True

        from builtin_interfaces.msg import Time

        from typing import Union, Optional, Type
        from rclpy.qos import QoSProfile
        from rclpy.callback_groups import CallbackGroup
        from rclpy.qos_event import PublisherEventCallbacks
        from rclpy.qos_overriding_options import QoSOverridingOptions
        from rclpy.publisher import Publisher

    from tf2_ros import StaticTransformBroadcaster
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import PoseArray, TransformStamped, Transform, Vector3, Quaternion, Pose, Point, PoseStamped
    from std_msgs.msg import Header, Float32MultiArray, MultiArrayLayout, MultiArrayDimension, ColorRGBA
    from nav_msgs.msg import Path, OccupancyGrid, MapMetaData, GridCells
    from grid_map_msgs.msg import GridMap

    USE_ROS = True
else:
    USE_ROS = False

from snamosim.utils.singleton import Singleton
from snamosim.utils import utils
from snamosim.display import tf_replacement
from snamosim.worldreps.entity_based.robot import Robot
from snamosim.worldreps.entity_based.obstacle import Obstacle
from snamosim.worldreps.occupation_based.binary_occupancy_grid import BinaryOccupancyGrid, BinaryInflatedOccupancyGrid


class NamespaceCache:
    def __init__(self):
        self.current_cell_to_marker = dict()
        self.current_cell_marker_current_id = 1
        self.cells_to_path_marker = dict()
        self.cells_path_marker_current_id = 1
        self.manip_search_neighbors_markers_p_ids = []
        self.current_fixed_robot_pose_to_marker = dict()
        self.current_fixed_robot_pose_marker_current_id = 1


try:
    # Try to use ROS2 compatibility class
    class MyNode(Node):
        def __init__(self, node_name):
            rclpy.init(args=None)
            super().__init__(node_name=node_name)

        def get_timestamp(self):
            return self.get_clock().now().to_msg()

        def log_warn(self, text):
            self.get_logger().warn(text)

        def get_transform_broadcaster(self):
            return StaticTransformBroadcaster(self)

        @staticmethod
        def get_publisher_subscription_count(publisher):
            return publisher.get_subscription_count()

        def check_duration(self, last_time, duration_in_secs=1.):
            return self.get_timestamp() - last_time > Duration.from_sec(duration_in_secs)

        @staticmethod
        def get_nodes_names():
            cmd_str = 'ros2 node list'
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
            nodes_names = [name for name in result.stdout.split('\n') if name]
            return nodes_names

        def create_publisher(
            self,
            msg_type,
            topic: str,
            qos_profile: Union[QoSProfile, int]=cfg.default_queue_size,
            *,
            callback_group: Optional[CallbackGroup] = None,
            event_callbacks: Optional[PublisherEventCallbacks] = None,
            qos_overriding_options: Optional[QoSOverridingOptions] = None,
            publisher_class: Type[Publisher] = Publisher,
        ) -> Publisher:
            return super(MyNode, self).create_publisher(
                msg_type=msg_type, topic=topic, qos_profile=qos_profile, callback_group=callback_group,
                event_callbacks=event_callbacks, qos_overriding_options=qos_overriding_options,
                publisher_class=publisher_class
            )

except NameError:
    # Else use ROS1 compatibility class
    class MyNode:
        def __init__(self, node_name):
            self.name = node_name
            rospy.init_node(node_name)

        @staticmethod
        def create_publisher(topic_type, topic_name, queue_size=cfg.default_queue_size):
            return rospy.Publisher(topic_name, topic_type, queue_size)

        @staticmethod
        def get_timestamp():
            return rospy.Time.now()

        @staticmethod
        def log_warn(text):
            rospy.logwarn(text)

        @staticmethod
        def get_transform_broadcaster():
            return StaticTransformBroadcaster()

        @staticmethod
        def get_publisher_subscription_count(publisher):
            return publisher.get_num_connections()

        @staticmethod
        def check_duration(last_time, duration_in_secs=1.):
            return rospy.Time.now() - last_time > rospy.Duration.from_sec(duration_in_secs)

        @staticmethod
        def shutdown():
            rospy.shutdown()

        @staticmethod
        def get_nodes_names():
            cmd_str = 'rosnode list'
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
            nodes_names = [name for name in result.stdout.split('\n') if name]
            return nodes_names

        def get_name(self):
            return self.name


# Basic conversion functions
def polygon_to_triangle_list(polygon, namespace, p_id, frame_id, color, z_index, stamp=Time()):
    marker = Marker(type=Marker.TRIANGLE_LIST,
                    ns=namespace,
                    id=p_id,
                    header=Header(frame_id=frame_id, stamp=stamp),
                    color=color,
                    scale=Vector3(x=1.0, y=1.0, z=1.0),
                    points=[])
    if isinstance(polygon, Polygon):
        verts = np.array(list(polygon.exterior.coords)).reshape(-1, 2)
        rings = np.array([verts.shape[0]])
        triangles_vertices = verts[earcut.triangulate_float64(verts, rings)]
        triangles = [triangles_vertices[n:n + 3] for n in range(0, len(triangles_vertices), 3)]
        marker.points = [Point(x=point[0], y=point[1], z=z_index) for triangle in triangles for point in triangle]
    return marker


def polygon_to_line_strip(polygon, namespace, p_id, frame_id, color, z_index, line_width, stamp=Time()):
    marker = Marker(type=Marker.LINE_STRIP,
                    ns=namespace,
                    id=p_id,
                    header=Header(frame_id=frame_id, stamp=stamp),
                    color=color,
                    scale=Vector3(x=line_width, y=0.0, z=0.0),
                    points=[])
    if polygon is not None:
        for i in range(len(polygon.exterior.coords) - 1):
            point = polygon.exterior.coords[i]
            next_point = polygon.exterior.coords[i + 1]
            marker.points.append(Point(x=point[0], y=point[1], z=z_index))
            marker.points.append(Point(x=next_point[0], y=next_point[1], z=z_index))
        marker.points.append(Point(x=polygon.exterior.coords[0][0], y=polygon.exterior.coords[0][1], z=z_index))
        marker.points.append(Point(x=polygon.exterior.coords[1][0], y=polygon.exterior.coords[1][1], z=z_index))
    return marker


def string_to_text(string, coordinates, namespace, p_id, frame_id, color, z_index, text_height, stamp=Time()):
    x, y, z = coordinates[0], coordinates[1], z_index
    marker = Marker(type=Marker.TEXT_VIEW_FACING,
                    ns=namespace,
                    id=p_id,
                    pose=Pose(
                        position=(Point(x=x, y=y, z=z) if ROS2 else Vector3(x=x, y=y, z=z)),
                        orientation=Quaternion()),
                    scale=Vector3(x=0.0, y=0.0, z=text_height),
                    header=Header(frame_id=frame_id, stamp=stamp),
                    color=color,
                    text=string)
    return marker


def costmap_to_grid_map(costmap, res, frame_id=cfg.social_gridmap_frame_id, stamp=Time()):
    grid_map = GridMap()
    if hasattr(grid_map.info, 'header'):
        grid_map.info.header = Header(stamp=stamp, frame_id=frame_id)
    elif hasattr(grid_map, 'header'):
        grid_map.header = Header(stamp=stamp, frame_id=frame_id)
    grid_map.info.resolution = res
    grid_map.info.length_x = costmap.shape[0] * res
    grid_map.info.length_y = costmap.shape[1] * res
    # grid_map.info.pose.position.z = 0. # The lib does not take this parameter into account...
    grid_map.layers = ["elevation"]
    inflated_costmap_data = Float32MultiArray(
        layout=MultiArrayLayout(
            dim=[MultiArrayDimension(label="column_index",
                                     size=costmap.shape[1],
                                     stride=costmap.shape[1] * costmap.shape[0]),
                 MultiArrayDimension(label="row_index",
                                     size=costmap.shape[0],
                                     stride=costmap.shape[0])],
            data_offset=0),
        data=(costmap.flatten('F')).astype(np.float32).tolist()
    )
    grid_map.data = [inflated_costmap_data]

    return grid_map


def make_delete_marker(namespace, p_id, frame_id, stamp=Time()):
    return Marker(
        ns=namespace, id=p_id, header=Header(frame_id=frame_id, stamp=stamp),
        action=Marker.DELETE
    )


def make_delete_all_marker(frame_id, ns='', stamp=Time()):
    return MarkerArray(markers=[Marker(
        ns=ns, header=Header(frame_id=frame_id, stamp=stamp), action=Marker.DELETEALL
    )])


def init_header(stamp=Time()):
    return Header(stamp=stamp, frame_id="map")


class RosObserver:
    def __init__(self, node, topic, is_active=True, rate=cfg.rate, msg_type=None):
        self.node = node
        self.topic = topic
        self._publisher = node.create_publisher(msg_type, topic)
        self.is_active = is_active
        self._rate = rate

        self._duration = 1. / self.rate
        self._last_time = time.time()
        # For backward ROS1 compatibility
        if ROS2:
            self.get_subscription_count = self._publisher.get_subscription_count
        else:
            self.get_subscription_count = self._publisher.get_num_connections

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, r):
        self._rate = r
        self._duration = 1. / self.rate

    def update(self, **kwargs):
        if USE_ROS and self.is_active:
            connections = self.get_subscription_count()
            if connections > 0:
                elapsed_time = time.time() - self._last_time
                time_to_wait = self._duration - elapsed_time
                if time_to_wait > 0.:
                    time.sleep(time_to_wait)
                self._publisher.publish(self.convert(**kwargs))
                self._last_time = time.time()

    def convert(self, **kwargs):
        raise NotImplementedError

    def reset(self, reset_msg=None):
        if USE_ROS and reset_msg is not None:
            self._publisher.publish(reset_msg)


class WorldObserver(RosObserver):
    def __init__(self, node, topic, is_active=True, rate=cfg.rate):
        RosObserver.__init__(self, node, topic, is_active, rate, msg_type=MarkerArray)
        self.prev_sim_world_draw_data = None

    def convert(self, **kwargs):
        world, robot_uid = kwargs['world'], kwargs['robot_uid']

        current_world_draw_data = {
            entity.uid: {
                "polygon": entity.polygon,
                "type": "robot" if isinstance(entity, Robot) else entity.type,
                "pose": entity.pose
            } for entity in world.entities.values()}
        entities_to_ignore = {
            entity_uid for entity_uid, drawable_data in current_world_draw_data.items()
            if (self.prev_sim_world_draw_data is not None
                and entity_uid in self.prev_sim_world_draw_data
                and drawable_data["polygon"] == self.prev_sim_world_draw_data[entity_uid]["polygon"]
                and drawable_data["type"] == self.prev_sim_world_draw_data[entity_uid]["type"]
                and drawable_data["pose"] == self.prev_sim_world_draw_data[entity_uid]["pose"])}
        self.prev_sim_world_draw_data = current_world_draw_data
        return self.world_to_marker_array(world, robot_uid, entities_to_ignore)

    def world_to_marker_array(self, world, robot_uid=None, entities_to_ignore=None):
        if entities_to_ignore is None:
            entities_to_ignore = set()
        marker_array = MarkerArray()
        markers = []
        for entity in world.entities.values():
            if entity.uid not in entities_to_ignore:
                entity_color = ColorRGBA(**colors.hex_to_rgba(entity.style.fill))
                if isinstance(entity, Robot):
                    namespace = '/robot'
                elif isinstance(entity, Obstacle):
                    namespace = 'obstacles'
                else:
                    raise ValueError(
                        'Only Robot and Obstacle can be displayed in Rviz, current entity is: {}'.format(entity)
                    )

                markers = markers + self.entity_to_markers(
                    entity, namespace, entity.uid, cfg.main_frame_id, entity_color, entity_color,
                    colors.text_color_on_filling, colors.text_color_on_empty, cfg.entities_z_index,
                    cfg.border_width, cfg.text_height, add_border=False, add_text=False)
        marker_array.markers = markers
        return marker_array

    def entity_to_markers(self, entity, namespace, p_id, frame_id, color, border_color, text_color_filling,
                          text_color_empty,
                          z_index, line_width, text_height,
                          add_filling=True, add_border=True, add_text=True,
                          add_uid=True, add_name=True):
        markers = []
        if add_filling:
            markers.append(polygon_to_triangle_list(
                    entity.polygon, namespace + "/polygon", p_id, frame_id, color, z_index, self.node.get_timestamp()
            ))
        if add_border:
            markers.append(polygon_to_line_strip(
                entity.polygon, namespace + "/border", p_id, frame_id, border_color, z_index,
                line_width, self.node.get_timestamp()
            ))
        if add_text:
            string = ((("UID: " + str(entity.uid) + "\n") if add_uid else "") +
                      (("Name: " + entity.name + "\n") if add_name else ""))
            text_coordinates = entity.polygon.centroid.coords[0]
            markers.append(string_to_text(
                string, text_coordinates, namespace + "/text", p_id, frame_id,
                text_color_filling if add_filling else text_color_empty, z_index, text_height, self.node.get_timestamp()
            ))
        return markers

    def reset(self, reset_msg=None):
        RosObserver.reset(self, make_delete_all_marker(cfg.main_frame_id))


class CostmapObserver(RosObserver):
    def __init__(self, node, topic, is_active=True, rate=cfg.rate):
        RosObserver.__init__(self, node, topic, is_active, rate, msg_type=OccupancyGrid)

    def convert(self, **kwargs):
        world, robot_uid = kwargs['world'], kwargs['robot_uid']
        return self.world_to_costmap(world, robot_uid)

    def world_to_costmap(self, world, robot_uid=None):
        polygons = {
            uid: entity.polygon for uid, entity in world.entities.items()
            if not isinstance(entity, Robot)
        }
        if robot_uid:
            robot_max_inflation_radius = utils.get_circumscribed_radius(world.entities[robot_uid].polygon)
            grid = BinaryInflatedOccupancyGrid(
                polygons, world.dd.res, robot_max_inflation_radius, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD
            )
        else:
            grid = BinaryOccupancyGrid(
                polygons, world.dd.res, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD
            )

        costmap = OccupancyGrid(header=init_header(self.node.get_timestamp()))
        costmap.info.map_load_time = costmap.header.stamp
        costmap.info.resolution = grid.res
        costmap.info.width = grid.d_width
        costmap.info.height = grid.d_height
        costmap.info.origin.position.x = grid.grid_pose[0]
        costmap.info.origin.position.y = grid.grid_pose[1]
        costmap.info.origin.position.z = -0.1
        costmap.data = np.fliplr(np.rot90(grid.grid, 3)).flatten().astype(np.int8).tolist()

        return costmap

    def reset(self, reset_msg=None):
        RosObserver.reset(self, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))


class GridMapObserver(RosObserver):
    def __init__(self, node, topic, is_active=True, rate=cfg.rate, msg_type=GridMap):
        RosObserver.__init__(self, node, topic, is_active, rate, msg_type=msg_type)

    def convert(self, **kwargs):
        costmap, res = kwargs['costmap'], kwargs['res']
        fixed_costmap = np.copy(costmap)
        fixed_costmap[fixed_costmap == -1.] = 0.
        grid_map = costmap_to_grid_map(fixed_costmap, res, stamp=self.node.get_timestamp())
        return grid_map

    def reset(self, reset_msg=None):
        RosObserver.reset(self, costmap_to_grid_map(np.full((1000, 1000), np.nan), 1.))


class CombinedCostGridMapObserver(GridMapObserver):
    def __init__(self, node, topic, is_active=True, rate=10, msg_type=GridMap):
        GridMapObserver.__init__(self, node, topic, is_active, rate, msg_type=msg_type)

    def convert(self, **kwargs):
        sorted_cell_to_combined_cost, inflated_grid_by_obstacle = kwargs['sorted_cell_to_combined_cost'], kwargs['inflated_grid_by_obstacle']
        combined_costmap = np.zeros((inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height))
        for cell, combined_cost in sorted_cell_to_combined_cost.items():
            combined_costmap[cell[0]][cell[1]] = combined_cost
        grid_map = costmap_to_grid_map(
            combined_costmap, inflated_grid_by_obstacle.res,
            frame_id=cfg.combined_gridmap_frame_id, stamp=self.node.get_timestamp()
        )
        return grid_map


class RosPublisher(with_metaclass(Singleton)):
    def __init__(self, simulator=None, prefix_topics_with_node_name=False):
        if simulator is None:
            raise ValueError("Cannot create RosPublisher instance with None 'simulator' parameter.")

        if not USE_ROS or cfg.deactivate_gui:
            return

        # HACK: Must necessarily be invoked in the init method of this singleton and not at module-level (rospy bug...)
        self.ros_node = MyNode(node_name=self.create_valid_node_name(simulator.simulation_filename))
        self.prefix = '' if not prefix_topics_with_node_name else self.ros_node.get_name()

        self.my_publishers = {}  # DEPRECATED
        self.observers = {}

        # Add simulation-specific publishers
        self.sim_knowledge_topic = self.prefix + '/simulation' + cfg.sim_knowledge_topic
        self.observers[self.sim_knowledge_topic] = WorldObserver(self.ros_node, self.sim_knowledge_topic)
        self.sim_costmap_topic = self.prefix + '/simulation' + cfg.sim_costmap_topic
        self.observers[self.sim_costmap_topic] = CostmapObserver(self.ros_node, self.sim_costmap_topic)
        self.sim_gridmap_topic = self.prefix + '/simulation' + cfg.test_social_gridmap_topic
        self.observers[self.sim_gridmap_topic] = GridMapObserver(self.ros_node, self.sim_gridmap_topic)
        self.sim_cc_topic = self.prefix + '/simulation' + cfg.test_connected_components_topic
        self.observers[self.sim_cc_topic] = GridMapObserver(self.ros_node, self.sim_cc_topic)

        self.my_publishers['/simulation' + cfg.sim_latest_message_topic] = self.ros_node.create_publisher(
            MarkerArray, '/simulation' + cfg.sim_latest_message_topic)

        self.agents_names = [a_to_b_config["agent_name"] for a_to_b_config in simulator.config["agents_behaviors"]]
        # Add robot-specific publishers for each robot namespace
        for agent_name in self.agents_names:
            ns = self.prefix + '/' + agent_name
            self.observers[ns + cfg.robot_knowledge_topic] = WorldObserver(self.ros_node, ns + cfg.robot_knowledge_topic)
            self.observers[ns + cfg.robot_costmap_topic] = CostmapObserver(self.ros_node, ns + cfg.robot_costmap_topic)
            self.observers[ns + cfg.robot_sim_world_topic] = WorldObserver(self.ros_node, ns + cfg.robot_sim_world_topic)
            self.observers[ns + cfg.robot_sim_costmap_topic] = CostmapObserver(self.ros_node, ns + cfg.robot_sim_costmap_topic)

            self.observers[ns + cfg.test_connected_components_topic] = GridMapObserver(self.ros_node, ns + cfg.test_connected_components_topic)
            self.observers[ns + cfg.test_combined_gridmap_topic] = CombinedCostGridMapObserver(self.ros_node, ns + cfg.test_combined_gridmap_topic)
            self.observers[ns + cfg.test_social_gridmap_topic] = GridMapObserver(self.ros_node, ns + cfg.test_social_gridmap_topic)

            # TODO: Refactor the following publishers with the Observer pattern
            self.my_publishers[ns + cfg.robot_goal_topic] = self.ros_node.create_publisher(
                MarkerArray, ns + cfg.robot_goal_topic)
            self.my_publishers[ns + cfg.obs_manip_poses_topic] = self.ros_node.create_publisher(
                PoseArray, ns + cfg.obs_manip_poses_topic)
            self.my_publishers[ns + cfg.plan_topic] = self.ros_node.create_publisher(
                MarkerArray, ns + cfg.plan_topic)
            self.my_publishers[ns + cfg.conflicts_check_topic] = self.ros_node.create_publisher(
                MarkerArray, ns + cfg.conflicts_check_topic)
            # TODO: Last publisher to refactor, as it requires separating it into smaller meaningful units
            self.my_publishers[ns + cfg.robot_sim_topic] = self.ros_node.create_publisher(
                MarkerArray, ns + cfg.robot_sim_topic, cfg.default_queue_size)

        # HACK: Necessary because ROS1 pub/sub system is not really reliable : wait a second for subscribers to listen
        time.sleep(cfg.hack_duration_wait)

        # Setup Static Transform for grid map (Hack so that it is properly placed in view)
        broadcaster = self.ros_node.get_transform_broadcaster()

        for frame_id, z_index in cfg.gridmap_frame_ids_to_z_indexes.items():
            transform = TransformStamped(
                header=Header(stamp=self.ros_node.get_timestamp(), frame_id=cfg.main_frame_id), child_frame_id=frame_id,
                transform=Transform(translation=Vector3(z=z_index), rotation=Quaternion(x=0., y=0., z=1., w=0.))
            )
            broadcaster.sendTransform(transform)
            time.sleep(0.5)  # Hack so that transform is properly sent...

        # Initialize caches for each top level namespace
        self.prev_sim_world_draw_data = {}

        self.namespaces_caches = {}
        for ns in self.agents_names:
            self.namespaces_caches[ns] = NamespaceCache()

    @staticmethod
    def create_valid_node_name(root_name):
        nodes_names = MyNode.get_nodes_names()
        node_name = root_name if (root_name and not root_name[0].isdigit()) else ('node_' + root_name)
        i = 0
        while node_name in nodes_names:
            node_name = root_name + '_' + str(i)
            i += 1
        return node_name

    def publish(self, topic, msg):
        publisher = self.my_publishers[topic]
        connections = self.ros_node.get_publisher_subscription_count(publisher)
        if connections > 0:
            time.sleep(1./cfg.rate)
            publisher.publish(msg)

    def is_activated(self, topic=''):
        if cfg.deactivate_gui or (topic and topic not in self.my_publishers):
            return False
        elif not cfg.deactivate_gui and not topic:
            return True
        return self.ros_node.get_publisher_subscription_count(self.my_publishers[topic]) > 0

    # region SIM WORLD
    def publish_sim_world(self, world, robot_uid=None):
        self.observers[self.sim_knowledge_topic].update(world=world, robot_uid=robot_uid)
        self.observers[self.sim_costmap_topic].update(world=world, robot_uid=robot_uid)

    def cleanup_sim_world(self):
        self.observers[self.sim_knowledge_topic].reset()
        self.observers[self.sim_costmap_topic].reset()
    # endregion

    # region ROBOT WORLD
    def publish_robot_world(self, world, robot_uid):
        world_topic = self.prefix + '/' + world.entities[robot_uid].name + cfg.robot_knowledge_topic
        self.observers[world_topic].update(world=world, robot_uid=robot_uid)
        costmap_topic = self.prefix + '/' + world.entities[robot_uid].name + cfg.robot_costmap_topic
        self.observers[costmap_topic].update(world=world, robot_uid=robot_uid)

    def cleanup_robot_world(self, ns=''):
        world_topic = self.prefix + '/' + ns + cfg.robot_knowledge_topic
        self.observers[world_topic].reset()
        costmap_topic = self.prefix + '/' + ns + cfg.robot_costmap_topic
        self.observers[costmap_topic].reset()
    # endregion

    # region ROBOT SIM
    def publish_robot_sim_world(self, world, robot_uid, ns=''):
        topic = self.prefix + '/' + world.entities[robot_uid].name + cfg.robot_sim_world_topic
        self.observers[topic].update(world=world, robot_uid=robot_uid)

    def cleanup_robot_sim_world(self, ns=''):
        topic = self.prefix + '/' + ns + cfg.robot_sim_world_topic
        self.observers[topic].reset()

    def publish_robot_sim_costmap(self, world, robot_uid):
        topic = self.prefix + '/' + world.entities[robot_uid].name + cfg.robot_sim_costmap_topic
        self.observers[topic].update(world=world, robot_uid=robot_uid)

    def cleanup_robot_sim_costmap(self, world, robot_uid):
        topic = self.prefix + '/' + world.entities[robot_uid].name + cfg.robot_sim_costmap_topic
        self.observers[topic].reset()
    # endregion

    # region GRID MAP
    def publish_social_grid_map(self, costmap, res, ns=''):
        topic = self.prefix + (cfg.test_social_gridmap_topic if not ns else '/' + ns + cfg.test_social_gridmap_topic)
        self.observers[topic].update(costmap=costmap, res=res)

    def cleanup_social_grid_map(self, ns=''):
        topic = self.prefix + (cfg.test_social_gridmap_topic if not ns else '/' + ns + cfg.test_social_gridmap_topic)
        self.observers[topic].reset()

    def publish_combined_costmap(self, sorted_cell_to_combined_cost, inflated_grid_by_obstacle, ns=''):
        topic = self.prefix + (cfg.test_combined_gridmap_topic if not ns else '/' + ns + cfg.test_combined_gridmap_topic)
        self.observers[topic].update(
            sorted_cell_to_combined_cost=sorted_cell_to_combined_cost,
            inflated_grid_by_obstacle=inflated_grid_by_obstacle
        )

    def cleanup_combined_costmap(self, ns=''):
        topic = self.prefix + (cfg.test_combined_gridmap_topic if not ns else '/' + ns + cfg.test_combined_gridmap_topic)
        self.observers[topic].reset()
    # endregion

    # region CONNECTED COMPONENTS GRID
    def publish_connected_components_grid(self, costmap, res, ns=''):
        topic = self.prefix + (cfg.test_connected_components_topic if not ns else '/' + ns + cfg.test_connected_components_topic)
        self.observers[topic].update(costmap=costmap, res=res)

    def cleanup_connected_components_grid(self, ns=''):
        topic = self.prefix + (cfg.test_connected_components_topic if not ns else '/' + ns + cfg.test_connected_components_topic)
        self.observers[topic].reset()
    # endregion

    # region STILMAN 2005 RCH DATA
    def publish_rch_data(self, current, gscore, close_set, open_queue, came_from, neighbors, traversed_obstacles_ids,
                         res, grid_pose, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = MarkerArray(markers=[])

            # Publish current cell
            current_marker = self.grid_cells_to_cube_list_markers(
                [current.cell], res, grid_pose, z_index=0.9, color=colors.flashy_purple, ns="/rch_current_cell"
            )
            marker_array.markers.append(current_marker)

            # Publish neighbors
            neighbors_marker = self.grid_cells_to_cube_list_markers(
                [neighbor.cell for neighbor in neighbors], res, grid_pose, z_index=0.9, color=colors.flashy_red,
                ns="/rch_current_cell_neighbors"
            )
            marker_array.markers.append(neighbors_marker)

            # Publish close_set
            if traversed_obstacles_ids:
                obstacle_id_to_color = dict(zip(
                    traversed_obstacles_ids, colors.generate_equally_spread_ros_colors(len(traversed_obstacles_ids))
                ))
                color = obstacle_id_to_color[current.first_obstacle_uid]
            else:
                color = colors.generate_equally_spread_ros_colors(1)[0]

            if current.cell in self.namespaces_caches[ns].current_cell_to_marker:
                original_marker = self.namespaces_caches[ns].current_cell_to_marker[current.cell]
                blended_color = colors.blend_colors(original_marker.color, color)
                original_marker.color = blended_color
                close_set_marker = original_marker
            else:
                _id = self.namespaces_caches[ns].current_cell_marker_current_id
                close_set_marker = self.grid_cell_to_cube_marker(
                    current.cell, res, grid_pose, color, _id, z_index=0.8, ns="/rch_close_set"
                )
                self.namespaces_caches[ns].current_cell_to_marker[current.cell] = close_set_marker
                self.namespaces_caches[ns].current_cell_marker_current_id += 1

            marker_array.markers.append(close_set_marker)

            # Publish open_heap
            # TODO

            # Publish came_from as paths between cells poses
            if current in came_from:
                path_color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=1.)
                cells = (current.cell, came_from[current].cell)
                if cells in self.namespaces_caches[ns].cells_to_path_marker:
                    original_marker = self.namespaces_caches[ns].cells_to_path_marker[cells]
                    blended_color = colors.blend_colors(original_marker.color, path_color)
                    original_marker.color = blended_color
                    came_from_marker = original_marker
                else:
                    _id = self.namespaces_caches[ns].cells_path_marker_current_id
                    cur_pose = utils.grid_to_real(current.cell[0], current.cell[1], res, grid_pose)
                    from_pose = utils.grid_to_real(came_from[current].cell[0], came_from[current].cell[1], res,
                                                   grid_pose)
                    came_from_marker = self.real_path_to_linestrip(
                        [cur_pose, from_pose],
                        '/rch_came_from', _id, cfg.main_frame_id, ColorRGBA(r=color.r, g=color.g, b=color.b, a=1.),
                        res / 10., cfg.path_line_z_index
                    )
                    self.namespaces_caches[ns].cells_to_path_marker[cells] = came_from_marker
                    self.namespaces_caches[ns].cells_path_marker_current_id += 1

                marker_array.markers.append(came_from_marker)

            self.publish(full_topic, marker_array)

    # endregion

    # region MANIP SEARCH
    def publish_manip_search_data(self, current, gscore, close_set, open_queue, came_from, neighbors, start_confs,
                                  res, grid_pose, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = MarkerArray(markers=[])

            manip_poses_ids = [c.manip_pose_id for c in start_confs.keys()]

            arrow_length, shaft_diameter, head_diameter, head_length = res / 1.5, res / 10., res / 5., res / 5.
            manip_pose_id_to_color = dict(zip(
                manip_poses_ids, colors.generate_equally_spread_ros_colors(len(manip_poses_ids))
            ))

            # Publish current configuration
            current_robot_pose_marker = self.pose_to_arrow(
                pose=current.robot.floating_point_pose, namespace="/manip_search/current/robot/pose",
                p_id=0, frame_id=cfg.main_frame_id, color=colors.flashy_cyan,
                z_index=1.1, arrow_length=arrow_length, shaft_diameter=shaft_diameter,
                head_diameter=head_diameter, head_length=head_length
            )
            current_obstacle_pose_marker = self.pose_to_arrow(
                pose=current.obstacle.floating_point_pose, namespace="/manip_search/current/obstacle/pose",
                p_id=0, frame_id=cfg.main_frame_id, color=colors.flashy_dark_cyan,
                z_index=1.1, arrow_length=arrow_length, shaft_diameter=shaft_diameter,
                head_diameter=head_diameter, head_length=head_length
            )
            marker_array.markers.append(current_robot_pose_marker)
            marker_array.markers.append(current_obstacle_pose_marker)

            current_robot_polygon_marker = self.polygon_to_line_strip(
                current.robot.polygon, "/manip_search/current/robot/polygon", 0, cfg.main_frame_id,
                colors.flashy_cyan, cfg.entities_z_index, cfg.border_width)
            current_obstacle_polygon_marker = self.polygon_to_line_strip(
                current.obstacle.polygon, "/manip_search/current/obstacle/polygon", 0, cfg.main_frame_id,
                colors.flashy_dark_cyan, cfg.entities_z_index, cfg.border_width)
            marker_array.markers.append(current_robot_polygon_marker)
            marker_array.markers.append(current_obstacle_polygon_marker)

            # Publish neighbors
            neighbors_markers = [
                self.pose_to_arrow(
                    pose=neighbor.robot.floating_point_pose, namespace="/manip_search_neighbors",
                    p_id=p_id, frame_id=cfg.main_frame_id, color=colors.flashy_green,
                    z_index=1.1, arrow_length=arrow_length, shaft_diameter=shaft_diameter,
                    head_diameter=head_diameter, head_length=head_length
                )
                for p_id, neighbor in enumerate(neighbors)
            ]
            marker_array.markers += neighbors_markers
            neighbor_markers_ids = {n.id for n in neighbors_markers}
            for p_id in self.namespaces_caches[ns].manip_search_neighbors_markers_p_ids:
                if p_id not in neighbor_markers_ids:
                    marker_array.markers.append(self.make_delete_marker(
                        frame_id=cfg.main_frame_id, namespace="/manip_search_neighbors", p_id=p_id
                    ))
            self.namespaces_caches[ns].manip_search_neighbors_markers_p_ids = neighbor_markers_ids

            # Publish close_set
            color = manip_pose_id_to_color[current.manip_pose_id]
            if current.robot.fixed_precision_pose in self.namespaces_caches[ns].current_fixed_robot_pose_to_marker:
                original_marker = self.namespaces_caches[ns].current_fixed_robot_pose_to_marker[
                    current.robot.fixed_precision_pose
                ]
                blended_color = colors.blend_colors(original_marker.color, color)
                original_marker.color = blended_color
                close_set_marker = original_marker
            else:
                _id = self.namespaces_caches[ns].current_fixed_robot_pose_marker_current_id
                close_set_marker = copy.deepcopy(current_robot_pose_marker)
                close_set_marker.ns = "/manip_search/close_set"
                close_set_marker.id = _id
                close_set_marker.color = color
                self.namespaces_caches[ns].current_fixed_robot_pose_to_marker[
                    current.robot.fixed_precision_pose] = close_set_marker
                self.namespaces_caches[ns].current_fixed_robot_pose_marker_current_id += 1

            marker_array.markers.append(close_set_marker)

            # # Publish open_heap
            # # TODO
            #
            # # Publish came_from as paths between cells poses
            # if current in came_from:
            #     path_color = ColorRGBA(color.r, color.g, color.b, 1.)
            #     cells = (current.cell, came_from[current].cell)
            #     if cells in self.namespaces_caches[ns].cells_to_path_marker:
            #         original_marker = self.namespaces_caches[ns].cells_to_path_marker[cells]
            #         blended_color = colors.blend_colors(original_marker.color, path_color)
            #         original_marker.color = blended_color
            #         came_from_marker = original_marker
            #     else:
            #         _id = self.namespaces_caches[ns].cells_path_marker_current_id
            #         cur_pose = utils.grid_to_real(current.cell[0], current.cell[1], res, grid_pose)
            #         from_pose = utils.grid_to_real(came_from[current].cell[0], came_from[current].cell[1], res, grid_pose)
            #         came_from_marker = self.real_path_to_linestrip(
            #             [cur_pose, from_pose],
            #             '/manip_search_came_from', _id, cfg.main_frame_id, ColorRGBA(color.r, color.g, color.b, 1.),
            #             res / 10., cfg.path_line_z_index
            #         )
            #         self.namespaces_caches[ns].cells_to_path_marker[cells] = came_from_marker
            #         self.namespaces_caches[ns].cells_path_marker_current_id += 1
            #
            #     marker_array.markers.append(came_from_marker)

            self.publish(full_topic, marker_array)

    # endregion

    # region Q MANIPS FOR OBS
    def publish_q_manips_for_obs(self, poses, ns=''):
        full_topic = cfg.obs_manip_poses_topic if not ns else '/' + ns + cfg.obs_manip_poses_topic
        if self.is_activated(full_topic):
            pose_array = self.poses_to_poses_array(poses)
            self.publish(full_topic, pose_array)

    def cleanup_q_manips_for_obs(self, ns=''):
        full_topic = cfg.obs_manip_poses_topic if not ns else '/' + ns + cfg.obs_manip_poses_topic
        if self.is_activated(full_topic):
            pose_array = PoseArray(header=Header(frame_id=cfg.main_frame_id, stamp=self.ros_node.get_timestamp()), poses=[])
            self.publish(full_topic, pose_array)
    # endregion

    # region P_OPT
    def publish_p_opt(self, plan, ns=''):
        full_topic = cfg.plan_topic if not ns else '/' + ns + cfg.plan_topic
        if plan and plan.path_components:
            if self.is_activated(full_topic):
                self.publish(full_topic, self.plan_to_markerarray(plan, cfg.main_frame_id, ns))

    def cleanup_p_opt(self, ns=''):
        full_topic = cfg.plan_topic if not ns else '/' + ns + cfg.plan_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, self.make_delete_all_marker(cfg.main_frame_id))
    # endregion

    # region ROBOT SIM
    def publish_sim(self, robot_polygon, obs_polygon, namespace="/init", ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            robot_color = colors.robot_border_color if namespace == "/target" else colors.robot_color
            obs_color = colors.movable_obstacle_border_color if namespace == "/target" else colors.movable_obstacle_color
            marker_array = MarkerArray(markers=[
                self.polygon_to_line_strip(
                    robot_polygon, namespace + "/robot/polygon", 0, cfg.main_frame_id, robot_color,
                    cfg.entities_z_index, cfg.border_width),
                self.polygon_to_line_strip(
                    obs_polygon, namespace + "/obstacle/polygon", 0, cfg.main_frame_id, obs_color,
                    cfg.entities_z_index, cfg.border_width)])
            self.publish(full_topic, marker_array)

    def publish_blocking_areas(self, init_blocking_areas, target_blocking_areas, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            init_blocking_areas_markers = []
            for i in range(len(init_blocking_areas)):
                init_blocking_areas_markers.append(polygon_to_triangle_list(
                    init_blocking_areas[i], "/blocking_areas/init", i, cfg.main_frame_id,
                    colors.init_blocking_areas_color, cfg.entities_z_index))

            target_blocking_areas_markers = []
            for i in range(len(target_blocking_areas)):
                target_blocking_areas_markers.append(polygon_to_triangle_list(
                    target_blocking_areas[i], "/blocking_areas/target", i, cfg.main_frame_id,
                    colors.target_blocking_areas_color, cfg.entities_z_index))

            marker_array = MarkerArray(markers=init_blocking_areas_markers + target_blocking_areas_markers)
            self.publish(full_topic, marker_array)

    def cleanup_blocking_areas(self, ns=''):
        # FIXME Not implemented correctly in ROS...
        #  https://answers.ros.org/question/263031/delete-all-rviz-markers-in-a-specific-namespace/
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, self.make_delete_all_marker(cfg.main_frame_id, '/blocking_areas'))

    def publish_diameter_inflated_polygons(self, init_entity_inflated_polygon, target_entity_inflated_polygon, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = MarkerArray(markers=[
                self.polygon_to_line_strip(init_entity_inflated_polygon, "/diameter_inflated_polygon/init", 0,
                                           cfg.main_frame_id,
                                           colors.init_diameter_inflated_polygon_color,
                                           cfg.entities_z_index, cfg.border_width / 2.),
                self.polygon_to_line_strip(target_entity_inflated_polygon, "/diameter_inflated_polygon/target", 0,
                                           cfg.main_frame_id,
                                           colors.target_diameter_inflated_polygon_color,
                                           cfg.entities_z_index, cfg.border_width / 2.)])
            self.publish(full_topic, marker_array)

    def cleanup_diameter_inflated_polygons(self, ns=''):
        # FIXME Not implemented correctly in ROS...
        #  https://answers.ros.org/question/263031/delete-all-rviz-markers-in-a-specific-namespace/
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(full_topic,
                         self.make_delete_all_marker(cfg.main_frame_id, '/diameter_inflated_polygon'))

    def publish_debug_polygons(self, polygons, ns=''):
        # FIXME Not implemented correctly in ROS...
        #  https://answers.ros.org/question/263031/delete-all-rviz-markers-in-a-specific-namespace/
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = self.polygons_to_line_strips_marker_array(
                polygons, "/debug/polygons", cfg.main_frame_id, colors.robot_color,
                cfg.entities_z_index, cfg.border_width / 5.)
            self.publish(full_topic, marker_array)

    def cleanup_debug_polygons(self, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(full_topic,
                         self.make_delete_all_marker(cfg.main_frame_id, '/debug/polygons'))

    def cleanup_robot_sim(self, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.namespaces_caches[ns] = NamespaceCache()
            self.publish(full_topic, self.make_delete_all_marker(cfg.main_frame_id))

    # endregion

    # region GOAL
    def publish_goal(self, q_init, q_goal, polygon, ns=''):
        full_topic = cfg.robot_goal_topic if not ns else '/' + ns + cfg.robot_goal_topic
        if self.is_activated(full_topic):
            if q_goal is not None:
                polygon_at_goal_pose = affinity.translate(polygon, q_goal[0] - q_init[0], q_goal[1] - q_init[1])
                color = colors.r0_dark_blue
                if ns == "robot_1":
                    color = colors.r1_dark_green
                elif ns == "robot_2":
                    color = colors.r2_dark_pink
                elif ns == "robot_3":
                    color = colors.r3_dark_red
                marker_array = MarkerArray(markers=[
                    self.polygon_to_line_strip(polygon_at_goal_pose, "/polygon", 0, cfg.main_frame_id,
                                               color, cfg.fov_z_index, cfg.border_width)])
                # pose_to_arrow(q_goal, "/pose", 0, self.frame_id, self.robot_border_color,
                #               self.entities_z_index, 0.5, 0.2, 0.0)])
                self.publish(full_topic, marker_array)

    def cleanup_goal(self, ns=''):
        full_topic = cfg.robot_goal_topic if not ns else '/' + ns + cfg.robot_goal_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, self.make_delete_all_marker(cfg.main_frame_id))
    # endregion

    # region MESSAGE TEXT
    def publish_message(self, message, pose=(0., 0., 0.), font_size=1.):
        if self.is_activated("/simulation" + cfg.sim_latest_message_topic):
            marker_array = MarkerArray(markers=[self.string_to_text_marker(
                message=message, pose=pose, ns="", p_id=0, z_index=cfg.fov_z_index,
                font_size=font_size, frame_id=cfg.main_frame_id, color=colors.black
            )])
            self.publish("/simulation" + cfg.sim_latest_message_topic, marker_array)

    def cleanup_message(self):
        if self.is_activated("/simulation" + cfg.sim_latest_message_topic):
            marker_array = MarkerArray(markers=[self.string_to_text_marker(
                message="_", pose=(0., 0., 0.), ns="", p_id=0, z_index=cfg.fov_z_index,
                font_size=.01, frame_id=cfg.main_frame_id, color=colors.black
            )])
            self.publish("/simulation" + cfg.sim_latest_message_topic, marker_array)
    # endregion

    # region CONFLICTS CHECK
    def publish_transit_horizon_cells(self, poses, start_index, end_index, check_horizon, inflated_grid_by_robot, ns):
        full_topic = cfg.conflicts_check_topic if not ns else '/' + ns + cfg.conflicts_check_topic
        if self.is_activated(full_topic):
            horizon_cells = set()
            for counter, index in enumerate(range(start_index, end_index)):
                if counter > check_horizon:
                    break

                pose = poses[index]
                cell = utils.real_to_grid(pose[0], pose[1], inflated_grid_by_robot.res,
                                          inflated_grid_by_robot.grid_pose)
                horizon_cells.add(cell)
            cube_list_marker = self.grid_cells_to_cube_list_markers(
                horizon_cells, inflated_grid_by_robot.res, inflated_grid_by_robot.grid_pose,
                colors.flashy_green, z_index=0.02, ns="/transit_horizon_cells"
            )
            marker_array = MarkerArray(markers=[cube_list_marker])
            self.publish(full_topic, marker_array)

    def publish_transit_conflicting_cells(self, conflicting_cells, inflated_grid_by_robot, ns):
        full_topic = cfg.conflicts_check_topic if not ns else '/' + ns + cfg.conflicts_check_topic
        if self.is_activated(full_topic):
            cube_list_marker = self.grid_cells_to_cube_list_markers(
                conflicting_cells, inflated_grid_by_robot.res, inflated_grid_by_robot.grid_pose,
                colors.flashy_red, z_index=0.03, ns="/transit_conflicting_cells"
            )
            marker_array = MarkerArray(markers=[cube_list_marker])
            self.publish(full_topic, marker_array)

    def publish_transit_conflicting_polygons_cells(self, conflicting_entities_cells, inflated_grid_by_robot, ns):
        full_topic = cfg.conflicts_check_topic if not ns else '/' + ns + cfg.conflicts_check_topic
        if self.is_activated(full_topic):
            cube_list_marker = self.grid_cells_to_cube_list_markers(
                conflicting_entities_cells, inflated_grid_by_robot.res, inflated_grid_by_robot.grid_pose,
                colors.flashy_cyan, z_index=-0.16, ns="/transit_conflicting_entities_cells"
            )
            marker_array = MarkerArray(markers=[cube_list_marker])
            self.publish(full_topic, marker_array)

    def publish_transfer_horizon_convex_polygons(self, robot_csv_polygons, obstacle_csv_polygons, start_index,
                                                 end_index, check_horizon, ns):
        full_topic = cfg.conflicts_check_topic if not ns else '/' + ns + cfg.conflicts_check_topic
        if self.is_activated(full_topic):
            subspace = "/transfer_horizon_csv_polygons"
            self.publish(full_topic, self.make_delete_all_marker(cfg.main_frame_id, ns=subspace))

            horizon_csv_polygons = []
            for counter, index in enumerate(range(start_index, end_index)):
                if counter > check_horizon:
                    break
                key = (index,)
                if key in robot_csv_polygons:
                    horizon_csv_polygons.append(robot_csv_polygons[key])
                if key in obstacle_csv_polygons:
                    horizon_csv_polygons.append(obstacle_csv_polygons[key])
            markers = []
            for p_id, polygon in enumerate(horizon_csv_polygons):
                marker = polygon_to_triangle_list(polygon, subspace, p_id, frame_id=cfg.main_frame_id,
                                                       color=colors.flashy_green, z_index=-0.06)
                markers.append(marker)
            marker_array = MarkerArray(markers=markers)
            self.publish(full_topic, marker_array)

    def publish_transfer_conflicting_intersections(self):
        pass

    def publish_transfer_conflicting_convex_polygons(self):
        pass

    def cleanup_conflicts_checks(self, ns):
        full_topic = cfg.conflicts_check_topic if not ns else '/' + ns + cfg.conflicts_check_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, self.make_delete_all_marker(cfg.main_frame_id))
    # endregion

    # region EXTRA COMBINED CLEANUP METHODS
    def cleanup_all(self):
        self.cleanup_sim_world()
        self.cleanup_message()
        for ns in self.agents_names:
            self.cleanup_robot_world(ns=ns)
            self.cleanup_robot_sim_world(ns=ns)
            self.cleanup_p_opt(ns=ns)
            self.cleanup_q_manips_for_obs(ns=ns)
            self.cleanup_goal(ns=ns)
            self.cleanup_social_grid_map(ns=ns)
            self.cleanup_combined_costmap(ns=ns)
            self.cleanup_conflicts_checks(ns=ns)
    # endregion

    # region CONVERSION TO ROS MSG HELPERS
    def init_header(self):
        return Header(stamp=self.ros_node.get_timestamp(), frame_id="map")

    def grid_cells_to_cube_list_markers(self, grid_cells, res, grid_pose, color, z_index=-0.5, cube_list=None, ns=""):
        if cube_list is None:
            cube_list = Marker(
                type=Marker.CUBE_LIST,
                ns=ns,
                id=0,
                header=Header(frame_id=cfg.main_frame_id, stamp=self.ros_node.get_timestamp()),
                color=color,
                scale=Vector3(x=res, y=res, z=res),
                points=[])
        for cell in grid_cells:
            point = Point()
            point.x, point.y = utils.grid_to_real(cell[0], cell[1], res, grid_pose)
            point.z = z_index
            cube_list.points.append(point)
        return cube_list

    def grid_cell_to_cube_marker(self, cell, res, grid_pose, color, _id, z_index, ns=""):
        x, y = utils.grid_to_real(cell[0], cell[1], res, grid_pose)
        z = z_index

        cube = Marker(type=Marker.CUBE, ns=ns, id=_id,
                      header=Header(frame_id=cfg.main_frame_id, stamp=self.ros_node.get_timestamp()),
                      color=color, scale=Vector3(x=res, y=res, z=res),
                      pose=Pose(position=(Point(x=x, y=y, z=z) if ROS2 else Vector3(x=x, y=y, z=z))))
        return cube

    def geom_quat_from_yaw(self, yaw):
        explicit_quat = tf_replacement.quaternion_from_euler(0.0, 0.0, math.radians(yaw))
        return Quaternion(x=explicit_quat[0], y=explicit_quat[1], z=explicit_quat[2], w=explicit_quat[3])

    def plan_to_markerarray(self, plan, frame_id, ns):
        markerarray = MarkerArray()
        markers = []
        p_id = 0
        for component in plan.path_components:
            current_color = colors.r0_light_blue
            if ns == "robot_1":
                current_color = colors.r1_light_green
            elif ns == "robot_2":
                current_color = colors.r2_light_pink
            elif ns == "robot_3":
                current_color = colors.r3_light_red
            if component.is_transfer:
                current_color = colors.r0_dark_blue
                if ns == "robot_1":
                    current_color = colors.r1_dark_green
                elif ns == "robot_2":
                    current_color = colors.r2_dark_pink
                elif ns == "robot_3":
                    current_color = colors.r3_dark_red
                obstacle_end_polygon_marker = self.polygon_to_line_strip(
                    component.obstacle_path.polygons[-1], '/end_obstacles', p_id, frame_id, current_color,
                    cfg.path_line_z_index, cfg.border_width
                )
                markers.append(obstacle_end_polygon_marker)
            path_marker = self.real_path_to_linestrip(
                component.robot_path.poses, '/plan', p_id, frame_id, current_color, cfg.path_line_width,
                cfg.path_line_z_index)
            markers.append(path_marker)
            p_id += 1
        markerarray.markers = markers
        return markerarray

    # def real_path_to_pose_markers(real_path, )

    def real_path_to_linestrip(self, real_path, namespace, p_id, frame_id, color, line_width, z_index, link_point=None):
        marker = Marker(type=Marker.LINE_STRIP,
                        ns=namespace,
                        id=p_id,
                        header=Header(frame_id=frame_id, stamp=self.ros_node.get_timestamp()),
                        color=color,
                        scale=Vector3(x=line_width, y=0.0, z=0.0),
                        points=[])
        for i in range(len(real_path) - 1):
            point = real_path[i]
            next_point = real_path[i + 1]
            marker.points.append(Point(x=point[0], y=point[1], z=z_index))
            marker.points.append(Point(x=next_point[0], y=next_point[1], z=z_index))
        if link_point:
            marker.points.append(Point(x=real_path[-1][0], y=real_path[-1][1], z=z_index))
            marker.points.append(Point(x=link_point[0], y=link_point[1], z=z_index))
        return marker

    def poses_to_poses_array(self, poses):
        pose_array = PoseArray(header=self.init_header(), poses=[])
        for pose in poses:
            pose_array.poses.append(self.pose_to_ros_pose(pose))
        return pose_array

    def pose_to_ros_pose(self, pose):
        x, y ,z = pose[0], pose[1], 0.0
        return Pose(
            position=(Point(x=x, y=y, z=z) if ROS2 else Vector3(x=x, y=y, z=z)),
            orientation=self.geom_quat_from_yaw(pose[2])
        )

    def polygon_to_line_strip(self, polygon, namespace, p_id, frame_id, color, z_index, line_width):
        marker = Marker(type=Marker.LINE_STRIP,
                        ns=namespace,
                        id=p_id,
                        header=Header(frame_id=frame_id, stamp=self.ros_node.get_timestamp()),
                        color=color,
                        scale=Vector3(x=line_width, y=0.0, z=0.0),
                        points=[])
        if polygon is not None:
            for i in range(len(polygon.exterior.coords) - 1):
                point = polygon.exterior.coords[i]
                next_point = polygon.exterior.coords[i + 1]
                marker.points.append(Point(x=point[0], y=point[1], z=z_index))
                marker.points.append(Point(x=next_point[0], y=next_point[1], z=z_index))
            marker.points.append(Point(x=polygon.exterior.coords[0][0], y=polygon.exterior.coords[0][1], z=z_index))
            marker.points.append(Point(x=polygon.exterior.coords[1][0], y=polygon.exterior.coords[1][1], z=z_index))
        return marker

    def polygons_to_line_strips_marker_array(self, polygons, namespace, frame_id, color, z_index, line_width):
        marker_array = MarkerArray()
        markers = []
        p_id = 0
        for polygon in polygons:
            markers.append(
                self.polygon_to_line_strip(
                    polygon, namespace, p_id, frame_id, color, z_index, line_width))
            p_id += 1
        marker_array.markers = markers
        return marker_array

    def pose_to_arrow(self, pose, namespace, p_id, frame_id, color, z_index, arrow_length, shaft_diameter,
                      head_diameter,
                      head_length):
        marker = Marker(
            type=Marker.ARROW,
            ns=namespace,
            id=p_id,
            # pose=Pose(Point(pose[0], pose[1], z_index), geom_quat_from_yaw(pose[2])),
            points=[
                Point(x=pose[0], y=pose[1], z=z_index),
                Point(
                    x=pose[0] + arrow_length * math.cos(math.radians(pose[2])),
                    y=pose[1] + arrow_length * math.sin(math.radians(pose[2])),
                    z=z_index
                )
            ],
            scale=Vector3(x=shaft_diameter, y=head_diameter, z=head_length),
            header=Header(frame_id=frame_id, stamp=self.ros_node.get_timestamp()),
            color=color
        )
        return marker

    def make_delete_marker(self, namespace, p_id, frame_id):
        return Marker(ns=namespace, id=p_id, header=Header(frame_id=frame_id, stamp=self.ros_node.get_timestamp()),
                      action=Marker.DELETE)

    def make_delete_all_marker(self, frame_id, ns=''):
        return MarkerArray(
            markers=[
                Marker(ns=ns, header=Header(frame_id=frame_id, stamp=self.ros_node.get_timestamp()), action=Marker.DELETEALL)])

    def string_to_text_marker(
            self, message="", pose=(0., 0., 0.), ns="", p_id=0, z_index=0., font_size=1., frame_id='/map',
            color=None):
        if color is None:
            color = colors.black
        x, y, z = pose[0], pose[1], z_index
        marker = Marker(
            type=Marker.TEXT_VIEW_FACING, ns=ns, id=p_id,
            pose=Pose(
                position=(Point(x=x, y=y, z=z) if ROS2 else Vector3(x=x, y=y, z=z)),
                orientation=self.geom_quat_from_yaw(pose[2])
            ),
            points=[Point(x=pose[0], y=pose[1], z=z_index)], scale=Vector3(x=0., y=0., z=font_size),
            header=Header(frame_id=frame_id, stamp=self.ros_node.get_timestamp()), color=color, text=message
        )
        return marker
    # endregion
