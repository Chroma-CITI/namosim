from future.utils import with_metaclass
import time
import numpy as np
from shapely import affinity
import copy

try:
    import rospy
    from tf2_ros import StaticTransformBroadcaster
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import PoseArray, TransformStamped, Transform, Vector3, Quaternion
    from std_msgs.msg import Header
    from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
    from grid_map_msgs.msg import GridMap
    import ros_conversion as conv
    import ros_publisher_config as cfg
    USE_ROS = True
except ImportError:
    USE_ROS = False

from snamosim.utils.singleton import Singleton
from snamosim.worldreps.entity_based.robot import Robot


class NamespaceCache:
    def __init__(self):
        self.prev_robot_world_draw_data, self.prev_robot_sim_world_draw_data = {}, {}
        self.prev_a_star_close_set, self.prev_multigoal_a_star_close_set = set(), set()
        self.prev_rch_close_set = set()
        self.a_star_close_set_start_id, self.multigoal_a_star_close_set_start_id = 1, 1
        self.rch_close_set_start_id = 1


class RosPublisher(with_metaclass(Singleton)):
    def __init__(self, top_level_namespaces=('simulation', 'agent')):
        if not USE_ROS:
            return

        self.top_level_namespaces = top_level_namespaces

        # HACK: Must necessarily be invoked in the init method of this singleton and not at module-level (rospy bug...)
        if len(top_level_namespaces) > 1:
            self.node_name = 'simulation_ros_node'
        else:
            self.node_name = top_level_namespaces[0] + '_ros_node'
        rospy.init_node(self.node_name)

        # Target refresh rate
        self.rate = rospy.Rate(cfg.rate)

        # Dictionary of Publishers
        self.publishers = {}

        # Add simulation-specific publishers
        if 'simulation' in top_level_namespaces:
            self.publishers['/simulation' + cfg.sim_knowledge_topic] = rospy.Publisher(
                '/simulation' + cfg.sim_knowledge_topic, MarkerArray, queue_size=cfg.default_queue_size)
            self.publishers['/simulation' + cfg.sim_costmap_topic] = rospy.Publisher(
                '/simulation' + cfg.sim_costmap_topic, OccupancyGrid, queue_size=cfg.default_queue_size)
            self.publishers['/simulation' + cfg.test_gridmap_topic] = rospy.Publisher(
                '/simulation' + cfg.test_gridmap_topic, GridMap, queue_size=cfg.default_queue_size)
            self.publishers['/simulation' + cfg.test_connected_components_topic] = rospy.Publisher(
                '/simulation' + cfg.test_connected_components_topic, GridMap, queue_size=cfg.default_queue_size)

        other_namespaces = [ns for ns in top_level_namespaces if ns != 'simulation']

        # Add robot-specific publishers for each robot namespace
        for ns in other_namespaces:
            full_ns = '/' + ns

            self.publishers[full_ns + cfg.min_max_inflated_polygons_topic] = rospy.Publisher(
                full_ns + cfg.min_max_inflated_polygons_topic, MarkerArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.path_grid_cells_topic] = rospy.Publisher(
                full_ns + cfg.path_grid_cells_topic, Marker, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.a_star_open_heap_topic] = rospy.Publisher(
                full_ns + cfg.a_star_open_heap_topic, Marker, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.a_star_close_set_topic] = rospy.Publisher(
                full_ns + cfg.a_star_close_set_topic, MarkerArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.multi_a_star_open_heap_topic] = rospy.Publisher(
                full_ns + cfg.multi_a_star_open_heap_topic, Marker, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.multi_a_star_close_set_topic] = rospy.Publisher(
                full_ns + cfg.multi_a_star_close_set_topic, MarkerArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.stilman_rch_close_set_topic] = rospy.Publisher(
                full_ns + cfg.stilman_rch_close_set_topic, MarkerArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.q_l_cells_topic] = rospy.Publisher(
                full_ns + cfg.q_l_cells_topic, Marker, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.q_l_poses_topic] = rospy.Publisher(
                full_ns + cfg.q_l_poses_topic, PoseArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.robot_goal_topic] = rospy.Publisher(
                full_ns + cfg.robot_goal_topic, MarkerArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.obs_manip_poses_topic] = rospy.Publisher(
                full_ns + cfg.obs_manip_poses_topic, PoseArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.c_1_topic] = rospy.Publisher(
                full_ns + cfg.c_1_topic, Path, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.c_2_topic] = rospy.Publisher(
                full_ns + cfg.c_2_topic, Path, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.c_3_topic] = rospy.Publisher(
                full_ns + cfg.c_3_topic, Path, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.eval_c_1_topic] = rospy.Publisher(
                full_ns + cfg.eval_c_1_topic, Path, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.eval_c_2_topic] = rospy.Publisher(
                full_ns + cfg.eval_c_2_topic, Path, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.eval_c_3_topic] = rospy.Publisher(
                full_ns + cfg.eval_c_3_topic, Path, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.robot_sim_topic] = rospy.Publisher(
                full_ns + cfg.robot_sim_topic, MarkerArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.robot_knowledge_topic] = rospy.Publisher(
                full_ns + cfg.robot_knowledge_topic, MarkerArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.robot_costmap_topic] = rospy.Publisher(
                full_ns + cfg.robot_costmap_topic, OccupancyGrid, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.robot_sim_costmap_topic] = rospy.Publisher(
                full_ns + cfg.robot_sim_costmap_topic, OccupancyGrid, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.test_gridmap_topic] = rospy.Publisher(
                full_ns + cfg.test_gridmap_topic, GridMap, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.social_cells_topic] = rospy.Publisher(
                full_ns + cfg.social_cells_topic, Marker, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.test_connected_components_topic] = rospy.Publisher(
                full_ns + cfg.test_connected_components_topic, GridMap, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.robot_sim_world_topic] = rospy.Publisher(
                full_ns + cfg.robot_sim_world_topic, MarkerArray, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.combined_costmap_topic] = rospy.Publisher(
                full_ns + cfg.combined_costmap_topic, GridMap, queue_size=cfg.default_queue_size)
            self.publishers[full_ns + cfg.plan_topic] = rospy.Publisher(
                full_ns + cfg.plan_topic, MarkerArray, queue_size=cfg.default_queue_size)

        # HACK: Necessary because ROS1 pub/sub system is not really reliable : wait a second for subscribers to listen
        time.sleep(cfg.hack_duration_wait)

        # Setup Static Transform for grid map (Hack so that it is properly placed in view)
        broadcaster = StaticTransformBroadcaster()

        for frame_id, z_index in cfg.gridmap_frame_ids_to_z_indexes.items():
            transform = TransformStamped(
                header=Header(stamp=rospy.Time.now(), frame_id=cfg.main_frame_id), child_frame_id=frame_id,
                transform=Transform(translation=Vector3(z=z_index), rotation=Quaternion(x=0., y=0., z=1., w=0.))
            )
            broadcaster.sendTransform(transform)
            time.sleep(0.5)  # Hack so that transform is properly sent...

        # Initialize caches for each top level namespace
        self.prev_sim_world_draw_data = {}

        self.namespaces_caches = {}
        for ns in other_namespaces:
            self.namespaces_caches[ns] = NamespaceCache()

    def publish(self, topic, msg):
        self.rate.sleep()
        publisher = self.publishers[topic]
        connections = publisher.get_num_connections()
        if connections > 0:
            publisher.publish(msg)

    def is_activated(self, topic=''):
        if cfg.deactivate_gui or (topic and topic not in self.publishers):
            return False
        elif not cfg.deactivate_gui and not topic:
            return True
        return self.publishers[topic].get_num_connections() > 0

    # region SIM WORLD
    def publish_sim_world(self, world, robot_uid):
        if self.is_activated('/simulation' + cfg.sim_knowledge_topic):
            current_world_draw_data = {
                entity.uid: {
                    "polygon": entity.polygon,
                    "type": "robot" if isinstance(entity, Robot) else entity.type,
                    "pose": entity.pose
                } for entity in world.entities.values()}
            entities_to_ignore = {
                entity_uid for entity_uid, drawable_data in current_world_draw_data.items()
                if (entity_uid in self.prev_sim_world_draw_data
                    and drawable_data["polygon"] == self.prev_sim_world_draw_data[entity_uid]["polygon"]
                    and drawable_data["type"] == self.prev_sim_world_draw_data[entity_uid]["type"]
                    and drawable_data["pose"] == self.prev_sim_world_draw_data[entity_uid]["pose"])}
            self.publish('/simulation' + cfg.sim_knowledge_topic,
                         conv.world_to_marker_array(world, robot_uid, entities_to_ignore))
            self.prev_sim_world_draw_data = current_world_draw_data
        if self.is_activated('/simulation' + cfg.sim_costmap_topic):
            self.publish('/simulation' + cfg.sim_costmap_topic, conv.world_to_costmap(world, robot_uid))

    def cleanup_sim_world(self):
        if self.is_activated('/simulation' + cfg.sim_knowledge_topic):
            self.publish('/simulation' + cfg.sim_knowledge_topic, conv.make_delete_all_marker(cfg.main_frame_id))
        if self.is_activated('/simulation' + cfg.sim_costmap_topic):
            self.publish('/simulation' + cfg.sim_costmap_topic,
                         OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    # endregion

    # region ROBOT WORLD
    def publish_robot_world(self, world, robot_uid, ns=''):
        full_topic = cfg.robot_knowledge_topic if not ns else '/' + ns + cfg.robot_knowledge_topic
        if self.is_activated(full_topic):
            current_world_draw_data = {
                entity.uid: {
                    "polygon": entity.polygon,
                    "type": "robot" if isinstance(entity, Robot) else entity.type,
                    "pose": entity.pose
                } for entity in world.entities.values()}
            prev_robot_world_draw_data = self.namespaces_caches[ns].prev_robot_world_draw_data
            entities_to_ignore = {
                entity_uid for entity_uid, drawable_data in current_world_draw_data.items()
                if (entity_uid in prev_robot_world_draw_data
                    and drawable_data["polygon"] == prev_robot_world_draw_data[entity_uid]["polygon"]
                    and drawable_data["type"] == prev_robot_world_draw_data[entity_uid]["type"]
                    and drawable_data["pose"] == prev_robot_world_draw_data[entity_uid]["pose"])}
            self.publish(full_topic, conv.world_to_marker_array(world, robot_uid, entities_to_ignore))
            self.namespaces_caches[ns].prev_robot_world_draw_data = current_world_draw_data

        full_topic = cfg.robot_costmap_topic if not ns else '/' + ns + cfg.robot_costmap_topic
        if self.is_activated(full_topic):
            self.publish(full_topic,  conv.world_to_costmap(world, robot_uid))

    def cleanup_robot_world(self, ns=''):
        full_topic = cfg.robot_knowledge_topic if not ns else '/' + ns + cfg.robot_knowledge_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_all_marker(cfg.main_frame_id))
        full_topic = cfg.robot_costmap_topic if not ns else '/' + ns + cfg.robot_costmap_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    # endregion

    # region ROBOT SIM
    def publish_robot_sim_costmap(self, world, robot_uid, ns=''):
        full_topic = cfg.robot_sim_costmap_topic if not ns else '/' + ns + cfg.robot_sim_costmap_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.world_to_costmap(world, robot_uid))

    # TODO Add cleanup method for publish_robot_sim_costmap

    # endregion

    # region GRID MAP
    def publish_grid_map(self, costmap, res, ns=''):
        full_topic = cfg.test_gridmap_topic if not ns else '/' + ns + cfg.test_gridmap_topic
        fixed_costmap = np.copy(costmap)
        fixed_costmap[fixed_costmap == -1.] = 0.
        if self.is_activated(full_topic):
            grid_map = conv.costmap_to_grid_map(fixed_costmap, res)
            self.publish(full_topic, grid_map)

    def cleanup_grid_map(self, ns=''):
        full_topic = cfg.test_gridmap_topic if not ns else '/' + ns + cfg.test_gridmap_topic
        if self.is_activated(full_topic):
            grid_map = conv.costmap_to_grid_map(np.full((1000, 1000), np.nan), 1.)
            self.publish(full_topic, grid_map)

    def publish_combined_costmap(self, sorted_cell_to_combined_cost, dd, ns=''):
        full_topic = cfg.combined_costmap_topic if not ns else '/' + ns + cfg.combined_costmap_topic
        if self.is_activated(full_topic):
            combined_costmap = np.zeros((dd.d_width, dd.d_height))
            for cell, combined_cost in sorted_cell_to_combined_cost:
                combined_costmap[cell[0]][cell[1]] = combined_cost
            grid_map = conv.costmap_to_grid_map(combined_costmap, dd.res, frame_id=cfg.combined_gridmap_frame_id)
            self.publish(full_topic, grid_map)

    def cleanup_combined_costmap(self, ns=''):
        full_topic = cfg.combined_costmap_topic if not ns else '/' + ns + cfg.combined_costmap_topic
        if self.is_activated(full_topic):
            grid_map = conv.costmap_to_grid_map(np.full((1000, 1000), np.nan), 1.)
            self.publish(full_topic, grid_map)
    # endregion

    # region CONNECTED COMPONENTS GRID
    def publish_connected_components_grid(self, costmap, dd, ns=''):
        full_topic = cfg.test_connected_components_topic if not ns else '/' + ns + cfg.test_connected_components_topic
        if self.is_activated(full_topic):
            grid_map = conv.costmap_to_grid_map(costmap, dd.res)
            self.publish(full_topic, grid_map)

    def cleanup_connected_components_grid(self, ns=''):
        full_topic = cfg.test_connected_components_topic if not ns else '/' + ns + cfg.test_connected_components_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.init_grid_map())

    # endregion

    # region A STAR OPEN HEAP
    def publish_a_star_open_heap(self, open_heap, res, grid_pose, ns=''):
        full_topic = cfg.a_star_open_heap_topic if not ns else '/' + ns + cfg.a_star_open_heap_topic
        if self.is_activated(full_topic):
            open_heap_data = []
            for element in open_heap:
                open_heap_data.append(element.cell)
            open_heap_cells = conv.grid_cells_to_cube_list_markers(
                open_heap_data, res, grid_pose, color=cfg.flashy_cyan)
            self.publish(full_topic, open_heap_cells)

    def cleanup_a_star_open_heap(self, ns=''):
        full_topic = cfg.a_star_open_heap_topic if not ns else '/' + ns + cfg.a_star_open_heap_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_marker("", 0, cfg.main_frame_id))

    # endregion

    # region A STAR CLOSE SET
    def publish_a_star_close_set(self, close_set, res, grid_pose, ns=''):
        full_topic = cfg.a_star_close_set_topic if not ns else '/' + ns + cfg.a_star_close_set_topic
        if self.is_activated(full_topic):
            new_cells = close_set.difference(self.namespaces_caches[ns].prev_a_star_close_set)
            # self.a_star_close_set_cube_list = conv.grid_cells_to_cube_list_markers(
            #     new_cells, res, grid_pose, cfg.unknown_obstacle_color, self.a_star_close_set_cube_list)
            marker_array, self.namespaces_caches[ns].a_star_close_set_start_id = conv.grid_cells_to_cube_markerarray(
                new_cells, res, grid_pose, cfg.dark_purple, self.namespaces_caches[ns].a_star_close_set_start_id)
            self.namespaces_caches[ns].prev_a_star_close_set = copy.copy(close_set)
            # self.publish(full_topic, self.a_star_close_set_cube_list)
            self.publish(full_topic, marker_array)

    def cleanup_a_star_close_set(self, ns=''):
        full_topic = cfg.a_star_close_set_topic if not ns else '/' + ns + cfg.a_star_close_set_topic
        if self.is_activated(full_topic):
            self.namespaces_caches[ns].prev_a_star_close_set = set()
            self.namespaces_caches[ns].a_star_close_set_cube_list = None
            self.namespaces_caches[ns].a_star_close_set_start_id = 1
            self.publish(full_topic, conv.make_delete_all_marker(cfg.main_frame_id))  #conv.make_delete_marker("", 0, cfg.main_frame_id))

    def publish_social_cells(self, social_cells_set, res, grid_pose, ns=''):
        full_topic = cfg.social_cells_topic if not ns else '/' + ns + cfg.social_cells_topic
        if self.is_activated(full_topic):
            ros_cells = conv.grid_cells_to_cube_list_markers(
                list(social_cells_set), res, grid_pose, color=cfg.flashy_purple)
            self.publish(full_topic, ros_cells)

    # endregion

    # region MULTIGOAL A STAR OPEN HEAP
    def publish_multigoal_a_star_open_heap(self, open_heap, res, grid_pose, ns=''):
        full_topic = cfg.multi_a_star_open_heap_topic if not ns else '/' + ns + cfg.multi_a_star_open_heap_topic
        if self.is_activated(full_topic):
            open_heap_data = []
            for element in open_heap:
                open_heap_data.append(element.cell)
            open_heap_cells = conv.grid_cells_to_cube_list_markers(
                open_heap_data, res, grid_pose, color=cfg.flashy_cyan)
            self.publish(full_topic, open_heap_cells)

    def cleanup_multigoal_a_star_open_heap(self, ns=''):
        full_topic = cfg.multi_a_star_open_heap_topic if not ns else '/' + ns + cfg.multi_a_star_open_heap_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_marker("", 0, cfg.main_frame_id))

    # endregion

    # region MULTIGOAL A STAR CLOSE SET
    def publish_multigoal_a_star_close_set(self, close_set, res, grid_pose, ns=''):
        full_topic = cfg.multi_a_star_close_set_topic if not ns else '/' + ns + cfg.multi_a_star_close_set_topic
        if self.is_activated(full_topic):
            new_cells = close_set.difference(self.namespaces_caches[ns].prev_multigoal_a_star_close_set)
            marker_array, self.namespaces_caches[ns].multigoal_a_star_close_set_start_id = conv.grid_cells_to_cube_markerarray(
                new_cells, res, grid_pose, cfg.dark_blue, self.namespaces_caches[ns].multigoal_a_star_close_set_start_id)
            self.namespaces_caches[ns].prev_multigoal_a_star_close_set = copy.copy(close_set)
            self.publish(full_topic, marker_array)

    def cleanup_multigoal_a_star_close_set(self, ns=''):
        full_topic = cfg.multi_a_star_close_set_topic if not ns else '/' + ns + cfg.multi_a_star_close_set_topic
        if self.is_activated(full_topic):
            self.namespaces_caches[ns].prev_multigoal_a_star_close_set = set()
            self.namespaces_caches[ns].multigoal_a_star_close_set_start_id = 1
            self.publish(full_topic, conv.make_delete_all_marker(cfg.main_frame_id))  #conv.make_delete_marker("", 0, cfg.main_frame_id))

    # endregion

    # region STILMAN 2005 RCH OPEN QUEUE
    def publish_rch_open_queue(self, open_queue, res, grid_pose, ns=''):
        full_topic = cfg.stilman_rch_open_heap_topic if not ns else '/' + ns + cfg.stilman_rch_open_heap_topic
        if self.is_activated(full_topic):
            open_queue_data = []
            for element in open_queue:
                open_queue_data.append(element.cell)
            open_heap_cells = conv.grid_cells_to_cube_list_markers(
                open_queue_data, res, grid_pose, color=cfg.flashy_cyan)
            self.publish(full_topic, open_heap_cells)

    def cleanup_rch_open_queue(self, ns=''):
        full_topic = cfg.stilman_rch_open_heap_topic if not ns else '/' + ns + cfg.stilman_rch_open_heap_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_marker("", 0, cfg.main_frame_id))

    # endregion

    # region STILMAN 2005 RCH CLOSED SET
    def publish_rch_closed_set(self, close_set, res, grid_pose, ns=''):
        full_topic = cfg.stilman_rch_close_set_topic if not ns else '/' + ns + cfg.stilman_rch_close_set_topic
        if self.is_activated(full_topic):
            new_cells = close_set.difference(self.namespaces_caches[ns].prev_rch_close_set)
            marker_array, self.namespaces_caches[ns].rch_close_set_start_id = conv.grid_cells_to_cube_markerarray(
                new_cells, res, grid_pose, cfg.dark_brown, self.namespaces_caches[ns].rch_close_set_start_id)
            self.namespaces_caches[ns].prev_rch_close_set = copy.copy(close_set)
            self.publish(full_topic, marker_array)

    def cleanup_rch_closed_set(self, ns=''):
        full_topic = cfg.stilman_rch_close_set_topic if not ns else '/' + ns + cfg.stilman_rch_close_set_topic
        if self.is_activated(full_topic):
            self.namespaces_caches[ns].prev_rch_close_set = set()
            self.namespaces_caches[ns].rch_close_set_start_id = 1
            self.publish(full_topic, conv.make_delete_all_marker(cfg.main_frame_id))

    # endregion

    # region STILMAN 2005 RCH CURRENT CELL AND CURRENT NEIGHBOR
    def publish_current_cell(self, cell, res, grid_pose, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker = conv.grid_cells_to_cube_list_markers([cell], res, grid_pose, color=cfg.flashy_purple)
            marker.ns = "/alg/current_cell"
            marker.id = 0
            marker_array = MarkerArray(markers=[marker])
            self.publish(full_topic, marker_array)

    def publish_current_neighbor(self, cell, res, grid_pose, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker = conv.grid_cells_to_cube_list_markers([cell], res, grid_pose, color=cfg.flashy_red)
            marker.ns = "/alg/current_neighbor"
            marker.id = 0
            marker_array = MarkerArray(markers=[marker])
            self.publish(full_topic, marker_array)

    # endregion

    # region GRID PATH
    def publish_grid_path(self, grid_path, res, grid_pose, ns=''):
        full_topic = cfg.path_grid_cells_topic if not ns else '/' + ns + cfg.path_grid_cells_topic
        if self.is_activated(full_topic):
            path_grid_cells = conv.grid_cells_to_cube_list_markers(grid_path, res, grid_pose, color=cfg.flashy_purple)
            self.publish(full_topic, path_grid_cells)

    def cleanup_grid_path(self, ns=''):
        full_topic = cfg.path_grid_cells_topic if not ns else '/' + ns + cfg.path_grid_cells_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_marker("", 0, cfg.main_frame_id))

    # endregion

    # region Q MANIPS FOR OBS
    def publish_q_manips_for_obs(self, poses, ns=''):
        full_topic = cfg.obs_manip_poses_topic if not ns else '/' + ns + cfg.obs_manip_poses_topic
        if self.is_activated(full_topic):
            pose_array = conv.poses_to_poses_array(poses)
            self.publish(full_topic, pose_array)

    def cleanup_q_manips_for_obs(self, ns=''):
        full_topic = cfg.obs_manip_poses_topic if not ns else '/' + ns + cfg.obs_manip_poses_topic
        if self.is_activated(full_topic):
            pose_array = PoseArray(header=Header(frame_id=cfg.main_frame_id, stamp=rospy.Time.now()), poses=[])
            self.publish(full_topic, pose_array)

    # endregion

    # region ROBOT EVAL C1 C2 C3
    def publish_c_1(self, c1, ns=''):
        full_topic = cfg.eval_c_1_topic if not ns else '/' + ns + cfg.eval_c_1_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.real_path_to_ros_path(c1.path))

    def publish_c_2(self, c2, ns=''):
        full_topic = cfg.eval_c_2_topic if not ns else '/' + ns + cfg.eval_c_2_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.real_path_to_ros_path(c2.path))

    def publish_c_3(self, c3, ns=''):
        full_topic = cfg.eval_c_3_topic if not ns else '/' + ns + cfg.eval_c_3_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.real_path_to_ros_path(c3.path))

    def cleanup_eval_c1_c2_c3_sim_init_target(self, ns=''):
        full_topic = cfg.eval_c_1_topic if not ns else '/' + ns + cfg.eval_c_1_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.init_ros_path())
        full_topic = cfg.eval_c_2_topic if not ns else '/' + ns + cfg.eval_c_2_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.init_ros_path())
        full_topic = cfg.eval_c_3_topic if not ns else '/' + ns + cfg.eval_c_3_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.init_ros_path())
        self.cleanup_robot_sim(ns=ns)

    # endregion

    # region P_OPT
    def publish_p_opt(self, plan, ns=''):
        full_topic = cfg.plan_topic if not ns else '/' + ns + cfg.plan_topic
        if plan and plan.path_components:
            if self.is_activated(full_topic):
                self.publish(full_topic, conv.plan_to_markerarray(plan, cfg.main_frame_id))

    def cleanup_p_opt(self, ns=''):
        full_topic = cfg.plan_topic if not ns else '/' + ns + cfg.plan_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_all_marker(cfg.main_frame_id))

    # def publish_p_opt_deprecated(self, plan):
    #     if plan and plan.path_components:
    #         if self.is_activated(cfg.c_1_topic):
    #             self.publish(cfg.c_1_topic, conv.real_path_to_ros_path(plan.path_components[0].path))
    #         if len(plan.path_components) == 3:
    #             if self.is_activated(cfg.c_2_topic):
    #                 self.publish(cfg.c_2_topic, conv.real_path_to_ros_path(plan.path_components[1].path))
    #             if self.is_activated(cfg.c_3_topic):
    #                 self.publish(cfg.c_3_topic, conv.real_path_to_ros_path(plan.path_components[2].path))

    # def cleanup_p_opt_deprecated(self):
    #     if self.is_activated(cfg.c_1_topic):
    #         self.publish(cfg.c_1_topic, conv.init_ros_path())
    #     if self.is_activated(cfg.c_2_topic):
    #         self.publish(cfg.c_2_topic, conv.init_ros_path())
    #     if self.is_activated(cfg.c_3_topic):
    #         self.publish(cfg.c_3_topic, conv.init_ros_path())
    #     if self.is_activated(cfg.robot_sim_costmap_topic):
    #         self.publish(cfg.robot_sim_costmap_topic, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    # endregion

    # region ROBOT SIM
    def publish_robot_sim_world(self, world, robot_uid, ns=''):
        full_topic = cfg.robot_sim_world_topic if not ns else '/' + ns + cfg.robot_sim_world_topic
        if self.is_activated(full_topic):
            current_world_draw_data = {
                entity.uid: {
                    "polygon_id": id(entity.polygon),
                    "type": "robot" if isinstance(entity, Robot) else entity.type
                } for entity in world.entities.values()}
            prev_robot_sim_world_draw_data = self.namespaces_caches[ns].prev_robot_sim_world_draw_data
            entities_to_ignore = {
                entity_uid for entity_uid, drawable_data in current_world_draw_data.items()
                if (entity_uid in prev_robot_sim_world_draw_data
                    and drawable_data["polygon_id"] == prev_robot_sim_world_draw_data[entity_uid]["polygon_id"]
                    and drawable_data["type"] == prev_robot_sim_world_draw_data[entity_uid]["type"])}
            markers = conv.world_to_marker_array(world, robot_uid, entities_to_ignore)
            self.publish(full_topic, markers)
            self.namespaces_caches[ns].prev_robot_sim_world_draw_data = current_world_draw_data

    def cleanup_robot_sim_world(self, ns=''):
        full_topic = cfg.robot_sim_world_topic if not ns else '/' + ns + cfg.robot_sim_world_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_all_marker(cfg.main_frame_id))

    def publish_sim(self, robot_polygon, obs_polygon, namespace="/init", ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            robot_color = cfg.robot_border_color if namespace == "/target" else cfg.robot_color
            obs_color = cfg.movable_obstacle_border_color if namespace == "/target" else cfg.movable_obstacle_color
            marker_array = MarkerArray(markers=[
                conv.polygon_to_line_strip(
                    robot_polygon, namespace + "/robot/polygon", 0, cfg.main_frame_id, robot_color,
                    cfg.entities_z_index, cfg.border_width),
                conv.polygon_to_line_strip(
                    obs_polygon, namespace + "/obstacle/polygon", 0, cfg.main_frame_id, obs_color,
                    cfg.entities_z_index, cfg.border_width)])
            self.publish(full_topic, marker_array)

    def publish_blocking_areas(self, init_blocking_areas, target_blocking_areas, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            init_blocking_areas_markers = []
            for i in range(len(init_blocking_areas)):
                init_blocking_areas_markers.append(conv.polygon_to_triangle_list(
                    init_blocking_areas[i], "/blocking_areas/init", i, cfg.main_frame_id,
                    cfg.init_blocking_areas_color, cfg.entities_z_index))

            target_blocking_areas_markers = []
            for i in range(len(target_blocking_areas)):
                target_blocking_areas_markers.append(conv.polygon_to_triangle_list(
                    target_blocking_areas[i], "/blocking_areas/target", i, cfg.main_frame_id,
                    cfg.target_blocking_areas_color, cfg.entities_z_index))

            marker_array = MarkerArray(markers=init_blocking_areas_markers + target_blocking_areas_markers)
            self.publish(full_topic, marker_array)

    def cleanup_blocking_areas(self, ns=''):
        # FIXME Not implemented correctly in ROS...
        #  https://answers.ros.org/question/263031/delete-all-rviz-markers-in-a-specific-namespace/
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_all_marker(cfg.main_frame_id, '/blocking_areas'))

    def publish_diameter_inflated_polygons(self, init_entity_inflated_polygon, target_entity_inflated_polygon, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = MarkerArray(markers=[
                conv.polygon_to_line_strip(init_entity_inflated_polygon, "/diameter_inflated_polygon/init", 0,
                                           cfg.main_frame_id, cfg.init_diameter_inflated_polygon_color,
                                           cfg.entities_z_index, cfg.border_width / 2.),
                conv.polygon_to_line_strip(target_entity_inflated_polygon, "/diameter_inflated_polygon/target", 0,
                                           cfg.main_frame_id, cfg.target_diameter_inflated_polygon_color,
                                           cfg.entities_z_index, cfg.border_width / 2.)])
            self.publish(full_topic, marker_array)

    def cleanup_diameter_inflated_polygons(self, ns=''):
        # FIXME Not implemented correctly in ROS...
        #  https://answers.ros.org/question/263031/delete-all-rviz-markers-in-a-specific-namespace/
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(full_topic,
                         conv.make_delete_all_marker(cfg.main_frame_id, '/diameter_inflated_polygon'))

    def publish_min_max_inflated(self, min_inflated_polygon, max_inflated_polygon, ns=''):
        full_topic = cfg.min_max_inflated_polygons_topic if not ns else '/' + ns + cfg.min_max_inflated_polygons_topic
        if self.is_activated(full_topic):
            marker_array = MarkerArray(markers=[
                conv.polygon_to_line_strip(min_inflated_polygon, "/min_inflated_polygon", 0, cfg.main_frame_id,
                                           cfg.min_inflated_polygon_border_color,
                                           cfg.entities_z_index, cfg.border_width),
                conv.polygon_to_line_strip(max_inflated_polygon, "/max_inflated_polygon", 0, cfg.main_frame_id,
                                           cfg.max_inflated_polygon_border_color,
                                           cfg.entities_z_index, cfg.border_width)])
            self.publish(full_topic, marker_array)

    def publish_debug_polygons(self, polygons, ns=''):
        # FIXME Not implemented correctly in ROS...
        #  https://answers.ros.org/question/263031/delete-all-rviz-markers-in-a-specific-namespace/
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = conv.polygons_to_line_strips_marker_array(
                polygons, "/debug/polygons", cfg.main_frame_id, cfg.robot_color,
                cfg.entities_z_index, cfg.border_width / 5.)
            self.publish(full_topic, marker_array)

    def cleanup_debug_polygons(self, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(full_topic,
                         conv.make_delete_all_marker(cfg.main_frame_id, '/debug/polygons'))

    def cleanup_min_max_inflated(self, ns=''):
        full_topic = cfg.min_max_inflated_polygons_topic if not ns else '/' + ns + cfg.min_max_inflated_polygons_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_marker("", 0, cfg.main_frame_id))

    def cleanup_robot_sim(self, ns=''):
        full_topic = cfg.robot_sim_topic if not ns else '/' + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_all_marker(cfg.main_frame_id))

    # endregion

    # region Q L CELLS
    def publish_q_l_cells(self, cells, res, grid_pose, ns=''):
        full_topic = cfg.q_l_cells_topic if not ns else '/' + ns + cfg.q_l_cells_topic
        if self.is_activated(full_topic):
            close_set_cells = conv.grid_cells_to_cube_list_markers(list(cells), res, grid_pose, color=cfg.flashy_cyan)
            self.publish(full_topic, close_set_cells)

    def publish_q_l_poses(self, poses, ns=''):
        full_topic = cfg.q_l_poses_topic if not ns else '/' + ns + cfg.q_l_poses_topic
        if self.is_activated(full_topic):
            pose_array = conv.poses_to_poses_array(poses)
            self.publish(full_topic, pose_array)

    def cleanup_q_l_cells_poses(self, ns=''):
        full_topic = cfg.q_l_cells_topic if not ns else '/' + ns + cfg.q_l_cells_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.init_grid_cells(0.1))
        full_topic = cfg.q_l_poses_topic if not ns else '/' + ns + cfg.q_l_poses_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, PoseArray(header=conv.init_header()))

    # endregion

    # region GOAL
    def publish_goal(self, q_init, q_goal, polygon, ns=''):
        full_topic = cfg.robot_goal_topic if not ns else '/' + ns + cfg.robot_goal_topic
        if self.is_activated(full_topic):
            if q_goal is not None:
                polygon_at_goal_pose = affinity.translate(polygon, q_goal[0] - q_init[0], q_goal[1] - q_init[1])
                # ros_pose = pose_to_ros_pose_stamped(q_goal)
                marker_array = MarkerArray(markers=[
                    conv.polygon_to_line_strip(polygon_at_goal_pose, "/polygon", 0, cfg.main_frame_id,
                                               cfg.robot_border_color, cfg.fov_z_index, cfg.border_width)])
                # pose_to_arrow(q_goal, "/pose", 0, self.frame_id, self.robot_border_color,
                #               self.entities_z_index, 0.5, 0.2, 0.0)])
                self.publish(full_topic, marker_array)

    def cleanup_goal(self, ns=''):
        full_topic = cfg.robot_goal_topic if not ns else '/' + ns + cfg.robot_goal_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, conv.make_delete_all_marker(cfg.main_frame_id))

    # endregion

    # region EXTRA COMBINED CLEANUP METHODS

    def cleanup_all(self):
        if 'simulation' in self.top_level_namespaces:
            self.cleanup_sim_world()

        other_namespaces = [ns for ns in self.top_level_namespaces if ns != 'simulation']

        for ns in other_namespaces:
            self.cleanup_robot_world(ns=ns)
            self.cleanup_robot_sim_world(ns=ns)
            self.cleanup_eval_c1_c2_c3_sim_init_target(ns=ns)
            self.cleanup_p_opt(ns=ns)
            self.cleanup_q_manips_for_obs(ns=ns)
            self.cleanup_goal(ns=ns)
            self.cleanup_q_l_cells_poses(ns=ns)
            self.cleanup_min_max_inflated(ns=ns)
            self.cleanup_a_star_open_heap(ns=ns)
            self.cleanup_a_star_close_set(ns=ns)
            self.cleanup_multigoal_a_star_open_heap(ns=ns)
            self.cleanup_multigoal_a_star_close_set(ns=ns)
            self.cleanup_grid_path(ns=ns)
            self.cleanup_grid_map(ns=ns)
            self.cleanup_combined_costmap(ns=ns)

    # endregion
