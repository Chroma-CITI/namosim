from ros_conversion import *
from future.utils import with_metaclass

from shapely import affinity


from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import MapMetaData

from src.utils.singleton import Singleton

import ros_publisher_config as cfg


class RosPublisher(with_metaclass(Singleton)):
    def __init__(self):
        # Must necessarily be invoked in the init method of this singleton and not at module-level
        rospy.init_node('world_gui_test_node', log_level=rospy.INFO)

        # Target refresh rate
        self.rate = rospy.Rate(cfg.rate)

        # Dictionary of Publishers
        self.publishers = {
            cfg.min_max_inflated_polygons_topic: rospy.Publisher(
                cfg.min_max_inflated_polygons_topic, MarkerArray, queue_size=cfg.default_queue_size),
            cfg.path_grid_cells_topic: rospy.Publisher(
                cfg.path_grid_cells_topic, Marker, queue_size=cfg.default_queue_size),
            cfg.a_star_open_heap_topic: rospy.Publisher(
                cfg.a_star_open_heap_topic, Marker, queue_size=cfg.default_queue_size),
            cfg.a_star_close_set_topic: rospy.Publisher(
                cfg.a_star_close_set_topic, Marker, queue_size=cfg.default_queue_size),
            cfg.multi_a_star_open_heap_topic: rospy.Publisher(
                cfg.multi_a_star_open_heap_topic, Marker, queue_size=cfg.default_queue_size),
            cfg.multi_a_star_close_set_topic: rospy.Publisher(
                cfg.multi_a_star_close_set_topic, Marker, queue_size=cfg.default_queue_size),
            cfg.q_l_cells_topic: rospy.Publisher(
                cfg.q_l_cells_topic, Marker, queue_size=cfg.default_queue_size),
            cfg.q_l_poses_topic: rospy.Publisher(
                cfg.q_l_poses_topic, PoseArray, queue_size=cfg.default_queue_size),
            cfg.robot_goal_topic: rospy.Publisher(
                cfg.robot_goal_topic, MarkerArray, queue_size=cfg.default_queue_size),
            cfg.obs_manip_poses_topic: rospy.Publisher(
                cfg.obs_manip_poses_topic, PoseArray, queue_size=cfg.default_queue_size),
            cfg.c_1_topic: rospy.Publisher(
                cfg.c_1_topic, Path, queue_size=cfg.default_queue_size),
            cfg.c_2_topic: rospy.Publisher(
                cfg.c_2_topic, Path, queue_size=cfg.default_queue_size),
            cfg.c_3_topic: rospy.Publisher(
                cfg.c_3_topic, Path, queue_size=cfg.default_queue_size),
            cfg.eval_c_1_topic: rospy.Publisher(
                cfg.eval_c_1_topic, Path, queue_size=cfg.default_queue_size),
            cfg.eval_c_2_topic: rospy.Publisher(
                cfg.eval_c_2_topic, Path, queue_size=cfg.default_queue_size),
            cfg.eval_c_3_topic: rospy.Publisher(
                cfg.eval_c_3_topic, Path, queue_size=cfg.default_queue_size),
            cfg.robot_sim_topic: rospy.Publisher(
                cfg.robot_sim_topic, MarkerArray, queue_size=cfg.default_queue_size),
            cfg.robot_knowledge_topic: rospy.Publisher(
                cfg.robot_knowledge_topic, MarkerArray, queue_size=cfg.default_queue_size),
            cfg.sim_knowledge_topic: rospy.Publisher(
                cfg.sim_knowledge_topic, MarkerArray, queue_size=cfg.default_queue_size),
            cfg.robot_costmap_topic: rospy.Publisher(
                cfg.robot_costmap_topic, OccupancyGrid, queue_size=cfg.default_queue_size),
            cfg.sim_costmap_topic: rospy.Publisher(
                cfg.sim_costmap_topic, OccupancyGrid, queue_size=cfg.default_queue_size),
            cfg.robot_sim_costmap_topic: rospy.Publisher(
                cfg.robot_sim_costmap_topic, OccupancyGrid, queue_size=cfg.default_queue_size),
            cfg.test_gridmap_topic: rospy.Publisher(
                cfg.test_gridmap_topic, GridMap, queue_size=cfg.default_queue_size),
            cfg.social_cells_topic: rospy.Publisher(
                cfg.social_cells_topic, Marker, queue_size=cfg.default_queue_size)
            }
        # HACK: Necessary because ROS1 pub/sub system is not really reliable : wait a second for subscribers to listen
        time.sleep(cfg.hack_duration_wait)

        # Setup Static Transform for grid map (Hack so that it is properly placed in view)
        broadcaster = StaticTransformBroadcaster()

        world_to_gridmap_transform = TransformStamped()
        world_to_gridmap_transform.header.stamp = rospy.Time.now()
        world_to_gridmap_transform.header.frame_id = cfg.frame_id
        world_to_gridmap_transform.child_frame_id = cfg.gridmap_frame_id
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

    def is_activated(self, topic):
        return self.publishers[topic].get_num_connections() > 0

    def publish_sim_world(self, world, robot_uid):
        if self.is_activated(cfg.sim_knowledge_topic):
            self.publish(cfg.sim_knowledge_topic, world_to_marker_array(world, robot_uid))
        if self.is_activated(cfg.sim_costmap_topic):
            self.publish(cfg.sim_costmap_topic, world_to_costmap(world, robot_uid))

    def publish_robot_world(self, world, robot_uid):
        if self.is_activated(cfg.robot_knowledge_topic):
            self.publish(cfg.robot_knowledge_topic, world_to_marker_array(world, robot_uid))
        if self.is_activated(cfg.robot_costmap_topic):
            self.publish(cfg.robot_costmap_topic,  world_to_costmap(world, robot_uid))

    def publish_robot_sim_costmap(self, world, robot_uid):
        if self.is_activated(cfg.robot_sim_costmap_topic):
            self.publish(cfg.robot_sim_costmap_topic, world_to_costmap(world, robot_uid))

    def publish_grid_map(self, costmap, dd):
        if self.is_activated(cfg.test_gridmap_topic):
            grid_map = costmap_to_grid_map(costmap, dd)
            self.publish(cfg.test_gridmap_topic, grid_map)

    def publish_a_star_open_heap(self, open_heap, res, grid_pose):
        if self.is_activated(cfg.a_star_open_heap_topic):
            open_heap_data = []
            for element in open_heap:
                open_heap_data.append(element.cell)
            open_heap_cells = grid_cells_to_cube_list_markers(open_heap_data, res, grid_pose, color=cfg.flashy_cyan)
            self.publish(cfg.a_star_open_heap_topic, open_heap_cells)

    def publish_a_star_close_set(self, close_set, res, grid_pose):
        if self.is_activated(cfg.a_star_close_set_topic):
            close_set_cells = grid_cells_to_cube_list_markers(list(close_set), res, grid_pose, color=cfg.flashy_green)
            self.publish(cfg.a_star_close_set_topic, close_set_cells)

    def publish_social_cells(self, social_cells_set, res, grid_pose):
        if self.is_activated(cfg.social_cells_topic):
            ros_cells = grid_cells_to_cube_list_markers(list(social_cells_set), res, grid_pose, color=cfg.flashy_purple)
            self.publish(cfg.social_cells_topic, ros_cells)

    def publish_multigoal_a_star_open_heap(self, open_heap, res, grid_pose):
        if self.is_activated(cfg.a_star_open_heap_topic):
            open_heap_data = []
            for element in open_heap:
                open_heap_data.append(element.cell)
            open_heap_cells = grid_cells_to_cube_list_markers(open_heap_data, res, grid_pose, color=cfg.flashy_cyan)
            self.publish(cfg.a_star_open_heap_topic, open_heap_cells)

    def publish_multigoal_a_star_close_set(self, close_set, res, grid_pose):
        if self.is_activated(cfg.a_star_close_set_topic):
            close_set_cells = grid_cells_to_cube_list_markers(list(close_set), res, grid_pose, color=cfg.flashy_green)
            self.publish(cfg.a_star_close_set_topic, close_set_cells)

    def publish_grid_path(self, grid_path, res, grid_pose):
        if self.is_activated(cfg.path_grid_cells_topic):
            path_grid_cells = grid_cells_to_cube_list_markers(grid_path, res, grid_pose, color=cfg.flashy_purple)
            self.publish(cfg.path_grid_cells_topic, path_grid_cells)

    def publish_q_manips_for_obs(self, poses):
        if self.is_activated(cfg.obs_manip_poses_topic):
            pose_array = poses_to_poses_array(poses)
            self.publish(cfg.obs_manip_poses_topic, pose_array)

    def publish_c_1(self, c1):
        if self.is_activated(cfg.eval_c_1_topic):
            self.publish(cfg.eval_c_1_topic, real_path_to_ros_path(c1.path))

    def publish_c_2(self, c2):
        if self.is_activated(cfg.eval_c_2_topic):
            self.publish(cfg.eval_c_2_topic, real_path_to_ros_path(c2.path))

    def publish_c_3(self, c3):
        if self.is_activated(cfg.eval_c_3_topic):
            self.publish(cfg.eval_c_3_topic, real_path_to_ros_path(c3.path))

    def publish_p_opt(self, plan):
        if plan.path_components:
            if self.is_activated(cfg.c_1_topic):
                self.publish(cfg.c_1_topic, real_path_to_ros_path(plan.path_components[0].path))
        if len(plan.path_components) == 3:
            if self.is_activated(cfg.c_2_topic):
                self.publish(cfg.c_2_topic, real_path_to_ros_path(plan.path_components[1].path))
            if self.is_activated(cfg.c_3_topic):
                self.publish(cfg.c_3_topic, real_path_to_ros_path(plan.path_components[2].path))

    def publish_sim(self, robot_polygon, obs_polygon, namespace="/init"):
        if self.is_activated(cfg.robot_sim_topic):
            robot_color = cfg.robot_border_color if namespace == "/target" else cfg.robot_color
            obs_color = cfg.movable_obstacle_border_color if namespace == "/target" else cfg.movable_obstacle_color
            marker_array = MarkerArray(markers=[
                polygon_to_line_strip(robot_polygon, namespace + "/robot/polygon", 0, cfg.frame_id, robot_color,
                                      cfg.entities_z_index, cfg.border_width),
                polygon_to_line_strip(obs_polygon, namespace + "/obstacle/polygon", 0, cfg.frame_id, obs_color,
                                      cfg.entities_z_index, cfg.border_width)])
            self.publish(cfg.robot_sim_topic, marker_array)

    def publish_blocking_areas(self, init_blocking_areas, target_blocking_areas):
        if self.is_activated(cfg.robot_sim_topic):
            init_blocking_areas_markers = []
            for i in range(len(init_blocking_areas)):
                init_blocking_areas_markers.append(polygon_to_triangle_list(
                    init_blocking_areas[i], "/blocking_areas/init", i, cfg.frame_id,
                    cfg.init_blocking_areas_color, cfg.entities_z_index))

            target_blocking_areas_markers = []
            for i in range(len(target_blocking_areas)):
                target_blocking_areas_markers.append(polygon_to_triangle_list(
                    target_blocking_areas[i], "/blocking_areas/target", i, cfg.frame_id,
                    cfg.target_blocking_areas_color, cfg.entities_z_index))

            marker_array = MarkerArray(markers=init_blocking_areas_markers + target_blocking_areas_markers)
            self.publish(cfg.robot_sim_topic, marker_array)

    def publish_diameter_inflated_polygons(self, init_entity_inflated_polygon, target_entity_inflated_polygon):
        if self.is_activated(cfg.robot_sim_topic):
            marker_array = MarkerArray(markers=[
                polygon_to_line_strip(init_entity_inflated_polygon, "/diameter_inflated_polygon/init", 0, cfg.frame_id,
                                      cfg.init_diameter_inflated_polygon_color,
                                      cfg.entities_z_index, cfg.border_width / 2.),
                polygon_to_line_strip(target_entity_inflated_polygon, "/diameter_inflated_polygon/target", 0,
                                      cfg.frame_id,
                                      cfg.target_diameter_inflated_polygon_color,
                                      cfg.entities_z_index, cfg.border_width / 2.)])
            self.publish(cfg.robot_sim_topic, marker_array)

    def publish_min_max_inflated(self, min_inflated_polygon, max_inflated_polygon):
        if self.is_activated(cfg.min_max_inflated_polygons_topic):
            marker_array = MarkerArray(markers=[
                polygon_to_line_strip(min_inflated_polygon, "/min_inflated_polygon", 0, cfg.frame_id,
                                      cfg.min_inflated_polygon_border_color,
                                      cfg.entities_z_index, cfg.border_width),
                polygon_to_line_strip(max_inflated_polygon, "/max_inflated_polygon", 0, cfg.frame_id,
                                      cfg.max_inflated_polygon_border_color,
                                      cfg.entities_z_index, cfg.border_width)])
            self.publish(cfg.min_max_inflated_polygons_topic, marker_array)

    def publish_q_l_cells(self, cells, res, grid_pose):
        if self.is_activated(cfg.q_l_cells_topic):
            close_set_cells = grid_cells_to_cube_list_markers(list(cells), res, grid_pose)
            self.publish(cfg.q_l_cells_topic, close_set_cells)

    def publish_q_l_poses(self, poses):
        if self.is_activated(cfg.q_l_poses_topic):
            pose_array = poses_to_poses_array(poses)
            self.publish(cfg.q_l_poses_topic, pose_array)

    def publish_goal(self, q_init, q_goal, polygon):
        if self.is_activated(cfg.robot_goal_topic):
            if q_goal is not None:
                polygon_at_goal_pose = affinity.translate(polygon, q_goal[0] - q_init[0], q_goal[1] - q_init[1])
                # ros_pose = pose_to_ros_pose_stamped(q_goal)
                marker_array = MarkerArray(markers=[
                    polygon_to_line_strip(polygon_at_goal_pose, "/polygon", 0, cfg.frame_id, cfg.robot_border_color,
                                          cfg.fov_z_index, cfg.border_width)])
                # pose_to_arrow(q_goal, "/pose", 0, self.frame_id, self.robot_border_color,
                #               self.entities_z_index, 0.5, 0.2, 0.0)])
                self.publish(cfg.robot_goal_topic, marker_array)

    def cleanup_sim_world(self):
        if self.is_activated(cfg.sim_knowledge_topic):
            self.publish(cfg.sim_knowledge_topic, make_delete_all_marker(cfg.frame_id))
        if self.is_activated(cfg.sim_costmap_topic):
            self.publish(cfg.sim_costmap_topic, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    def cleanup_robot_world(self):
        if self.is_activated(cfg.robot_knowledge_topic):
            self.publish(cfg.robot_knowledge_topic, make_delete_all_marker(cfg.frame_id))
        if self.is_activated(cfg.robot_costmap_topic):
            self.publish(cfg.robot_costmap_topic, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    def cleanup_eval_c1_c2_c3_sim_init_target(self):
        if self.is_activated(cfg.eval_c_1_topic):
            self.publish(cfg.eval_c_1_topic, init_ros_path())
        if self.is_activated(cfg.eval_c_2_topic):
            self.publish(cfg.eval_c_2_topic, init_ros_path())
        if self.is_activated(cfg.eval_c_3_topic):
            self.publish(cfg.eval_c_3_topic, init_ros_path())
        if self.is_activated(cfg.robot_sim_topic):
            self.publish(cfg.robot_sim_topic, make_delete_all_marker(cfg.frame_id))

    def cleanup_p_opt(self):
        if self.is_activated(cfg.c_1_topic):
            self.publish(cfg.c_1_topic, init_ros_path())
        if self.is_activated(cfg.c_2_topic):
            self.publish(cfg.c_2_topic, init_ros_path())
        if self.is_activated(cfg.c_3_topic):
            self.publish(cfg.c_3_topic, init_ros_path())
        if self.is_activated(cfg.robot_sim_costmap_topic):
            self.publish(cfg.robot_sim_costmap_topic, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))

    def cleanup_q_manips_for_obs(self):
        if self.is_activated(cfg.obs_manip_poses_topic):
            pose_array = PoseArray(header=Header(frame_id=cfg.frame_id, stamp=rospy.Time.now()), poses=[])
            self.publish(cfg.obs_manip_poses_topic, pose_array)

    def cleanup_goal(self):
        if self.is_activated(cfg.robot_goal_topic):
            self.publish(cfg.robot_goal_topic, make_delete_all_marker(cfg.frame_id))

    def cleanup_q_l_cells_poses(self):
        if self.is_activated(cfg.q_l_cells_topic):
            self.publish(cfg.q_l_cells_topic, init_grid_cells(0.1))
        if self.is_activated(cfg.q_l_poses_topic):
            self.publish(cfg.q_l_poses_topic, PoseArray(header=init_header()))

    def cleanup_min_max_inflated(self):
        if self.is_activated(cfg.min_max_inflated_polygons_topic):
            self.publish(cfg.min_max_inflated_polygons_topic, make_delete_marker("", 0, cfg.frame_id))

    def cleanup_a_star_open_heap(self):
        if self.is_activated(cfg.a_star_open_heap_topic):
            self.publish(cfg.a_star_open_heap_topic, make_delete_marker("", 0, cfg.frame_id))

    def cleanup_a_star_close_set(self):
        if self.is_activated(cfg.a_star_close_set_topic):
            self.publish(cfg.a_star_close_set_topic, make_delete_marker("", 0, cfg.frame_id))

    def cleanup_multigoal_a_star_open_heap(self):
        if self.is_activated(cfg.a_star_open_heap_topic):
            self.publish(cfg.a_star_open_heap_topic, make_delete_marker("", 0, cfg.frame_id))

    def cleanup_multigoal_a_star_close_set(self):
        if self.is_activated(cfg.a_star_close_set_topic):
            self.publish(cfg.a_star_close_set_topic, make_delete_marker("", 0, cfg.frame_id))

    def cleanup_grid_path(self):
        if self.is_activated(cfg.path_grid_cells_topic):
            self.publish(cfg.path_grid_cells_topic, make_delete_marker("", 0, cfg.frame_id))

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
