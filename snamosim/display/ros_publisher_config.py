# Target display rate (in Hz)
rate = 500

# Deactivate GUI
deactivate_gui = True

# Simulation topics names (without namespace)
sim_knowledge_topic = "/knowledge"
sim_costmap_topic = "/costmap"
sim_social_costmap_topic = '/social_costmap'
sim_connected_components_topic = "/connected_components"

# Robot topics names (without namespace)
min_max_inflated_polygons_topic = "/compute_c_0_c1/min_max_inflated_polygons"
path_grid_cells_topic = "/a_star/path_grid_cells"
a_star_open_heap_topic = "/a_star/open_heap_cells"
a_star_close_set_topic = "/a_star/close_set_cells"
multi_a_star_open_heap_topic = "/multi_a_star/open_heap"
multi_a_star_close_set_topic = "/multi_a_star/close_set"
stilman_rch_open_heap_topic = "/stilman_rch/open_heap"
stilman_rch_close_set_topic = "/stilman_rch/close_set"
q_l_cells_topic = "/compute_c_0_c1/q_l_cells"
q_l_poses_topic = "/compute_c_0_c1/q_l_poses"
robot_goal_topic = "/goal"
obs_manip_poses_topic = "/test/obs_manip_poses"
c_1_topic = "/p_opt/c_1"
c_2_topic = "/p_opt/c_2"
c_3_topic = "/p_opt/c_3"
eval_c_1_topic = "/eval_c_1"
eval_c_2_topic = "/eval_c_2"
eval_c_3_topic = "/eval_c_3"
robot_sim_topic = "/sim"
robot_knowledge_topic = "/knowledge"
robot_costmap_topic = "/costmap"
robot_sim_costmap_topic = "/robot_sim/costmap"
test_gridmap_topic = "/test/gridmap"
social_cells_topic = "/test/social_cells"
test_connected_components_topic = "/test/connected_components"
robot_sim_world_topic = "/sim/world"
combined_costmap_topic = "/combined_costmap"
plan_topic = "/plan"

default_queue_size = 10

# HACK: Necessary because ROS1 pub/sub system is not reliable : wait (time in seconds) for subscribers to listen
hack_duration_wait = 1.0

main_frame_id = "/map"
social_gridmap_frame_id = "/social_gridmap"
combined_gridmap_frame_id = "/combined_gridmap"
gridmap_frame_ids_to_z_indexes = {
    social_gridmap_frame_id: -1.5,
    combined_gridmap_frame_id: -1.4
}

fov_z_index = -0.04
entities_z_index = -0.05
taboos_z_index = -0.06
path_line_z_index = 0.0

fov_line_width = 0.05
border_width = 0.08
text_height = 0.2
path_line_width = 0.1
