# Target display rate (in Hz)
rate = 100000

# Simulation topics names (without namespace)
sim_knowledge_topic = "/knowledge"
sim_costmap_topic = "/costmap"
sim_social_costmap_topic = "/social_costmap"
sim_connected_components_topic = "/connected_components"
sim_latest_message_topic = "/messages"
obstacles_topic = "/namo/obstacles"

# Robot topics names (without namespace)
robot_goal_topic = "/goal"
obs_manip_poses_topic = "/test/obs_manip_poses"
robot_sim_topic = "/sim"
robot_knowledge_topic = "/knowledge"
robot_costmap_topic = "/costmap"
robot_sim_costmap_topic = "/robot_sim/costmap"
test_social_gridmap_topic = "/test/gridmap"
test_connected_components_topic = "/test/connected_components"
robot_sim_world_topic = "/robot_sim/world"
test_combined_gridmap_topic = "/test/combined_costmap"
plan_topic = "/plan"
swept_area_topic = "/swept_area"
conflict_horizon_topic = "/conflict_horizon"
conflicts_check_topic = "/conflicts_check"
manip_search_topic = "/manip_search"
default_queue_size = 100

# HACK: Necessary because ROS1 pub/sub system is not reliable : wait (time in seconds) for subscribers to listen
hack_duration_wait = 1.0

main_frame_id = "/map"
social_gridmap_frame_id = "/social_gridmap"
combined_gridmap_frame_id = "/combined_gridmap"
gridmap_frame_ids_to_z_indexes = {
    social_gridmap_frame_id: -1.5,
    combined_gridmap_frame_id: -1.4,
}

horizon_markers_z_index = 0.02
path_line_z_index = 0.01
entities_z_index = 0.0
swept_area_z_index = -0.01
goal_z_index = -0.02
conflicting_cells_z_index = -0.029
conflict_markers_z_index = -0.03
