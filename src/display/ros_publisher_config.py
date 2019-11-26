from std_msgs.msg import ColorRGBA


def hex_to_rgba(hex_string):
    hex_string = hex_string.lstrip('#')
    argb_tuple = tuple(int(hex_string[i:i + 2], 16) / 255. for i in (0, 2, 4, 6))
    rgba_tuple = (argb_tuple[1], argb_tuple[2], argb_tuple[3], argb_tuple[0])
    return rgba_tuple


# Target display rate (in Hz)
rate = 1000000

# Actual topic names
min_max_inflated_polygons_topic = "/robot/compute_c_0_c1/min_max_inflated_polygons"
path_grid_cells_topic = "/test/path_grid_cells"
a_star_open_heap_topic = "/test/open_heap_cells"
a_star_close_set_topic = "/test/close_set_cells"
multi_a_star_open_heap_topic = "/test/multigoal_a_star_open_heap"
multi_a_star_close_set_topic = "/test/multigoal_a_star_close_set"
q_l_cells_topic = "/robot/compute_c_0_c1/q_l_cells"
q_l_poses_topic = "/robot/compute_c_0_c1/q_l_poses"
robot_goal_topic = "/robot/goal"
obs_manip_poses_topic = "/test/obs_manip_poses"
c_1_topic = "/robot/p_opt/c_1"
c_2_topic = "/robot/p_opt/c_2"
c_3_topic = "/robot/p_opt/c_3"
eval_c_1_topic = "/robot/eval_c_1"
eval_c_2_topic = "/robot/eval_c_2"
eval_c_3_topic = "/robot/eval_c_3"
robot_sim_topic = "/robot/sim"
robot_knowledge_topic = "/robot/knowledge"
sim_knowledge_topic = "/sim/knowledge"
robot_costmap_topic = "/robot/costmap"
sim_costmap_topic = "/sim/costmap"
robot_sim_costmap_topic = "/robot_sim/costmap"
test_gridmap_topic = "/test/gridmap"
social_cells_topic = "/test/social_cells"

default_queue_size = 10

# HACK: Necessary because ROS1 pub/sub system is not reliable : wait (time in seconds) for subscribers to listen
hack_duration_wait = 1.0

frame_id = "/map"
gridmap_frame_id = "/gridmap"

# Elements colors
robot_color = ColorRGBA(*hex_to_rgba("#ff6d9eeb"))
movable_obstacle_color = ColorRGBA(*hex_to_rgba("#fff1c232"))
unmovable_obstacle_color = ColorRGBA(*hex_to_rgba("#ff000000"))
unknown_obstacle_color = ColorRGBA(*hex_to_rgba("#ff8e7cc3"))
taboo_color = ColorRGBA(*hex_to_rgba("#ffea9999"))

robot_border_color = ColorRGBA(*hex_to_rgba("#ff1155cc"))
movable_obstacle_border_color = ColorRGBA(*hex_to_rgba("#ff7f6000"))
unmovable_obstacle_border_color = ColorRGBA(*hex_to_rgba("#ff4d2802"))
unknown_obstacle_border_color = ColorRGBA(*hex_to_rgba("#ff351c75"))
taboo_border_color = ColorRGBA(*hex_to_rgba("#ffcc0000"))
g_fov_border_color = ColorRGBA(*hex_to_rgba("#ff6d9eeb"))
s_fov_border_color = ColorRGBA(*hex_to_rgba("#ff6aa84f"))

min_inflated_polygon_border_color = ColorRGBA(*hex_to_rgba("#ff666666"))
max_inflated_polygon_border_color = ColorRGBA(*hex_to_rgba("#ff666666"))

text_color_on_filling = ColorRGBA(*hex_to_rgba("#ffffffff"))
text_color_on_empty = ColorRGBA(*hex_to_rgba("#ff000000"))

init_blocking_areas_color = ColorRGBA(*hex_to_rgba("#aafd5454"))
target_blocking_areas_color = ColorRGBA(*hex_to_rgba("#aac85ab7"))
init_diameter_inflated_polygon_color = ColorRGBA(*hex_to_rgba("#aa88dc7a"))
target_diameter_inflated_polygon_color = ColorRGBA(*hex_to_rgba("#aa24641a"))

flashy_green = ColorRGBA(*hex_to_rgba("#ff25ff00"))
flashy_cyan = ColorRGBA(*hex_to_rgba("#ff85ffff"))
flashy_purple = ColorRGBA(*hex_to_rgba("#ffff00ff"))

fov_z_index = -0.04
entities_z_index = -0.05
taboos_z_index = -0.06

fov_line_width = 0.05
border_width = 0.075
text_height = 0.2
