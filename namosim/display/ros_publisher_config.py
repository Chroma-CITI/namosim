# Target display rate (in Hz)
rate = 100000

main_frame_id = "map"
social_gridmap_frame_id = "social_gridmap"
combined_gridmap_frame_id = "combined_gridmap"
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
