import os


if __name__ == '__main__':
    simulation_file_parent_dirname = '04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset_snamo'

    rel_path_to_main_sim_logs_dir = os.path.join('../logs/', simulation_file_parent_dirname)
    abs_path_to_main_sim_logs_dir = os.path.join(os.path.dirname(__file__), rel_path_to_main_sim_logs_dir)

