import sys
import logging

if "/home/brenault/s-namo-sim-private" not in sys.path:
    sys.path.append("/home/brenault/s-namo-sim-private")

import unittest
from snamosim.simulator import Simulator
import os
from datetime import datetime


class AfterTheFeastTest(unittest.TestCase):

    def setUp(self):
        self.path_to_folder = os.path.join(__file__, "../../../../../data/simulations/s-namo_cases/04_after_the_feast/")

    # def test_navigation_only_behavior(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"navigation_only_behavior.json")
    #     report = sim.run()
    #
    # def test_navigation_only_behavior_no_movables(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"navigation_only_behavior_no_movables.json")
    #     sim.run()
    #     # Test should end up with a success
    #
    # def test_navigation_only_behavior_no_movables_multi_robots(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"navigation_only_behavior_no_movables_multi_robots.json")
    #     sim.run()
    #     # Test should end up with a success
    #
    # def test_wu_levihn_2014_behavior(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"wu_levihn_2014.json")
    #     sim.run()
    #     # Test should end up with a success
    #
    # def test_wu_levihn_2014_behavior_no_movables(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"wu_levihn_2014_no_movables.json")
    #     sim.run()
    #     # Test should end up with a success

    def test_stilman_2005_behavior(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"stilman_2005_behavior.json")
        sim.run()
        # Test should end up with a success

    def test_stilman_2005_behavior_no_movables(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"stilman_2005_behavior_no_movables.json")
        sim.run()
        # Test should end up with a success

    def test_stilman_2005_behavior_complexified(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified.json")
        sim.run()

    def test_stilman_2005_behavior_complexified_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_snamo.json")
        sim.run()

    def test_stilman_2005_behavior_complexified_snamo_r2g2(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_snamo_r2g2.json")
        sim.run()

    # def test_stilman_2005_behavior_complexified_debug_case_after_16_goals(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder + "/debug/after_16_goals_exception/stilman_2005_behavior_complexified_random_goal_no_reset_snamo.json")
    #     sim.run()

    def test_stilman_2005_behavior_multi_robots(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots.json")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified.json")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_snamo.json")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_snamo.json")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_reset(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_reset.json")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_reset_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_reset_snamo.json")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_no_reset(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset.json")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_no_reset_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset_snamo.json")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_no_reset_namo_simple_then_snamo(self):
        timestring = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f")
        sim_namo = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset.json", timestring=timestring)
        sim_namo_report = sim_namo.run()
        sim_snamo = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset_snamo.json", goals=sim_namo_report["temp_goals"], timestring=timestring)
        sim_snamo_report = sim_snamo.run()

    def namo_and_snamo(self):
        timestring = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f")

        LOG_FILENAME = '/tmp/logging_example.out'

        try:
            sim_namo = Simulator(
                simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset.json",
                timestring=timestring)
            sim_namo_report = sim_namo.run()
            sim_snamo = Simulator(
                simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset_snamo.json",
                goals=sim_namo_report["temp_goals"], timestring=timestring)
            sim_snamo_report = sim_snamo.run()
        except Exception as e:
            print(e)

    def test_for_2_hours(self):
        import multiprocessing
        import time
        import psutil

        nb_cpu = multiprocessing.cpu_count()

        start_time = time.time()
        now_time = time.time()

        current_processes = []
        use_computer = True

        while (now_time - start_time) < (8. * 60. * 60.):
            if use_computer and len(current_processes) < nb_cpu - 3:
                process = multiprocessing.Process(target=self.namo_and_snamo)
                current_processes.append(process)
                process.start()

            for index, process in enumerate(current_processes):
                if not process.is_alive():
                    process.terminate()
                    del current_processes[index]

            connected_users = psutil.users()
            other_connected_users = False
            for user in connected_users:
                if user.name not in ["brenault", "xia0ben"]:
                    use_computer = False
                    other_connected_users = True

                    break
            if other_connected_users:
                for process in current_processes:
                    process.terminate()
                current_processes = []


            time.sleep(1.)
            now_time = time.time()

        for index, process in enumerate(current_processes):
            process.terminate()

        os.system("pkill -9 python3")

    def test_stilman_2005_behavior_multi_robots_complexified_random_goal_reset(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_random_goal_reset.json")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_random_goal_no_reset(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_random_goal_no_reset.json")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_random_goal_reset_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_random_goal_reset_snamo.json")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_random_goal_no_reset_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_random_goal_no_reset_snamo.json")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_conflict_middle(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_conflict_middle.json")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_conflict_02(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_conflict_02.json")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_4_robots(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_4_robots.json")
        sim.run()

    def test_stilman_2005_behavior_replay_complexified_random(self):
        pass

if __name__ == '__main__':
    unittest.main()
