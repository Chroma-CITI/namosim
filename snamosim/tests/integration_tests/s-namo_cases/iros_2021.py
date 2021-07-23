import sys
import logging

if "/home/brenault/s-namo-sim-private" not in sys.path:
    sys.path.append("/home/brenault/s-namo-sim-private")

import unittest
from snamosim.simulator import Simulator
import os
from datetime import datetime


class IROS2021Tests(unittest.TestCase):

    def setUp(self):
        self.path_to_folder = os.path.join(os.path.dirname(__file__), "../../../../data/simulations/iros_2021/")

    def test_single_generated_scenario(self):
        sim = Simulator(simulation_file_path=os.path.join(
            self.path_to_folder, "after_the_feast/", "2_robots/", "50_goals/", "0000/", "sim_namo_0000.json")
        )
        sim.run()

    def test_single_generated_scenario_pair(self):
        self.namo_and_snamo('0000')

    def namo_and_snamo(self, scenario_id):
        timestring = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f")

        try:
            sim_namo = Simulator(
                simulation_file_path=os.path.join(
                    self.path_to_folder, "after_the_feast/", "2_robots/", "50_goals/", scenario_id + "/", "sim_namo_" + scenario_id + ".json"
                ),
                timestring=timestring
            )
            sim_namo_report = sim_namo.run()
            sim_snamo = Simulator(
                simulation_file_path=os.path.join(
                    self.path_to_folder, "after_the_feast/", "2_robots/", "50_goals/", scenario_id + "/", "sim_snamo_" + scenario_id + ".json"
                ),
                timestring=timestring
            )
            sim_snamo_report = sim_snamo.run()
        except Exception as e:
            print(e)

    def test_for_10_hours(self):
        import multiprocessing
        import time
        import psutil

        nb_cpu = multiprocessing.cpu_count()

        start_time = time.time()
        now_time = time.time()

        current_processes = []
        use_computer = True

        nb_scenarios = 1000
        scenario_counter = 0

        while (now_time - start_time) < (5. * 60. * 60.):
            if use_computer and len(current_processes) < nb_cpu - 1:
                process = multiprocessing.Process(target=self.namo_and_snamo, args=(("{:0" + str(len(str(nb_scenarios))) + "d}").format(scenario_counter),))
                current_processes.append(process)
                process.start()
                scenario_counter += 1

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


if __name__ == '__main__':
    unittest.main()
