import sys

if "/home/brenault/s-namo-sim-private" not in sys.path:
    sys.path.append("/home/brenault/s-namo-sim-private")

import unittest
from snamosim.simulator import Simulator
import os
from datetime import datetime


class IROS2021Tests(unittest.TestCase):
    MIN_SCENARIO = 0
    MAX_SCENARIO = 1000

    def setUp(self):
        self.path_to_folder = os.path.join(os.path.dirname(__file__), "../../../../data/simulations/iros_2021/")

    def test_single_generated_scenario(self):
        sim = Simulator(simulation_file_path=os.path.join(
            self.path_to_folder, "after_the_feast/", "4_robots/", "25_goals/", "0000/", "sim_namo_0000.json")
        )
        sim.run()

    def test_single_generated_scenario_pair(self):
        self.namo_and_snamo('0000')

    def namo_and_snamo(self, scenario_id):
        timestring = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f")

        try:
            sim_namo = Simulator(
                simulation_file_path=os.path.join(
                    self.path_to_folder, "after_the_feast/", "4_robots/", "25_goals/", scenario_id + "/", "sim_namo_" + scenario_id + ".json"
                ),
                timestring=timestring
            )
            sim_namo_report = sim_namo.run()
        except Exception as e:
            print(e)

        try:
            sim_snamo = Simulator(
                simulation_file_path=os.path.join(
                    self.path_to_folder, "after_the_feast/", "4_robots/", "25_goals/", scenario_id + "/", "sim_snamo_" + scenario_id + ".json"
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

        print('Starting test for 10 hours.')

        nb_cpu = multiprocessing.cpu_count()

        start_time = time.time()
        now_time = time.time()

        current_processes = []
        use_computer = True

        nb_scenarios = 1000
        scenario_counter = self.MIN_SCENARIO

        while (now_time - start_time) < (5. * 60. * 60.) and (scenario_counter < self.MAX_SCENARIO or current_processes):
            if use_computer and len(current_processes) < nb_cpu - 1 and scenario_counter < self.MAX_SCENARIO:
                print('Execute test for scenario {}'.format(scenario_counter))
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
                if user.name not in ["brenault", "xia0ben", "vdiuser"]:
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
    if len(sys.argv) > 1:
        arg_1 = int(sys.argv.pop())
        arg_2 = int(sys.argv.pop())
        IROS2021Tests.MAX_SCENARIO = max(arg_1, arg_2)
        IROS2021Tests.MIN_SCENARIO = min(arg_1, arg_2)
    print('Received args : {}, {}'.format(IROS2021Tests.MIN_SCENARIO, IROS2021Tests.MAX_SCENARIO))
    unittest.main()
