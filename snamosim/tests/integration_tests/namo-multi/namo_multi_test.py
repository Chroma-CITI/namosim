import sys

if "/home/brenault/s-namo-sim-private" not in sys.path:
    sys.path.append("/home/brenault/s-namo-sim-private")

import unittest
from snamosim.simulator import Simulator
import os
from datetime import datetime
import multiprocessing
import time
import psutil


class NAMOMultiTests(unittest.TestCase):
    MIN_SCENARIO = 0
    MAX_SCENARIO = 200
    NB_SCENARIOS = 200

    def setUp(self):
        self.path_to_folder = os.path.join(
            os.path.dirname(__file__), "../../../../data/NAMO-multi/"
        )
        self.logging_folder = os.path.join(
            os.path.dirname(__file__), "../../../../logs/"
        )

    # INTRO
    def test_basic_with_opening_namo(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "basic_with_opening/02_basic_with_opening_namo.json",
            )
        )
        report = sim.run()

    def test_basic_with_opening_snamo(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "basic_with_opening/02_basic_with_opening_snamo.json",
            )
        )
        report = sim.run()

    # CONFLICTS
    def test_robot_robot(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder, "basic_with_opening/conflicts/robot_robot.json"
            )
        )
        report = sim.run()

    def test_robot_obstacle(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder, "basic_with_opening/conflicts/robot_obstacle.json"
            )
        )
        report = sim.run()

    def test_stolen_stealing_obstacle(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder, "basic_with_opening/conflicts/stolen_obstacle.json"
            )
        )
        report = sim.run()

    def test_simultaneous_grab(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "basic_with_opening/conflicts/simultaneous_grab.json",
            )
        )
        report = sim.run()

    def test_simultaneous_space_access(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "basic_with_opening/conflicts/simultaneous_space_access.json",
            )
        )
        report = sim.run()

    # DEALOCKS
    # def test_2_sym_rooms_corridor_deadlock_namo(self):
    #     sim = Simulator(simulation_file_path=os.path.join(self.path_to_folder,"2_sym_rooms_corridor_deadlock/2_sym_rooms_corridor_deadlock_namo.json"))
    #     report = sim.run()

    def test_2_sym_rooms_corridor_deadlock_snamo(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "2_sym_rooms_corridor_deadlock/2_sym_rooms_corridor_deadlock_snamo.json",
            )
        )
        report = sim.run()

    def test_2_sym_rooms_corridor_with_obstacle_deadlock_snamo(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "2_sym_rooms_corridor_with_obstacle_deadlock/2_sym_rooms_corridor_with_obstacle_deadlock_snamo.json",
            )
        )
        report = sim.run()

    # def test_2_asym_right_rooms_corridor_deadlock_namo(self):
    #     sim = Simulator(simulation_file_path=os.path.join(self.path_to_folder,"2_asym_right_rooms_corridor_deadlock/2_asym_right_rooms_corridor_deadlock_namo.json"))
    #     report = sim.run()

    def test_2_asym_right_rooms_corridor_deadlock_snamo(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "2_asym_right_rooms_corridor_deadlock/2_asym_right_rooms_corridor_deadlock_snamo.json",
            )
        )
        report = sim.run()

    # def test_2_asym_left_rooms_corridor_deadlock_namo(self):
    #     sim = Simulator(simulation_file_path=os.path.join(self.path_to_folder,"2_asym_left_rooms_corridor_deadlock/2_asym_left_rooms_corridor_deadlock_namo.json"))
    #     report = sim.run()

    def test_2_asym_left_rooms_corridor_deadlock_snamo(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "2_asym_left_rooms_corridor_deadlock/2_asym_left_rooms_corridor_deadlock_snamo.json",
            )
        )
        report = sim.run()

    def test_3_rooms_and_robots_corridor_deadlocks_namo(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "3_rooms_and_robots_corridor_deadlocks/3_rooms_and_robots_corridor_deadlocks_namo.json",
            )
        )
        report = sim.run()

    def test_3_rooms_and_robots_corridor_deadlocks_snamo(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.path_to_folder,
                "3_rooms_and_robots_corridor_deadlocks/3_rooms_and_robots_corridor_deadlocks_snamo.json",
            )
        )
        report = sim.run()

    # RESULTS
    ## INT
    def test_int_2r_50g_namo_scenario(self):
        namo_report = self.run_scenario(
            scenario_folder="after_the_feast/2_robots/50_goals/",
            scenario_id="000",
            scenario_type="namo",
        )

    def test_int_2r_50g_snamo_scenario(self):
        snamo_report = self.run_scenario(
            scenario_folder="after_the_feast/2_robots/50_goals/",
            scenario_id="000",
            scenario_type="snamo",
        )

    def test_int_4r_25g_namo_scenario(self):
        namo_report = self.run_scenario(
            scenario_folder="after_the_feast/4_robots/25_goals/",
            scenario_id="000",
            scenario_type="namo",
        )

    def test_int_4r_25g_snamo_scenario(self):
        snamo_report = self.run_scenario(
            scenario_folder="after_the_feast/4_robots/25_goals/",
            scenario_id="000",
            scenario_type="snamo",
        )

    def test_int_5r_20g_namo_scenario(self):
        namo_report = self.run_scenario(
            scenario_folder="after_the_feast/5_robots/20_goals/",
            scenario_id="000",
            scenario_type="namo",
        )

    def test_int_5r_20g_snamo_scenario(self):
        snamo_report = self.run_scenario(
            scenario_folder="after_the_feast/5_robots/20_goals/",
            scenario_id="000",
            scenario_type="snamo",
        )

    def test_int_10r_10g_namo_scenario(self):
        namo_report = self.run_scenario(
            scenario_folder="after_the_feast/10_robots/10_goals/",
            scenario_id="000",
            scenario_type="namo",
        )

    def test_int_10r_10g_snamo_scenario(self):
        snamo_report = self.run_scenario(
            scenario_folder="after_the_feast/10_robots/10_goals/",
            scenario_id="000",
            scenario_type="snamo",
        )

    ## CITI
    def test_citi_2r_50g_namo_scenario(self):
        namo_report = self.run_scenario(
            scenario_folder="citi/2_robots/50_goals/",
            scenario_id="000",
            scenario_type="namo",
        )

    def test_citi_2r_50g_snamo_scenario(self):
        snamo_report = self.run_scenario(
            scenario_folder="citi/2_robots/50_goals/",
            scenario_id="000",
            scenario_type="snamo",
        )

    def run_scenario(
        self,
        scenario_folder="after_the_feast/4_robots/25_goals/",
        scenario_id="0000",
        timestring=datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f"),
        scenario_type="namo",
    ):
        try:
            sim = Simulator(
                simulation_file_path=os.path.join(
                    self.path_to_folder,
                    scenario_folder,
                    scenario_id + "/",
                    "sim_" + scenario_type + "_" + scenario_id + ".json",
                ),
                simulation_log_stub=scenario_folder,
                timestring=timestring,
            )
            report = sim.run()
            return report
        except Exception as e:
            print(e)

    def namo_and_snamo(self, scenario_folder, scenario_id):
        timestring = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f")

        namo_report = self.run_scenario(
            scenario_folder, scenario_id, timestring, "namo"
        )
        snamo_report = self.run_scenario(
            scenario_folder, scenario_id, timestring, "snamo"
        )

    def test_for_10_hours(self):
        print("Starting test for 10 hours.")

        nb_cpu = multiprocessing.cpu_count()

        start_time = time.time()
        now_time = time.time()

        current_processes = []
        use_computer = True

        scenario_folders = [
            # "after_the_feast/1_robots/100_goals/",
            "after_the_feast/2_robots/50_goals/",
            "after_the_feast/4_robots/25_goals/",
            "after_the_feast/5_robots/20_goals/",
            "after_the_feast/10_robots/10_goals/",
            "citi/2_robots/50_goals/",
        ]

        for scenario_folder in scenario_folders:
            scenario_counter = self.MIN_SCENARIO
            while (now_time - start_time) < (5.0 * 60.0 * 60.0) and (
                scenario_counter < self.MAX_SCENARIO or current_processes
            ):
                if (
                    use_computer
                    and len(current_processes) < nb_cpu - 1
                    and scenario_counter < self.MAX_SCENARIO
                ):
                    print("Execute test for scenario {}".format(scenario_counter))
                    process = multiprocessing.Process(
                        target=self.namo_and_snamo,
                        args=(
                            scenario_folder,
                            ("{:0" + str(len(str(self.NB_SCENARIOS))) + "d}").format(
                                scenario_counter
                            ),
                        ),
                    )
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

                time.sleep(1.0)
                now_time = time.time()

        for index, process in enumerate(current_processes):
            process.terminate()

        os.system("pkill -9 python3")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg_1 = int(sys.argv.pop())
        arg_2 = int(sys.argv.pop())
        NAMOMultiTests.MAX_SCENARIO = max(arg_1, arg_2)
        NAMOMultiTests.MIN_SCENARIO = min(arg_1, arg_2)
    print(
        "Received args : {}, {}".format(
            NAMOMultiTests.MIN_SCENARIO, NAMOMultiTests.MAX_SCENARIO
        )
    )
    unittest.main()
