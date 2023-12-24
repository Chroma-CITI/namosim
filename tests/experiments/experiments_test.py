import os
import time
import unittest

import namosim.config as config
from namosim.simulator import Simulator


class ExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.scenarios_folder = os.path.join(__file__, "../scenarios")

    def test_3_robots(self):
        config.DISPLAY_WINDOW = True
        sim_parallel = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/3_robots.svg"
            )
        )

        start_time = time.perf_counter()
        sim_parallel.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_two_rooms(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/two_rooms.svg"
            )
        )
        sim.run()
        assert True

    def test_evasion(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/evasion.svg"
            )
        )
        sim.run()
        assert True

    def test_intersections_1_robots_50_goals_namo(self):
        config.DISPLAY_WINDOW = True
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/1_robot_50_goals_namo.svg",
            )
        )
        sim.run()
        assert True

    def test_intersections_1_robots_50_goals_snamo(self):
        config.DISPLAY_WINDOW = True
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/1_robot_50_goals_snamo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_intersections_2_robots_50_goals_snamo(self):
        config.DISPLAY_WINDOW = True
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/2_robots_50_goals_snamo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_intersections_2_robots_50_goals_namo(self):
        config.DISPLAY_WINDOW = True
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/2_robots_50_goals_namo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_intersections_2_robots_50_goals_namo_generated(self):
        config.DISPLAY_WINDOW = True
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/generated/1_robots_50_goals_namo/0.svg",
            )
        )
        sim.run()
        assert True

    def test_intersections_4_robots_25_goals_snamo(self):
        config.DISPLAY_WINDOW = True
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/4_robots_25_goals_snamo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_citi_lab_2_robots_50_goals_snamo(self):
        config.DISPLAY_WINDOW = True
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "citi_lab/citi_lab_2_robots_50_goals_snamo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True


if __name__ == "__main__":
    unittest.main()
