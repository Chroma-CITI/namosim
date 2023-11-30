import os
import time
import unittest

import namosim.config as config
from namosim.simulator import Simulator


class MultiRobotTests(unittest.TestCase):
    def setUp(self):
        self.scenarios_folder = os.path.join(__file__, "../scenarios")

    def test_3_robots_parallel(self):
        config.DISPLAY_WINDOW = True
        sim_parallel = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/3_robots_sim.json"
            )
        )

        start_time = time.perf_counter()
        sim_parallel.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time with parallel thinking: {elapsed_time} seconds")
        assert True

    def test_3_robots_sequential(self):
        config.DISPLAY_WINDOW = True
        sim_parallel = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/3_robots_sim.json"
            )
        )

        start_time = time.perf_counter()
        sim_parallel.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time when thinking sequentially {elapsed_time} seconds")
        assert True


if __name__ == "__main__":
    unittest.main()
