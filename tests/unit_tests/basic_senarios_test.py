import os
import unittest

from namosim.simulator import Simulator


class BasicTest(unittest.TestCase):
    def setUp(self):
        self.scenarios_folder = os.path.join(__file__, "../test_data/scenarios")

    def test_minimal(self):
        sim = Simulator(
            simulation_file_path=os.path.join(self.scenarios_folder, "minimal_sim.json")
        )
        sim.run()


if __name__ == "__main__":
    unittest.main()
