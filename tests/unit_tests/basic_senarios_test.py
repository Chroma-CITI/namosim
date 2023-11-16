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
        assert (
            sim.simulation_log[7].message
            == "Agent robot_0 finished executing all its goals."
        )

    def test_custom(self):
        sim = Simulator(
            simulation_file_path=os.path.join(self.scenarios_folder, "custom_sim.json")
        )
        sim.run()
        assert (
            sim.simulation_log[7].message
            == "Agent robot_0 finished executing all its goals."
        )


if __name__ == "__main__":
    unittest.main()
