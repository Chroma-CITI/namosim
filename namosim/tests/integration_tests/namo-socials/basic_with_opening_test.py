import unittest
import os
from namosim.simulator import Simulator


class BasicWithOpeningTest(unittest.TestCase):
    def setUp(self):
        self.path_to_folder = os.path.join(
            __file__, "../../../../../data/NAMO-socials/02_basic_with_opening/"
        )

    def test_navigation_only_behavior(self):
        sim = Simulator(
            simulation_file_path=self.path_to_folder + "navigation_only_behavior.json"
        )
        sim.run()
        # Test should end up with a failure

    def test_navigation_only_behavior_no_movables(self):
        sim = Simulator(
            simulation_file_path=self.path_to_folder
            + "navigation_only_behavior_no_movables.json"
        )
        sim.run()
        # Test should end up with a success

    def test_wu_levihn_2014_behavior(self):
        sim = Simulator(
            simulation_file_path=self.path_to_folder + "wu_levihn_2014.json"
        )
        sim.run()
        # Test should end up with a success

    def test_wu_levihn_2014_behavior_no_movables(self):
        sim = Simulator(
            simulation_file_path=self.path_to_folder + "wu_levihn_2014_no_movables.json"
        )
        sim.run()
        # Test should end up with a success

    def test_stilman_2005_behavior(self):
        sim = Simulator(
            simulation_file_path=self.path_to_folder + "stilman_2005_behavior.json"
        )
        sim.run()
        # Test should end up with a success

    def test_stilman_2005_behavior_no_movables(self):
        sim = Simulator(
            simulation_file_path=self.path_to_folder
            + "stilman_2005_behavior_no_movables.json"
        )
        sim.run()
        # Test should end up with a success


if __name__ == "__main__":
    unittest.main()
