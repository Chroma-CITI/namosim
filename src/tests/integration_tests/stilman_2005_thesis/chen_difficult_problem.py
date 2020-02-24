import unittest
from src.simulator import Simulator


class ChenDifficultProblem(unittest.TestCase):

    def setUp(self):
        self.path_to_folder = "../../../../data/simulations/stilman_2005_thesis/04_chen_difficult_problem/"

    def test_navigation_only_behavior(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"navigation_only_behavior.yaml")
        sim.run()
        # Test should end up with a failure

    def test_navigation_only_behavior_no_movables(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"navigation_only_behavior_no_movables.yaml")
        sim.run()
        # Test should end up with a success

    def test_wu_levihn_2014_behavior(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"wu_levihn_2014.yaml")
        sim.run()
        # Test should end up with a success

    def test_wu_levihn_2014_behavior_no_movables(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"wu_levihn_2014_no_movables.yaml")
        sim.run()
        # Test should end up with a success

    def test_stilman_2005_behavior(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"stilman_2005_behavior.yaml")
        sim.run()
        # Test should end up with a success

    def test_stilman_2005_behavior_no_movables(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"stilman_2005_behavior_no_movables.yaml")
        sim.run()
        # Test should end up with a success


if __name__ == '__main__':
    unittest.main()
