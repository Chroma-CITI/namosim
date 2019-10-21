import unittest
from src.simulator import Simulator


class TwoRoomsCorridorTest(unittest.TestCase):

    def test_navigation_only_behavior(self):
        sim = Simulator(simulation_file_path="../../../data/simulations/first_level/01_two_rooms_corridor/navigation_only_behavior.yaml")
        sim.run()
        # Test should end up with a failure

    def test_navigation_only_behavior_no_boxes(self):
        sim = Simulator(simulation_file_path="../../../data/simulations/first_level/01_two_rooms_corridor/navigation_only_behavior_no_boxes.yaml")
        sim.run()
        # Test should end up with a success

    def test_wu_levihn_2014_behavior(self):
        sim = Simulator(simulation_file_path="../../../data/simulations/first_level/01_two_rooms_corridor/wu_levihn_2014.yaml")
        sim.run()
        # Test should end up with a success

    def test_wu_levihn_2014_behavior_no_boxes(self):
        sim = Simulator(simulation_file_path="../../../data/simulations/first_level/01_two_rooms_corridor/wu_levihn_2014_no_boxes.yaml")
        sim.run()
        # Test should end up with a success


if __name__ == '__main__':
    unittest.main()
