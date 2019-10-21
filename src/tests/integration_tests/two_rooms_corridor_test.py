import unittest
from src.simulator import Simulator


class TwoRoomsCorridorTest(unittest.TestCase):

    def test_two_rooms_corridor_test(self):
        sim = Simulator(simulation_file_path="../data/simulations/first_level/01_two_rooms_corridor/navigation_only_behavior.yaml")
        sim.run()


if __name__ == '__main__':
    unittest.main()
