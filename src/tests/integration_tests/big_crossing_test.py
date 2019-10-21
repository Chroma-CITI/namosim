import unittest
from src.simulator import Simulator


class BigCrossingTest(unittest.TestCase):

    def test_big_crossing_01(self):
        sim = Simulator(world_file_path="../../../data/worlds/first_level/03_big_crossing/03_big_crossing.yaml")
        sim.run()


if __name__ == '__main__':
    unittest.main()
