import unittest
from src.simulator import Simulator


class MoghaddamPlanning2016Benchmark01(unittest.TestCase):

    def moghaddam_planning_2016_benchmark_01_test_01(self):
        sim = Simulator(world_file_path="../../../data/worlds/moghaddam_planning_2016_benchmark/01/01.yaml")
        sim.run()


if __name__ == '__main__':
    unittest.main()
