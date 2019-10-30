import unittest
import numpy as np
from src.worldreps.occupation_based.connected_components_grid import ConnectedComponentsGrid
from src.utils import utils


class ConnectedComponentsGridTest(unittest.TestCase):
    def setUp(self):
        self.test_data = [
            {
                "test_array": np.array([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1],
                                        [0, 0, 0, 0, 0],
                                        [1, 0, 1, 0, 0]]),
                "expected_result_array": np.array([[1, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 1],
                                                   [0, 0, 0, 0, 0],
                                                   [2, 2, 2, 2, 2],
                                                   [0, 2, 0, 2, 2]])
            },
            {
                "test_array": np.array([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]]),
                "expected_result_array": np.array([[1, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 1]])
            },
            {
                "test_array": np.array([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [1, 1, 0, 1, 1],
                                        [0, 0, 0, 0, 0],
                                        [1, 0, 1, 0, 0]]),
                "expected_result_array": np.array([[1, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 1],
                                                   [0, 0, 1, 0, 0],
                                                   [1, 1, 1, 1, 1],
                                                   [0, 1, 0, 1, 1]])
            },
            {
                "test_array": np.array([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [1, 1, 0, 1, 1],
                                        [0, 0, 1, 0, 0],
                                        [1, 0, 1, 0, 0]]),
                "expected_result_array": np.array([[1, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 1],
                                                   [0, 0, 1, 0, 0],
                                                   [2, 2, 0, 3, 3],
                                                   [0, 2, 0, 3, 3]])
            },
        ]

    def test_init(self):
        counter = 1
        for test in self.test_data:
            print("Testing case", str(counter))
            ccg = ConnectedComponentsGrid(test["test_array"], neighborhood=utils.TAXI_NEIGHBORHOOD)
            for i in range(ccg.grid.shape[0]):
                self.assertEqual(list(test["expected_result_array"][i]), list(ccg.grid[i]))
            counter += 1


if __name__ == '__main__':
    unittest.main()
