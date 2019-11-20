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
                                                   [0, 2, 0, 2, 2]]),
                "expected_result_component": {1: {(0,0), (0,1), (0,2), (0,3), (0,4),
                                                  (1,0), (1,1), (1,2), (1,3), (1,4)},
                                              2: {(3,0), (3,1), (3,2), (3,3), (3,4),
                                                  (4,1), (4,3), (4,4)}}
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
                                                   [1, 1, 1, 1, 1]]),
                "expected_result_component": {
                    1: {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
                        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
                        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)}}
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
                                                   [0, 1, 0, 1, 1]]),
                "expected_result_component": {
                    1: {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                                        (2, 2),
                        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
                                (4, 1),         (4, 3), (4, 4)}}

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
                                                   [0, 2, 0, 3, 3]]),
                "expected_result_component": {
                    1: {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                                        (2, 2)},
                    2: {(3, 0), (3, 1),
                                (4, 1)},
                    3: {(3, 3), (3, 4),
                        (4, 3), (4, 4)}}
            }
        ]

        self.test_data_for_update = [self.test_data[0], self.test_data[3]]

    def test_init_grid(self):
        counter = 1
        for test in self.test_data:
            print("Testing case", str(counter))
            ccg = ConnectedComponentsGrid(test["test_array"], neighborhood=utils.TAXI_NEIGHBORHOOD)
            for line_nb in range(ccg.grid.shape[0]):
                self.assertEqual(list(test["expected_result_array"][line_nb]), list(ccg.grid[line_nb]))
            for component_id in test["expected_result_component"].keys():
                self.assertEqual(test["expected_result_component"][component_id], ccg.components[component_id])
            counter += 1

    def test_update(self):
        counter_1 = 1
        for test_data_1 in self.test_data:
            counter_2 = 1
            for test_data_2 in self.test_data:
                if test_data_1 is not test_data_2:
                    print("Testing case", str(counter_1), "against case", str(counter_2))
                    ccg = ConnectedComponentsGrid(test_data_1["test_array"], neighborhood=utils.TAXI_NEIGHBORHOOD)
                    ccg.invaded_cells, ccg.freed_cells = self.compute_invaded_and_freed_cells(
                        test_data_1["expected_result_array"], test_data_2["expected_result_array"])
                    ccg._update_grid()
                    for line_nb in range(ccg.grid.shape[0]):
                        self.assertEqual(list(test_data_2["expected_result_array"][line_nb]), list(ccg.grid[line_nb]))
                counter_2 += 1
            counter_1 += 1

    @staticmethod
    def compute_invaded_and_freed_cells(cc_grid_before, cc_grid_after):
        invaded_cells, freed_cells = set(), set()
        for i in range(len(cc_grid_before)):
            for j in range(len(cc_grid_before[0])):
                if cc_grid_before[i][j] != 0 and cc_grid_after[i][j] == 0:
                    invaded_cells.add((i, j))
                elif cc_grid_before[i][j] == 0 and cc_grid_after[i][j] != 0:
                    freed_cells.add((i, j))
        return invaded_cells, freed_cells


if __name__ == '__main__':
    unittest.main()
