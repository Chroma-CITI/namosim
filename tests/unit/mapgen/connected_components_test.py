import unittest

import numpy as np

from namosim.mapgen.connected_components import ConnectedComponents


class Test:
    def test_num_components(self):
        cc = ConnectedComponents(np.array([[0, 1], [1, 0]]))
        assert len(cc.components) == 2

    def test_cell_to_component(self):
        cc = ConnectedComponents(np.array([[0, 1], [1, 0]]))
        assert len(cc.cell_to_component) == 2
        assert (0, 0) in cc.cell_to_component
        assert (1, 1) in cc.cell_to_component
        assert (0, 1) not in cc.cell_to_component

    def test_nearest_neighbor(self):
        cc = ConnectedComponents(np.array([[0, 1], [1, 0]]))
        nn, _path = cc.find_nearest_neighbor(0)
        assert nn == 1

        nn, _path = cc.find_nearest_neighbor(1)
        assert nn == 0
