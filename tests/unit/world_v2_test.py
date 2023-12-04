import os
import unittest

from namosim.world.world_v2 import WorldV2


class WorldV2Test(unittest.TestCase):
    def setUp(self):
        self.scenarios_folder = os.path.join(__file__, "../data/scenarios")

    def test_load_from_svg(self):
        w = WorldV2.load_from_svg("./tests/unit/data/scenarios/minimal_v2.svg")
        assert w is not None
