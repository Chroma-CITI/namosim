import os
import unittest

from namosim.world.world import World


class WorldV2Test(unittest.TestCase):
    def setUp(self):
        self.scenarios_folder = os.path.join(__file__, "../data/scenarios")

    def test_load_from_svg(self):
        w = World.load_from_svg("./tests/unit/data/scenarios/minimal_stilman_2005.svg")
        assert w is not None
