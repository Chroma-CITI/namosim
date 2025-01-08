import os
import unittest

from namosim.world.world import World
from namosim.data_models import namo_config_from_yaml


class TestWorld:
    def test_load_from_svg(self):
        world = World.load_from_svg("tests/scenarios/minimal_stilman_2005.svg")
        assert len(world.agents) == 1
        assert "robot_0" in world.agents
        assert world.agents["robot_0"].is_initialized

    def test_load_from_config(self):
        w = World.load_from_yaml("tests/scenarios/citi_ing/namo.yaml")
        assert len(w.agents) == 1
        assert len(w.get_movable_obstacles()) == 2


if __name__ == "__main__":
    unittest.main()
