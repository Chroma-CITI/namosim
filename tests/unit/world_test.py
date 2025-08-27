import os
from namosim.world.world import World

dirname = os.path.dirname(os.path.abspath(__file__))


def test_load_from_svg():
    world = World.load_from_svg(f"{dirname}/../scenarios/minimal_stilman_2005.svg")
    assert len(world.agents) == 1
    assert "robot_0" in world.agents
    assert world.agents["robot_0"].is_initialized


def test_load_from_config():
    w = World.load_from_yaml(f"{dirname}/../scenarios/citi_ing/namo.yaml")
    assert len(w.agents) == 1
    assert len(w.get_movable_obstacles()) == 2
