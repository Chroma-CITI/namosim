import unittest
import os
from src.simulator import Simulator
from src.behaviors.stilman_2005_behavior import Stilman2005Behavior
from src.utils import utils

class Stilman2005BehaviorTest(unittest.TestCase):

    def setUp(self):
        self.sim = Simulator(
            simulation_file_path="../../../data/simulations/first_level/01_two_rooms_corridor/stilman_2005_behavior.yaml")
        self.robot_uid, self.behavior = next(iter(self.sim.agent_uid_to_behavior.items()))

    def test_manip_search(self):
        ref_world = self.sim.ref_world
        test_obstacle_uid = ref_world.get_entity_uid_from_name("movable_box")
        connected_components_grid = ref_world.get_connected_components_grid((self.robot_uid,))
        connected_components = ref_world.get_connected_components((self.robot_uid,))
        goal_cell = utils.real_to_grid(self.behavior.)

        self.behavior._manip_search(ref_world, test_obstacle_uid, )


if __name__ == '__main__':
    unittest.main()
