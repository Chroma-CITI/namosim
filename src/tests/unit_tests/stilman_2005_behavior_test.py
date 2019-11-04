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
        goal_cell = utils.real_to_grid(self.behavior._navigation_goals[0][0],
                                       self.behavior._navigation_goals[0][1],
                                       ref_world.dd.res,
                                       ref_world.dd.grid_pose)
        goal_cell_component_id = connected_components_grid[goal_cell[0]][goal_cell[1]]
        goal_cell_component_cells = connected_components[goal_cell_component_id]

        self.behavior._manip_search(ref_world, test_obstacle_uid, goal_cell_component_cells)


if __name__ == '__main__':
    unittest.main()
