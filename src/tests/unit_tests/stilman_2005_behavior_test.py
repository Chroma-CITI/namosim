import unittest
from src.simulator import Simulator
from src.utils import utils
from src.display.ros_publisher import RosPublisher


class Stilman2005BehaviorTest(unittest.TestCase):

    def setUp(self):
        self.sim = Simulator(
            simulation_file_path="../../../data/simulations/first_level/01_two_rooms_corridor/stilman_2005_behavior.yaml")
        self.robot_uid, self.behavior = next(iter(self.sim.agent_uid_to_behavior.items()))
        self._rp = RosPublisher()

    def test_manip_search(self):
        ref_world = self.sim.ref_world
        self._rp.publish_robot_world(ref_world, self.robot_uid)
        test_obstacle_uid = ref_world.get_entity_uid_from_name("movable_box")
        connected_components_grid = ref_world.get_connected_components_grid((self.robot_uid,))
        connected_components_grid_costmap = connected_components_grid.grid
        connected_components = connected_components_grid.components
        goal_cell = utils.real_to_grid(self.behavior._navigation_goals[0][0],
                                       self.behavior._navigation_goals[0][1],
                                       ref_world.dd.res,
                                       ref_world.dd.grid_pose)
        goal_cell_component_id = connected_components_grid_costmap[goal_cell[0]][goal_cell[1]]
        goal_cell_component_cells = connected_components[goal_cell_component_id]
        self.behavior._q_goal = self.behavior._navigation_goals.pop(0)
        w_t_plus_2, tho_n, tho_m, cost = self.behavior._manip_search(
            ref_world, test_obstacle_uid, goal_cell_component_cells)
        self._rp.publish_c_1(tho_n)
        self._rp.publish_c_2(tho_m)
        self._rp.publish_robot_world(w_t_plus_2, self.robot_uid)
        print("Total Cost of tho_n and tho_m = ", cost)


if __name__ == '__main__':
    unittest.main()
