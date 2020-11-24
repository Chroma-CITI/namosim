import unittest
from snamosim.simulator import Simulator
from snamosim.utils import utils
from snamosim.display.ros_publisher import RosPublisher
import copy


class Stilman2005BehaviorTest(unittest.TestCase):

    def setUp(self):
        self.sim = Simulator("../../../data/simulations/first_level/01_two_rooms_corridor/stilman_2005_behavior.yaml")
        self.robot_uid, self.behavior = next(iter(self.sim.agent_uid_to_behavior.items()))
        self._robot_name = self.behavior._robot_name
        self._rp = RosPublisher()

    def test_manip_search(self):
        ref_world = self.sim.ref_world
        self._rp.publish_robot_world(ref_world, self.robot_uid, ns=self.behavior.robot_name)
        test_obstacle_uid = ref_world.get_entity_uid_from_name("movable_box")
        connected_components_grid = ref_world.get_connected_components_grid((self.robot_uid,))
        connected_components_grid_costmap = connected_components_grid.get_grid()
        connected_components = connected_components_grid.get_components()
        goal_cell = utils.real_to_grid(self.behavior._navigation_goals[0][0],
                                       self.behavior._navigation_goals[0][1],
                                       ref_world.dd.res,
                                       ref_world.dd.grid_pose)
        goal_cell_component_id = connected_components_grid_costmap[goal_cell[0]][goal_cell[1]]
        goal_cell_component_cells = copy.deepcopy(connected_components[goal_cell_component_id])
        r_f = self.behavior._navigation_goals.pop(0)

        w_t_plus_2, tho_n, tho_m, cost = self.behavior.manip_search(
            ref_world, test_obstacle_uid, goal_cell_component_cells, r_f)

        self._rp.publish_c_1(tho_n, ns=self._robot_name)
        self._rp.publish_c_2(tho_m, ns=self._robot_name)
        self._rp.publish_robot_world(w_t_plus_2, self.robot_uid, ns=self.behavior.robot_name)
        print("Total Cost of tho_n and tho_m = " + str(cost))

    def test_rch(self):
        ref_world = self.sim.ref_world
        self._rp.publish_robot_world(ref_world, self.robot_uid, ns=self.behavior.robot_name)
        r_f = self.behavior._navigation_goals[0]
        o_1, c_1 = self.behavior.rch(ref_world, set(), set(), r_f)
        print(
            "First obstacle in the path is <{o_name}>, and the first disconnected free-space is {cid}".format(
                o_name=ref_world.entities[o_1].name, cid=c_1))
        self.assertEqual("movable_box", ref_world.entities[o_1].name)
        self.assertEqual(2, c_1)

    def test_select_connect(self):
        ref_world = self.sim.ref_world
        self._rp.publish_robot_world(ref_world, self.robot_uid, ns=self.behavior.robot_name)
        r_f = self.behavior._navigation_goals[0]
        plan = self.behavior.select_connect(ref_world, set(), r_f)
        print("")


if __name__ == '__main__':
    unittest.main()
