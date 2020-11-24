import unittest
from snamosim.worldreps.entity_based.world import World
from snamosim.worldreps.entity_based.obstacle import Obstacle
from snamosim.worldreps.entity_based.robot import Robot
from snamosim.display.ros_publisher import RosPublisher
from snamosim.worldreps.occupation_based.binary_occupancy_grid import BinaryOccupancyGrid
from snamosim.worldreps.occupation_based.social_topological_occupation_cost_grid import SocialTopologicalOccupationCostGrid, voronoi_skeleton


class Stilman2005BehaviorTest(unittest.TestCase):

    def setUp(self):
        self._rp = RosPublisher()

    def compute_and_display_costmap(self, world):
        robot = world.entities[world.get_entity_uid_from_name("robot_01")]
        self._rp.cleanup_sim_world()
        self._rp.publish_sim_world(world, robot.uid)
        movable_entities_uids = tuple({entity_uid for entity_uid, entity in world.entities.items()
                                       if isinstance(entity, Robot) or (isinstance(entity, Obstacle)
                                       and robot.deduce_movability(entity.type) == "movable")})
        occ_grid = BinaryOccupancyGrid(
            world.dd.d_width, world.dd.d_height, world.dd.res, world.dd.grid_pose,
            world.dd.inflation_radius, world.entities, entities_to_ignore=movable_entities_uids)
        social_costmap = SocialTopologicalOccupationCostGrid.from_binary_occ_grid(occ_grid, ns='simulation')
        self._rp.publish_grid_map(social_costmap.get_grid(), world.dd.res, ns=robot.name)

    def test_two_rooms_corridor(self):
        world = World.load_from_yaml("../../../data/worlds/first_level/01_two_rooms_corridor/01_two_rooms_corridor.yaml")
        self.compute_and_display_costmap(world)

    def test_big_crossing(self):
        world = World.load_from_yaml("../../../data/worlds/first_level/03_big_crossing/03_big_crossing.yaml")
        self.compute_and_display_costmap(world)

    def test_moghaddam_01(self):
        world = World.load_from_yaml("../../../data/worlds/moghaddam_planning_2016_benchmark/01/01.yaml")
        self.compute_and_display_costmap(world)

    def test_chen_difficult_problem(self):
        world = World.load_from_yaml("../../../data/worlds/stilman_2005_thesis/04_chen_difficult_problem/04_chen_difficult_problem.yaml")
        self.compute_and_display_costmap(world)

    def test_basic(self):
        world = World.load_from_yaml("../../../data/worlds/s-namo_cases/01_basic/01_basic.yaml")
        self.compute_and_display_costmap(world)

    def test_basic_with_opening(self):
        world = World.load_from_yaml("../../../data/worlds/s-namo_cases/02_basic_with_opening/02_basic_with_opening.yaml")
        self.compute_and_display_costmap(world)

    def test_crossing(self):
        world = World.load_from_yaml("../../../data/worlds/s-namo_cases/03_crossing/03_crossing.yaml")
        self.compute_and_display_costmap(world)

    def test_after_the_feast(self):
        world = World.load_from_yaml("../../../data/worlds/s-namo_cases/04_after_the_feast/04_after_the_feast.yaml")
        self.compute_and_display_costmap(world)

    def test_citi_second_floor(self):
        world = World.load_from_yaml("../../../data/worlds/s-namo_cases/05_citi_second_floor/05_citi_second_floor.yaml")
        self.compute_and_display_costmap(world)


if __name__ == '__main__':
    unittest.main()
