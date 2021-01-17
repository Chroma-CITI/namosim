import unittest
from snamosim.simulator import Simulator
import os


class AfterTheFeastTest(unittest.TestCase):

    def setUp(self):
        self.path_to_folder = os.path.join(__file__, "../../../../../data/simulations/s-namo_cases/04_after_the_feast/")

    # def test_navigation_only_behavior(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"navigation_only_behavior.yaml")
    #     report = sim.run()
    #
    # def test_navigation_only_behavior_no_movables(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"navigation_only_behavior_no_movables.yaml")
    #     sim.run()
    #     # Test should end up with a success
    #
    # def test_navigation_only_behavior_no_movables_multi_robots(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"navigation_only_behavior_no_movables_multi_robots.yaml")
    #     sim.run()
    #     # Test should end up with a success
    #
    # def test_wu_levihn_2014_behavior(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"wu_levihn_2014.yaml")
    #     sim.run()
    #     # Test should end up with a success
    #
    # def test_wu_levihn_2014_behavior_no_movables(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder+"wu_levihn_2014_no_movables.yaml")
    #     sim.run()
    #     # Test should end up with a success

    def test_stilman_2005_behavior(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"stilman_2005_behavior.yaml")
        sim.run()
        # Test should end up with a success

    def test_stilman_2005_behavior_no_movables(self):
        sim = Simulator(simulation_file_path=self.path_to_folder+"stilman_2005_behavior_no_movables.yaml")
        sim.run()
        # Test should end up with a success

    def test_stilman_2005_behavior_complexified(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified.yaml")
        sim.run()

    def test_stilman_2005_behavior_complexified_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_snamo.yaml")
        sim.run()

    def test_stilman_2005_behavior_complexified_snamo_r2g2(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_snamo_r2g2.yaml")
        sim.run()

    # def test_stilman_2005_behavior_complexified_debug_case_after_16_goals(self):
    #     sim = Simulator(simulation_file_path=self.path_to_folder + "/debug/after_16_goals_exception/stilman_2005_behavior_complexified_random_goal_no_reset_snamo.yaml")
    #     sim.run()

    def test_stilman_2005_behavior_multi_robots(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots.yaml")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified.yaml")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_snamo.yaml")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_snamo.yaml")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_reset(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_reset.yaml")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_reset_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_reset_snamo.yaml")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_no_reset(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset.yaml")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_no_reset_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset_snamo.yaml")
        sim.run()

    def test_stilman_2005_behavior_complexified_random_goal_no_reset_namo_simple_then_snamo(self):
        sim_namo = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset.yaml")
        sim_namo_report = sim_namo.run()
        sim_snamo = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_complexified_random_goal_no_reset_snamo.yaml", goals=sim_namo_report["temp_goals"])
        sim_snamo_report = sim_snamo.run()

    def test_stilman_2005_behavior_multi_robots_complexified_random_goal_reset(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_random_goal_reset.yaml")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_random_goal_no_reset(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_random_goal_no_reset.yaml")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_random_goal_reset_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_random_goal_reset_snamo.yaml")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_random_goal_no_reset_snamo(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_random_goal_no_reset_snamo.yaml")
        sim.run()

    def test_stilman_2005_behavior_multi_robots_complexified_conflict_middle(self):
        sim = Simulator(simulation_file_path=self.path_to_folder + "stilman_2005_behavior_multi_robots_complexified_conflict_middle.yaml")
        sim.run()

if __name__ == '__main__':
    unittest.main()
