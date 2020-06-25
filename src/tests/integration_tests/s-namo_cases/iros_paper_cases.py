import unittest
from src.simulator import Simulator


class BasicWithOpeningTest(unittest.TestCase):

    def setUp(self):
        self.path_to_folder = "../../../../data/simulations/s-namo_cases/"

    # region BASIC WITH OPENING CASE
    def test_basic_with_opening(self):
        filename = "02_basic_with_opening/stilman_2005_behavior.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()
        # Test should end up with a success

    def test_basic_with_opening_focused_snamo(self):
        filename = "02_basic_with_opening/stilman_2005_behavior_focused_snamo.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()
        # Test should end up with a success
    # endregion

    # region CITI CASE
    def test_citi(self):
        filename = "05_citi_second_floor/stilman_2005_behavior.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()

    def test_citi_focused_snamo(self):
        filename = "05_citi_second_floor/stilman_2005_behavior_focused_snamo.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()
    # endregion

    # region CROSSING CASE
    def test_crossing_simple(self):
        filename = "03_crossing/stilman_2005_behavior.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()
        # Test should end up with a success

    def test_crossing_simple_focused_snamo(self):
        filename = "03_crossing/stilman_2005_behavior_focused_snamo.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()
        # Test should end up with a success

    def test_crossing_multigoal_once(self):
        filename = "03_crossing/stilman_2005_behavior_multigoal.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()
        # Test should end up with a success

    def test_crossing_multigoal_once_focused_snamo(self):
        filename = "03_crossing/stilman_2005_behavior_multigoal_focused_snamo.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()
        # Test should end up with a success

    # def test_crossing_multigoal_random(self):
    #     filename = "03_crossing/stilman_2005_behavior_multigoal_random.yaml"
    #     sim = Simulator(simulation_file_path=self.path_to_folder + filename)
    #     sim.run()
    #     # Test should end up with a success

    # def test_crossing_multigoal_random_focused_snamo(self):
    #     filename = "03_crossing/stilman_2005_behavior_multigoal_random_focused_snamo.yaml"
    #     sim = Simulator(simulation_file_path=self.path_to_folder + filename)
    #     sim.run()
    #     # Test should end up with a success
    # endregion

    # region AFTER THE FEAST CASE
    def test_after_the_feast(self):
        filename = "04_after_the_feast/stilman_2005_behavior.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()
        # Test should end up with a success

    def test_after_the_feast_focused_snamo(self):
        filename = "04_after_the_feast/stilman_2005_behavior_focused_snamo.yaml"
        sim = Simulator(simulation_file_path=self.path_to_folder + filename)
        sim.run()
        # Test should end up with a success

    # endregion

    def test_multiple_times(self, nb_times=1):
        for i in range(nb_times):
            filename = "02_basic_with_opening/stilman_2005_behavior.yaml"
            sim = Simulator(simulation_file_path=self.path_to_folder + filename)
            sim.run()

            filename = "02_basic_with_opening/stilman_2005_behavior_focused_snamo.yaml"
            sim = Simulator(simulation_file_path=self.path_to_folder + filename)
            sim.run()

            filename = "05_citi_second_floor/stilman_2005_behavior.yaml"
            sim = Simulator(simulation_file_path=self.path_to_folder + filename)
            sim.run()

            filename = "05_citi_second_floor/stilman_2005_behavior_focused_snamo.yaml"
            sim = Simulator(simulation_file_path=self.path_to_folder + filename)
            sim.run()

            filename = "03_crossing/stilman_2005_behavior_multigoal.yaml"
            sim = Simulator(simulation_file_path=self.path_to_folder + filename)
            sim.run()

            filename = "03_crossing/stilman_2005_behavior_multigoal_focused_snamo.yaml"
            sim = Simulator(simulation_file_path=self.path_to_folder + filename)
            sim.run()

            filename = "04_after_the_feast/stilman_2005_behavior.yaml"
            sim = Simulator(simulation_file_path=self.path_to_folder + filename)
            sim.run()

            filename = "04_after_the_feast/stilman_2005_behavior_focused_snamo.yaml"
            sim = Simulator(simulation_file_path=self.path_to_folder + filename)
            sim.run()


if __name__ == '__main__':
    unittest.main()
