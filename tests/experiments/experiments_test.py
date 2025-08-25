import os
import time
import unittest

import namosim.config as config
from namosim.simulator import create_sim_from_file


class TestExperiments:
    def setup_method(self):
        self.scenarios_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scenarios"
        )

    def test_3_robots(self):
        config.DISPLAY_WINDOW = True
        sim_parallel = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/3_robots.svg"
            )
        )

        start_time = time.perf_counter()
        sim_parallel.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_two_rooms(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/two_rooms.svg"
            )
        )
        sim.run()
        assert True

    def test_overlapping_movables(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/overlapping_movables.svg"
            )
        )
        sim.run()
        assert True

    def test_intersections_base(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/intersections_base.svg",
            )
        )
        sim.run()
        assert True

    def test_intersections_1_robots_50_goals_namo(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/1_robot_50_goals_namo.svg",
            )
        )
        sim.run()
        assert True

    def test_intersections_1_robots_50_goals_snamo(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/1_robot_50_goals_snamo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_intersections_2_robots_50_goals_snamo(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/2_robots_50_goals_snamo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_intersections_2_robots_50_goals_namo(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/2_robots_50_goals_namo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_intersections_generated(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/generated/2_robots_50_goals_snamo_distance_dr/11.svg",
            )
        )
        sim.run()
        assert True

    def test_willow_garage_generated(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "willow_garage/generated/2_robots_50_goals_namo/00.svg",
            )
        )
        sim.run()
        assert True

    def test_intersections_4_robots_25_goals_snamo(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "intersections/4_robots_25_goals_snamo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_citi_lab_2_robots_50_goals_snamo(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "citi_lab/citi_lab_2_robots_50_goals_snamo.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_citi_lab_generated(self):
        config.DISPLAY_WINDOW = True
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "citi_lab/generated/1_robots_50_goals_snamo/06.svg",
            )
        )

        start_time = time.perf_counter()
        sim.run()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")
        assert True

    def test_teleop(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "teleop.svg",
            )
        )
        sim.run()
        assert True

    def test_willow_garage_center_small(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "willow_garage_center_small.svg",
            )
        )
        sim.run()
        assert True

    def test_willow_garage_multi_shape(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "willow_garage_multi_shape.svg",
            )
        )
        sim.run()
        assert True

    def test_mapgen(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(self.scenarios_folder, "mapgen.svg")
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )


if __name__ == "__main__":
    unittest.main()
