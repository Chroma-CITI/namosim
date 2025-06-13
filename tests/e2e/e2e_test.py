import os
import unittest

from namosim.simulator import create_sim_from_file
import cProfile


class TestE2E:
    def setup_method(self):
        self.scenarios_folder = os.path.join(__file__, "../../scenarios")

    def test_minimal_stilman_2005(self):
        """Tests a minimal scenario with Stilman-20005 behavior"""
        # profiler = cProfile.Profile()
        # profiler.enable()
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "minimal_stilman_2005.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )
        # profiler.disable()
        # profiler.dump_stats(file="stats.prof")

    def test_stilman_rrt_star(self):
        """Tests Stilman-20005 behavior with RRT* navigation"""
        sim = create_sim_from_file(
            simulation_file_path=os.path.abspath(
                os.path.join(self.scenarios_folder, "minimal_stilman_rrt_star.svg")
            )
        )
        assert (
            sim is not None
        ), "Simulation could not be created. Check the scenario file."
        sim.run()
        for log in sim.logger:
            print(log.message)  # Log all messages for debugging
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.logger
            ]
        ), "Agent robot_0 did not finish executing all its goals."
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        ), "Agent robot_0 did not successfully execute any goal."

    def test_minimal_nav_only(self):
        """Tests a minimal scenario with navigation-only behavior"""
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "minimal_nav_only.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_rrt_star_nav_only(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(self.scenarios_folder, "rrt.svg")
        )
        sim.run()
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_1_robot_2_goals(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "1_robot_2_goals.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_1_robot_2_obstacles(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "1_robot_2_obstacles.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 finished executing all its goals.")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_1_robot_2_rooms(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "1_robot_2_rooms.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 finished executing all its goals.")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_1_robot_2_obstacles_social(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "1_robot_2_obstacles_social.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 finished executing all its goals.")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_1_robot_2_obstacles_social_rrt_star(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "1_robot_2_obstacles_social_rrt_star.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 finished executing all its goals.")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_repulsive_dr_fail_b(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "repulsive_dr_fail_b.svg"
            )
        )
        sim.run()
        for x in sim.logger:
            print(x.message)
        assert any(["Agent robot_1: Failing goal" in x.message for x in sim.logger])

    def test_social_dr_success_a(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "social_dr_success_a.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_social_dr_success_d(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "social_dr_success_d.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_repulsive_dr_fail_c(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "repulsive_dr_fail_c.svg"
            )
        )
        sim.run()
        assert any(["Agent robot_1: Failing goal" in x.message for x in sim.logger])

    def test_multi_robot(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/multi_robot.svg"
            )
        )
        sim.run()

        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_stealing_movable_conflict(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "stealing_movable.svg",
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.logger
            ]
        )
        assert any(["StealingMovableConflict" in x.message for x in sim.logger])

    def test_obstacle_on_goal(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "obstacle_on_goal.svg",
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_evasion(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(self.scenarios_folder, "evasion.svg")
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_evasion_nonsocial(self):
        sim = create_sim_from_file(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "evasion_nonsocial.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_citi_ing_yaml(self):
        sim = create_sim_from_file(
            simulation_file_path="tests/scenarios/citi_ing/namo.yaml"
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_citi_ing_push_only(self):
        sim = create_sim_from_file(
            simulation_file_path="tests/scenarios/citi_ing_push_only/namo.yaml"
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_citi_full(self):
        sim = create_sim_from_file(
            simulation_file_path="tests/scenarios/citi_full/citi_full_namo.yaml"
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_namoros_demo(self):
        sim = create_sim_from_file(
            simulation_file_path="tests/scenarios/namoros_demo_map.svg"
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_robot_starts_in_collision(self):
        sim = create_sim_from_file(
            simulation_file_path="tests/scenarios/robot_starts_in_collision.svg"
        )
        sim.ref_world.resolve_collisions("robot_0")
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_space_conflict(self):
        sim = create_sim_from_file(
            simulation_file_path="tests/scenarios/space_conflict.svg"
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.logger
            ]
        )

    def test_three_robots(self):
        sim = create_sim_from_file(
            simulation_file_path="tests/scenarios/three_robots.svg"
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.logger
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_2 successfully executed goal")
                for x in sim.logger
            ]
        )


if __name__ == "__main__":
    unittest.main()
