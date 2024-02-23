import os
import unittest

from namosim.simulator import Simulator


class BasicTest(unittest.TestCase):
    def setUp(self):
        self.scenarios_folder = os.path.join(__file__, "../scenarios")

    def test_minimal_stilman_2005(self):
        """Tests a minimal scenario with Stilman-20005 behavior"""
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "minimal_stilman_2005.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_minimal_stilman_only(self):
        """Tests a minimal scenario with Stilman-only behavior"""
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "minimal_stilman_only.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_minimal_nav_only(self):
        """Tests a minimal scenario with navigation-only behavior"""
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "minimal_nav_only.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_1_robot_2_goals(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "1_robot_2_goals.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_stilman_2005_1_robot_2_obstacles(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "stilman_2005_1_robot_2_obstacles.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 finished executing all its goals.")
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_1_robot_2_obstacles(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "stilman_only_1_robot_2_obstacles.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 finished executing all its goals.")
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_1_robot_2_obstacles_social(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "stilman_2005_1_robot_2_obstacles.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 finished executing all its goals.")
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_repulsive_dr_fail_b(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "repulsive_dr_fail_b.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message
                == "Agent robot_1: Failing goal, no tries remaining to plan an evasion."
                for x in sim.simulation_log
            ]
        )

    def test_social_dr_success_a(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "social_dr_success_a.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_social_dr_success_d(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "social_dr_success_d.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_repulsive_dr_fail_c(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "repulsive_dr_fail_c.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message
                == "Agent robot_1: Failing goal, no tries remaining to plan an evasion."
                for x in sim.simulation_log
            ]
        )

    def test_custom(self):
        sim = Simulator(
            simulation_file_path=os.path.join(self.scenarios_folder, "custom.svg")
        )
        sim.run()
        assert any(
            [
                x.message == "Agent robot_0 finished executing all its goals."
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_multi_robot(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "multi_robot/multi_robot.svg"
            )
        )
        sim.run()

        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_stealing_movable_conflict(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "stealing_movable.svg",
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )
        assert any(
            ["Stealing Movable conflict" in x.message for x in sim.simulation_log]
        )

    def test_obstacle_on_goal(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder,
                "obstacle_on_goal.svg",
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_evasion(self):
        sim = Simulator(
            simulation_file_path=os.path.join(self.scenarios_folder, "evasion.svg")
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.simulation_log
            ]
        )

    def test_evasion_nonsocial(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "evasion_nonsocial.svg"
            )
        )
        sim.run()
        assert any(
            [
                x.message.startswith("Agent robot_0 successfully executed goal")
                for x in sim.simulation_log
            ]
        )
        assert any(
            [
                x.message.startswith("Agent robot_1 successfully executed goal")
                for x in sim.simulation_log
            ]
        )


if __name__ == "__main__":
    unittest.main()
