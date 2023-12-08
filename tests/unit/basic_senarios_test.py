import os
import unittest

from namosim.simulator import Simulator


class BasicTest(unittest.TestCase):
    def setUp(self):
        self.scenarios_folder = os.path.join(__file__, "../data/scenarios")

    def test_minimal_stilman_2005(self):
        """Tests a minimal scenario with Stilman-20005 behavior"""
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "minimal_stilman_2005.svg"
            )
        )
        sim.run()
        assert (
            sim.simulation_log[7].message
            == "Agent robot_0 finished executing all its goals."
        )

    def test_minimal_stilman_only(self):
        """Tests a minimal scenario with Stilman-only behavior"""
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "minimal_stilman_only.svg"
            )
        )
        sim.run()
        assert (
            sim.simulation_log[7].message
            == "Agent robot_0 finished executing all its goals."
        )

    def test_minimal_nav_only(self):
        """Tests a minimal scenario with navigation-only behavior"""
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "minimal_nav_only.svg"
            )
        )
        sim.run()
        assert (
            sim.simulation_log[7].message
            == "Agent robot_0 finished executing all its goals."
        )

    def test_1_robot_2_goals(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "1_robot_2_goals.svg"
            )
        )
        sim.run()
        assert (
            sim.simulation_log[8].message
            == "Agent robot_0 finished executing all its goals."
        )

    def test_1_robot_2_obstacles(self):
        sim = Simulator(
            simulation_file_path=os.path.join(
                self.scenarios_folder, "stilman_only_1_robot_2_obstacles.svg"
            )
        )
        sim.run()
        assert (
            sim.simulation_log[7].message
            == "Agent robot_0 finished executing all its goals."
        )

    def test_custom(self):
        sim = Simulator(
            simulation_file_path=os.path.join(self.scenarios_folder, "custom.svg")
        )
        sim.run()
        assert (
            sim.simulation_log[7].message
            == "Agent robot_0 finished executing all its goals."
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


if __name__ == "__main__":
    unittest.main()
