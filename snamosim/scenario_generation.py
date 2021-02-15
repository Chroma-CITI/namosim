import json
import os
from xml.dom import minidom


def scenarios_from_simulation_results(scenario_original_filepath, scenario_logs_dir_filepath,
                                      temp_simulations_dir_filepath, temp_worlds_dir_filepath):
    # Get data from original files
    with open(scenario_original_filepath) as f:
        scenario_data = json.load(f)

    world_file_path = os.path.join(os.path.dirname(scenario_original_filepath), scenario_data["files"]["world_file"])
    with open(world_file_path) as f:
        world_data = json.load(f)

    geometry_file_path = os.path.join(os.path.dirname(world_file_path), world_data["files"]["geometry_file"])
    with open(geometry_file_path) as f:
        geometry_data = minidom.parse(f)

    logged_scenarios_ids = {
        name for name in os.listdir(scenario_logs_dir_filepath)
        if os.path.isdir(os.path.join(scenario_logs_dir_filepath, name))
    }

    for scenario_id in logged_scenarios_ids:
        sim_results_path = os.path.join(scenario_logs_dir_filepath, scenario_id, "sim_results.json")

        simulation_filepath = os.path.join(temp_simulations_dir_filepath, scenario_id + "/", os.path.basename(scenario_original_filepath))
        world_json_filepath = os.path.join(temp_worlds_dir_filepath, scenario_id + "/", os.path.basename(world_file_path))
        world_svg_filepath = os.path.join(temp_worlds_dir_filepath, scenario_id + "/", os.path.basename(geometry_file_path))

        try:
            with open(sim_results_path) as f:
                sim_results_data = json.load(f)

            for agent_data in sim_results_data["agents"]:
                agent_index = None
                for agent_counter, behavior_data in enumerate(scenario_data["agents_behaviors"]):
                    if behavior_data["agent_name"] == agent_data["agent_name"]:
                        agent_index = agent_counter

                if agent_index is None:
                    continue

                if "randomization" in scenario_data["agents_behaviors"][agent_index]["behavior"]:
                    del scenario_data["agents_behaviors"][agent_index]["behavior"]["randomization"]
                scenario_data["agents_behaviors"][agent_index]["behavior"]["navigation_goals"] = []

                world_data["things"]["zones"]["goals"] = []

                for counter, goal_report in enumerate(agent_data["goals_reports"]):
                    goal_pose = goal_report["goal"]
                    goal_name = "goal_" + str(counter)

                    world_data["things"]["zones"]["goals"].append(
                        {"name": goal_name, "pose": goal_pose}
                    )

                    scenario_data["agents_behaviors"][agent_index]["behavior"]["navigation_goals"].append({"name": goal_name})

            # TODO Udpate filepath data for world svg in world json, and world json in simulation json
            scenario_data["files"]["world_file"] = os.path.join(
                os.path.relpath(os.path.dirname(world_json_filepath), os.path.dirname(simulation_filepath)),
                os.path.basename(world_json_filepath)
            )
            world_data["files"]["geometry_file"] = os.path.join(
                os.path.relpath(os.path.dirname(world_svg_filepath), os.path.dirname(world_json_filepath)),
                os.path.basename(world_svg_filepath)
            )

            if not os.path.exists(os.path.dirname(simulation_filepath)):
                os.makedirs(os.path.dirname(simulation_filepath))
            if not os.path.exists(os.path.dirname(world_json_filepath)):
                os.makedirs(os.path.dirname(world_json_filepath))

            with open(simulation_filepath, "w") as f:
                json.dump(scenario_data, f)
            with open(world_json_filepath, "w") as f:
                json.dump(world_data, f)
            with open(world_svg_filepath, "w") as f:
                geometry_data.writexml(f)
        except (IOError, ValueError) as e:
            continue


if __name__ == '__main__':
    scenarios_from_simulation_results(
        scenario_original_filepath=os.path.join(
            os.path.dirname(__file__),
            "../data/simulations/s-namo_cases/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset.json"
        ),
        scenario_logs_dir_filepath=os.path.join(
            os.path.dirname(__file__),
            "../logs/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset/"
        ),
        temp_simulations_dir_filepath=os.path.join(
            os.path.dirname(__file__),
            "../tmp/simulations/s-namo_cases/04_after_the_feast/variations-stilman_2005_behavior_complexified_random_goal_no_reset/"
        ),
        temp_worlds_dir_filepath=os.path.join(
            os.path.dirname(__file__),
            "../tmp/worlds/s-namo_cases/04_after_the_feast/variations-stilman_2005_behavior_complexified_random_goal_no_reset/"
        )
    )
    scenarios_from_simulation_results(
        scenario_original_filepath=os.path.join(
            os.path.dirname(__file__),
            "../data/simulations/s-namo_cases/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset_snamo.json"
        ),
        scenario_logs_dir_filepath=os.path.join(
            os.path.dirname(__file__),
            "../logs/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset_snamo/"
        ),
        temp_simulations_dir_filepath=os.path.join(
            os.path.dirname(__file__),
            "../tmp/simulations/s-namo_cases/04_after_the_feast/variations-stilman_2005_behavior_complexified_random_goal_no_reset_snamo/"
        ),
        temp_worlds_dir_filepath=os.path.join(
            os.path.dirname(__file__),
            "../tmp/worlds/s-namo_cases/04_after_the_feast/variations-stilman_2005_behavior_complexified_random_goal_no_reset_snamo/"
        )
    )
