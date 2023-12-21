import copy
import json
import os
import random
import typing as t
from xml.dom import minidom

from shapely import Polygon

from namosim.data_models import PoseModel
from namosim.utils import collision, conversion, utils
from namosim.world.binary_occupancy_grid import (
    BinaryInflatedOccupancyGrid,
    BinaryOccupancyGrid,
)


def get_map_bounds(polygons: t.Dict[int, Polygon]):
    map_min_x, map_min_y, map_max_x, map_max_y = (
        float("inf"),
        float("inf"),
        -float("inf"),
        -float("inf"),
    )
    for polygon in polygons.values():
        min_x, min_y, max_x, max_y = polygon.bounds
        map_min_x, map_min_y = min(map_min_x, min_x), min(map_min_y, min_y)
        map_max_x, map_max_y = max(map_max_x, max_x), max(map_max_y, max_y)
    return map_min_x, map_min_y, map_max_x, map_max_y


def sample_poses_uniform(
    obstacles_polygons: t.Dict[int, Polygon],
    robot_polygon: Polygon,
    robot_pose: PoseModel,
    nb_poses: int = 1,
    grid: BinaryOccupancyGrid | None = None,
    no_collisions_between_poses: bool = False,
):
    # Make AABB Tree from polygons
    aabb_tree = collision.polygons_to_aabb_tree(obstacles_polygons)

    # Compute map bounds
    map_min_x, map_min_y, map_max_x, map_max_y = get_map_bounds(obstacles_polygons)

    generated_poses = []

    if no_collisions_between_poses:
        generated_polygons = []

    while len(generated_poses) < nb_poses:
        rand_pose = (
            random.uniform(map_min_x, map_max_x),
            random.uniform(map_min_y, map_max_y),
            random.uniform(0.0, 360.0),
        )
        if grid:
            rand_cell = utils.real_to_grid(
                rand_pose[0], rand_pose[1], grid.res, grid.grid_pose
            )
            if grid.grid[rand_cell[0]][rand_cell[1]] != 0:
                continue
        robot_polygon_at_rand_pose = utils.set_polygon_pose(
            robot_polygon, robot_pose, rand_pose
        )
        robot_aabb_at_rand_pose = collision.polygon_to_aabb(robot_polygon_at_rand_pose)
        potential_collision_uids = aabb_tree.overlap_values(robot_aabb_at_rand_pose)
        pose_collides = False
        for uid in potential_collision_uids:
            if obstacles_polygons[uid].intersects(robot_polygon_at_rand_pose):
                pose_collides = True
                break

        if no_collisions_between_poses:
            for polygon in generated_polygons:
                if polygon.intersects(robot_polygon_at_rand_pose):
                    pose_collides = True
                    break

        if not pose_collides:
            generated_poses.append(rand_pose)
            if no_collisions_between_poses:
                generated_polygons.append(robot_polygon_at_rand_pose)
    return generated_poses


def infer_type_from_uid(obstacle_uid):
    potential_types = ["table", "stool", "box", "wall", "chair"]
    for potential_type in potential_types:
        if potential_type in obstacle_uid:
            return potential_type
    return "wall"


def generate_scenarios_alternatives(
    base_svg_filepath, nb_robots, nb_goals_per_robot, grid_res, nb_scenarios
):
    # Load SVGs
    svg_filepath = os.path.join(os.path.dirname(__file__), base_svg_filepath)
    svg_data_init = minidom.parse(svg_filepath)
    conversion.set_all_id_attributes_as_ids(svg_data_init)

    svg_base_elements_filepath = os.path.join(
        os.path.dirname(__file__), "../data/simulations/iros_2021/common_elements.svg"
    )
    svg_base_elements_data = minidom.parse(svg_base_elements_filepath)
    conversion.set_all_id_attributes_as_ids(svg_base_elements_data)
    svg_base_robot_shape = svg_base_elements_data.getElementById(
        "base_circular_like_robot_shape"
    )
    svg_base_robot_direction = svg_base_elements_data.getElementById(
        "base_circular_like_robot_direction"
    )
    svg_base_goal_shape = svg_base_elements_data.getElementById(
        "base_circular_like_robot_goal_shape"
    )
    svg_base_goal_direction = svg_base_elements_data.getElementById(
        "base_circular_like_robot_goal_direction"
    )

    # Convert svg_data paths into polygons
    obstacles_polygons = {
        path.getAttribute("id"): conversion.svg_pathd_to_shapely_geometry(
            path.getAttribute("d"), scaling_value=0.01
        )
        for path in svg_data_init.getElementsByTagName("path")
    }
    base_robot_polygon = conversion.svg_pathd_to_shapely_geometry(
        svg_base_robot_shape.getAttribute("d"), scaling_value=0.01
    )
    base_robot_orientation_polygon = conversion.svg_pathd_to_shapely_geometry(
        svg_base_robot_direction.getAttribute("d"), scaling_value=0.01
    )
    base_robot_orientation_geom_coords = list(base_robot_orientation_polygon.coords)
    base_robot_orientation_vector = (
        base_robot_orientation_geom_coords[1][0]
        - base_robot_orientation_geom_coords[0][0],
        base_robot_orientation_geom_coords[1][1]
        - base_robot_orientation_geom_coords[0][1],
    )
    base_robot_pose = (
        base_robot_polygon.centroid.coords[0][0],
        base_robot_polygon.centroid.coords[0][1],
        utils.yaw_from_direction(base_robot_orientation_vector),
    )

    # Do uniform sampling in coordinates that are within map bounds or load "_samples.json",
    # for initial robot poses (can not be in any obstacles) and goals robot poses (can be in movable obstacles)
    polygons_for_init_poses = {
        uid: p for uid, p in obstacles_polygons.items() if "direction" not in uid
    }
    all_obstacles_grid = BinaryInflatedOccupancyGrid(
        polygons_for_init_poses,
        grid_res,
        utils.get_circumscribed_radius(base_robot_polygon),
    )
    polygons_for_goals_poses = {
        uid: p
        for uid, p in obstacles_polygons.items()
        if not any(
            word in uid for word in ["movable", "box", "chair", "stool", "direction"]
        )
    }
    only_static_obstacles_grid = BinaryInflatedOccupancyGrid(
        polygons_for_goals_poses,
        grid_res,
        utils.get_circumscribed_radius(base_robot_polygon),
    )

    for c_scenario in range(nb_scenarios):
        svg_data = copy.deepcopy(svg_data_init)

        scenario_id = ("{:0" + str(len(str(nb_scenarios))) + "d}").format(c_scenario)

        initial_robot_poses = sample_poses_uniform(
            polygons_for_init_poses,
            base_robot_polygon,
            base_robot_pose,
            nb_poses=nb_robots,
            grid=all_obstacles_grid,
            no_collisions_between_poses=True,
        )
        goals_poses_for_robots = []
        for i in range(nb_robots):
            goals_poses_for_robots.append(
                sample_poses_uniform(
                    polygons_for_goals_poses,
                    base_robot_polygon,
                    base_robot_pose,
                    nb_poses=nb_goals_per_robot,
                    grid=only_static_obstacles_grid,
                )
            )

        # Save sampled coordinates in same folder as svg_filepath with same filename - ".svg" + "_samples.json"
        # TODO

        # Create robot polygons at said poses in svg_data using svg styles of robot and goals and setting unique ids
        goals_group = conversion.add_group(svg_data, "goals")
        for i in range(nb_robots):
            robot_id = "robot_" + str(i)
            robot_group = conversion.add_group(svg_data, robot_id, is_layer=False)
            # Add robot shape
            conversion.add_shapely_geometry_to_svg(
                utils.set_polygon_pose(
                    base_robot_polygon, base_robot_pose, initial_robot_poses[i]
                ),
                robot_id + "_shape",
                svg_base_robot_shape.getAttribute("style"),
                svg_data,
                robot_group,
                scale=0.01,
            )
            # Add robot direction shape
            conversion.add_shapely_geometry_to_svg(
                utils.rotate_then_translate_polygon(
                    base_robot_orientation_polygon,
                    *utils.get_translation_and_rotation(
                        base_robot_pose, initial_robot_poses[i]
                    ),
                    rotation_center=(base_robot_pose[0], base_robot_pose[1]),
                ),
                robot_id + "_direction",
                svg_base_robot_direction.getAttribute("style"),
                svg_data,
                robot_group,
                scale=0.01,
            )
            # Add robot goals
            robot_goals_layer = conversion.add_group(
                svg_data, robot_id + "_goals", goals_group
            )
            for goal_counter, goal_pose in enumerate(goals_poses_for_robots[i]):
                goal_id = robot_id + "_goal_" + str(goal_counter)
                goal_group = conversion.add_group(
                    svg_data, goal_id, robot_goals_layer, is_layer=False
                )
                # Add goal shape
                conversion.add_shapely_geometry_to_svg(
                    utils.set_polygon_pose(
                        base_robot_polygon, base_robot_pose, goal_pose
                    ),
                    goal_id + "_shape",
                    svg_base_goal_shape.getAttribute("style"),
                    svg_data,
                    goal_group,
                    scale=0.01,
                )
                # Add goal direction shape
                conversion.add_shapely_geometry_to_svg(
                    utils.rotate_then_translate_polygon(
                        base_robot_orientation_polygon,
                        *utils.get_translation_and_rotation(base_robot_pose, goal_pose),
                        rotation_center=(base_robot_pose[0], base_robot_pose[1]),
                    ),
                    goal_id + "_direction",
                    svg_base_goal_direction.getAttribute("style"),
                    svg_data,
                    goal_group,
                    scale=0.01,
                )

        # Create SVG file from modified data
        base_svg_dirpath = os.path.dirname(svg_filepath)
        scenario_dirpath = os.path.join(
            base_svg_dirpath,
            str(nb_robots) + "_robots/",
            str(nb_goals_per_robot) + "_goals/",
            scenario_id + "/",
        )
        if not os.path.exists(scenario_dirpath):
            os.makedirs(scenario_dirpath)
        scenario_world_svg_filename = "world_" + scenario_id + ".svg"
        with open(
            os.path.join(scenario_dirpath, scenario_world_svg_filename), "w+"
        ) as f:
            svg_data.writexml(f)

        # Create json file to describe world data
        world_json_data = {
            "discretization_data": {
                "res": 0.1,
            },
            "files": {"geometry_file": "./" + scenario_world_svg_filename},
            "geometry_scale": 0.01,
            "no_scaling_workaround": True,
            "things": {
                "entities": [
                    {
                        "force_pushes_only": True,
                        "geometry": {
                            "from": "file",
                            "id": "robot_" + str(c_robot) + "_shape",
                            "orientation_id": "robot_" + str(c_robot) + "_direction",
                        },
                        "movable_whitelist": ["stool", "box", "chair"],
                        "name": "robot_" + str(c_robot),
                        "push_only_list": ["stool", "box", "chair"],
                        "sensors": [{"type": "omniscient"}],
                        "type": "robot",
                    }
                    for c_robot in range(nb_robots)
                ]
                + [
                    {
                        "geometry": {"from": "file", "id": obstacle_uid},
                        "name": obstacle_uid,
                        "type": infer_type_from_uid(obstacle_uid),
                    }
                    for obstacle_uid in obstacles_polygons.keys()
                ],
                "zones": {
                    "goals": [
                        {
                            "geometry": {
                                "from": "file",
                                "id": "robot_"
                                + str(c_robot)
                                + "_goal_"
                                + str(c_goals)
                                + "_shape",
                                "orientation_id": "robot_"
                                + str(c_robot)
                                + "_goal_"
                                + str(c_goals)
                                + "_direction",
                            },
                            "name": "robot_" + str(c_robot) + "_goal_" + str(c_goals),
                        }
                        for c_robot in range(nb_robots)
                        for c_goals in range(nb_goals_per_robot)
                    ]
                },
            },
        }
        scenario_world_json_filename = "world_" + scenario_id + ".json"
        with open(
            os.path.join(scenario_dirpath, scenario_world_json_filename), "w+"
        ) as f:
            json.dump(world_json_data, f)

        # Create json files to describe simulation data for snamo then namo
        simulation_json_data = {
            "agents_behaviors": [
                {
                    "agent_name": "robot_" + str(c_robot),
                    "behavior": {
                        "name": "stilman_2005_behavior",
                        "parameters": {
                            "alpha_for_obstacle_choice_heur": 0.5,
                            "basic_rotation_moment": 2.0,
                            "basic_translation_force": 2.0,
                            "check_new_local_opening_before_global": True,
                            "collision_check_angular_res": 5.0,
                            "activate_grids_logging": False,
                            "forbid_rotations": False,
                            "heuristic_cost_for_traversing_obstacle_in_choice_heur": 2.0,
                            "neighborhood_for_obstacle_choice_heur": "TAXI",
                            "robot_rotation_unit_angle": 30.0,
                            "robot_translation_unit_length": 0.1,
                            "solution_interval_bound_percentage": 0.01,
                        },
                        "navigation_goals": [
                            {"name": "robot_" + str(c_robot) + "_goal_" + str(c_goal)}
                            for c_goal in range(nb_goals_per_robot)
                        ],
                    },
                }
                for c_robot in range(nb_robots)
            ],
            "display_sim_knowledge_only_once": False,
            "files": {"world_file": "./" + scenario_world_json_filename},
            "provide_walls": True,
        }

        for behavior in simulation_json_data["agents_behaviors"]:
            behavior["behavior"]["parameters"]["use_social_cost"] = True
            behavior["behavior"]["parameters"]["manipulation_search_procedure"] = "DFS"
        scenario_json_filename = "sim_snamo_" + scenario_id + ".json"
        with open(os.path.join(scenario_dirpath, scenario_json_filename), "w+") as f:
            json.dump(simulation_json_data, f)

        for behavior in simulation_json_data["agents_behaviors"]:
            behavior["behavior"]["parameters"]["use_social_cost"] = False
            behavior["behavior"]["parameters"]["manipulation_search_procedure"] = "BFS"
        scenario_json_filename = "sim_namo_" + scenario_id + ".json"
        with open(os.path.join(scenario_dirpath, scenario_json_filename), "w+") as f:
            json.dump(simulation_json_data, f)


if __name__ == "__main__":
    generate_scenarios_alternatives(
        base_svg_filepath="../data/simulations/iros_2021/after_the_feast/after_the_feast_base.svg",
        nb_robots=4,
        nb_goals_per_robot=25,
        grid_res=0.1,
        nb_scenarios=1000,
    )
