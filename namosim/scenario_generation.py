import copy
import os
import random
import typing as t
from xml.dom import minidom

from shapely import Polygon

from namosim.data_models import (
    UID,
    AgentConfigModel,
    GoalConfigModel,
    GridCellModel,
    NamosimConfigModel,
    PoseModel,
    StilmanBehaviorConfigModel,
    StilmanBehaviorParametersModel,
)
from namosim.utils import collision, conversion, utils
from namosim.world.binary_occupancy_grid import (
    BinaryInflatedOccupancyGrid,
    BinaryOccupancyGrid,
)

random.seed(0)


def reinit_svg(doc: minidom.Document) -> minidom.Document:
    """Clears an existing scenario file by removing all elements except walls and movables."""
    doc = minidom.parseString(doc.toxml())
    for element in doc.documentElement.getElementsByTagName("*"):
        if element.getAttribute("type") not in ["movable", "wall"]:
            element.parentNode.removeChild(element)
    return doc


def sample_poses_uniform(
    obstacles_polygons: t.Dict[UID, Polygon],
    robot_polygon: Polygon,
    robot_pose: PoseModel,
    grid: BinaryOccupancyGrid,
    nb_poses: int = 1,
    min_distance_between: float = 0.0,
) -> t.List[PoseModel]:
    """Samples robot poses which do not collide with any of the provided obstacle polygons"""
    # Make AABB Tree from polygons
    aabb_tree = collision.polygons_to_aabb_tree(obstacles_polygons)

    accessible_cells: t.Set[GridCellModel] = set()
    for i in range(grid.d_width):
        for j in range(grid.d_height):
            if grid.grid[i][j] == 0:
                accessible_cells.add((i, j))

    if len(accessible_cells) == 0:
        raise Exception("No accessible cells")

    generated_poses: t.List[PoseModel] = []
    generated_polygons: t.List[Polygon] = []

    while len(generated_poses) < nb_poses:
        rand_cell = random.choice(tuple(accessible_cells))
        cell_center = grid.get_cell_center(rand_cell)
        rand_pose = (
            cell_center[0],
            cell_center[1],
            random.uniform(0.0, 360.0),
        )

        check_cell = utils.real_to_grid(
            rand_pose[0], rand_pose[1], grid.res, grid.grid_pose
        )
        assert check_cell == rand_cell

        robot_polygon_at_rand_pose = utils.set_polygon_pose(
            robot_polygon, robot_pose, rand_pose
        )
        robot_aabb_at_rand_pose = collision.polygon_to_aabb(robot_polygon_at_rand_pose)
        potential_collision_uids = aabb_tree.overlap_values(robot_aabb_at_rand_pose)

        pose_invalid = False

        # Invalidate pose if it intersects with an obstacle
        for uid in potential_collision_uids:
            if obstacles_polygons[uid].intersects(robot_polygon_at_rand_pose):
                pose_invalid = True
                break

        min_dist_to_others = float("inf")

        for polygon in generated_polygons:
            d = polygon.distance(robot_polygon_at_rand_pose)
            min_dist_to_others = min(min_dist_to_others, d)

        # Invalidate pose if too close to other pose
        if min_dist_to_others < min_distance_between:
            pose_invalid = True

        if not pose_invalid:
            generated_poses.append(rand_pose)
            if min_distance_between > 0:
                accessible_cells.remove(rand_cell)
                generated_polygons.append(robot_polygon_at_rand_pose)

    return generated_poses


def generate_alternative_scenarios(
    out_dir: str,
    base_svg_filepath: str,
    nb_robots: int,
    nb_goals_per_robot: int,
    nb_scenarios: int,
    cell_size: float,
    use_social_cost: bool = True,
    resolve_conflicts: bool = True,
    resolve_deadlocks: bool = True,
):
    """Randomly generates alternative versions of a given scenario with a given number of robots and goals."""
    # Load SVGs
    svg_data_init = minidom.parse(base_svg_filepath)
    svg_init_config = NamosimConfigModel.from_xml(
        svg_data_init.getElementsByTagName("namo_config")[0].toxml()
    )
    conversion.set_all_id_attributes_as_ids(svg_data_init)

    base_agent = svg_init_config.agents[0]
    svg_base_robot = svg_data_init.getElementById(base_agent.agent_id)
    svg_base_goal = svg_data_init.getElementById(base_agent.goals[0].goal_id)

    if not svg_base_robot:
        raise Exception(f"Path for robot {base_agent.agent_id} not found")
    if not svg_base_goal:
        raise Exception(f"Path for goal {base_agent.goals[0].goal_id} not found")

    svg_base_robot_shape = get_elements_by_attribute(svg_base_robot, "type", "shape")[0]
    svg_base_robot_direction = get_elements_by_attribute(
        svg_base_robot, "type", "orientation"
    )[0]
    svg_base_goal_shape = get_elements_by_attribute(svg_base_goal, "type", "shape")[0]
    svg_base_goal_direction = get_elements_by_attribute(
        svg_base_goal, "type", "orientation"
    )[0]

    if not svg_base_robot_shape:
        raise Exception("Failed to get base robot shape")
    if not svg_base_robot_direction:
        raise Exception("Failed to get base robot direction")
    if not svg_base_goal_shape:
        raise Exception("Failed to get base goal shape")
    if not svg_base_goal_direction:
        raise Exception("Failed to get base goal direction")

    # Convert svg_data paths into polygons
    all_polygons: t.Dict[UID, Polygon] = {}
    static_polygons: t.Dict[UID, Polygon] = {}
    static_and_movable_polygons: t.Dict[UID, Polygon] = {}

    for path in svg_data_init.getElementsByTagNameNS("*", "path"):
        uid = path.getAttribute("id")
        polygon = conversion.svg_pathd_to_shapely_geometry(path.getAttribute("d"))
        all_polygons[uid] = polygon

        if path.getAttribute("type") == "wall":
            static_polygons[uid] = polygon
            static_and_movable_polygons[uid] = polygon
        elif path.getAttribute("type") == "movable":
            static_and_movable_polygons[uid] = polygon

    base_robot_polygon = conversion.svg_pathd_to_shapely_geometry(
        svg_base_robot_shape.getAttribute("d")
    )
    base_robot_orientation_polygon = conversion.svg_pathd_to_shapely_geometry(
        svg_base_robot_direction.getAttribute("d")
    )
    base_robot_orientation_geom_coords = list(base_robot_orientation_polygon.coords)
    base_robot_orientation_vector = (
        base_robot_orientation_geom_coords[1][0]
        - base_robot_orientation_geom_coords[0][0],
        base_robot_orientation_geom_coords[1][1]
        - base_robot_orientation_geom_coords[0][1],
    )
    base_robot_pose: PoseModel = (
        base_robot_polygon.centroid.coords[0][0],
        base_robot_polygon.centroid.coords[0][1],
        utils.yaw_from_direction(base_robot_orientation_vector),
    )

    base_goal_polygon = conversion.svg_pathd_to_shapely_geometry(
        svg_base_goal_shape.getAttribute("d")
    )
    base_goal_orientation_polygon = conversion.svg_pathd_to_shapely_geometry(
        svg_base_goal_direction.getAttribute("d")
    )
    base_goal_orientation_geom_coords = list(base_goal_orientation_polygon.coords)
    base_goal_orientation_vector = (
        base_goal_orientation_geom_coords[1][0]
        - base_goal_orientation_geom_coords[0][0],
        base_goal_orientation_geom_coords[1][1]
        - base_goal_orientation_geom_coords[0][1],
    )
    base_goal_pose: PoseModel = (
        base_goal_polygon.centroid.coords[0][0],
        base_goal_polygon.centroid.coords[0][1],
        utils.yaw_from_direction(base_goal_orientation_vector),
    )

    # Do uniform sampling in coordinates that are within map bounds or load "_samples.json",
    # for initial robot poses (can not be in any obstacles) and goals robot poses (can be in movable obstacles)
    static_and_movable_grid = BinaryInflatedOccupancyGrid(
        static_and_movable_polygons,
        cell_size,
        utils.get_circumscribed_radius(base_robot_polygon),
    )
    static_grid = BinaryInflatedOccupancyGrid(
        static_polygons,
        cell_size,
        utils.get_circumscribed_radius(base_robot_polygon),
    )

    for c_scenario in range(nb_scenarios):
        svg_data = reinit_svg(svg_data_init)
        scenario_id = ("{:0" + str(len(str(nb_scenarios))) + "d}").format(c_scenario)

        # Create the NamoConfig
        namo_config = copy.deepcopy(svg_init_config)
        namo_config.cell_size = cell_size
        namo_config.agents = []

        goals_poses_for_robots: t.List[t.List[PoseModel]] = []
        for i in range(nb_robots):
            poses = sample_poses_uniform(
                obstacles_polygons=static_polygons,
                robot_polygon=base_goal_polygon,
                robot_pose=base_goal_pose,
                nb_poses=nb_goals_per_robot,
                grid=static_grid,
            )

            poses = sorted(poses, key=lambda x: x[0])
            goals_poses_for_robots.append(poses)

        for i_robot in range(nb_robots):
            goals: t.List[GoalConfigModel] = []
            for i_goal in range(len(goals_poses_for_robots[i_robot])):
                goals.append(
                    GoalConfigModel.model_validate(
                        {"goal_id": f"robot_{i_robot}_goal_{i_goal}"}
                    )
                )

            behavior_config = StilmanBehaviorConfigModel.model_validate(
                {
                    "type": "stilman_2005_behavior",
                    "parameters": StilmanBehaviorParametersModel.model_validate(
                        {
                            "robot_translation_unit_length": cell_size,
                            "use_social_cost": use_social_cost,
                            "solution_interval_bound_percentage": 0.02,
                            "manipulation_search_procedure": "DFS"
                            if use_social_cost
                            else "BFS",
                            "resolve_deadlocks": resolve_deadlocks,
                            "resolve_conflicts": resolve_conflicts,
                        }
                    ),
                }
            )
            agent_config = AgentConfigModel.model_validate(
                {
                    "agent_id": f"robot_{i_robot}",
                    "behavior": behavior_config,
                    "goals": goals,
                }
            )

            namo_config.agents.append(agent_config)

        svg_data.documentElement.appendChild(
            minidom.parseString(namo_config.to_xml()).documentElement
        )

        robot_radius = utils.get_circumscribed_radius(base_robot_polygon)
        initial_robot_poses = sample_poses_uniform(
            obstacles_polygons=static_and_movable_polygons,
            robot_polygon=base_robot_polygon,
            robot_pose=base_robot_pose,
            nb_poses=nb_robots,
            grid=static_and_movable_grid,
            min_distance_between=robot_radius + utils.SQRT_OF_2 * cell_size + 1e-6,
        )

        # Create robot polygons at said poses in svg_data using svg styles of robot and goals and setting unique ids
        goals_group = conversion.add_group(svg_data, "goals")

        for i in range(nb_robots):
            robot_id = "robot_" + str(i)
            robot_group = conversion.add_group(svg_data, robot_id, is_layer=False)
            # Add robot shape
            conversion.add_shapely_geometry_to_svg(
                shapely_geometry=utils.set_polygon_pose(
                    base_robot_polygon, base_robot_pose, initial_robot_poses[i]
                ),
                uname=robot_id + "_shape",
                style=svg_base_robot_shape.getAttribute("style"),
                svg_data=svg_data,
                svg_group=robot_group,
                namo_type="shape",
            )

            # Add robot direction shape
            robot_shape = utils.set_polygon_pose(
                base_robot_orientation_polygon,
                base_robot_pose,
                initial_robot_poses[i],
                rotation_center=(base_robot_pose[0], base_robot_pose[1]),
            )
            conversion.add_shapely_geometry_to_svg(
                shapely_geometry=robot_shape,
                uname=robot_id + "_direction",
                style=svg_base_robot_direction.getAttribute("style"),
                svg_data=svg_data,
                svg_group=robot_group,
                namo_type="orientation",
            )

            # Add robot goals
            robot_goals_layer = conversion.add_group(
                svg_data, robot_id + "_goals", goals_group
            )

            for i_goal, goal_pose in enumerate(goals_poses_for_robots[i]):
                goal_id = robot_id + "_goal_" + str(i_goal)
                goal_group = conversion.add_group(
                    svg_data, goal_id, robot_goals_layer, is_layer=False
                )
                # Add goal shape
                goal_shape = utils.set_polygon_pose(
                    base_goal_polygon,
                    base_goal_pose,
                    goal_pose,
                    rotation_center=(base_goal_pose[0], base_goal_pose[1]),
                )
                conversion.add_shapely_geometry_to_svg(
                    shapely_geometry=goal_shape,
                    uname=goal_id + "_shape",
                    style=svg_base_goal_shape.getAttribute("style"),
                    svg_data=svg_data,
                    svg_group=goal_group,
                    namo_type="shape",
                )
                # Add goal direction shape
                conversion.add_shapely_geometry_to_svg(
                    shapely_geometry=utils.set_polygon_pose(
                        base_goal_orientation_polygon,
                        base_goal_pose,
                        goal_pose,
                        rotation_center=(base_goal_pose[0], base_goal_pose[1]),
                    ),
                    uname=goal_id + "_direction",
                    style=svg_base_goal_direction.getAttribute("style"),
                    svg_data=svg_data,
                    svg_group=goal_group,
                    namo_type="orientation",
                )

        # Create SVG file from modified data
        new_scenario_basedir = f"{nb_robots}_robots_{nb_goals_per_robot}_goals"
        if not use_social_cost and resolve_conflicts and resolve_deadlocks:
            new_scenario_basedir += "_namo"
        elif not use_social_cost and resolve_conflicts and not resolve_deadlocks:
            new_scenario_basedir += "_namo_ndr"
        elif not use_social_cost and not resolve_conflicts and not resolve_deadlocks:
            new_scenario_basedir += "_namo_ncr"
        elif use_social_cost and resolve_conflicts and resolve_deadlocks:
            new_scenario_basedir += "_snamo"
        elif use_social_cost and resolve_conflicts and not resolve_deadlocks:
            new_scenario_basedir += "_snamo_ndr"
        elif use_social_cost and not resolve_conflicts and not resolve_deadlocks:
            new_scenario_basedir += "_snamo_ncr"

        new_scenario_path = os.path.join(
            out_dir,
            new_scenario_basedir,
            f"{scenario_id}.svg",
        )

        if not os.path.exists(os.path.dirname(new_scenario_path)):
            os.makedirs(os.path.dirname(new_scenario_path))

        with open(new_scenario_path, "w+") as f:
            svg_data.writexml(f, addindent="  ")


def get_elements_by_attribute(
    root: minidom.Element, attribute_name: str, attribute_value: str
) -> t.List[minidom.Element]:
    result = []
    for el in root.getElementsByTagName("*"):
        if el.getAttribute(attribute_name) == attribute_value:
            result.append(el)
    return result
