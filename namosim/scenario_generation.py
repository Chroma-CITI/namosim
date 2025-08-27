import copy
import os
import random
import typing as t
from xml.dom import minidom

from shapely.geometry import Polygon

from namosim.data_models import (
    AgentConfigXmlModel,
    GoalConfigModel,
    GridCellModel,
    NamoConfigXmlModel,
    Pose2D,
    StilmanBehaviorConfigModel,
    StilmanBehaviorParametersModel,
)
from namosim.utils import collision, conversion, utils
from namosim.world.binary_occupancy_grid import (
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
    robot_polygon: Polygon,
    robot_pose: Pose2D,
    grid: BinaryOccupancyGrid,
    nb_poses: int = 1,
    min_distance_between: float = 0.0,
) -> t.List[Pose2D]:
    """Samples robot poses which do not collide with any of the provided obstacle polygons"""
    accessible_cells: t.Set[GridCellModel] = set()
    for i in range(grid.d_width):
        for j in range(grid.d_height):
            if grid.grid[i][j] == 0:
                accessible_cells.add((i, j))

    if len(accessible_cells) == 0:
        raise Exception("No accessible cells")

    generated_poses: t.Set[Pose2D] = set()
    generated_polygons: t.List[Polygon] = []

    while len(generated_poses) < nb_poses:
        rand_cell = random.choice(tuple(accessible_cells))
        cell_center = grid.get_cell_center(rand_cell)
        rand_pose = Pose2D(
            cell_center[0],
            cell_center[1],
            random.uniform(0.0, 360.0),
        )

        check_cell = utils.real_to_grid(
            rand_pose[0], rand_pose[1], grid.cell_size, grid.grid_pose
        )
        assert check_cell == rand_cell

        robot_polygon_at_rand_pose = utils.set_polygon_pose(
            robot_polygon, robot_pose, rand_pose
        )

        pose_invalid = False

        min_dist_to_others = float("inf")
        for polygon in generated_polygons:
            d = polygon.distance(robot_polygon_at_rand_pose)
            min_dist_to_others = min(min_dist_to_others, d)

        # Invalidate pose if too close to other pose
        if min_dist_to_others < min_distance_between:
            pose_invalid = True

        if not pose_invalid:
            generated_poses.add(rand_pose)
            if min_distance_between > 0:
                accessible_cells.remove(rand_cell)
                generated_polygons.append(robot_polygon_at_rand_pose)

    return list(generated_poses)


def generate_alternative_scenarios(
    *,
    out_dir: str,
    base_svg_filepath: str,
    nb_robots: int,
    nb_goals_per_robot: int,
    nb_scenarios: int,
    cell_size_cm: float,
    deadlock_strategy: t.Optional[t.Literal["SOCIAL", "DISTANCE"]] = None,
    use_social_cost: bool = True,
    resolve_conflicts: bool = True,
    resolve_deadlocks: bool = True,
):
    """Randomly generates alternative versions of a given scenario with a given number of robots and goals."""
    # Load SVGs
    svg_data_init = minidom.parse(base_svg_filepath)
    svg_init_config = NamoConfigXmlModel.from_xml(
        svg_data_init.getElementsByTagName("namo_config")[0].toxml()
    )
    conversion.set_all_id_attributes_as_ids(svg_data_init)

    if not svg_data_init.documentElement.hasAttribute("viewBox"):
        raise Exception("svg has no viewBox attribute")

    # Split the viewBox attribute into its components
    viewbox_values = [
        float(x) / 100
        for x in svg_data_init.documentElement.getAttribute("viewBox").split()
    ]
    assert viewbox_values[0] == 0
    assert viewbox_values[1] == 0
    height = viewbox_values[3]

    base_agent = svg_init_config.agents[0]
    svg_base_robot = svg_data_init.getElementById(base_agent.agent_id)
    svg_base_goal = svg_data_init.getElementById(base_agent.goals[0].goal_id)

    if not svg_base_robot:
        raise Exception(f"Path for robot {base_agent.agent_id} not found")
    if not svg_base_goal:
        raise Exception(f"Path for goal {base_agent.goals[0].goal_id} not found")

    # Convert svg_data paths into polygons
    all_polygons: t.Dict[str, Polygon] = {}
    static_polygons: t.List[Polygon] = []
    movable_polygons: t.Dict[str, Polygon] = {}

    for path in svg_data_init.getElementsByTagNameNS("*", "path"):
        uid = path.getAttribute("id")
        polygon: Polygon = conversion.svg_pathd_to_shapely_geometry(svg_path=path.getAttribute("d"), ymax_meters=height)  # type: ignore
        all_polygons[uid] = polygon

        if path.getAttribute("type") == "wall":
            static_polygons.append(polygon)
        elif path.getAttribute("type") == "movable":
            movable_polygons[uid] = polygon

    base_robot_polygon: Polygon = conversion.svg_pathd_to_shapely_geometry(
        svg_path=svg_base_robot.getAttribute("d"), ymax_meters=height
    )  # type: ignore
    base_robot_pose: Pose2D = Pose2D(
        base_robot_polygon.centroid.coords[0][0],
        base_robot_polygon.centroid.coords[0][1],
        0,
    )

    base_goal_polygon: Polygon = conversion.svg_pathd_to_shapely_geometry(
        svg_path=svg_base_goal.getAttribute("d"), ymax_meters=height
    )  # type: ignore
    base_goal_pose: Pose2D = Pose2D(
        base_goal_polygon.centroid.coords[0][0],
        base_goal_polygon.centroid.coords[0][1],
        0,
    )

    # Do uniform sampling in coordinates that are within map bounds or load "_samples.json",
    # for initial robot poses (can not be in any obstacles) and goals robot poses (can be in movable obstacles)

    static_grid = BinaryOccupancyGrid(
        static_polygons=static_polygons,
        width=viewbox_values[2],
        height=viewbox_values[3],
        cell_size=cell_size_cm / 100,
        inflation_radius=utils.get_circumscribed_radius(base_robot_polygon),
    )
    static_and_movable_grid = BinaryOccupancyGrid(
        static_polygons=static_polygons,
        width=viewbox_values[2],
        height=viewbox_values[3],
        cell_size=cell_size_cm / 100,
        inflation_radius=utils.get_circumscribed_radius(base_robot_polygon),
    )
    static_and_movable_grid.update_polygons(movable_polygons)

    for c_scenario in range(nb_scenarios):
        svg_data = reinit_svg(svg_data_init)
        scenario_id = ("{:0" + str(len(str(nb_scenarios))) + "d}").format(c_scenario)

        # Create the NamoConfig
        namo_config = copy.deepcopy(svg_init_config)
        namo_config.cell_size_cm = cell_size_cm
        namo_config.agents = []

        goals_poses_for_robots: t.List[t.List[Pose2D]] = []
        for i in range(nb_robots):
            poses = sample_poses_uniform(
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
                            "use_social_cost": use_social_cost,
                            "manip_search_bound_percentage": 0.05,
                            "resolve_deadlocks": resolve_deadlocks,
                            "resolve_conflicts": resolve_conflicts,
                        }
                    ),
                }
            )

            if deadlock_strategy:
                behavior_config.parameters.deadlock_strategy = deadlock_strategy

            agent_config = AgentConfigXmlModel.model_validate(
                {
                    "agent_id": f"robot_{i_robot}",
                    "behavior": behavior_config,
                    "goals": goals,
                }
            )

            namo_config.agents.append(agent_config)

        svg_data.documentElement.appendChild(
            minidom.parseString(namo_config.to_xml()).documentElement  # type: ignore
        )

        robot_radius = utils.get_circumscribed_radius(base_robot_polygon)
        initial_robot_poses = sample_poses_uniform(
            robot_polygon=base_robot_polygon,
            robot_pose=base_robot_pose,
            nb_poses=nb_robots,
            grid=static_and_movable_grid,
            min_distance_between=robot_radius
            + utils.SQRT_OF_2 * (cell_size_cm / 100)
            + 1e-6,
        )

        # Create robot polygons at said poses in svg_data using svg styles of robot and goals and setting unique ids
        goals_group = conversion.add_group(svg_data, "goals")

        for i in range(nb_robots):
            agent_id = "robot_" + str(i)
            # Add robot shape
            conversion.add_shapely_geometry_to_svg(
                shape=utils.set_polygon_pose(
                    base_robot_polygon, base_robot_pose, initial_robot_poses[i]
                ),
                uid=agent_id,
                style=svg_base_robot.getAttribute("style"),
                svg_data=svg_data,
                namo_type="shape",
                ymax_meters=height,
            )

            # Add robot goals
            robot_goals_layer = conversion.add_group(
                svg_data, agent_id + "_goals", goals_group
            )

            for i_goal, goal_pose in enumerate(goals_poses_for_robots[i]):
                goal_id = agent_id + "_goal_" + str(i_goal)

                # Add goal shape
                goal_shape = utils.set_polygon_pose(
                    base_goal_polygon,
                    base_goal_pose,
                    goal_pose,
                    rotation_center=(base_goal_pose[0], base_goal_pose[1]),
                )
                conversion.add_shapely_geometry_to_svg(
                    shape=goal_shape,
                    uid=goal_id,
                    style=svg_base_goal.getAttribute("style"),
                    svg_data=svg_data,
                    namo_type="shape",
                    ymax_meters=height,
                )

        new_scenario_path = os.path.join(
            out_dir,
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
