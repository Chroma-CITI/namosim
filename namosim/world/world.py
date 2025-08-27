import copy
import io
import os
import typing as t
from xml.dom import minidom

import cairosvg
import numpy as np
from bidict import bidict  # type: ignore[reportPrivateImportUsage]
from PIL import Image, ImageDraw
import shapely
from shapely.geometry import Polygon, Point
from typing_extensions import Self

from namosim import svg_styles
import namosim.agents as agts
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
from namosim.svg_styles import AgentStyle
import namosim.utils.conversion as conversion
import namosim.utils.utils as utils
from namosim import agents
from namosim.data_models import (
    AgentConfigXmlModel,
    AgentConfigYamlModel,
    NamoConfigYamlModel,
    NamoConfigXmlModel,
    Pose2D,
    StilmanBehaviorConfigModel,
    StilmanBehaviorParametersModel,
)
from namosim.utils import collision
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from namosim.world.entity import Entity, Movability, Style
from namosim.world.goal import Goal
from namosim.world.obstacle import Obstacle
from namosim.world.sensors.omniscient_sensor import OmniscientSensor
from namosim.data_models import namo_config_from_yaml
import shapely.ops


class World:
    def __init__(
        self,
        *,
        map: BinaryOccupancyGrid,
        collision_margin: float,
        agents: t.Optional[t.Dict[str, "agts.Agent"]] = None,
        dynamic_entities: t.Optional[t.Dict[str, Entity]] = None,
        entity_to_agent: t.Optional[bidict[str, str]] = None,
        goals: t.Optional[t.Dict[str, Goal]] = None,
        init_geometry_file: t.Optional[minidom.Document] = None,
        logger: utils.NamosimLogger | None = None,
        svg_config: NamoConfigXmlModel | None = None,
        generate_report: bool = True,
        random_seed: int = 0,
    ):
        self.svg_config = svg_config
        self.dynamic_entities: t.Dict[str, Entity] = dynamic_entities or {}
        self.map = map
        self.collision_margin = collision_margin
        self.agents: t.Dict[str, "agts.Agent"] = agents if agents else {}
        self.entity_to_agent = entity_to_agent or bidict()
        self.init_geometry_file = init_geometry_file
        self.generate_report = generate_report
        self.random_seed = random_seed
        if init_geometry_file:
            conversion.set_all_id_attributes_as_ids(init_geometry_file)
            conversion.clean_attributes(init_geometry_file)
        self.init_geometry_file = init_geometry_file
        self._goals: t.Dict[str, Goal] = goals if goals else dict()
        self._goal_pose_to_goal: t.Dict[Pose2D, Goal] = {}
        for goal in self._goals.values():
            self._goal_pose_to_goal[goal.pose] = goal

        if logger is None:
            self.logger = utils.NamosimLogger()
        else:
            self.logger = logger

    def get_goals(self):
        return self._goals

    def add_goal(self, goal: Goal):
        self._goals[goal.uid] = goal
        self._goal_pose_to_goal[goal.pose] = goal

    def add_agent(self, agent: "agents.Agent"):
        self.agents[agent.uid] = agent
        self.add_entity(agent)

    def get_dynamic_occupancy_grid(
        self, inflation_radius: float = 0, ignored_entities: t.Set[str] | None = None
    ) -> BinaryOccupancyGrid:
        grid = copy.deepcopy(self.map)
        if inflation_radius:
            grid.inflate_map_destructive(inflation_radius)
        if ignored_entities:
            grid.update_polygons(removed_polygons=ignored_entities)
        dynamic_polygons = {}
        for entity in self.dynamic_entities.values():
            dynamic_polygons[entity.uid] = entity.polygon
        grid.update_polygons(dynamic_polygons)
        return grid

    # Constructor
    @classmethod
    def load_from_svg(
        cls,
        svg_path: str,
        logs_dir: str = "namo_logs",
        logger: utils.NamosimLogger | None = None,
    ) -> Self:
        if logger is None:
            logger = utils.NamosimLogger()

        # Import entire world from svg file
        svg_doc = minidom.parse(svg_path)
        if not svg_doc or not svg_doc.documentElement:
            raise Exception("SVG document is empty or not found")
        conversion.set_all_id_attributes_as_ids(svg_doc)
        conversion.clean_attributes(svg_doc)
        config = NamoConfigXmlModel.from_xml(
            svg_doc.getElementsByTagName("namo_config")[0].toxml()
        )

        if not svg_doc.documentElement.hasAttribute("viewBox"):
            raise Exception("svg has no viewBox attribute")

        # Split the viewbox attribute into its components
        viewbox_values = [
            float(x) / 100
            for x in svg_doc.documentElement.getAttribute("viewBox").split()
        ]

        assert viewbox_values[0] == 0
        assert viewbox_values[1] == 0
        height = viewbox_values[3]
        cell_size = config.cell_size_cm / 100
        collision_margin = cell_size
        if config.collision_margin_cm:
            collision_margin = config.collision_margin_cm / 100

        svg_paths = {}
        for el in svg_doc.getElementsByTagNameNS("*", "path"):
            svg_paths[el.getAttribute("id")] = el.getAttribute("d")

        shapes: t.Dict[str, Polygon] = dict()

        # Convert imported geometry to shapely polygons
        for svg_id, svg_path in svg_paths.items():
            try:
                shapes[svg_id] = conversion.svg_pathd_to_shapely_geometry(  # type: ignore
                    svg_path=svg_path, ymax_meters=height
                )
            except RuntimeError:
                raise RuntimeError(
                    "Could not convert svg path to shapely geometry for svg id: {}".format(
                        svg_id
                    )
                )

        static_obstacles: t.Dict[str, Polygon] = {}
        dynamic_entities: t.List[Entity] = []

        # Get static and movable obstacles
        for el in svg_doc.getElementsByTagName("*"):
            id = el.getAttribute("id")
            type_ = el.getAttribute("type")
            if not id:
                continue

            if el.tagName in ["svg:path", "path"] and type_ == "movable":
                polygon = shapes[id]
                style = Style.from_string(el.getAttribute("style"))
                pose = Pose2D(
                    t.cast(float, list(polygon.centroid.coords)[0][0]),
                    t.cast(float, list(polygon.centroid.coords)[0][1]),
                    0.0,
                )
                movable_box = Obstacle(
                    type_="movable",
                    uid=id,
                    polygon=polygon,
                    pose=pose,
                    style=style,
                    movability=Movability.MOVABLE,
                )
                dynamic_entities.append(movable_box)
            if el.tagName in ["svg:path", "path"] and type_ == "wall":
                polygon = shapes[id]
                style = Style.from_string(el.getAttribute("style"))
                pose = Pose2D(
                    t.cast(float, list(polygon.centroid.coords)[0][0]),
                    t.cast(float, list(polygon.centroid.coords)[0][1]),
                    0.0,
                )
                static_obstacles[id] = polygon

        agents: t.List["agts.Agent"] = []

        # get agents
        for agent_config in config.agents:
            el = svg_doc.getElementById(agent_config.agent_id)
            if not el:
                raise Exception(f"Robot {agent_config.agent_id} not found in svg")

            if not el.tagName in ["path", "svg:path"]:
                raise Exception(
                    f"Robot {agent_config.agent_id} svg element is not a path element"
                )

            robot_shape_style = el.getAttribute("style")
            robot_polygon = shapes[agent_config.agent_id]
            robot_angle = 0
            if el.hasAttribute("angle"):
                robot_angle = float(el.getAttribute("angle"))

            agent_style = AgentStyle(
                shape=robot_shape_style,
                orientation="",
            )

            init_pose = Pose2D(
                t.cast(float, list(robot_polygon.centroid.coords)[0][0]),
                t.cast(float, list(robot_polygon.centroid.coords)[0][1]),
                robot_angle,
            )

            # make robot polygon a perfect circle
            robot_polygon = robot_polygon
            goals: t.List[Goal] = []

            for goal in agent_config.goals:
                goal_el = svg_doc.getElementById(goal.goal_id)
                if not goal_el:
                    raise Exception(f"Goal {goal.goal_id} not found in svg")
                if not goal_el.tagName in ["path", "svg:path"]:
                    raise Exception(
                        f"Goal {goal.goal_id} svg element is not a path element"
                    )

                goal_style = goal_el.getAttribute("style")
                goal_polygon = shapes[goal.goal_id]
                goal_angle = 0
                if goal_el.hasAttribute("angle"):
                    goal_angle = float(goal_el.getAttribute("angle"))

                goal_pose = (
                    t.cast(float, list(goal_polygon.centroid.coords)[0][0]),
                    t.cast(float, list(goal_polygon.centroid.coords)[0][1]),
                    goal_angle,
                )
                goal = Goal(
                    uid=goal.goal_id,
                    polygon=goal_polygon,
                    pose=tuple(goal_pose),  # type: ignore
                    svg_style=AgentStyle(shape=goal_style, orientation=""),
                )
                goals.append(goal)

            new_robot = World.construct_agent_from_xml_model(
                agent_config=agent_config,
                goals=goals,
                logs_dir=logs_dir,
                robot_polygon=robot_polygon,
                agent_style=agent_style,
                init_pose=init_pose,
                cell_size=cell_size,
                collision_margin=collision_margin,
                logger=logger,
            )
            agents.append(new_robot)

        goals_node = svg_doc.getElementById("goals")
        if goals_node and goals_node.parentNode:
            goals_node.parentNode.removeChild(goals_node)

        map = BinaryOccupancyGrid(
            static_polygons=list(static_obstacles.values()),
            cell_size=config.cell_size_cm / 100,
            width=viewbox_values[2],
            height=viewbox_values[3],
        )
        world = cls(
            init_geometry_file=svg_doc,
            logger=logger,
            map=map,
            random_seed=config.random_seed,
            generate_report=config.generate_report,
            svg_config=config,
            collision_margin=collision_margin,
        )

        for movable in dynamic_entities:
            world.add_entity(movable)

        for agent_config in agents:
            world.add_agent(agent_config)

        for agent_config in agents:
            agent_config.init(world)

        return world

    @classmethod
    def construct_agent_from_xml_model(
        cls,
        *,
        agent_config: AgentConfigXmlModel,
        goals: t.List[Goal],
        logs_dir: str,
        robot_polygon: Polygon,
        agent_style: AgentStyle,
        init_pose: Pose2D,
        cell_size: float,
        collision_margin: float,
        logger: utils.NamosimLogger,
    ) -> "agts.Agent":
        if agent_config.behavior.type == "stilman_2005_behavior":
            new_robot = agts.Stilman2005Agent(
                navigation_goals=goals,
                config=agent_config.behavior,
                logs_dir=logs_dir,
                uid=agent_config.agent_id,
                polygon=robot_polygon,
                style=agent_style,
                pose=init_pose,
                sensors=[OmniscientSensor()],
                cell_size=cell_size,
                collision_margin=collision_margin,
                logger=logger,
            )
        elif agent_config.behavior.type == "stilman_rrt_star_behavior":
            new_robot = agts.StilmanRRTStarAgent(
                navigation_goals=goals,
                config=agent_config.behavior,
                logs_dir=logs_dir,
                uid=agent_config.agent_id,
                polygon=robot_polygon,
                style=agent_style,
                pose=init_pose,
                sensors=[OmniscientSensor()],
                cell_size=cell_size,
                collision_margin=collision_margin,
                logger=logger,
            )
        elif agent_config.behavior.type == "navigation_only_behavior":
            new_robot = agts.NavigationOnlyAgent(
                navigation_goals=goals,
                config=agent_config.behavior,
                logs_dir=logs_dir,
                uid=agent_config.agent_id,
                polygon=robot_polygon,
                style=agent_style,
                pose=init_pose,
                sensors=[OmniscientSensor()],
                cell_size=cell_size,
                logger=logger,
            )
        elif agent_config.behavior.type == "rrt":
            new_robot = agts.RRTAgent(
                navigation_goals=goals,
                config=agent_config.behavior,
                logs_dir=logs_dir,
                uid=agent_config.agent_id,
                polygon=robot_polygon,
                style=agent_style,
                pose=init_pose,
                sensors=[OmniscientSensor()],
                cell_size=cell_size,
                logger=logger,
            )
        elif agent_config.behavior.type == "teleop_behavior":
            new_robot = agts.TeleopAgent(
                navigation_goals=goals,
                config=agent_config.behavior,
                logs_dir=logs_dir,
                uid=agent_config.agent_id,
                polygon=robot_polygon,
                style=agent_style,
                pose=init_pose,
                sensors=[OmniscientSensor()],
                cell_size=cell_size,
                logger=logger,
            )
        else:
            raise NotImplementedError(
                "You tried to associate entity '{agent_name}' with a behavior named"
                "'{b_name}' that is not implemented yet."
                "Maybe you mispelled something ?".format(
                    agent_name=agent_config.agent_id, b_name=agent_config.behavior.type
                )
            )
        return new_robot

    @classmethod
    def construct_agent_from_yaml_model(
        cls,
        *,
        agent: AgentConfigYamlModel,
        goals: t.List[Goal],
        logs_dir: str,
        robot_polygon: Polygon,
        agent_style: AgentStyle,
        init_pose: Pose2D,
        cell_size: float,
        collision_margin: float,
        logger: utils.NamosimLogger,
    ) -> "agts.Agent":
        if agent.behavior.type == "stilman_2005_behavior":
            return agts.Stilman2005Agent(
                navigation_goals=goals,
                config=agent.behavior,
                logs_dir=logs_dir,
                uid=agent.id,
                polygon=robot_polygon,
                style=agent_style,
                pose=init_pose,
                sensors=[OmniscientSensor()],
                cell_size=cell_size,
                collision_margin=collision_margin,
                logger=logger,
            )
        else:
            raise NotImplementedError(
                "You tried to associate entity '{agent_name}' with a behavior named"
                "'{b_name}' that is not implemented yet."
                "Maybe you mispelled something ?".format(
                    agent_name=agent.id, b_name=agent.behavior.type
                )
            )

    @staticmethod
    def get_wall_polygons_from_svg(
        svg_path: str,
    ) -> t.List[Polygon]:
        svg_doc = minidom.parse(svg_path).documentElement
        if not svg_doc:
            raise Exception("SVG document is empty or not found")
        conversion.set_all_id_attributes_as_ids(svg_doc)
        conversion.clean_attributes(svg_doc)
        if not svg_doc.hasAttribute("viewBox"):
            raise Exception("svg has no viewBox attribute")

        # Split the viewbox attribute into its components
        viewbox_values = [
            float(x) / 100 for x in svg_doc.getAttribute("viewBox").split()
        ]

        assert viewbox_values[0] == 0
        assert viewbox_values[1] == 0
        height = viewbox_values[3]

        wall_polygons: t.List[Polygon] = []
        for el in svg_doc.getElementsByTagNameNS("*", "path"):
            svg_id = el.getAttribute("id")
            svg_path = el.getAttribute("d")
            if el.getAttribute("type") == "wall":
                try:
                    polygon = conversion.svg_pathd_to_shapely_geometry(  # type: ignore
                        svg_path=svg_path, ymax_meters=height
                    )
                    wall_polygons.append(polygon)  # type: ignore
                except RuntimeError:
                    raise RuntimeError(
                        "Could not convert svg path to shapely geometry for svg id: {}".format(
                            svg_id
                        )
                    )
        return wall_polygons

    def get_empty_svg_doc(self, ignore_config: bool = False):
        doc = minidom.Document()
        svg = doc.createElement("svg")
        svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
        svg.setAttribute("xmlns:svg", "http://www.w3.org/2000/svg")
        width = self.map.width * 100
        height = self.map.height * 100
        svg.setAttribute("width", str(int(width)))
        svg.setAttribute("height", str(int(height)))

        if ignore_config is False:
            if not self.svg_config:
                raise Exception("World has no svg config")
            config = minidom.parseString(self.svg_config.to_xml()).documentElement
            if config:
                svg.appendChild(config)
        doc.appendChild(svg)
        return doc

    def to_svg(self, ignore_config: bool = False) -> minidom.Document:
        if self.init_geometry_file:
            svg_data = minidom.parseString(self.init_geometry_file.toxml())
        else:
            svg_data = self.get_empty_svg_doc(ignore_config=True)

        # clear geometries
        els_to_del = list(svg_data.getElementsByTagNameNS("*", "path"))
        for el in els_to_del:
            if el.parentNode:
                el.parentNode.removeChild(el)

        for entity in self.dynamic_entities.values():
            if isinstance(entity, Obstacle):
                if entity.movability in [Movability.STATIC, Movability.UNMOVABLE]:
                    style = svg_styles.FIXED_ENTITY_STYLE
                elif entity.movability == Movability.MOVABLE:
                    style = svg_styles.DEFAULT_MOVABLE_ENTITY_STYLE
                elif entity.movability == Movability.UNKNOWN:
                    style = svg_styles.UNKNOWN_ENTITY_STYLE
                else:
                    raise NotImplementedError(
                        "Can only export new obstacles entities that have a 'movability' attribute of "
                        "value ['static', 'unmovable', 'movable', 'unknown'], got {}.".format(
                            entity.movability
                        )
                    )
                conversion.add_shapely_geometry_to_svg(
                    shape=entity.polygon,
                    uid=entity.uid,
                    style=style,
                    svg_data=svg_data,
                    ymax_meters=self.map.height,
                )
            elif isinstance(entity, agts.Agent):
                robot_group = conversion.add_group(svg_data, entity.uid, is_layer=False)
                # Add robot shape
                conversion.add_shapely_geometry_to_svg(
                    shape=entity.polygon,
                    uid=entity.uid + "_shape",
                    style=entity.agent_style.shape,
                    svg_data=svg_data,
                    svg_group=robot_group,
                    ymax_meters=self.map.height,
                )
                # Add robot direction shape
                orientation_polygon = entity.get_orientation_polygon()
                conversion.add_shapely_geometry_to_svg(
                    shape=orientation_polygon,
                    uid=entity.uid + "_direction",
                    style=entity.agent_style.orientation,
                    svg_data=svg_data,
                    svg_group=robot_group,
                    ymax_meters=self.map.height,
                )

                # add agent goal
                goal = entity.get_current_goal()
                if goal:
                    goal_group = conversion.add_group(svg_data, "goal", is_layer=False)
                    # Add robot shape
                    polygon = goal.polygon
                    conversion.add_shapely_geometry_to_svg(
                        shape=polygon,
                        uid=goal.uid + "_shape",
                        style=goal.svg_style.shape,
                        svg_data=svg_data,
                        svg_group=goal_group,
                        ymax_meters=self.map.height,
                    )
                    # Add robot direction shape
                    radius = utils.get_inscribed_radius(entity.polygon)
                    point_a = np.array([goal.pose[0], goal.pose[1]])
                    direction = np.array(utils.direction_from_yaw(goal.pose[2]))
                    point_b = point_a + direction * radius
                    poly = utils.path_to_polygon(
                        points=[point_a, point_b], line_width=radius / 4
                    )
                    conversion.add_shapely_geometry_to_svg(
                        shape=poly,
                        uid=goal.uid + "_direction",
                        style=goal.svg_style.orientation,
                        svg_data=svg_data,
                        svg_group=goal_group,
                        ymax_meters=self.map.height,
                    )
            else:
                raise NotImplementedError(
                    "Only entities of class [Robot, Obstacle] can be created in SVG file for now."
                )  # TODO Add creation of new SVG goals

        return svg_data

    def to_image(
        self,
        grayscale: bool = False,
        width: int = 100,
        ignore_goal: bool = False,
        draw_grid_lines=True,
    ) -> Image.Image:

        svg = self.to_svg(ignore_config=True).toprettyxml()
        image_data = cairosvg.svg2png(
            svg, dpi=300, output_width=width, background_color="white"
        )

        if not image_data:
            raise Exception("Failed to convert world to image")

        image = Image.open(io.BytesIO(image_data))
        background = self.map.to_image()
        background = background.resize(image.size, resample=Image.NEAREST)

        image = image.convert("RGBA")
        background = background.convert("RGBA")
        image.alpha_composite(background)
        if draw_grid_lines:
            image = self._draw_grid_lines(image, self.map.width)
        if grayscale:
            image = image.convert("L")

        return image

    def _draw_grid_lines(self, img: Image.Image, map_width_meters: float):

        # Create a drawing object
        draw = ImageDraw.Draw(img)

        # Get image dimensions
        width, height = img.size

        pixels_per_meter = int(width / map_width_meters)

        # Draw vertical lines
        for x in range(0, width, pixels_per_meter):
            draw.line([(x, 0), (x, height)], fill=(128, 128, 128), width=1)

        # Draw horizontal lines
        for y in range(0, height, pixels_per_meter):
            draw.line([(0, y), (width, y)], fill=(128, 128, 128), width=1)

        return img

    def to_numpy_array(self, grayscale: bool = False):
        img = self.to_image(grayscale=grayscale)
        arr = np.array(img, dtype=np.float32)
        arr /= 255.0
        return arr

    def add_entity(self, new_entity: t.Any):
        self.dynamic_entities[new_entity.uid] = new_entity

    def remove_entity(self, entity_uid: str):
        if entity_uid in self.dynamic_entities:
            del self.dynamic_entities[entity_uid]
        if entity_uid in self.agents:
            del self.agents[entity_uid]
        if entity_uid in self.entity_to_agent:
            del self.entity_to_agent[entity_uid]

    def light_copy(self, ignored_entities: t.Iterable[str]):
        dynamic_entities = {}
        agents = {}
        for uid, x in self.dynamic_entities.items():
            if uid in ignored_entities:
                continue

            x = x.copy()
            dynamic_entities[x.uid] = x
            if isinstance(x, agts.Agent):
                agents[x.uid] = x

        return World(
            dynamic_entities=dynamic_entities,
            agents=agents,
            goals=copy.deepcopy(self._goals),
            logger=self.logger,
            map=self.map,
            collision_margin=self.collision_margin,
            random_seed=self.random_seed,
            generate_report=self.generate_report,
            svg_config=self.svg_config,
        )

    def set_entity_polygon(self, id: str, polygon: Polygon):
        self.dynamic_entities[id].polygon = polygon

    def save_to_files(
        self,
        svg_filepath: str,
        svg_data: t.Optional[t.Any] = None,
    ):
        # Generate SVG data
        if not svg_data:
            svg_data = self.to_svg()

        # Save SVG to specified path
        with open(svg_filepath, "w+") as f:
            svg_data.writexml(f)

    def get_map_bounds(self):
        return self.map.get_bounds()

    def is_holding_obstacle(self, agent_id: str) -> bool:
        return agent_id in self.entity_to_agent.inverse

    def get_combined_agent_obstacle_polygon(self, agent_id: str) -> Polygon:
        agent_polygon = self.agents[agent_id].polygon
        obstacle = self.get_agent_held_obstacle(agent_id)
        if obstacle is None:
            return agent_polygon
        combined_polygon = shapely.ops.unary_union([agent_polygon, obstacle.polygon])
        return t.cast(Polygon, combined_polygon)

    def get_polygon_collisions(self, uid: str, others: t.Iterable[str]) -> t.Any:
        other_polygons = {uid: self.dynamic_entities[uid].polygon for uid in others}
        collisions = collision.get_collisions_for_entity(
            entity_polygon=self.dynamic_entities[uid].polygon,
            other_entity_polygons=other_polygons,
        )
        return collisions

    def get_movable_obstacles(self) -> t.Dict[str, Obstacle]:
        result: t.Dict[str, Obstacle] = {}
        for e in self.dynamic_entities.values():
            if isinstance(e, Obstacle) and e.movability == Movability.MOVABLE:
                result[e.uid] = e
        return result

    def get_all_obstacles(self) -> t.List[Obstacle]:
        result = []
        for e in self.dynamic_entities.values():
            if isinstance(e, Obstacle):
                result.append(e)
        return result

    def get_agent_held_obstacle(self, agent_id: str) -> Entity | None:
        eid = self.entity_to_agent.inverse.get(agent_id, None)
        if eid:
            return self.dynamic_entities[eid]

    def step(
        self,
        actions: t.Dict[str, ba.Action],
        step_count: int,
        ignore_collisions: bool = False,
    ) -> t.Dict[str, ar.ActionResult]:
        results: t.Dict[str, ar.ActionResult] = {}

        actions_to_collision_check = {}
        grabs: t.Dict[str, t.Set[str]] = {}  # maps entity -> grabbing agents

        for agent_id, action in actions.items():
            if isinstance(
                action, (ba.Wait, ba.GoalSuccess, ba.GoalFailed, ba.GoalsFinished)
            ):
                results[agent_id] = ar.ActionSuccess(action, self.agents[agent_id].pose)
            elif isinstance(action, ba.Release):
                # check if release is valid
                entity_uid = action.entity_uid
                if entity_uid not in self.entity_to_agent:
                    results[agent_id] = ar.NotGrabbedFailure(action)
                elif self.entity_to_agent[entity_uid] != agent_id:
                    results[agent_id] = ar.GrabbedByOtherFailure(
                        action, self.entity_to_agent[entity_uid]
                    )
                else:
                    actions_to_collision_check[agent_id] = action
            elif isinstance(action, ba.Grab):
                if action.entity_uid in grabs:
                    grabs[action.entity_uid].add(agent_id)
                else:
                    grabs[action.entity_uid] = set([agent_id])
            else:
                actions_to_collision_check[agent_id] = action

        # check for simultaneous grabs
        for agent_ids in grabs.values():
            if len(agent_ids) > 1:
                for agent_id in agent_ids:
                    other_agent_ids = agent_ids.copy()
                    other_agent_ids.remove(agent_id)
                    results[agent_id] = ar.SimultaneousGrabFailure(
                        actions[agent_id], other_agent_ids
                    )
            else:
                agent_id = list(agent_ids)[0]
                action = actions[agent_id]
                actions_to_collision_check[agent_id] = action

        # Check actions regarding dynamic collisions and apply the valid ones
        collides_with = collision.csv_simulate_simple_kinematics(
            world=self,
            agent_actions=actions_to_collision_check,
            apply=True,
            ignore_collisions=ignore_collisions,
        )

        # Finish separating succeeded and failed actions, and apply result to world state on success
        for agent_id, action in actions_to_collision_check.items():
            agent_obstacle = self.get_agent_held_obstacle(agent_id)
            agent_obstacle_id = agent_obstacle.uid if agent_obstacle else None
            agent_in_collision_with: t.Set[str] = set()
            if agent_id in collides_with:
                if isinstance(action, ba.Grab):
                    if (
                        len(collides_with[agent_id]) > 1
                        or action.entity_uid not in collides_with[agent_id]
                    ):
                        agent_in_collision_with = collides_with[agent_id]
                else:
                    agent_in_collision_with = collides_with[agent_id]
            elif agent_obstacle_id is not None and agent_obstacle_id in collides_with:
                if not isinstance(action, ba.Release):
                    agent_in_collision_with = collides_with[agent_obstacle_id]

            if len(agent_in_collision_with) > 0 and not ignore_collisions:
                results[agent_id] = ar.DynamicCollisionFailure(
                    action, agent_in_collision_with
                )
            else:
                if len(agent_in_collision_with) > 1 and ignore_collisions:
                    self.logger.append(
                        utils.NamosimLog(
                            "Dynamic collision ignored, entities: {}".format(
                                {
                                    self.dynamic_entities[uid].uid: {
                                        self.dynamic_entities[uid2].uid for uid2 in uids
                                    }
                                    for uid, uids in collides_with.items()
                                }
                            ),
                            step=step_count,
                        )
                    )

                # SUCCESS
                # If Grab or Release, first update entity_to_agent
                if isinstance(action, ba.Grab):
                    self.entity_to_agent[action.entity_uid] = agent_id
                if isinstance(action, ba.Release):
                    del self.entity_to_agent[action.entity_uid]

                results[agent_id] = ar.ActionSuccess(
                    action=action,
                    robot_pose=self.dynamic_entities[agent_id].pose,
                    is_transfer=agent_id in self.entity_to_agent.inverse,
                    obstacle_uid=self.entity_to_agent.inverse.get(agent_id, None),
                )

        return results

    @classmethod
    def load_movables(
        cls, map: BinaryOccupancyGrid, config: NamoConfigYamlModel, config_dir: str
    ) -> t.List[Obstacle]:
        if config.svg_file is None:
            return []

        svg_path = os.path.join(config_dir, config.svg_file)

        # Import entire world from svg file
        svg_doc = minidom.parse(svg_path)
        conversion.set_all_id_attributes_as_ids(svg_doc)
        conversion.clean_attributes(svg_doc)

        bounds = map.get_bounds()
        height = bounds[3]

        obtacles: t.List[Obstacle] = []

        movables_layer = svg_doc.getElementById("movables_layer")
        if movables_layer is None:
            return []

        for el in movables_layer.getElementsByTagNameNS("*", "path"):
            path_id = el.getAttribute("id")
            path_data = el.getAttribute("d")
            try:
                polygon: Polygon = conversion.svg_pathd_to_shapely_geometry(  # type: ignore
                    svg_path=path_data, ymax_meters=height, scale=map.cell_size
                )
                style = Style.from_string(el.getAttribute("style"))
                pose = Pose2D(
                    t.cast(float, list(polygon.centroid.coords)[0][0]),
                    t.cast(float, list(polygon.centroid.coords)[0][1]),
                    0.0,
                )
                movable = Obstacle(
                    type_="movable",
                    uid=path_id,
                    polygon=polygon,
                    pose=pose,
                    style=style,
                    movability=Movability.MOVABLE,
                )
                obtacles.append(movable)
            except RuntimeError:
                raise RuntimeError(
                    "Could not convert svg path to shapely geometry for svg id: {}".format(
                        path_id
                    )
                )

        return obtacles

    @classmethod
    def load_goals(
        cls,
        map: BinaryOccupancyGrid,
        config: NamoConfigYamlModel,
        config_dir: str,
        agents: t.Dict[str, "agts.Agent"],
    ):
        if config.svg_file is None:
            return

        svg_path = os.path.join(config_dir, config.svg_file)

        # Import entire world from svg file
        svg_doc = minidom.parse(svg_path)
        conversion.set_all_id_attributes_as_ids(svg_doc)
        # conversion.clean_attributes(svg_doc)

        bounds = map.get_bounds()
        height = bounds[3]

        goals_layer = svg_doc.getElementById("goals_layer")
        if goals_layer is None:
            return

        for el in goals_layer.getElementsByTagNameNS("*", "path"):
            uid = el.getAttribute("id")
            path_data = el.getAttribute("d")
            agent_id = el.getAttribute("agent_id")
            try:
                polygon: Polygon = conversion.svg_pathd_to_shapely_geometry(  # type: ignore
                    svg_path=path_data, ymax_meters=height, scale=map.cell_size
                )
                pose = Pose2D(
                    t.cast(float, list(polygon.centroid.coords)[0][0]),
                    t.cast(float, list(polygon.centroid.coords)[0][1]),
                    0.0,
                )
                goal_polygon = agents[agent_id].polygon
                goal = Goal(
                    uid=uid,
                    polygon=goal_polygon,
                    pose=pose,
                    svg_style=AgentStyle(
                        svg_styles.DEFAULT_GOAL_SHAPE_STYLE,
                        svg_styles.DEFAULT_GOAL_ORIENTATION_STYLE,
                    ),
                )
                agents[agent_id].add_nav_goal(goal)
            except RuntimeError:
                raise RuntimeError(
                    "Could not convert svg path to shapely geometry for svg id: {}".format(
                        uid
                    )
                )

    @classmethod
    def load_agents(
        cls,
        map: BinaryOccupancyGrid,
        collision_margin: float,
        config: NamoConfigYamlModel,
        config_dir: str,
        logs_dir: str,
        logger: utils.NamosimLogger,
    ) -> t.List["agts.Agent"]:
        if config.svg_file is None:
            return []

        svg_path = os.path.join(config_dir, config.svg_file)

        # Import entire world from svg file
        svg_doc = minidom.parse(svg_path)
        conversion.set_all_id_attributes_as_ids(svg_doc)
        bounds = map.get_bounds()
        height = bounds[3]

        agent_configs: t.Dict[str, AgentConfigYamlModel] = {
            x.id: x for x in config.agents
        }

        agent_poses: t.Dict[str, Pose2D] = {}
        agent_polygons: t.Dict[str, Polygon] = {}

        robots_layer = svg_doc.getElementById("robots_layer")
        if robots_layer:
            for el in robots_layer.getElementsByTagNameNS("*", "path"):
                uid = el.getAttribute("id")
                path_data = el.getAttribute("d")
                agent_id = el.getAttribute("agent_id")
                try:
                    agent_config = agent_configs[agent_id]
                    polygon: Polygon = conversion.svg_pathd_to_shapely_geometry(  # type: ignore
                        svg_path=path_data, ymax_meters=height, scale=map.cell_size
                    )
                    agent_polygons[agent_id] = polygon
                    agent_poses[agent_id] = Pose2D(
                        t.cast(float, list(polygon.centroid.coords)[0][0]),
                        t.cast(float, list(polygon.centroid.coords)[0][1]),
                        0,
                    )
                except RuntimeError:
                    raise RuntimeError(
                        "Could not convert svg path to shapely geometry for svg id: {}".format(
                            uid
                        )
                    )
        agents: t.List["agts.Agent"] = []
        for agent_config in config.agents:
            pose: Pose2D = Pose2D(0, 0, 0)
            agent_polygon = t.cast(
                Polygon, Point(pose[0], pose[1]).buffer(agent_config.radius)
            )
            if agent_config.initial_pose:
                pose = Pose2D(
                    agent_config.initial_pose[0],
                    agent_config.initial_pose[1],
                    agent_config.initial_pose[2],
                )
                agent_polygon = t.cast(
                    Polygon, Point(pose[0], pose[1]).buffer(agent_config.radius)
                )
            elif agent_config.id in agent_poses:
                pose = agent_poses[agent_config.id]
                agent_polygon = agent_polygons[agent_config.id]

            agent = World.construct_agent_from_yaml_model(
                agent=agent_config,
                goals=[],
                logs_dir=logs_dir,
                robot_polygon=agent_polygon,
                agent_style=AgentStyle(),
                init_pose=pose,
                cell_size=map.cell_size,
                collision_margin=collision_margin,
                logger=logger,
            )
            agents.append(agent)

        return agents

    @classmethod
    def load_from_yaml(
        cls,
        yaml_file: str,
        logs_dir: str = "namo_logs",
        logger: utils.NamosimLogger | None = None,
    ) -> Self:

        config = namo_config_from_yaml(yaml_file)
        config_dir = os.path.dirname(yaml_file)
        map_yaml_path = os.path.join(config_dir, config.map_yaml)
        map = BinaryOccupancyGrid.load_from_yaml(map_yaml_path)
        collision_margin = map.cell_size
        if config.collision_margin_cm:
            collision_margin = config.collision_margin_cm / 100

        world = cls(map=map, collision_margin=collision_margin, logger=logger)

        agents = World.load_agents(
            map=map,
            collision_margin=collision_margin,
            config=config,
            config_dir=config_dir,
            logs_dir=logs_dir,
            logger=world.logger,
        )
        for agent in agents:
            world.add_agent(agent)

        movables = cls.load_movables(map, config, config_dir)
        for mo in movables:
            world.add_entity(mo)

        cls.load_goals(map, config, config_dir, world.agents)

        for agent in world.agents.values():
            agent.init(world)

        return world

    def get_inflated_grid_for_entity(self, entity_id: str):
        entity = self.dynamic_entities[entity_id]
        inflation_radius = entity.circumscribed_radius + self.collision_margin
        polygons = {
            x.uid: x.polygon
            for x in self.dynamic_entities.values()
            if x.uid != entity_id
        }
        grid = copy.deepcopy(self.map)
        grid = grid.inflate_map_destructive(inflation_radius)
        grid.update_polygons(polygons)
        return grid

    def drop_obstacle(self, agent_id: str, resolve_collisions: bool):
        if agent_id in self.entity_to_agent.inverse:
            del self.entity_to_agent.inverse[agent_id]
        if resolve_collisions:
            self.resolve_collisions(agent_id)

    def resolve_collisions(self, entity_id: str):
        entity = self.dynamic_entities[entity_id]
        grid = self.get_inflated_grid_for_entity(entity_id)
        entity_cell = grid.pose_to_cell(entity.pose[0], entity.pose[1])
        if grid.get_cell_value(entity_cell) == 0:
            return

        # do BFS to find the nearest unoccupied cell
        nearest = grid.find_nearest_free_cell(entity_cell)
        if nearest is None:
            return
        x, y = grid.get_cell_center(nearest)
        entity.move_to_pose(Pose2D(x, y, entity.pose[2]))

        # re-init all agents
        for agent in self.agents.values():
            agent.init(self)
