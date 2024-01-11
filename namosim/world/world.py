import copy
import os
import typing as t
from xml.dom import minidom

import numpy as np
import shapely.affinity as affinity
from bidict import bidict
from shapely.geometry import LineString, Polygon, box
from typing_extensions import Self

import namosim.agents as agts
import namosim.utils.conversion as conversion
import namosim.utils.utils as utils
from namosim.data_models import UID, NamosimConfigModel, PoseModel
from namosim.display import conversions
from namosim.utils import collision
from namosim.world.discretization_data import DiscretizationData
from namosim.world.entity import Entity, Movability, Style
from namosim.world.goal import Goal
from namosim.world.obstacle import Obstacle
from namosim.world.sensors.omniscient_sensor import OmniscientSensor


class World:
    def __init__(
        self,
        *,
        discretization_data: DiscretizationData,
        config: NamosimConfigModel,
        entities: t.Optional[t.Dict[UID, Entity]] = None,
        agents: t.Optional[t.Dict[UID, "agts.Agent"]] = None,
        entity_to_agent: t.Optional[bidict[UID, UID]] = None,
        goals: t.Optional[t.Dict[UID, Goal]] = None,
        init_geometry_filename: str = "world_name_placeholder.svg",
        init_geometry_file: t.Optional[minidom.Document] = None,
        logger: utils.CustomLogger,
    ):
        self.config = config
        self.entities = entities or dict()
        self.agents: t.Dict[UID, "agts.Agent"] = agents if agents else {}
        self.entity_to_agent = entity_to_agent or bidict()
        self.discretization_data = discretization_data
        self.agent_configs = ({x.agent_id: x for x in config.agents},)
        self.init_geometry_file = init_geometry_file
        if init_geometry_file:
            conversion.set_all_id_attributes_as_ids(init_geometry_file)
            conversion.clean_attributes(init_geometry_file)
        self.init_geometry_filename = init_geometry_filename
        self.init_geometry_file = init_geometry_file
        self.goals: t.Dict[UID, Goal] = goals or dict()
        self.logger = logger

    # Constructor
    @classmethod
    def load_from_svg(
        cls, world_svg_path: str, logs_dir: str, logger: utils.CustomLogger
    ) -> Self:
        # Import entire world from svg file
        svg_doc = minidom.parse(world_svg_path)
        config = NamosimConfigModel.from_xml(
            svg_doc.getElementsByTagName("namo_config")[0].toxml()
        )
        svg_filename = os.path.basename(world_svg_path)

        if not svg_doc.documentElement.hasAttribute("viewBox"):
            raise Exception("svg has no viewBox attribute")

        # Split the viewBox attribute into its components
        viewbox_values = [
            float(x) for x in svg_doc.documentElement.getAttribute("viewBox").split()
        ]

        discretization_data = World.get_discretization_data(
            min_x=viewbox_values[0],
            min_y=viewbox_values[1],
            max_x=viewbox_values[2],
            max_y=viewbox_values[3],
            config=config,
        )

        svg_paths = {}
        for el in svg_doc.getElementsByTagNameNS("*", "path"):
            svg_paths[el.getAttribute("id")] = el.getAttribute("d")

        shapely_geoms: t.Dict[str, t.Union[Polygon, LineString]] = dict()

        # Convert imported geometry to shapely polygons
        for svg_id, svg_path in svg_paths.items():
            try:
                shapely_geoms[svg_id] = conversion.svg_pathd_to_shapely_geometry(
                    svg_path, scaling_value=1.0
                )
            except RuntimeError:
                raise RuntimeError(
                    "Could not convert svg path to shapely geometry for svg id: {}".format(
                        svg_id
                    )
                )

        # Center the imported geometries
        bounding_box = box(
            viewbox_values[0],
            viewbox_values[1],
            viewbox_values[2],
            viewbox_values[3],
        )
        translation_to_center: t.List[float] = [
            bounding_box.centroid.coords[0][0],
            bounding_box.centroid.coords[0][1],
        ]
        for svg_id, polygon in shapely_geoms.items():
            shapely_geoms[svg_id] = t.cast(
                Polygon | LineString,
                affinity.translate(
                    polygon, -translation_to_center[0], translation_to_center[1]
                ),
            )

        world = cls(
            init_geometry_filename=svg_filename,
            init_geometry_file=svg_doc,
            discretization_data=discretization_data,
            config=config,
            logger=logger,
        )

        # Get all things
        for el in svg_doc.getElementsByTagName("*"):
            id = el.getAttribute("id")
            type_ = el.getAttribute("type")

            if not id:
                continue

            if el.tagName in ["svg:path", "path"] and type_ == "movable":
                polygon = shapely_geoms[id]
                style = Style.from_string(el.getAttribute("style"))
                pose = (
                    t.cast(float, list(polygon.centroid.coords)[0][0]),
                    t.cast(float, list(polygon.centroid.coords)[0][1]),
                    0.0,
                )
                movable_box = Obstacle(
                    type_="movable",
                    name=id,
                    polygon=polygon,
                    pose=pose,
                    style=style,
                    movability=Movability.MOVABLE,
                    full_geometry_acquired=True,
                )
                world.add_entity(movable_box)
            if el.tagName in ["svg:path", "path"] and type_ == "wall":
                polygon = shapely_geoms[id]
                style = Style.from_string(el.getAttribute("style"))
                pose = (
                    t.cast(float, list(polygon.centroid.coords)[0][0]),
                    t.cast(float, list(polygon.centroid.coords)[0][1]),
                    0.0,
                )
                wall = Obstacle(
                    type_="wall",
                    name=id,
                    polygon=polygon,
                    pose=pose,
                    style=style,
                    movability=Movability.STATIC,
                    full_geometry_acquired=True,
                )
                world.add_entity(wall)

        for agent in config.agents:
            el = svg_doc.getElementById(agent.agent_id)
            if not el:
                raise Exception(f"Robot {agent.agent_id} not found in svg")

            robot_polygon: Polygon | None = None
            direction_polygon: Polygon | None = None
            robot_style: str = ""
            robot_pose = [
                0.0,
                0.0,
                0.0,
            ]
            for sub_el in el.getElementsByTagNameNS("*", "path"):
                sub_id = sub_el.getAttribute("id")
                if sub_el.getAttribute("type") == "shape":
                    robot_style = sub_el.getAttribute("style")
                    robot_polygon = shapely_geoms[sub_id]
                elif sub_el.getAttribute("type") == "orientation":
                    direction_polygon = shapely_geoms[sub_id]
                    theta = get_orientation(direction_polygon)
                    robot_pose[2] = theta

            if not robot_polygon:
                raise Exception("No robot shape polygon was found")

            robot_pose[0] = t.cast(float, list(robot_polygon.centroid.coords)[0][0])
            robot_pose[1] = t.cast(float, list(robot_polygon.centroid.coords)[0][1])

            goal_poses: t.List[PoseModel] = []
            for goal in agent.goals:
                goal_el = svg_doc.getElementById(goal.goal_id)
                if not goal_el:
                    raise Exception(f"Goal {goal.goal_id} not found in svg")

                goal_polygon: Polygon | None = None
                direction_polygon: Polygon | None = None
                goal_pose = [
                    0.0,
                    0.0,
                    0.0,
                ]
                for sub_el in goal_el.getElementsByTagNameNS("*", "path"):
                    sub_id = sub_el.getAttribute("id")
                    if sub_el.getAttribute("type") == "shape":
                        goal_polygon = shapely_geoms[sub_id]
                    elif sub_el.getAttribute("type") == "orientation":
                        theta = get_orientation(shapely_geoms[sub_id])
                        goal_pose[2] = theta

                if not goal_polygon:
                    raise Exception(
                        f"No goal_shape polygon was found for goal {goal.goal_id}"
                    )

                goal_pose[0] = t.cast(float, list(goal_polygon.centroid.coords)[0][0])
                goal_pose[1] = t.cast(float, list(goal_polygon.centroid.coords)[0][1])
                goal = Goal(
                    name=goal.goal_id,
                    polygon=goal_polygon,
                    pose=tuple(goal_pose),  # type: ignore
                )
                world.goals[goal.uid] = goal
                goal_poses.append((goal_pose[0], goal_pose[1], goal_pose[2]))

            if agent.behavior.type == "stilman_2005_behavior":
                new_robot = agts.Stilman2005Agent(
                    navigation_goals=goal_poses,
                    params=agent.behavior.parameters,
                    logs_dir=logs_dir,
                    full_geometry_acquired=True,
                    name=agent.agent_id,
                    polygon=robot_polygon,
                    style=Style.from_string(robot_style),
                    pose=(robot_pose[0], robot_pose[1], robot_pose[2]),
                    sensors=[OmniscientSensor()],
                    push_only_list=[],
                    force_pushes_only=False,
                    movable_whitelist=["box"],
                    cell_size=config.cell_size,
                    logger=logger,
                )
            elif agent.behavior.type == "navigation_only_behavior":
                new_robot = agts.NavigationOnlyAgent(
                    navigation_goals=goal_poses,
                    logs_dir=logs_dir,
                    full_geometry_acquired=True,
                    name=agent.agent_id,
                    polygon=robot_polygon,
                    style=Style.from_string(robot_style),
                    pose=(robot_pose[0], robot_pose[1], robot_pose[2]),
                    sensors=[OmniscientSensor()],
                    push_only_list=[],
                    force_pushes_only=False,
                    movable_whitelist=["box"],
                    cell_size=config.cell_size,
                    logger=logger,
                )
            elif agent.behavior.type == "stilman_only_behavior":
                new_robot = agts.StilmanOnlyAgent(
                    navigation_goals=goal_poses,
                    params=agent.behavior.parameters,
                    logs_dir=logs_dir,
                    full_geometry_acquired=True,
                    name=agent.agent_id,
                    polygon=robot_polygon,
                    style=Style.from_string(robot_style),
                    pose=(robot_pose[0], robot_pose[1], robot_pose[2]),
                    sensors=[OmniscientSensor()],
                    push_only_list=[],
                    force_pushes_only=False,
                    movable_whitelist=["box"],
                    cell_size=config.cell_size,
                    logger=logger,
                )
            else:
                raise NotImplementedError(
                    "You tried to associate entity '{agent_name}' with a behavior named"
                    "'{b_name}' that is not implemented yet."
                    "Maybe you mispelled something ?".format(
                        agent_name=agent.agent_id, b_name=agent.behavior.type
                    )
                )

            world.add_entity(new_robot)
            world.agents[new_robot.uid] = new_robot

        goals_node = svg_doc.getElementById("goals")
        if goals_node:
            goals_node.parentNode.removeChild(goals_node)

        for agent in world.agents.values():
            agent.init(world)

        return world

    def to_svg(self) -> minidom.Document:
        if self.init_geometry_file:
            svg_data: minidom.Document = minidom.parseString(
                self.init_geometry_file.toxml()
            )

            # clear geometries
            els_to_del = list(svg_data.getElementsByTagNameNS("*", "path"))
            for el in els_to_del:
                if el.parentNode:
                    el.parentNode.removeChild(el)

            current_geometries_names_to_ids = {
                entity.name: uid for uid, entity in self.entities.items()
            }
            # The 4 following lines are a hack to compensate for the fact the geometries are not associated with entity
            # for uid, entity in self.entities.items():
            #     if isinstance(entity, Robot):
            #         current_geometries_names_to_ids[entity.name + "_shape"] = uid
            #         current_geometries_names_to_ids[entity.name + "_direction"] = uid
            current_geometries_names = set(current_geometries_names_to_ids.keys())

            for geometry_name in current_geometries_names:
                entity = self.entities[current_geometries_names_to_ids[geometry_name]]
                if isinstance(entity, Obstacle):
                    if entity.movability in [Movability.STATIC, Movability.UNMOVABLE]:
                        style = conversion.FIXED_ENTITY_STYLE
                    elif entity.movability == Movability.MOVABLE:
                        style = conversion.MOVABLE_ENTITY_STYLE
                    elif entity.movability == Movability.UNKNOWN:
                        style = conversion.UNKNOWN_ENTITY_STYLE
                    else:
                        raise NotImplementedError(
                            "Can only export new obstacles entities that have a 'movability' attribute of "
                            "value ['static', 'unmovable', 'movable', 'unknown'], got {}.".format(
                                entity.movability
                            )
                        )
                    conversion.add_shapely_geometry_to_svg(
                        entity.polygon,
                        entity.name,
                        style,
                        svg_data,
                        map_width=self.discretization_data.width,
                        map_height=self.discretization_data.height,
                    )
                elif isinstance(entity, agts.Agent):
                    robot_group = conversion.add_group(
                        svg_data, entity.name, is_layer=False
                    )
                    # Add robot shape
                    conversion.add_shapely_geometry_to_svg(
                        entity.polygon,
                        entity.name + "_shape",
                        conversion.ROBOT_ENTITY_STYLE,
                        svg_data,
                        robot_group,
                        map_width=self.discretization_data.width,
                        map_height=self.discretization_data.height,
                    )
                    # Add robot direction shape
                    radius = utils.get_inscribed_radius(entity.polygon)
                    point_a = np.array([entity.pose[0], entity.pose[1]])
                    direction = np.array(utils.direction_from_yaw(entity.pose[2]))
                    point_b = point_a + direction * radius

                    poly = conversions.path_to_polygon(
                        points=[point_a, point_b], line_width=radius / 4
                    )
                    conversion.add_shapely_geometry_to_svg(
                        shapely_geometry=poly,
                        uname=entity.name + "_direction",
                        style=conversion.ORIENTATION_STYLE,
                        svg_data=svg_data,
                        svg_group=robot_group,
                        map_width=self.discretization_data.width,
                        map_height=self.discretization_data.height,
                    )
                else:
                    raise NotImplementedError(
                        "Only entities of class [Robot, Obstacle] can be created in SVG file for now."
                    )  # TODO Add creation of new SVG goals
        else:
            raise NotImplementedError(
                "TODO : use bootstrap SVG data to build new SVG file from scratch"
            )
        return svg_data

    def add_entity(self, new_entity: t.Any):
        # for obj in self.entities.values():
        #     is_within = new_entity.within(obj)
        #     if is_within:
        #         raise EntityPlacementException("Entity {} would be within entity {}. Cannot load world.".format(
        #             new_entity.name, obj.name))
        self.entities[new_entity.uid] = new_entity

    def remove_entity(self, entity_uid: UID):
        if entity_uid in self.entities:
            del self.entities[entity_uid]
        if entity_uid in self.agents:
            del self.agents[entity_uid]
        if entity_uid in self.entity_to_agent:
            del self.entity_to_agent[entity_uid]

    @staticmethod
    def get_discretization_data(
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        config: NamosimConfigModel,
    ) -> DiscretizationData:
        width, height = max_x - min_x, max_y - min_y
        grid_pose = (min_x, min_y, 0.0)
        d_width = int(round(width / config.cell_size))
        d_height = int(round(height / config.cell_size))

        return DiscretizationData(
            res=config.cell_size,
            grid_pose=grid_pose,
            width=width,
            height=height,
            d_width=d_width,
            d_height=d_height,
        )

    # TO DEPRECATE
    def get_entity_uid_from_name(self, name: str) -> UID:
        for entity_uid, entity in self.entities.items():
            if entity.name == name:
                return entity_uid
        raise LookupError(
            "Could not find an entity in this world with name : {name}.".format(
                name=name
            )
        )

    def light_copy(self, ignored_entities: t.Iterable[UID]):
        entity_to_agent: bidict[UID, UID] = bidict()
        for e, a in self.entity_to_agent.items():
            if a not in ignored_entities and e not in ignored_entities:
                entity_to_agent[e] = a
        entities = {}
        agents = {}
        for uid, e in self.entities.items():
            if uid in ignored_entities:
                continue

            e = e.light_copy()
            entities[uid] = e
            if isinstance(e, agts.Agent):
                agents[uid] = e

        return World(
            config=self.config,
            entities=entities,
            agents=agents,
            entity_to_agent=entity_to_agent,
            discretization_data=copy.deepcopy(self.discretization_data),
            goals=copy.deepcopy(self.goals),
            init_geometry_filename=self.init_geometry_filename,
            init_geometry_file=self.init_geometry_file,
            logger=self.logger,
        )

    def set_entity_polygon(self, id: int, polygon: Polygon):
        self.entities[id].polygon = polygon

    def save_to_files(
        self,
        svg_filepath: t.Optional[str] = None,
        svg_data: t.Optional[t.Any] = None,
    ):
        svg_filepath = svg_filepath or "./" + self.init_geometry_filename

        # Generate SVG data
        if not svg_data:
            svg_data = self.to_svg()

        # Save SVG to specified path
        with open(svg_filepath, "w+") as f:
            svg_data.writexml(f)

    def get_map_bounds(self):
        return (
            0,
            0,
            self.discretization_data.width,
            self.discretization_data.height,
        )

    def is_holding_obstacle(self, agent_id: UID) -> bool:
        return agent_id in self.entity_to_agent.inverse

    def get_robot_conflict_radius(self, robot_id: UID, obstacle_id: UID | None = None):
        robot = self.agents[robot_id]
        center = robot.polygon.centroid
        radius_for_move = (
            robot.circumscribed_radius + utils.SQRT_OF_2 * self.config.cell_size
        )
        radius_for_grab_or_release = (
            robot.circumscribed_radius + robot.grab_and_release_distance
        )

        conflict_radius = radius_for_move

        if self.is_holding_obstacle(robot_id):
            obstacle_id = self.entity_to_agent.inverse[robot.uid]

        if obstacle_id is not None:
            obstacle = self.entities[obstacle_id]
            conflict_radius = (
                center.hausdorff_distance(obstacle.polygon)
                + utils.SQRT_OF_2 * self.config.cell_size
            )

            # Account for possible release
            conflict_radius = max(conflict_radius, radius_for_grab_or_release)
        else:
            # Enlarge radius to account for possible grabs
            for uid, obstacle in self.entities.items():
                if (
                    isinstance(obstacle, Obstacle)
                    and uid not in self.entity_to_agent
                    and obstacle.movability == Movability.MOVABLE
                ):
                    if obstacle.polygon.buffer(
                        robot.grab_and_release_distance,
                        join_style="mitre",
                    ).intersects(robot.polygon):
                        conflict_radius = radius_for_grab_or_release
                        break
        return conflict_radius

    def get_polygon_collisions(self, uid: UID, others: t.Iterable[UID]) -> t.Any:
        other_polygons = {uid: self.entities[uid].polygon for uid in others}
        others_aabb_tree = collision.polygons_to_aabb_tree(other_polygons)
        collisions = collision.check_static_collision(
            main_uid=uid,
            polygon=self.entities[uid].polygon,
            other_entities_polygons=other_polygons,
            aabb_tree=others_aabb_tree,
        )
        return collisions


def get_orientation(geom: (Polygon | LineString)) -> float:
    orientation_geom: t.List[t.List[float]] = list(geom.coords)  # type: ignore
    orientation_vector = (
        orientation_geom[1][0] - orientation_geom[0][0],
        orientation_geom[1][1] - orientation_geom[0][1],
    )
    theta = utils.yaw_from_direction(orientation_vector)
    return theta
