import copy
import os
import typing as t
from xml.dom import minidom

import numpy as np
import shapely.affinity as affinity
from bidict import bidict
from shapely import union_all
from shapely.geometry import LineString, Polygon, box
from typing_extensions import Self

import namosim.utils.conversion as conversion
import namosim.utils.utils as utils
import namosim.world.robot as robot
from namosim.data_models_v2 import NamosimConfigModel
from namosim.display import conversions
from namosim.world.discretization_data import DiscretizationData
from namosim.world.entity import Entity, Style
from namosim.world.goal import Goal
from namosim.world.obstacle import Obstacle
from namosim.world.sensors.omniscient_sensor import OmniscientSensor
from namosim.world.taboo import Taboo


class WorldV2:
    def __init__(
        self,
        discretization_data: DiscretizationData,
        config: NamosimConfigModel,
        entities: t.Optional[t.Dict[int, Entity]] = None,
        entity_to_agent: t.Optional[bidict[int, int]] = None,
        taboo_zones: t.Optional[t.Dict[int, Taboo]] = None,
        goals: t.Optional[t.Dict[int, Goal]] = None,
        geometry_scale: float = 1.0,
        init_geometry_filename: str = "world_name_placeholder.svg",
        init_geometry_file: t.Optional[minidom.Document] = None,
    ):
        self.config = config
        self.entities = entities or dict()
        self.entity_to_agent = entity_to_agent or bidict()
        self.discretization_data = discretization_data
        self.agent_configs = ({x.agent_id: x for x in config.agents},)

        self.geometry_scale = geometry_scale
        self.scaling_value = self.geometry_scale

        self.init_geometry_file = init_geometry_file
        if init_geometry_file:
            conversion.set_all_id_attributes_as_ids(init_geometry_file)
            conversion.clean_attributes(init_geometry_file)
        self.init_geometry_filename = init_geometry_filename
        self.init_geometry_file = init_geometry_file

        self.taboo_zones: t.Dict[int, Taboo] = taboo_zones or dict()
        self.goals: t.Dict[int, Goal] = goals or dict()

    # Constructor
    @classmethod
    def load_from_svg(cls, world_svg_path: str) -> Self:
        # Import entire world from svg file
        svg_doc = minidom.parse(world_svg_path)
        # svg_doc = tree.getroot()
        # ns = {"svg": "http://www.w3.org/2000/svg"}

        # namo_config_el = svg_doc.findall(".//svg:namo_config", ns)[0]
        # ET.register_namespace("", "http://www.w3.org/2000/svg")
        # namo_config_xml = (
        #     ET.tostring(namo_config_el, xml_declaration=False)
        #     .decode("utf-8")
        #     .replace(ns["svg"], "")
        #     .replace('xmlns=""', "")
        # )
        config = NamosimConfigModel.from_xml(
            svg_doc.getElementsByTagName("namo_config")[0].toxml()
        )
        svg_filename = os.path.basename(world_svg_path)

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
        # TODO Fix this so that it only accounts for obstacles in polygon layer otherwise, things might get messy with
        #  direction vectors that get outside of the obstacle polygons
        # Center the imported geometries
        unioned_polygons = t.cast(Polygon, union_all(list(shapely_geoms.values())))
        bounding_box = box(
            unioned_polygons.bounds[0],
            unioned_polygons.bounds[1],
            unioned_polygons.bounds[2],
            unioned_polygons.bounds[3],
        )
        # print(str((bounding_box.bounds[2] - bounding_box.bounds[0], bounding_box.bounds[3] - bounding_box.bounds[1])))
        translation_to_center: t.List[float] = [
            bounding_box.centroid.coords[0][0],
            bounding_box.centroid.coords[0][1],
        ]
        for svg_id, polygon in shapely_geoms.items():
            shapely_geoms[svg_id] = t.cast(
                Polygon | LineString,
                affinity.translate(
                    polygon, -translation_to_center[0], -translation_to_center[1]
                ),
            )

        # Get map discretization parameters
        dd = DiscretizationData(
            res=config.cell_size,
        )

        world = cls(
            geometry_scale=1.0,
            init_geometry_filename=svg_filename,
            init_geometry_file=svg_doc,
            discretization_data=dd,
            config=config,
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
                    movability="movable",
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
                    movability="static",
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
            new_robot = robot.Robot(
                name=agent.agent_id,
                full_geometry_acquired=True,
                polygon=robot_polygon,
                pose=tuple(robot_pose),  # type: ignore
                sensors=[OmniscientSensor()],
                push_only_list=[],
                force_pushes_only=True,
                movable_whitelist=["box"],
                style=Style.from_string(robot_style),
            )
            world.add_entity(new_robot)

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

        world.update_dd()

        goals_node = svg_doc.getElementById("goals")
        if goals_node:
            goals_node.parentNode.removeChild(goals_node)

        return world

    def to_svg(self) -> minidom.Document:
        if self.init_geometry_file:
            svg_data: minidom.Document = copy.deepcopy(self.init_geometry_file)

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
                    if (
                        entity.movability == "static"
                        or entity.movability == "unmovable"
                    ):
                        style = conversion.FIXED_ENTITY_STYLE
                    elif entity.movability == "movable":
                        style = conversion.MOVABLE_ENTITY_STYLE
                    elif entity.movability == "unknown":
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
                        scale=self.scaling_value,
                        map_width=self.discretization_data.width,
                        map_height=self.discretization_data.height,
                    )
                elif isinstance(entity, robot.Robot):
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
                        scale=self.scaling_value,
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
                        poly,
                        entity.name + "_direction",
                        conversion.ORIENTATION_STYLE,
                        svg_data,
                        robot_group,
                        scale=self.scaling_value,
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

    def remove_entity(self, entity_uid: int):
        if entity_uid in self.entities:
            del self.entities[entity_uid]
        else:
            raise KeyError(
                "Warning, you tried to remove an entity that is not registered in this world !"
            )

    def get_map_bounds(self) -> t.Tuple[float, float, float, float]:
        if len(self.entities) == 0:
            raise ValueError(
                "There are no entities to populate the grid, it can't be created !"
            )
        polygons = [entity.polygon for entity in self.entities.values()]
        map_min_x, map_min_y, map_max_x, map_max_y = (
            float("inf"),
            float("inf"),
            -float("inf"),
            -float("inf"),
        )
        for polygon in polygons:
            min_x, min_y, max_x, max_y = polygon.bounds
            map_min_x, map_min_y = min(map_min_x, min_x), min(map_min_y, min_y)
            map_max_x, map_max_y = max(map_max_x, max_x), max(map_max_y, max_y)
        return map_min_x, map_min_y, map_max_x, map_max_y

    # TO DEPRECATE
    def update_dd(self):
        min_x, min_y, max_x, max_y = self.get_map_bounds()
        width, height = max_x - min_x, max_y - min_y

        self.discretization_data.grid_pose = (min_x, min_y, 0.0)
        self.discretization_data.width, self.discretization_data.height = width, height
        self.discretization_data.d_width, self.discretization_data.d_height = (
            int(round(self.discretization_data.width / self.discretization_data.res)),
            int(round(self.discretization_data.height / self.discretization_data.res)),
        )
        new_hash = hash(self.discretization_data)
        if new_hash != self.discretization_data.saved_hash:
            self.discretization_data.saved_hash = new_hash

    # TO DEPRECATE
    def get_entity_uid_from_name(self, name: str) -> int:
        for entity_uid, entity in self.entities.items():
            if entity.name == name:
                return entity_uid
        raise LookupError(
            "Could not find an entity in this world with name : {name}.".format(
                name=name
            )
        )

    def light_copy(self, ignored_entities: t.Iterable[int]):
        entity_to_agent: bidict[int, int] = bidict()
        for e, a in self.entity_to_agent.items():
            if a not in ignored_entities and e not in ignored_entities:
                entity_to_agent[e] = a

        return WorldV2(
            config=self.config,
            entities={
                uid: entity.light_copy()
                for uid, entity in self.entities.items()
                if uid not in ignored_entities
            },
            entity_to_agent=entity_to_agent,
            discretization_data=copy.deepcopy(self.discretization_data),
            taboo_zones=copy.deepcopy(self.taboo_zones),
            goals=copy.deepcopy(self.goals),
            geometry_scale=self.geometry_scale,
            init_geometry_filename=self.init_geometry_filename,
            init_geometry_file=self.init_geometry_file,
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


def get_orientation(geom: (Polygon | LineString)) -> float:
    orientation_geom: t.List[t.List[float]] = list(geom.coords)  # type: ignore
    orientation_vector = (
        orientation_geom[1][0] - orientation_geom[0][0],
        orientation_geom[1][1] - orientation_geom[0][1],
    )
    theta = utils.yaw_from_direction(orientation_vector)
    return theta
