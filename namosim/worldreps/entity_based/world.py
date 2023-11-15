import copy
import json
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
from namosim.worldreps.discretization_data import DiscretizationData
from namosim.worldreps.entity_based.entity import Style
from namosim.worldreps.entity_based.goal import Goal
from namosim.worldreps.entity_based.models import WorldModel
from namosim.worldreps.entity_based.obstacle import Obstacle
from namosim.worldreps.entity_based.robot import Robot
from namosim.worldreps.entity_based.sensors.g_fov_sensor import GFOVSensor
from namosim.worldreps.entity_based.sensors.omniscient_sensor import OmniscientSensor
from namosim.worldreps.entity_based.sensors.s_fov_sensor import SFOVSensor
from namosim.worldreps.entity_based.taboo import Taboo


class World:
    SCALING_CONSTANT = 1.0 / 3.5433

    def __init__(
        self,
        discretization_data: DiscretizationData,
        entities: t.Optional[t.Dict[int, t.Any]] = None,
        entity_to_agent: t.Optional[bidict[int, int]] = None,
        taboo_zones: t.Optional[t.Dict[int, Taboo]] = None,
        goals: t.Optional[t.Dict[int, Goal]] = None,
        geometry_scale: float = 1.0,
        init_json_filename: str = "world_name_placeholder.json",
        init_geometry_filename: str = "world_name_placeholder.svg",
        init_geometry_file: t.Optional[minidom.Document] = None,
    ):
        self.entities = entities or dict()
        self.entity_to_agent = entity_to_agent or bidict()
        self.discretization_data = discretization_data

        self.geometry_scale = geometry_scale
        self.scaling_value = self.geometry_scale

        self.init_geometry_file = init_geometry_file
        if init_geometry_file:
            conversion.set_all_id_attributes_as_ids(init_geometry_file)
            conversion.clean_attributes(init_geometry_file)
        self.init_geometry_filename = init_geometry_filename
        self.init_geometry_file = init_geometry_file
        self.init_json_filename = init_json_filename

        self.taboo_zones: t.Dict[int, Taboo] = taboo_zones or dict()
        self.goals: t.Dict[int, Goal] = goals or dict()

    # Constructor
    @classmethod
    def load_from_json(cls, world_file_path: str) -> Self:
        # Import world configuration file
        with open(world_file_path) as f:
            world_json = json.load(f)
        config = WorldModel.model_validate(world_json)

        # Import SVG geometry file
        svg_path = config.files.geometry_file

        if not os.path.isabs(svg_path):
            working_directory = os.path.dirname(world_file_path)
            svg_path = os.path.join(working_directory, svg_path)

        svg_filename = os.path.basename(svg_path)
        svg_doc = minidom.parse(svg_path)
        svg_paths = {
            path.getAttribute("id"): path.getAttribute("d")
            for path in svg_doc.getElementsByTagName("path")
            + svg_doc.getElementsByTagName("svg:path")
        }

        shapely_geoms: t.Dict[str, t.Union[Polygon, LineString]] = dict()

        if config.no_scaling_workaround:
            scaling_value = config.geometry_scale
        else:
            # TODO Remove the scaling constant once all the worlds SVGs have been fixed
            scaling_value = World.SCALING_CONSTANT * config.geometry_scale
        # Convert imported geometry to shapely polygons
        for svg_id, svg_path in svg_paths.items():
            try:
                shapely_geoms[svg_id] = conversion.svg_pathd_to_shapely_geometry(
                    svg_path, scaling_value
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
            res=config.discretization_data.res,
            inflation_radius=config.discretization_data.inflation_radius,
            cost_lethal=config.discretization_data.cost_lethal,
            cost_inscribed=config.discretization_data.cost_inscribed,
            cost_circumscribed=config.discretization_data.cost_circumscribed,
            cost_possibly_nonfree=config.discretization_data.cost_possibly_nonfree,
        )

        world = cls(
            geometry_scale=scaling_value,
            init_geometry_filename=svg_filename,
            init_geometry_file=svg_doc,
            init_json_filename=world_file_path,
            discretization_data=dd,
        )

        first_robot = None

        # Get all things
        for entity_data in config.things.entities:
            # Pose of object definition
            theta: float = 0
            if entity_data.geometry.orientation_id is not None:
                # If a drawn vector in the SVG is defined as orientation, use it
                geom = shapely_geoms[entity_data.geometry.orientation_id]
                theta = get_orientation(geom)

            # Polygonal geometry object definition
            if entity_data.geometry.from_ == "file":
                # If geometry is defined in SVG file, prioritize using it
                polygon = shapely_geoms[entity_data.geometry.id]
                polygon_el = svg_doc.getElementById(entity_data.geometry.id)
                if not polygon_el:
                    print(
                        "Could not find geometry {} in svg file. Next entity.".format(
                            entity_data.geometry.id
                        )
                    )
                    continue
                else:
                    style = Style.from_string(polygon_el.getAttribute("style"))
            else:
                raise NotImplementedError(
                    "You can't define a geometry in the json file manually for now."
                )

            # Adjust initial position in pose if not given only by SVG file
            pose = [
                list(polygon.centroid.coords)[0][0],
                list(polygon.centroid.coords)[0][1],
                theta,
            ]

            if entity_data.type_ == "robot":
                sensors_data = entity_data.sensors

                sensors: t.List[t.Union[OmniscientSensor, GFOVSensor, SFOVSensor]] = []
                for sensor_data in sensors_data:
                    if sensor_data.type_ == "perfect_g_fov":
                        sensors.append(
                            GFOVSensor(
                                sensor_data.max_radius,
                                sensor_data.min_radius,
                                sensor_data.opening_angle,
                                pose,
                            )
                        )
                    elif sensor_data.type_ == "perfect_s_fov":
                        sensors.append(
                            SFOVSensor(
                                sensor_data.max_radius,
                                sensor_data.min_radius,
                                sensor_data.opening_angle,
                                pose,
                            )
                        )
                    elif sensor_data.type_ == "omniscient":
                        sensors.append(OmniscientSensor())

                new_robot = Robot(
                    name=entity_data.name,
                    full_geometry_acquired=True,
                    polygon=polygon,
                    pose=tuple(pose),
                    sensors=sensors,
                    push_only_list=entity_data.push_only_list,
                    force_pushes_only=entity_data.force_pushes_only,
                    movable_whitelist=entity_data.movable_whitelist,
                    style=style,
                )
                if not first_robot:
                    first_robot = new_robot

                # Prevent specified inflation radius to be smaller than actual polygon

                if new_robot.min_inflation_radius > dd.inflation_radius:
                    dd.inflation_radius = new_robot.min_inflation_radius

                world.add_entity(new_robot)
            else:
                new_object = Obstacle(
                    name=entity_data.name,
                    polygon=polygon,
                    pose=pose,
                    type_in=entity_data.type_,
                    full_geometry_acquired=True,
                    movability="static"
                    if entity_data.type_ in ["wall", "pillar", "table"]
                    else "unknown",
                    style=style,
                )

                world.add_entity(new_object)

        # Get zones
        if config.things.zones is not None:
            if config.things.zones.goals:
                for goal_data in config.things.zones.goals:
                    try:
                        if goal_data.geometry is not None:
                            goal_polygon = shapely_geoms[goal_data.geometry.id]
                            pose: t.List[float] = [
                                goal_polygon.centroid.coords[0][0],
                                goal_polygon.centroid.coords[0][1],
                                0.0,
                            ]  # type: ignore

                            if goal_data.geometry.orientation_id is not None:
                                # If a drawn vector in the SVG is defined as orientation, use it
                                geom = shapely_geoms[goal_data.geometry.orientation_id]
                                pose[2] = get_orientation(geom)
                            else:
                                raise NotImplementedError(
                                    "You can't define a geometry in the json file manually for now."
                                )
                            goal = Goal(
                                polygon=goal_polygon,
                                name=goal_data.name,
                                pose=(pose[0], pose[1], pose[2]),
                            )
                            world.goals[goal.uid] = goal
                        elif goal_data.pose is not None:
                            # TODO: Change goal polygon to an arrow
                            if first_robot:
                                goal_polygon = utils.set_polygon_pose(
                                    first_robot.polygon,
                                    first_robot.pose,
                                    goal_data.pose,
                                )
                            else:
                                goal_polygon = None
                            goal = Goal(
                                polygon=goal_polygon,
                                name=goal_data.name,
                                pose=goal_data.pose,
                            )
                            world.goals[goal.uid] = goal
                    except KeyError:
                        print(
                            "No goal named in geometry data... {}".format(
                                goal_data.name
                            )
                        )
            if config.things.zones.taboos is not None:
                for taboo_data in config.things.zones.taboos:
                    try:
                        taboo_polygon = shapely_geoms[taboo_data.geometry.id]
                        new_taboo = Taboo(
                            name=taboo_data.name, polygon=Polygon(taboo_polygon)
                        )
                        world.taboo_zones[new_taboo.uid] = new_taboo
                    except Exception:
                        print("No taboo zone named... {}".format(taboo_data.name))

        world.update_dd()

        goals_node = svg_doc.getElementById("goals")
        if goals_node:
            goals_node.parentNode.removeChild(goals_node)

        return world

    def save_to_files(
        self,
        json_filepath: t.Optional[str] = None,
        svg_filepath: t.Optional[str] = None,
        json_data: t.Optional[t.Any] = None,
        svg_data: t.Optional[t.Any] = None,
    ):
        json_filepath = json_filepath or "./" + self.init_json_filename
        svg_filepath = svg_filepath or "./" + self.init_geometry_filename
        working_directory = os.path.dirname(json_filepath)
        abs_svg_filepath = os.path.join(working_directory, svg_filepath)

        if not json_data:
            json_data = self.to_json(svg_filepath)

        # Generate SVG data
        if not svg_data:
            svg_data = self.to_svg()

        # Save both json and SVG to specified path
        with open(json_filepath, "w+") as f:
            json.dump(json_data, f)
        with open(abs_svg_filepath, "w+") as f:
            svg_data.writexml(f)

    def to_json(self, svg_filepath: str) -> t.Any:
        return {
            "files": {"geometry_file": svg_filepath},
            "geometry_scale": self.geometry_scale,
            "discretization_data": {
                "res": self.discretization_data.res,
                "inflation_radius": self.discretization_data.inflation_radius,
                "cost_lethal": self.discretization_data.cost_lethal,
                "cost_inscribed": self.discretization_data.cost_inscribed,
                "cost_circumscribed": self.discretization_data.cost_circumscribed,
                "cost_possibly_nonfree": self.discretization_data.cost_possibly_nonfree,
            },
            "things": {
                "entities": [entity.to_json() for entity in self.entities.values()],
                "zones": {
                    "goals": [goal.to_json() for goal in self.goals.values()],
                    "taboos": [taboo.to_json() for taboo in self.taboo_zones.values()],
                },
            },
        }

    def to_svg(self) -> minidom.Document:
        if self.init_geometry_file:
            svg_data: minidom.Document = copy.deepcopy(self.init_geometry_file)
            init_geometries_ids = {
                path.getAttribute("id")
                for path in svg_data.getElementsByTagName("path")
            }
            current_geometries_names_to_ids = {
                entity.name: uid for uid, entity in self.entities.items()
            }
            # The 4 following lines are a hack to compensate for the fact the geometries are not associated with entity
            # for uid, entity in self.entities.items():
            #     if isinstance(entity, Robot):
            #         current_geometries_names_to_ids[entity.name + "_shape"] = uid
            #         current_geometries_names_to_ids[entity.name + "_direction"] = uid
            current_geometries_names = set(current_geometries_names_to_ids.keys())

            new_geometries_names = current_geometries_names.difference(
                init_geometries_ids
            )
            deleted_geometries_names = init_geometries_ids.difference(
                current_geometries_names
            )
            updated_geometries_names = init_geometries_ids.intersection(
                current_geometries_names
            )

            for geometry_name in new_geometries_names:
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
                elif isinstance(entity, Robot):
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
                    point_b = (
                        point_a
                        + np.array(utils.direction_from_yaw(entity.pose[2])) * radius
                    )
                    direction_linestring = LineString([point_a, point_b])
                    conversion.add_shapely_geometry_to_svg(
                        direction_linestring,
                        entity.name + "_direction",
                        conversion.GOAL_STYLE,
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
            for geometry_name in deleted_geometries_names:
                xml_element = svg_data.getElementById(geometry_name)
                if xml_element:
                    xml_element.parentNode.removeChild(xml_element)
            for geometry_name in updated_geometries_names:
                entity = self.entities[current_geometries_names_to_ids[geometry_name]]
                geometry = affinity.translate(
                    entity.polygon,
                    self.discretization_data.width / 2.0,
                    -self.discretization_data.height / 2.0,
                )
                new_svg_path = conversion.shapely_geometry_to_svg_pathd(
                    geometry, self.scaling_value
                )
                geom_el = svg_data.getElementById(geometry_name)
                if geom_el:
                    geom_el.setAttribute("d", str(new_svg_path))
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

    def get_map_bounds(self):
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

        return World(
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


def get_orientation(geom: (Polygon | LineString)) -> float:
    orientation_geom: t.List[t.List[float]] = list(geom.coords)  # type: ignore
    orientation_vector = [
        orientation_geom[1][0] - orientation_geom[0][0],
        orientation_geom[1][1] - orientation_geom[0][1],
    ]
    theta = utils.yaw_from_direction(orientation_vector)
    return theta
