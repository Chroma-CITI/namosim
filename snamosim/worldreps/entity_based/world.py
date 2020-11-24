import copy
import os
from xml.dom import minidom
import json
import numpy as np

import shapely.affinity as affinity
import yaml
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import cascaded_union

import snamosim.utils.utils as utils
import snamosim.utils.conversion as conversion
from snamosim.worldreps.entity_based.custom_exceptions import EntityPlacementException
from snamosim.worldreps.discretization_data import DiscretizationData
from snamosim.display.ros_publisher import RosPublisher
from obstacle import Obstacle
from robot import Robot
from taboo import Taboo
from goal import Goal
from sensors.g_fov_sensor import GFOVSensor
from sensors.s_fov_sensor import SFOVSensor
from sensors.omniscient_sensor import OmniscientSensor


class World:
    SCALING_CONSTANT = 1. / 3.5433

    def __init__(self, entities=None, dd=None, taboo_zones=None, goals=None,
                 probabilist_occupancy_grids=None, binary_occupancy_grids=None, binary_inflated_occupancy_grids=None,
                 social_topological_occupation_cost_grids=None, connected_components_grids=None, geometry_scale=1.,
                 init_geometry_filename="/world_name_placeholder.svg", init_geometry_file=None):

        self.entities = entities if entities is not None else dict()

        self.dd = dd

        self.geometry_scale = geometry_scale
        self.scaling_value = self.geometry_scale * World.SCALING_CONSTANT

        self.init_geometry_file = init_geometry_file
        if init_geometry_file:
            conversion.set_all_id_attributes_as_ids(init_geometry_file)
            conversion.clean_attributes(init_geometry_file)
        self.init_geometry_filename = init_geometry_filename

        self.taboo_zones = taboo_zones if taboo_zones is not None else dict()
        self.goals = goals if goals is not None else dict()

        self._probabilist_occupancy_grids = probabilist_occupancy_grids if probabilist_occupancy_grids is not None else dict()
        self._binary_occupancy_grids = binary_occupancy_grids if binary_occupancy_grids is not None else dict()
        self._binary_inflated_occupancy_grids = binary_inflated_occupancy_grids if binary_inflated_occupancy_grids is not None else dict()
        self._social_topological_occupation_cost_grids = social_topological_occupation_cost_grids if social_topological_occupation_cost_grids is not None else dict()
        self._connected_components_grids = connected_components_grids if connected_components_grids is not None else dict()

    # Constructor
    @classmethod
    def load_from_yaml(cls, abs_path_to_file):
        # Import YAML world configuration file
        config = yaml.load(open(abs_path_to_file))

        # Import SVG geometry file specified in YAML configuration
        geometry_file_path = config["files"]["geometry_file"]
        abs_geometry_file_path = geometry_file_path
        if not os.path.isabs(geometry_file_path):
            yaml_working_directory = os.path.dirname(abs_path_to_file)
            abs_geometry_file_path = os.path.join(yaml_working_directory, geometry_file_path)
        init_geometry_filename = os.path.basename(abs_geometry_file_path)
        init_geometry_file = minidom.parse(abs_geometry_file_path)
        svg_paths = {path.getAttribute("id"): path.getAttribute('d')
                     for path in init_geometry_file.getElementsByTagName('path')}
        shapely_geoms = dict()
        scaling_value = World.SCALING_CONSTANT * config["geometry_scale"]
        # Convert imported geometry to shapely polygons
        for svg_id, svg_path in svg_paths.items():
            try:
                shapely_geoms[svg_id] = conversion.svg_pathd_to_shapely_geometry(svg_path, scaling_value)
            except RuntimeError:
                raise RuntimeError("Could not convert svg path to shapely geometry for svg id: {}".format(svg_id))
        # TODO Fix this so that it only accounts for obstacles in polygon layer otherwise, things might get messy with
        #  direction vectors that get outside of the obstacle polygons
        # Center the imported geometries
        unioned_polygons = cascaded_union(shapely_geoms.values())
        bounding_box = box(unioned_polygons.bounds[0], unioned_polygons.bounds[1],
                           unioned_polygons.bounds[2], unioned_polygons.bounds[3])
        # print(str((bounding_box.bounds[2] - bounding_box.bounds[0], bounding_box.bounds[3] - bounding_box.bounds[1])))
        translation_to_center = [bounding_box.centroid.coords[0][0], bounding_box.centroid.coords[0][1]]
        for svg_id, polygon in shapely_geoms.items():
            shapely_geoms[svg_id] = affinity.translate(polygon, -translation_to_center[0], -translation_to_center[1])

        # Get map discretization parameters
        dd = DiscretizationData(
            res=config["discretization_data"]["res"],
            inflation_radius=config["discretization_data"]["inflation_radius"],
            cost_lethal=config["discretization_data"]["cost_lethal"],
            cost_inscribed=config["discretization_data"]["cost_inscribed"],
            cost_circumscribed=config["discretization_data"]["cost_circumscribed"],
            cost_possibly_nonfree=config["discretization_data"]["cost_possibly_nonfree"]
        )

        world = cls(
            geometry_scale=config["geometry_scale"],
            init_geometry_filename=init_geometry_filename,
            init_geometry_file=init_geometry_file,
            dd=dd
        )

        # Get all things
        for entity_data in config["things"]["entities"]:
            # Pose of object definition
            pose = [None, None, 0.0]  # x, y, theta
            if "orientation_id" in entity_data["geometry"]:
                # If a drawn vector in the SVG is defined as orientation, use it
                orientation_geom = list(shapely_geoms[entity_data["geometry"]["orientation_id"]].coords)
                orientation_vector = [orientation_geom[1][0] - orientation_geom[0][0],
                                      orientation_geom[1][1] - orientation_geom[0][1]]
                pose[2] = utils.yaw_from_direction(orientation_vector)

            # Polygonal geometry object definition
            if entity_data["geometry"]["from"] == "file":
                # If geometry is defined in SVG file, prioritize using it
                try:
                    polygon = shapely_geoms[entity_data["geometry"]["id"]]
                except KeyError:
                    print("Could not find geometry {} in svg file. Next entity.".format(entity_data["geometry"]["id"]))
                    continue
            else:
                raise NotImplementedError("You can't define a geometry in the json file manually for now.")

            # Adjust initial position in pose if not given only by SVG file
            if pose[0] is None or pose[1] is None:
                pose[0], pose[1] = [list(polygon.centroid.coords)[0][0], list(polygon.centroid.coords)[0][1]]

            if entity_data["type"] == "robot":
                sensors_data = entity_data["sensors"]

                sensors = []
                for sensor_data in sensors_data:
                    if sensor_data["type"] == "perfect_g_fov":
                        sensors.append(GFOVSensor(
                            sensor_data["max_radius"],
                            sensor_data["min_radius"],
                            sensor_data["opening_angle"], pose))
                    elif sensor_data["type"] == "perfect_s_fov":
                        sensors.append(SFOVSensor(
                            sensor_data["max_radius"],
                            sensor_data["min_radius"],
                            sensor_data["opening_angle"], pose))
                    elif sensor_data["type"] == "omniscient":
                        sensors.append(OmniscientSensor())

                new_robot = Robot(name=entity_data["name"],
                                  full_geometry_acquired=True,
                                  polygon=polygon,
                                  pose=tuple(pose),
                                  sensors=sensors,
                                  push_only_list=entity_data["push_only_list"],
                                  force_pushes_only=entity_data["force_pushes_only"],
                                  movable_whitelist=entity_data["movable_whitelist"])

                # Prevent specified inflation radius to be smaller than actual polygon

                if new_robot.min_inflation_radius > dd.inflation_radius:
                    dd.inflation_radius = new_robot.min_inflation_radius

                world.add_entity(new_robot)
            else:
                new_object = Obstacle(name=entity_data["name"],
                                      polygon=polygon,
                                      pose=pose,
                                      type_in=entity_data["type"],
                                      full_geometry_acquired=True)

                world.add_entity(new_object)

        # Get zones
        if "zones" in config["things"] :
            if ("goals" in config["things"]["zones"]
                    and isinstance(config["things"]["zones"]["goals"], list)):
                for goal_data in config["things"]["zones"]["goals"]:
                    try:
                        goal_polygon = shapely_geoms[goal_data["geometry"]["id"]]
                        pose = [goal_polygon.centroid.coords[0][0], goal_polygon.centroid.coords[0][1], 0.0]

                        if "orientation_id" in goal_data["geometry"]:
                            # If a drawn vector in the SVG is defined as orientation, use it
                            orientation_geom = list(shapely_geoms[goal_data["geometry"]["orientation_id"]].coords)
                            orientation_vector = [orientation_geom[1][0] - orientation_geom[0][0],
                                                  orientation_geom[1][1] - orientation_geom[0][1]]
                            pose[2] = utils.yaw_from_direction(orientation_vector)
                        else:
                            raise NotImplementedError("You can't define a geometry in the json file manually for now.")
                        goal = Goal(polygon=goal_polygon, name=goal_data["name"], pose=tuple(pose))
                        world.goals[goal.uid] = goal
                    except Exception:
                        print("No goal named... {}".format(goal_data['geometry']['id']))
            if ("taboos" in config["things"]["zones"]
                    and isinstance(config["things"]["zones"]["taboos"], list)):
                for thing_data in config["things"]["zones"]["taboos"]:
                    try:
                        taboo_polygon = shapely_geoms[thing_data["geometry"]["id"]]
                        new_taboo = Taboo(name=thing_data["name"], polygon=Polygon(taboo_polygon))
                        world.taboo_zones[new_taboo.uid] = new_taboo
                    except Exception:
                        print("No taboo zone named... {}".format(thing_data['geometry']['id']))

        world.update_dd()

        return world

    def save_to_files(self, json_filepath, svg_filepath=None, json_data=None, svg_data=None):
        if not svg_filepath:
            svg_filepath = "./" + self.init_geometry_filename
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

    def to_json(self, svg_filepath):
        return {
            "files": {
                "geometry_file": svg_filepath
            },
            "geometry_scale": self.geometry_scale,
            "discretization_data": {
                "res": self.dd.res,
                "inflation_radius": self.dd.inflation_radius,
                "cost_lethal": self.dd.cost_lethal,
                "cost_inscribed": self.dd.cost_inscribed,
                "cost_circumscribed": self.dd.cost_circumscribed,
                "cost_possibly_nonfree": self.dd.cost_possibly_nonfree
            },
            "things": {
                "entities": [entity.to_json() for entity in self.entities.values()],
                "zones": {
                    "goals": [goal.to_json() for goal in self.goals.values()],
                    "taboos": [taboo.to_json() for taboo in self.taboo_zones.values()]
                }
            }
        }

    def to_svg(self):
        if self.init_geometry_file:
            svg_data = copy.deepcopy(self.init_geometry_file)
            init_geometries_ids = {path.getAttribute("id") for path in svg_data.getElementsByTagName('path')}
            current_geometries_ids_and_polygons = self.get_current_geometries_ids_and_polygon()
            current_geometries_ids = set(current_geometries_ids_and_polygons.keys())

            new_geometries_ids = current_geometries_ids.difference(init_geometries_ids)
            deleted_geometries_ids = init_geometries_ids.difference(current_geometries_ids)
            updated_geometries_ids = init_geometries_ids.intersection(current_geometries_ids)

            for geometry_id in new_geometries_ids:
                pass
                # raise NotImplementedError("TODO : use bootstrap SVG data to build new SVG paths from scratch")
            for geometry_id in deleted_geometries_ids:
                xml_element = svg_data.getElementById(geometry_id)
                xml_element.parentNode.removeChild(xml_element)
            for geometry_id in updated_geometries_ids:
                geometry = affinity.translate(
                    current_geometries_ids_and_polygons[geometry_id], self.dd.width / 2., -self.dd.height / 2.)
                new_svg_path = conversion.shapely_geometry_to_svg_pathd(geometry, self.scaling_value)
                svg_data.getElementById(geometry_id).setAttribute('d', new_svg_path)
        else:
            raise NotImplementedError("TODO : use bootstrap SVG data to build new SVG file from scratch")
        return svg_data

    def add_entity(self, new_entity):
        for obj in self.entities.values():
            is_within = new_entity.within(obj)
            if is_within:
                raise EntityPlacementException("Entity {} would be within entity {}. Cannot load world.".format(
                    new_entity.name, obj.name))
        self.entities[new_entity.uid] = new_entity

    def remove_entity(self, entity_uid):
        removed_entity = self.entities[entity_uid]
        if entity_uid in self.entities:
            del self.entities[entity_uid]
        else:
            raise KeyError("Warning, you tried to remove an entity that is not registered in this world !")

    def remove_entities(self, entities_uids):
        for entity_uid in entities_uids:
            self.remove_entity(entity_uid)

    def translate_entity(self, entity_uid, translation):
        entity = self.entities[entity_uid]
        prev_entity = copy.deepcopy(entity)
        entity.translate(translation[0], translation[1], self.dd.res)

    def rotate_entity(self, entity_uid, rotation, rot_center='centroid'):
        entity = self.entities[entity_uid]
        prev_entity = copy.deepcopy(entity)
        entity.rotate(rotation, rot_center)

    def set_entity_polygon(self, entity_uid, polygon, full_geometry_acquired=False):
        entity = self.entities[entity_uid]
        prev_entity = copy.deepcopy(entity)
        entity.set_polygon(polygon)
        entity.full_geometry_acquired = full_geometry_acquired

    def get_map_bounds(self):
        if len(self.entities) == 0:
            raise ValueError("There are no entities to populate the grid, it can't be created !")
        polygons = [entity.polygon for entity in self.entities.values()]
        map_min_x, map_min_y, map_max_x, map_max_y = float("inf"), float("inf"), -float("inf"), -float("inf")
        for polygon in polygons:
            min_x, min_y, max_x, max_y = polygon.bounds
            map_min_x, map_min_y = min(map_min_x, min_x), min(map_min_y, min_y)
            map_max_x, map_max_y = max(map_max_x, max_x), max(map_max_y, max_y)
        return map_min_x, map_min_y, map_max_x, map_max_y

    # TO DEPRECATE
    def update_dd(self):
        if self.dd is None:
            raise ValueError("Discretization data (dd) is None, this should not be happening !")

        min_x, min_y, max_x, max_y = self.get_map_bounds()
        width, height = max_x - min_x, max_y - min_y

        self.dd.grid_pose = (min_x, min_y, 0.0)
        self.dd.width, self.dd.height = width, height
        self.dd.d_width, self.dd.d_height = (int(round(self.dd.width / self.dd.res)),
                                             int(round(self.dd.height / self.dd.res)))
        new_hash = hash(self.dd)
        if new_hash != self.dd.saved_hash:
            self.dd.saved_hash = new_hash

    @staticmethod
    def _has_not_ignored_entity_changed(entities_to_ignore, prev_entities, next_entities):
        for entity_uid, entity in prev_entities.items():
            if entity_uid not in entities_to_ignore:
                return True
        for entity_uid, entity in next_entities.items():
            if entity_uid not in entities_to_ignore:
                return True
        return False

    # TO DEPRECATE
    def get_entity_uid_from_name(self, name):
        for entity_uid, entity in self.entities.items():
            if entity.name == name:
                return entity_uid
        raise LookupError("Could not find an entity in this world with name : {name}.".format(name=name))

    # TO DEPRECATE
    def agg_grid_cost_for_entities(self, entities_uids, grid, aggregation_function=sum):
        entities_cells = set()
        for entity_uid in entities_uids:
            entities_cells = entities_cells.union(self.entities[entity_uid].get_discrete_cells_set(
                self.dd.inflation_radius, self.dd.res, self.dd.grid_pose, self.dd.d_width, self.dd.d_height))
        RosPublisher().publish_social_cells(entities_cells, self.dd.res, self.dd.grid_pose)
        cells_values = [grid[cell[0]][cell[1]] for cell in entities_cells]
        return aggregation_function(cells_values)

    def get_current_geometries_ids_and_polygon(self):
        current_geometries_ids_and_polygons = {}
        for entity in self.entities.values():
            current_geometries_ids_and_polygons[entity.name] = entity.polygon
            radius = utils.get_inscribed_radius(entity.polygon)
            if isinstance(entity, Robot):
                point_a = np.array([entity.pose[0], entity.pose[1]])
                point_b = point_a + np.array(utils.direction_from_yaw(entity.pose[2])) * radius
                current_geometries_ids_and_polygons[entity.name + "_dir"] = LineString([point_a, point_b])
        # for goal in self.goals.values():
        #     current_geometries_ids_and_polygons[goal.name] = goal.polygon
        #     radius = utils.get_inscribed_radius(goal.polygon)
        #     point_a = np.array([goal.pose[0], goal.pose[1]])
        #     point_b = point_a + np.array(utils.direction_from_yaw(goal.pose[2])) * radius
        #     current_geometries_ids_and_polygons[goal.name + "_dir"] = LineString([point_a, point_b])
        # for taboo in self.taboo_zones.values():
        #     current_geometries_ids_and_polygons[taboo.name] = taboo.polygon
        return current_geometries_ids_and_polygons