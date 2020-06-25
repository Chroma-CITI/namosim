import copy
import os
from xml.dom import minidom

import shapely.affinity as affinity
import yaml
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import cascaded_union
from svgpath2mpl import parse_path

import src.utils.utils as utils
from src.worldreps.entity_based.custom_exceptions import EntityPlacementException
from src.worldreps.discretization_data import DiscretizationData
from src.display.ros_publisher import RosPublisher
from obstacle import Obstacle
from src.worldreps.occupation_based.probabilist_occupancy_grid import ProbabilistOccupancyGrid
from src.worldreps.occupation_based.binary_occupancy_grid import BinaryOccupancyGrid
from src.worldreps.occupation_based.binary_inflated_occupancy_grid import BinaryInflatedOccupancyGrid
from src.worldreps.occupation_based.social_topological_occupation_cost_grid import SocialTopologicalOccupationCostGrid
from src.worldreps.occupation_based.connected_components_grid import ConnectedComponentsMeta
from robot import Robot
from taboo import Taboo
from sensors.g_fov_sensor import GFOVSensor
from sensors.s_fov_sensor import SFOVSensor
from sensors.omniscient_sensor import OmniscientSensor


class World:
    def __init__(self, entities=None, dd=None, taboo_zones=None,
                 probabilist_occupancy_grids=None, binary_occupancy_grids=None, binary_inflated_occupancy_grids=None,
                 social_topological_occupation_cost_grids=None, connected_components_grids=None):

        self.entities = entities if entities is not None else dict()

        self.dd = dd

        self.taboo_zones = taboo_zones if taboo_zones is not None else dict()

        self._probabilist_occupancy_grids = probabilist_occupancy_grids if probabilist_occupancy_grids is not None else dict()
        self._binary_occupancy_grids = binary_occupancy_grids if binary_occupancy_grids is not None else dict()
        self._binary_inflated_occupancy_grids = binary_inflated_occupancy_grids if binary_inflated_occupancy_grids is not None else dict()
        self._social_topological_occupation_cost_grids = social_topological_occupation_cost_grids if social_topological_occupation_cost_grids is not None else dict()
        self._connected_components_grids = connected_components_grids if connected_components_grids is not None else dict()

    # Constructor
    def load_from_yaml(self, abs_path_to_file):
        # Import YAML world configuration file
        config = yaml.load(open(abs_path_to_file))

        # Import SVG geometry file specified in YAML configuration
        yaml_working_directory = os.path.dirname(abs_path_to_file)
        geometry_file_path = os.path.join(yaml_working_directory, config["files"]["geometry_file"])
        geometry_file = minidom.parse(geometry_file_path)
        svg_paths = {path.getAttribute("id"): path.getAttribute('d') for path in geometry_file.getElementsByTagName('path')}
        # svg_circles = {
        #     circle.getAttribute("id"): {
        #         "cx": circle.getAttribute('cx'),
        #         "cy": circle.getAttribute('cy'),
        #         "r": circle.getAttribute('r')} for circle in geometry_file.getElementsByTagName('circle')}
        shapely_geoms = dict()
        SCALING_CONSTANT = 1. / 3.5433
        scaling_value = SCALING_CONSTANT * config["geometry_scale"]
        # Convert imported geometry to shapely polygons
        for svg_id, svg_path in svg_paths.items():
            parse_result = parse_path(svg_path)
            geom_pts = parse_result.vertices * scaling_value
            geom_pts[:, 1] = -geom_pts[:, 1] # Mirror on y-axis
            if len(geom_pts) >= 3:
                # shapely_geoms[svg_id] = affinity.scale(Polygon(geom_pts), scaling_value, scaling_value)
                shapely_geoms[svg_id] = Polygon(geom_pts)
            elif len(geom_pts) == 2:
                # shapely_geoms[svg_id] = affinity.scale(LineString(geom_pts), scaling_value, scaling_value)
                shapely_geoms[svg_id] = LineString(geom_pts)
            elif len(geom_pts) == 1:
                # shapely_geoms[svg_id] = affinity.scale(Point(geom_pts), scaling_value, scaling_value)
                shapely_geoms[svg_id] = Point(geom_pts)
        # for svg_id, svg_circle in svg_circles.items():
        #     shapely_geoms[svg_id] = Point(float(svg_circle["cx"]) * config["geometry_scale"],
        #                                      float(svg_circle["cy"]) * config["geometry_scale"]).buffer(
        #         float(svg_circle["r"]) * config["geometry_scale"])
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
        self.dd = DiscretizationData(res=config["discretization_data"]["res"],
                                     inflation_radius=config["discretization_data"]["inflation_radius"],
                                     cost_lethal=config["discretization_data"]["cost_lethal"],
                                     cost_inscribed=config["discretization_data"]["cost_inscribed"],
                                     cost_circumscribed=config["discretization_data"]["cost_circumscribed"],
                                     cost_possibly_nonfree=config["discretization_data"]["cost_possibly_nonfree"])
        # Get all things
        for entity_data in config["things"]["entities"]:
            # Pose of object definition
            pose = [None, None, 0.0] # x, y, theta
            if "orientation_id" in entity_data["geometry"]:
                # If a drawn vector in the SVG is defined as orientation, use it
                orientation_geom = list(shapely_geoms[entity_data["geometry"]["orientation_id"]].coords)
                orientation_vector = [orientation_geom[1][0] - orientation_geom[0][0],
                                      orientation_geom[1][1] - orientation_geom[0][1]]
                pose[2] = utils.yaw_from_direction(orientation_vector)
            if "pose" in entity_data["geometry"]:
                # If a pose is manually described in the YAML file, override the possibly SVG-computed orientation
                yaml_pose = entity_data["geometry"]["pose"]
                if "position" in yaml_pose:
                    pose[0], pose[1] = yaml_pose["position"]["x"], yaml_pose["position"]["y"]
                if "orientation" in yaml_pose:
                    pose[2] = yaml_pose["orientation"]["z"]

            # Polygonal geometry object definition
            polygon = Polygon()
            try:
                if entity_data["geometry"]["from"] == "file":
                    # If geometry is defined in SVG file, prioritize using it
                    polygon = shapely_geoms[entity_data["geometry"]["id"]]
                elif entity_data["geometry"]["from"] == "polygon":
                    # If geometry manually defined in yaml file, use it before radius-defined but after SVG if exists
                    polygon = Polygon(entity_data["geometry"]["polygon"])
                elif entity_data["geometry"]["from"] == "radius":
                    # Last case:
                    polygon = Point(pose[0], pose[1]).buffer(entity_data["geometry"]["polygon"]["radius"])
            except Exception as e:
                continue

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

                if new_robot.min_inflation_radius > self.dd.inflation_radius:
                    self.dd.inflation_radius = new_robot.min_inflation_radius

                self.add_entity(new_robot)
            else:
                new_object = Obstacle(name=entity_data["name"],
                                      polygon=polygon,
                                      pose=pose,
                                      type_in=entity_data["type"],
                                      full_geometry_acquired=True)

                self.add_entity(new_object)

        # Get zones
        goals = dict()
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
                        if "pose" in goal_data["geometry"]:
                            # If a pose is manually described in the YAML file, override the possibly SVG-computed orientation
                            yaml_pose = goal_data["geometry"]["pose"]
                            if "position" in yaml_pose:
                                pose[0], pose[1] = yaml_pose["position"]["x"], yaml_pose["position"]["y"]
                            if "orientation" in yaml_pose:
                                pose[2] = yaml_pose["orientation"]["z"]

                        goals[goal_data["name"]] = tuple(pose)
                    except Exception:
                        print("No goal named... {}".format(goal_data['geometry']['id']))
            if ("taboos" in config["things"]["zones"]
                    and isinstance(config["things"]["zones"]["taboos"], list)):
                for thing_data in config["things"]["zones"]["taboos"]:
                    new_taboo = Taboo(name=thing_data["name"],
                                      polygon=Polygon(thing_data["polygon"]))
                    self.taboo_zones[new_taboo.uid] = new_taboo

        self.update_dd()

        return goals

    def add_entity(self, new_entity):
        for obj in self.entities.values():
            is_within = new_entity.within(obj)
            if is_within:
                raise EntityPlacementException("Entity {} would be within entity {}. Cannot load world.".format(
                    new_entity.name, obj.name))
        self.entities[new_entity.uid] = new_entity

        self._invalidate_and_inform_grids(prev_entities=dict(), next_entities={new_entity.uid: new_entity})

    def remove_entity(self, entity_uid):
        removed_entity = self.entities[entity_uid]
        if entity_uid in self.entities:
            del self.entities[entity_uid]
        else:
            raise KeyError("Warning, you tried to remove an entity that is not registered in this world !")

        self._invalidate_and_inform_grids(prev_entities={removed_entity.uid: removed_entity},
                                          next_entities=dict())

    def remove_entities(self, entities_uids):
        for entity_uid in entities_uids:
            self.remove_entity(entity_uid)

    def translate_entity(self, entity_uid, translation):
        entity = self.entities[entity_uid]
        prev_entity = copy.deepcopy(entity)
        entity.translate(translation[0], translation[1], self.dd.res)
        self._invalidate_and_inform_grids(prev_entities={entity_uid: prev_entity},
                                          next_entities={entity_uid: entity})

    def rotate_entity(self, entity_uid, rotation, rot_center='centroid'):
        entity = self.entities[entity_uid]
        prev_entity = copy.deepcopy(entity)
        entity.rotate(rotation, rot_center)
        self._invalidate_and_inform_grids(prev_entities={entity_uid: prev_entity},
                                          next_entities={entity_uid: entity})

    def set_entity_polygon(self, entity_uid, polygon, full_geometry_acquired=False):
        entity = self.entities[entity_uid]
        prev_entity = copy.deepcopy(entity)
        entity.set_polygon(polygon)
        entity.full_geometry_acquired = full_geometry_acquired
        self._invalidate_and_inform_grids(prev_entities={entity_uid: prev_entity},
                                          next_entities={entity_uid: entity})

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
            self.delete_all_grids()
            self.dd.saved_hash = new_hash

    # TO DEPRECATE
    def delete_all_grids(self):
        self._probabilist_occupancy_grids = dict()
        self._binary_occupancy_grids = dict()
        self._binary_inflated_occupancy_grids = dict()
        self._social_topological_occupation_cost_grids = dict()

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
    def _invalidate_and_inform_grids(self, prev_entities, next_entities):
        # If any entity that is required to build these grids has changed, invalidate them
        for entities_to_ignore, grid in self._probabilist_occupancy_grids.items():
            if self._has_not_ignored_entity_changed(entities_to_ignore, prev_entities, next_entities):
                del self._probabilist_occupancy_grids[entities_to_ignore]

        for entities_to_ignore, grid in self._social_topological_occupation_cost_grids.items():
            if self._has_not_ignored_entity_changed(entities_to_ignore, prev_entities, next_entities):
                del self._social_topological_occupation_cost_grids[entities_to_ignore]

        # TODO Make the connected components grids upgradable on demand
        for entities_to_ignore, grid in self._connected_components_grids.items():
            if self._has_not_ignored_entity_changed(entities_to_ignore, prev_entities, next_entities):
                del self._connected_components_grids[entities_to_ignore]

        # If any entity that is required to build these grids has changed, inform them of the change
        for grid in self._binary_occupancy_grids.values():
            grid.update_buffered_entities(prev_entities, next_entities)

        for grid in self._binary_inflated_occupancy_grids.values():
            grid.update_buffered_entities(prev_entities, next_entities)

    # TO DEPRECATE
    def get_probabilist_occupancy_grid(self, entities_to_ignore):
        self.update_dd()
        if entities_to_ignore not in self._probabilist_occupancy_grids:
            self._probabilist_occupancy_grids[entities_to_ignore] = ProbabilistOccupancyGrid(self.dd, self.entities, entities_to_ignore)
        return self._probabilist_occupancy_grids[entities_to_ignore]

    # TO DEPRECATE
    def get_binary_occupancy_grid(self, entities_to_ignore):
        self.update_dd()
        if entities_to_ignore not in self._binary_occupancy_grids:
            self._binary_occupancy_grids[entities_to_ignore] = BinaryOccupancyGrid(
                self.dd.d_width, self.dd.d_height, self.dd.res, self.dd.grid_pose, self.dd.inflation_radius,
                self.entities, entities_to_ignore)
        return self._binary_occupancy_grids[entities_to_ignore]

    # TO DEPRECATE
    def get_binary_inflated_occupancy_grid(self, entities_to_ignore):
        self.update_dd()
        if entities_to_ignore not in self._binary_inflated_occupancy_grids:
            self._binary_inflated_occupancy_grids[entities_to_ignore] = BinaryInflatedOccupancyGrid(
                self.dd.d_width, self.dd.d_height, self.dd.res, self.dd.grid_pose, self.dd.inflation_radius,
                self.entities, entities_to_ignore)
        return self._binary_inflated_occupancy_grids[entities_to_ignore]

    # TO DEPRECATE
    def get_social_topological_occupation_cost_grid(self, entities_to_ignore):
        self.update_dd()
        if entities_to_ignore not in self._social_topological_occupation_cost_grids:
            self._social_topological_occupation_cost_grids[entities_to_ignore] = SocialTopologicalOccupationCostGrid()
        return self._social_topological_occupation_cost_grids[entities_to_ignore]

    # TO DEPRECATE
    def get_connected_components_grid(self, entities_to_ignore):
        self.update_dd()
        if entities_to_ignore not in self._connected_components_grids:
            grid = ConnectedComponentsMeta(
                self.get_binary_inflated_occupancy_grid(entities_to_ignore).get_grid())
            self._connected_components_grids[entities_to_ignore] = grid
        return self._connected_components_grids[entities_to_ignore]

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
