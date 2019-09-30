import yaml
import numpy as np
from math import ceil, floor, sqrt, atan2
import copy

from robot import Robot
from obstacle import Obstacle
from taboo import Taboo
import utils

from custom_exceptions import EntityPlacementException

from discretization_data import DiscretizationData

from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import cascaded_union
from shapely.errors import TopologicalError

from xml.dom import minidom
from svgpath2mpl import parse_path
import os
import shapely.affinity as affinity

from probabilist_occupancy_grid import ProbabilistOccupancyGrid
from binary_occupancy_grid import BinaryOccupancyGrid
from binary_inflated_occupancy_grid import BinaryInflatedOccupancyGrid
from social_topological_occupation_cost_grid import SocialTopologicalOccupationCostGrid


class World:

    def __init__(self, entities=None, taboos=None, robot_uid=None, dd=None, inflated_grid=None):

        self.entities = entities if entities is not None else dict()

        self.robot_uid = robot_uid

        self.dd = dd

        self._taboo_zones = taboos if taboos is not None else dict()

        self._probabilist_occupancy_grids = dict()
        self._binary_occupancy_grid = dict()
        self._binary_inflated_occupancy_grid = dict()
        self._social_topological_occupation_cost_grids = dict()

    # Constructor
    def load_from_yaml(self, path_to_file):
        # Import YAML world configuration file
        yaml_abs_path = os.path.abspath(path_to_file)
        config = yaml.load(open(yaml_abs_path))

        # Import SVG geometry file specified in YAML configuration
        yaml_working_directory = os.path.dirname(yaml_abs_path)
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
            geom_pts = parse_result._vertices * scaling_value
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
            if entity_data["geometry"]["from"] == "file":
                # If geometry is defined in SVG file, prioritize using it
                polygon = shapely_geoms[entity_data["geometry"]["id"]]
            elif entity_data["geometry"]["from"] == "polygon":
                # If geometry manually defined in yaml file, use it before radius-defined but after SVG if exists
                polygon = Polygon(entity_data["geometry"]["polygon"])
            elif entity_data["geometry"]["from"] == "radius":
                # Last case:
                polygon = Point(pose[0], pose[1]).buffer(entity_data["geometry"]["polygon"]["radius"])

            # Adjust initial position in pose if not given only by SVG file
            if pose[0] is None or pose[1] is None:
                pose[0], pose[1] = [list(polygon.centroid.coords)[0][0], list(polygon.centroid.coords)[0][1]]

            if entity_data["type"] == "robot":
                sensors = entity_data["sensors"]

                new_robot = Robot(name=entity_data["name"],
                                  full_geometry_acquired=True,
                                  polygon=polygon,
                                  pose=pose,
                                  g_fov_max_radius=sensors["perfect_g_fov"]["max_radius"],
                                  g_fov_min_radius=sensors["perfect_g_fov"]["min_radius"],
                                  g_fov_opening_angle=sensors["perfect_g_fov"]["opening_angle"],
                                  s_fov_max_radius=sensors["perfect_s_fov"]["max_radius"],
                                  s_fov_min_radius=sensors["perfect_s_fov"]["min_radius"],
                                  s_fov_opening_angle=sensors["perfect_s_fov"]["opening_angle"],
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
        goals = []
        if "zones" in config["things"] :
            if ("goals" in config["things"]["zones"]
                    and isinstance(config["things"]["zones"]["goals"], list)):
                for goal_data in config["things"]["zones"]["goals"]:
                    goal_polygon = shapely_geoms[goal_data["geometry"]["id"]]
                    # TODO: Fix this so that orientation can be specified by vector in SVG and position can be
                    #       determined from YAML file if wanted
                    goals.append([goal_polygon.centroid.coords[0][0], goal_polygon.centroid.coords[0][1], 0.0])
            if ("taboos" in config["things"]["zones"]
                    and isinstance(config["things"]["zones"]["taboos"], list)):
                for thing_data in config["things"]["zones"]["taboos"]:
                    new_taboo = Taboo(name=thing_data["name"],
                                      polygon=Polygon(thing_data["polygon"]))
                    self._taboo_zones[new_taboo.uid] = new_taboo

        return goals

    def add_entity(self, new_entity):
        for obj in self.entities.values():
            is_within = new_entity.within(obj)
            if is_within:
                raise EntityPlacementException("Entity {} would be within entity {}. Cannot load world.".format(
                    new_entity.name, obj.name))
        self.entities[new_entity.uid] = new_entity

        if isinstance(new_entity, Robot):
            self.robot_uid = new_entity.uid

        self._is_inflated_grid_valid = False
        self._is_int_grid_valid = False
        self.invalidate_saved_costmaps((new_entity.uid,))
        self.new_entities[new_entity.uid] = new_entity

    def remove_entity(self, entity_uid):
        removed_entity = self.entities[entity_uid]
        if entity_uid in self.entities:
            del self.entities[entity_uid]
        else:
            print("Warning, you tried to remove an entity that is not registered in this world !")

        self._is_inflated_grid_valid = False
        self._is_int_grid_valid = False
        self.invalidate_saved_costmaps((entity_uid,))
        self.prev_entities[removed_entity.uid] = removed_entity
        if removed_entity.uid in self.new_entities:
            # Prevents artifacts if translation/rotation is applied to removed object before removal, which could in
            # some cases lead to the obstacle be re-added to the grid after it has been removed
            del self.new_entities[removed_entity.uid]

    def remove_entities(self, entities_uids):
        for entity_uid in entities_uids:
            self.remove_entity(entity_uid)

    def translate_entity(self, entity_uid, translation):
        entity = self.entities[entity_uid]
        if entity_uid != self.robot_uid:
            self._is_inflated_grid_valid = False
            self.invalidate_saved_costmaps((entity_uid,))
            self._is_int_grid_valid = False
            if entity.uid not in self.prev_entities:
                entity_prev = copy.deepcopy(entity)
                self.prev_entities[entity_prev.uid] = entity_prev
            if entity.uid not in self.new_entities:
                self.new_entities[entity.uid] = entity
        entity.translate(translation[0], translation[1], self.dd)

    def rotate_entity(self, entity_uid, rotation):
        entity = self.entities[entity_uid]
        if entity_uid != self.robot_uid:
            self._is_inflated_grid_valid = False
            self.invalidate_saved_costmaps((entity_uid,))
            self._is_int_grid_valid = False
            if entity.uid not in self.prev_entities:
                entity_prev = copy.deepcopy(entity)
                self.prev_entities[entity_prev.uid] = entity_prev
            if entity.uid not in self.new_entities:
                self.new_entities[entity.uid] = entity
        entity.rotate(rotation)

    def update_from_g_fov(self, entities):
        for entity_uid, entity in entities.items():
            if isinstance(entity, Obstacle):
                # If entity is already registered, update it
                try:
                    self_entity = self.entities[entity_uid]
                    self_entity_movability = self.entities[self.robot_uid].deduce_movability(self_entity.type)
                    # If self entity full geometry has not been acquired, update it
                    if not self_entity.full_geometry_acquired:
                        if entity.uid not in self.prev_entities:
                            entity_prev = copy.deepcopy(self_entity)
                            self.prev_entities[entity_prev.uid] = entity_prev
                        if entity.uid not in self.new_entities:
                            self.new_entities[entity.uid] = entity
                        if entity.full_geometry_acquired:
                            self_entity.set_polygon(entity.polygon, self.dd, self_entity_movability)
                            self_entity.full_geometry_acquired = True
                        else:
                            self_entity.set_polygon(
                                cascaded_union([self_entity.polygon, entity.polygon]).convex_hull,
                                self.dd, self_entity_movability)
                        self.invalidate_saved_costmaps((entity.uid,))
                        self._is_inflated_grid_valid = False
                        self._is_int_grid_valid = False
                    # If it is already known, only translate/rotate the polygon appropriately
                    else:
                        if (entity.full_geometry_acquired
                                and self_entity_movability != "unmovable"):
                            translation = [entity.pose[0] - self_entity.pose[0], entity.pose[1] - self_entity.pose[1]]
                            rotation = (entity.pose[2] - self_entity.pose[2]) % 360.0
                            # Only apply translation if there is one
                            if not all(np.isclose(translation, [0.0, 0.0], rtol=0.00001)):
                                self.translate_entity(entity_uid, translation)
                            # Only apply rotation if there is one
                            if rotation != 0:
                                self.rotate_entity(entity_uid, rotation)
                # If entity is not registered yet, create it
                except KeyError:
                    self.add_entity(entity)

    def update_from_s_fov(self, entities):
        for entity_uid, entity in entities.items():
            if isinstance(entity, Obstacle):
                # If entity is already registered, update it
                try:
                    self.entities[entity_uid].name = entity.name
                    self.entities[entity_uid].type = entity.type

                # If entity is not registered yet, create it
                except KeyError as e:
                    raise(e, "update_from_s_fov should never need to create an object !")

    def get_entities_in_g_fov_seethrough(self, robot_uid):
        robot = self.entities[robot_uid]

        entities_in_g_fov = dict()

        for entity_uid, entity in self.entities.items():
            if entity_uid != robot_uid:
                if entity.polygon.intersects(robot.g_fov_polygon):
                    if entity.polygon.within(robot.g_fov_polygon):
                        entity_visible_polygon = entity.polygon
                        full_geometry_acquired = True
                    else:
                        try:
                            entity_visible_polygon = entity.polygon.difference(
                                entity.polygon.difference(robot.g_fov_polygon))
                            full_geometry_acquired = False
                        except TopologicalError:
                            continue  # If we could not make a polygon, do not try to create Entity

                    if isinstance(entity_visible_polygon, Polygon):
                        entities_in_g_fov[entity_uid] = Obstacle(name="unknown",
                                                                 polygon=entity_visible_polygon,
                                                                 pose=[entity_visible_polygon.centroid.coords[0][0],
                                                                       entity_visible_polygon.centroid.coords[0][1],
                                                                       0.0],
                                                                 full_geometry_acquired=full_geometry_acquired,
                                                                 type_in="unknown",
                                                                 uid=entity_uid)
        return entities_in_g_fov

    def get_entities_in_s_fov_seethrough(self, robot_uid):
        robot = self.entities[robot_uid]

        entities_in_s_fov = dict()

        for entity_uid, entity in self.entities.items():
            if entity_uid != robot_uid:
                if entity.polygon.within(robot.s_fov_polygon):

                    entities_in_s_fov[entity_uid] = Obstacle(name=entity.name,
                                                             polygon=entity.polygon,
                                                             pose=entity.pose,
                                                             full_geometry_acquired=True,
                                                             type_in=entity.type,
                                                             uid=entity_uid)
        return entities_in_s_fov

    def get_map_bounds(self):
        if len(self.entities) == 0:
            raise ValueError("There are no entities to populate the grid, it can't be created !")
        all_entity_polygons_in_map = []
        for entity_uid, entity in self.entities.items():
            all_entity_polygons_in_map.append(entity.polygon)
        unioned_polygons = cascaded_union(all_entity_polygons_in_map)
        return unioned_polygons.bounds

    def _update_dd_and_reset_grids(self):
        if self.dd is None:
            raise ValueError("Discretization data (dd) is None, this should not be happening !")

        min_x, min_y, max_x, max_y = self.get_map_bounds()
        width, height = max_x - min_x, max_y - min_y

        is_dd_same = (self.dd.grid_pose[0] == min_x and self.dd.grid_pose[1] == min_y
                      and self.dd.width == width and self.dd.height == height)
        if not is_dd_same:
            self.dd.grid_pose = [min_x, min_y, 0.0]
            self.dd.width, self.dd.height = width, height
            self.dd.d_width, self.dd.d_height = (int(round(self.dd.width / self.dd.res)),
                                                 int(round(self.dd.height / self.dd.res)))
            self._int_grid = np.zeros((self.dd.d_width, self.dd.d_height), dtype=np.int16)
            self.new_entities = copy.copy(self.entities)
            self.prev_entities = dict()
            self.saved_social_costmaps = dict()

    def get_social_costmap(self, entity_uids_to_ignore, rp, restrict_4_neighbors=False):
        if entity_uids_to_ignore in self.saved_social_costmaps:
            return self.saved_social_costmaps[entity_uids_to_ignore]
        else:
            new_costmap = self._compute_social_costmap_for_entities(entity_uids_to_ignore, rp, restrict_4_neighbors)
            self.saved_social_costmaps[entity_uids_to_ignore] = new_costmap
            return new_costmap

    def get_binary_occupancy_grid(self, entity_uids_to_ignore):
        if entity_uids_to_ignore in self._binary_occupancy_grid:

    def invalidate_saved_costmaps(self, changed_entities):
        for entities in self.saved_social_costmaps.keys():
            # If the changed obstacles are not associated with the saved costmap, the costmap's computation
            # depends on them, so we cannot preserve the previously saved one
            if not set(changed_entities).issubset(entities):
                del self.saved_social_costmaps[entities]

    def compute_discrete_connected_components(self):
        connected_grid = copy.deepcopy(self.get_inflated_grid())
        neighborhood = utils.CHESSBOARD_NEIGHBORHOOD

        closed_set = set()
        current_component_index = -1

        for i in range(self.dd.d_width):
            for j in range(self.dd.d_height):
                current_cell = (i, j)
                if current_cell not in closed_set and connected_grid[i][j] == 0:
                    connected_grid[i][j] = current_component_index
                    open_set = utils.get_neighbors(current_cell, self.dd.d_width, self.dd.d_height, neighborhood)
                    closed_set.add(current_cell)
                    while open_set:
                        neighbor_cell = open_set.pop()
                        if connected_grid[neighbor_cell[0]][neighbor_cell[1]] == 0:
                            connected_grid[neighbor_cell[0]][neighbor_cell[1]] = current_component_index
                            open_set = open_set.union(
                                utils.get_neighbors(neighbor_cell, self.dd.d_width, self.dd.d_height, neighborhood))
                            closed_set.add(neighbor_cell)
                    current_component_index -= 1

        # plt.imshow(connected_grid); plt.show()
        nb_components = -current_component_index
        return connected_grid, nb_components

    # def edge_crosses_buffered_polygons(self, edge, buffered_polygons):
    #     for buffered_polygon in buffered_polygons:
    #         if LineString(edge).crosses(buffered_polygon):
    #             return True
    #     return False
    #
    # # Misguided attempt: actually, we don't need the visibility to compute connected components, but the interior
    # # of the buffered polygons unions
    # def compute_visibility_graph(self, radius, excluded_entities_uids, resolution=1, cap_style=1, join_style=1, mitre_limit=5.0):
    #     # Compute buffered polygons
    #     buffered_polygons = []
    #     buffered_graph = nx.Graph()
    #     for entity_uid, entity in self.entities.items():
    #         if entity_uid not in excluded_entities_uids:
    #             if resolution <= 0:
    #                 raise NotImplementedError
    #                 # points = entity.polygon.exterior.coords
    #                 # buffered_entities[entity_uid] = # COMPLETE
    #             else:
    #                 buffered_polygon = entity.polygon.buffer(radius, resolution=resolution, cap_style=cap_style,
    #                                                          join_style=join_style, mitre_limit=mitre_limit)
    #                 plt.plot(*buffered_polygon.exterior.xy)
    #                 # Iterate over buffered polygon edges to check for intersection
    #                 # with previously computed buffered polygons
    #                 points = list(buffered_polygon.exterior.coords)
    #                 for i in range(len(points) - 1):
    #                     edge = [points[i], points[i + 1]]
    #                     if not self.edge_crosses_buffered_polygons(edge, buffered_polygons):
    #                         buffered_graph.add_edge(edge[0], edge[1])
    #                 last_edge = [points[-1], points[0]]
    #                 if not self.edge_crosses_buffered_polygons(last_edge, buffered_polygons):
    #                     buffered_graph.add_edge(last_edge[0], last_edge[1])
    #
    #                 buffered_polygons.append(buffered_polygon)
    #
    #     plt.show()
    #     nx.draw(buffered_graph)
    #     plt.show()
    #     return buffered_graph
    #
    # def compute_polygonal_connected_components(self):
    #     visibility_graph = self.compute_visibility_graph(self.dd.inflation_radius, [self.robot_uid])
    #     connected_components = nx.connected_component_subgraphs(visibility_graph)
    #     connected_polygons = []
    #     for component in connected_components:
    #         if len(component.edges) >= 3:
    #             points = []
    #             for edge in component.edges:
    #                 points.append(edge[0])
    #             connected_polygon = Polygon(points)
    #             connected_polygons.append(connected_polygon)
    #             plt.plot(*connected_polygon.exterior.xy)
    #     plt.show()
    #     return connected_components, connected_polygons
    #
    # def get_entities_in_g_fov(self, robot_uid):
    #     WIP: Main idea would be:
    #     - iterate over see-through entities and create list of points
    #     - cast ray from robot center to each point, and associate obstacles that intersection by order of closest
    #     euc dist
    #     - if a ray passes within an obstacle it is stopped. add intersection point to obstacle point list
    #     - if it only intersects with the border, it continues
    #     - remove all points that could not be reached by a ray from obstacle's point list
    #     robot = self.entities[robot_uid]
    #
    #     entities_in_g_fov_seethrough = self.get_entities_in_g_fov_seethrough(robot_uid)
    #
    #     entities_in_g_fov = dict()
    #
    #     for entity_uid, entity in entities_in_g_fov_seethrough.items():
    #         for point in entity.polygon.exterior.coords:
