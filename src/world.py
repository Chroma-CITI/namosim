import yaml
import numpy as np
from math import ceil, floor
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

# TODO REMOVE BELOW
import skimage.morphology as skimage_morph
from skimage.morphology import medial_axis
import scipy.ndimage.morphology as scipy_morph
import matplotlib.pyplot as plt
import cv2
# TODO REMOVE ABOVE

from xml.dom import minidom
from svgpath2mpl import parse_path
import os
import shapely.affinity as affinity
import networkx as nx


class World:

    # Constructor
    def __init__(self, entities=None, taboos=None, robot_uid=None, dd=None, inflated_grid=None):

        self.entities = entities if entities is not None else dict()
        self.taboos = taboos if taboos is not None else dict()

        self.robot_uid = robot_uid

        self.dd = dd
        self._inflated_grid = inflated_grid
        self._is_inflated_grid_valid = False

        # Social costmaps related fields
        # TODO Properly parameterize all this...
        self.saved_social_costmaps = dict()
        self.use_social_layer = True

        self.half_1_u_p = 0.45
        self.half_2_u_p = 0.70
        self.half_3_u_p = 0.90
        self.half_4_u_p = 1.20

        self.cost_value_at_0_u_p = 0.0
        self.cost_value_before_1_u_p = 0.1
        self.cost_value_at_1_u_p = 1.0
        self.cost_value_at_2_u_p = 0.9
        self.cost_value_at_3_u_p = 0.75
        self.cost_value_at_4_u_p_and_beyond = 0.25

        self.curve_0_to_1_u_p = (self.cost_value_before_1_u_p - self.cost_value_at_0_u_p) / (self.half_1_u_p - 0.0)
        self.offset_0_to_1_u_p = self.cost_value_before_1_u_p - self.curve_0_to_1_u_p * self.half_1_u_p

        self.curve_1_to_2_u_p = (self.cost_value_at_2_u_p - self.cost_value_at_1_u_p) / (self.half_2_u_p - self.half_1_u_p)
        self.offset_1_to_2_u_p = self.cost_value_at_2_u_p - self.curve_1_to_2_u_p * self.half_2_u_p

        self.curve_2_to_3_u_p = (self.cost_value_at_3_u_p - self.cost_value_at_2_u_p) / (self.half_3_u_p - self.half_2_u_p)
        self.offset_2_to_3_u_p = self.cost_value_at_3_u_p - self.curve_2_to_3_u_p * self.half_3_u_p

        self.curve_3_to_4_u_p = (self.cost_value_at_4_u_p_and_beyond - self.cost_value_at_3_u_p) / (self.half_4_u_p - self.half_3_u_p)
        self.offset_3_to_4_u_p = self.cost_value_at_4_u_p_and_beyond - self.curve_3_to_4_u_p * self.half_4_u_p

        self.decay_factor = 0.1
        self.keep_number_of_decimals = 10
        self.decimals_multiplicator = 10 ** self.keep_number_of_decimals
        self.decay_limit = self.decimals_multiplicator * self.cost_value_at_4_u_p_and_beyond

    def load_from_yaml(self, path_to_file):
        # Import YAML world configuration file
        yaml_abs_path = os.path.abspath(path_to_file)
        config = yaml.load(open(yaml_abs_path))

        # Import SVG geometry file specified in YAML configuration
        yaml_working_directory = os.path.dirname(yaml_abs_path)
        geometry_file_path = os.path.join(yaml_working_directory, config["files"]["geometry_file"])
        geometry_file = minidom.parse(geometry_file_path)
        svg_geometries = {path.getAttribute("id"): path.getAttribute('d') for path in geometry_file.getElementsByTagName('path')}
        shapely_polygons = dict()
        # Convert imported geometry to shapely polygons
        for svg_id, svg_geometry in svg_geometries.items():
            parse_result = parse_path(svg_geometry)
            polygon_pts = parse_result.to_polygons()[0] * config["geometry_scale"]
            polygon_pts[:, 1] = -polygon_pts[:, 1] # Mirror on y-axis
            shapely_polygons[svg_id] = Polygon(polygon_pts)
        # TODO Fix this so that it only accounts for obstacles in polygon layer otherwise, things might get messy with
        #  direction vectors that get outside of the obstacle polygons
        # Center the imported geometries
        unioned_polygons = cascaded_union(shapely_polygons.values())
        bounding_box = box(unioned_polygons.bounds[0], unioned_polygons.bounds[1],
                           unioned_polygons.bounds[2], unioned_polygons.bounds[3])
        translation_to_center = [bounding_box.centroid.coords[0][0], bounding_box.centroid.coords[0][1]]
        for svg_id, polygon in shapely_polygons.items():
            shapely_polygons[svg_id] = affinity.translate(polygon, -translation_to_center[0], -translation_to_center[1])

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
                orientation_polygon = shapely_polygons[entity_data["geometry"]["orientation_id"]]
                if orientation_polygon:
                    pose[2] = 0.0 # TODO Check if orientation object exists and that polygon has only one side
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
                polygon = shapely_polygons[entity_data["geometry"]["id"]]
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
                    goal_polygon = shapely_polygons[goal_data["geometry"]["id"]]
                    # TODO: Fix this so that orientation can be specified by vector in SVG and position can be
                    #       determined from YAML file if wanted
                    goals.append([goal_polygon.centroid.coords[0][0], goal_polygon.centroid.coords[0][1], 0.0])
            if ("taboos" in config["things"]["zones"]
                    and isinstance(config["things"]["zones"]["taboos"], list)):
                for thing_data in config["things"]["zones"]["taboos"]:
                    new_taboo = Taboo(name=thing_data["name"],
                                      polygon=Polygon(thing_data["polygon"]))
                    self.taboos[new_taboo.uid] = new_taboo

        self._is_inflated_grid_valid = False
        self.saved_social_costmaps = dict()

        self.compute_discrete_connected_components()

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
        self.invalidate_saved_costmaps((new_entity.uid,))

    def remove_entity(self, entity_uid):
        if entity_uid in self.entities:
            del self.entities[entity_uid]
        else:
            print("Warning, you tried to remove an entity that is not registered in this world !")

        self._is_inflated_grid_valid = False
        self.invalidate_saved_costmaps((entity_uid,))

    def remove_entities(self, entities_uids):
        for entity_uid in entities_uids:
            self.remove_entity(entity_uid)

    def translate_entity(self, entity_uid, translation):
        entity = self.entities[entity_uid]
        entity.translate(translation[0], translation[1], self.dd)
        if entity_uid != self.robot_uid:
            self._is_inflated_grid_valid = False
            self.invalidate_saved_costmaps((entity_uid,))

    def rotate_entity(self, entity_uid, rotation):
        self.entities[entity_uid].rotate(rotation)
        if entity_uid != self.robot_uid:
            self._is_inflated_grid_valid = False
            self.invalidate_saved_costmaps((entity_uid,))

    def update_from_g_fov(self, entities):
        for entity_uid, entity in entities.items():
            if isinstance(entity, Obstacle):
                # If entity is already registered, update it
                try:
                    self_entity = self.entities[entity_uid]
                    self_entity_movability = self.entities[self.robot_uid].deduce_movability(self_entity.type)
                    # If self entity full geometry has not been acquired, update it
                    if not self_entity.full_geometry_acquired:
                        if entity.full_geometry_acquired:
                            self_entity.set_polygon(entity.polygon, self.dd, self_entity_movability)
                            self_entity.full_geometry_acquired = True
                        else:
                            self_entity.set_polygon(
                                cascaded_union([self_entity.polygon, entity.polygon]).convex_hull,
                                self.dd, self_entity_movability)
                        self.invalidate_saved_costmaps((entity.uid,))
                        self._is_inflated_grid_valid = False
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
                    self.invalidate_saved_costmaps((entity.uid,))
                    self._is_inflated_grid_valid = False


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

    def get_inflated_grid(self):
        if not self._is_inflated_grid_valid:
            self._update_grid()
            self._is_inflated_grid_valid = True
        return self._inflated_grid

    def _get_map_bounds(self):
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), -float("inf"), -float("inf")
        for entity_uid, entity in self.entities.items():
            cur_min_x, cur_min_y, cur_max_x, cur_max_y = entity.polygon.bounds
            min_x = cur_min_x - self.dd.res if cur_min_x < min_x else min_x
            min_y = cur_min_y - self.dd.res if cur_min_y < min_y else min_y
            max_x = cur_max_x + self.dd.res if cur_max_x > max_x else max_x
            max_y = cur_max_y + self.dd.res if cur_max_y > max_y else max_y
        return min_x, min_y, max_x, max_y

    def _update_grid(self):
        # Don't create grid if we don't have dd data
        if self.dd is None:
            return

        if self.dd.width != 0.0 and self.dd.height != 0.0 and self.dd.d_width != 0 and self.dd.d_height != 0:
            min_x, min_y = self.dd.grid_pose[0], self.dd.grid_pose[1]
            max_x, max_y = self.dd.grid_pose[0] + self.dd.width, self.dd.grid_pose[1] + self.dd.height
        else:
            min_x, min_y, max_x, max_y = self._get_map_bounds()

        # No grid can be created without entities to populate it...
        if min_x == float("inf") or min_y == float("inf") or max_x == -float("inf") or max_y == -float("inf"):
            return

        width, height = max_x - min_x, max_y - min_y
        self.dd.grid_pose = [min_x, min_y, 0.0]
        self.dd.width, self.dd.height = width, height

        d_width, d_height = int(floor(width / self.dd.res)), int(floor(height / self.dd.res))
        self.dd.d_width, self.dd.d_height = d_width, d_height

        grid = np.zeros((d_width, d_height))

        for entity_uid, entity in self.entities.items():
            if entity_uid != self.robot_uid:
                e_min_x, e_min_y, e_max_x, e_max_y = entity.get_inflated_polygon(self.dd).bounds

                min_cell_x = int(round((e_min_x - min_x) / self.dd.res))
                min_cell_y = int(round((e_min_y - min_y) / self.dd.res))
                discrete_inflated_polygon = entity.get_discrete_inflated_polygon(self.dd)
                max_cell_x = min_cell_x + discrete_inflated_polygon.shape[0]
                max_cell_y = min_cell_y + discrete_inflated_polygon.shape[1]

                if self.use_social_layer and not entity.is_discrete_cell_set_valid():
                    entity_discrete_cells_set = []
                    entity_discrete_inflated_cells_set = []
                    i = 0
                    for x in range(min_cell_x, max_cell_x):
                        j = 0
                        for y in range(min_cell_y, max_cell_y):
                            try:
                                if grid[x][y] < discrete_inflated_polygon[i][j]:
                                    grid[x][y] = discrete_inflated_polygon[i][j]
                                if discrete_inflated_polygon[i][j] == self.dd.cost_lethal:
                                    entity_discrete_cells_set.append((x, y))
                                if self.dd.cost_possibly_nonfree <= discrete_inflated_polygon[i][j] < self.dd.cost_lethal:
                                    entity_discrete_inflated_cells_set.append((x, y))
                            except IndexError:
                                pass  # Trim non-lethal obstacle cells around map
                            j = j + 1
                        i = i + 1
                    entity.set_discrete_cells_set(entity_discrete_cells_set)
                    entity.set_discrete_inflated_cells_set(entity_discrete_inflated_cells_set)
                else:
                    i = 0
                    for x in range(min_cell_x, max_cell_x):
                        j = 0
                        for y in range(min_cell_y, max_cell_y):
                            try:
                                if grid[x][y] < discrete_inflated_polygon[i][j]:
                                    grid[x][y] = discrete_inflated_polygon[i][j]
                            except IndexError:
                                pass  # Trim non-lethal obstacle cells around map
                            j = j + 1
                        i = i + 1

        self._inflated_grid = grid

    def get_discrete_cells_set_for_entity_uid(self, entity_uid):
        entity = self.entities[entity_uid]
        if not entity.is_discrete_cell_set_valid():
            self._update_grid()
        return entity.get_discrete_cells_set()

    def get_discrete_inflated_cells_set_for_entity_uid(self, entity_uid):
        entity = self.entities[entity_uid]
        if not entity.is_discrete_inflated_cell_set_valid():
            self._update_grid()
        return entity.get_discrete_inflated_cells_set()

    def get_social_costmap_for_entities_uids(self, entities_uids, rp, restrict_4_neighbors=False):
        if entities_uids in self.saved_social_costmaps:
            return self.saved_social_costmaps[entities_uids]
        else:
            new_costmap = self._compute_social_costmap_for_entities(entities_uids, rp, restrict_4_neighbors)
            self.saved_social_costmaps[entities_uids] = new_costmap
            return new_costmap

    def _compute_social_costmap_for_entities(self, entities_uids, rp, restrict_4_neighbors=True):
        # Acceptable transitions from current grid element to neighbors
        neighborhood_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighborhood_8 = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        if restrict_4_neighbors:
            neighborhood = neighborhood_4
        else:
            neighborhood = neighborhood_8

        # TODO :
        #  - Add support for restrict_4_neighbors
        #  - Add loop for building the final_array from the skeleton values
        world_copy_without_entities = copy.deepcopy(self)
        world_copy_without_entities.remove_entities(entities_uids)

        grid_without_entities = world_copy_without_entities.get_inflated_grid()
        grid_without_entities[grid_without_entities < self.dd.cost_lethal] = 1.0
        grid_without_entities[grid_without_entities == self.dd.cost_lethal] = 0.0
        plt.imshow(grid_without_entities)
        plt.show()

        # Distance transform
        # test_distance_transform = scipy_morph.distance_transform_cdt(grid_without_entities, 'chessboard')
        test_distance_transform = scipy_morph.distance_transform_edt(grid_without_entities)
        plt.imshow(test_distance_transform)
        plt.show()

        # Skeleton
        test_skeleton = skimage_morph.skeletonize(grid_without_entities)
        # test_skeleton = medial_axis(grid_without_entities, return_distance=True)[0]
        plt.imshow(test_skeleton)
        plt.show()

        skeleton_cells_arrays = np.where(test_skeleton == True)
        final_array = np.full(test_distance_transform.shape, -1)
        width, height = final_array.shape[0], final_array.shape[1]
        skeleton_cells_nb = len(skeleton_cells_arrays[0])
        closed_cell_set = set()

        ordered_value_list = []

        for i in range(skeleton_cells_nb):
            x, y = skeleton_cells_arrays[0][i], skeleton_cells_arrays[1][i]
            closed_cell_set.add((x, y))
            value = int(self.skeleteton_social_cost_function(test_distance_transform[x][y]) * self.decimals_multiplicator)
            final_array[x][y] = value
            if value not in ordered_value_list:
                ordered_value_list.append(value)
                ordered_value_list.sort()


        # Min variant
        cur_set = closed_cell_set
        prev_set = cur_set
        while cur_set:
            rp.publish_grid_map(final_array / float(self.decimals_multiplicator / 100), self.dd)
            plt.imshow(final_array)
            plt.show()
            next_set = set()
            for current in cur_set:
                for i, j in neighborhood_4:
                    neighbor = current[0] + i, current[1] + j
                    if neighbor not in cur_set and neighbor not in prev_set:
                        # Check that neighbor exists within the map
                        if utils.is_in_matrix(neighbor, width, height) and grid_without_entities[neighbor[0]][neighbor[1]] != 0.0:
                            # MIN CASE
                            _min = float("inf")
                            # # AVG CASE
                            # _avg = 0
                            # _count = 0
                            for k, l in neighborhood_4:
                                neighbor_of_neighbor = neighbor[0] + k, neighbor[1] + l
                                if utils.is_in_matrix(neighbor_of_neighbor, width, height):
                                    n_o_n_value = final_array[neighbor_of_neighbor[0]][neighbor_of_neighbor[1]]
                                    # print("Value at current cell {cell} : {val}".format(
                                    #     cell=str(current), val=final_array[current[0]][current[1]]))
                                    # print("Value at neighbor cell {cell} : {val}".format(
                                    #         cell=str(neighbor), val=final_array[neighbor[0]][neighbor[1]]))
                                    # print("Value at neighbor of neighbor cell {cell} : {val}".format(
                                    #     cell=str(neighbor_of_neighbor), val=n_o_n_value))
                                    # print("----------------------------------------------------------------------")
                                    # MIN CASE
                                    if neighbor_of_neighbor not in next_set:
                                        if n_o_n_value != -1:
                                            if n_o_n_value < _min:
                                                _min = n_o_n_value
                                    # # AVG CASE
                                    # if neighbor_of_neighbor not in next_set:
                                    #     if n_o_n_value != -1:
                                    #         _avg += n_o_n_value
                                    #         _count += 1
                            # MIN CASE
                            final_array[neighbor[0]][neighbor[1]] = self.decay_function(_min)
                            # # AVG CASE
                            # _avg = int(float(_avg) / float(_count))
                            # final_array[neighbor[0]][neighbor[1]] = self.decay_function(_avg)
                            next_set.add(neighbor)
            prev_set = cur_set
            cur_set = next_set

        rp.publish_grid_map(final_array / float(self.decimals_multiplicator / 100), self.dd)

        return final_array / float(self.decimals_multiplicator / 100)

    def skeleteton_social_cost_function(self, dist_in_cells):
        dist_real = dist_in_cells * self.dd.res

        if 0.0 < dist_real < self.half_1_u_p:
            return self.curve_0_to_1_u_p * dist_real + self.offset_0_to_1_u_p
        elif self.half_1_u_p <= dist_real < self.half_2_u_p:
            return self.curve_1_to_2_u_p * dist_real + self.offset_1_to_2_u_p
        elif self.half_2_u_p <= dist_real < self.half_3_u_p:
            return self.curve_2_to_3_u_p * dist_real + self.offset_2_to_3_u_p
        elif self.half_3_u_p <= dist_real < self.half_4_u_p:
            return self.curve_3_to_4_u_p * dist_real + self.offset_3_to_4_u_p
        elif self.half_4_u_p <= dist_real:
            return self.cost_value_at_4_u_p_and_beyond
        else:
            return -1.0

    def decay_function(self, cost):
        return cost - cost * self.decay_factor

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

        plt.imshow(connected_grid)
        plt.show()
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

    # def _update_grid_dynamic(self):
    #     # Don't create grid if we don't have dd data
    #     if self.dd is None:
    #         return
    #
    #     if self.dd.width != 0.0 and self.dd.height != 0.0 and self.dd.d_width != 0 and self.dd.d_height != 0:
    #         min_x, min_y = self.dd.grid_pose[0], self.dd.grid_pose[1]
    #         max_x, max_y = self.dd.grid_pose[0] + self.dd.width, self.dd.grid_pose[1] + self.dd.height
    #     else:
    #         min_x, min_y, max_x, max_y = self._get_map_bounds()
    #
    #     # No grid can be created without entities to populate it...
    #     if min_x == float("inf") or min_y == float("inf") or max_x == -float("inf") or max_y == -float("inf"):
    #         return
    #
    #     width, height = max_x - min_x, max_y - min_y
    #     grid_pose = [min_x, min_y, 0.0]
    #     d_width, d_height = int(ceil(width / self.dd.res)), int(ceil(height / self.dd.res))
    #
    #     is_grid_same = (self.dd.grid_pose == grid_pose and self.dd.width == width and self.dd.height == height
    #                     and self.dd.d_width == d_width and self.dd.d_height == d_height)
    #     if not is_grid_same:
    #         self.dd.grid_pose = grid_pose
    #         self.dd.width, self.dd.height = width, height
    #         self.dd.d_width, self.dd.d_height = d_width, d_height
    #         self._grid = np.zeros((d_width, d_height))
    #         self._supergrid = np.full((d_width, d_height), {})
    #
    #     for entity_uid, entity in self.entities.items():
    #         entity_is_not_robot = entity_uid != self.robot_uid
    #         grid_not_same_or_same_but_entities_moved = (not is_grid_same or
    #                                                     (is_grid_same and not entity.are_discrete_cells_sets_valid()))
    #
    #         if entity_is_not_robot and grid_not_same_or_same_but_entities_moved:
    #             e_min_x, e_min_y, e_max_x, e_max_y = entity.get_inflated_polygon(self.dd).bounds
    #
    #             min_cell_x = int(round((e_min_x - min_x) / self.dd.res))
    #             min_cell_y = int(round((e_min_y - min_y) / self.dd.res))
    #             discrete_inflated_polygon = entity.get_discrete_inflated_polygon(self.dd)
    #             max_cell_x = min_cell_x + discrete_inflated_polygon.shape[0]
    #             max_cell_y = min_cell_y + discrete_inflated_polygon.shape[1]
    #
    #             entity_discrete_cells_set = set()
    #             entity_discrete_inflated_cells_set = set()
    #
    #             i = 0
    #             for x in range(min_cell_x, max_cell_x):
    #                 j = 0
    #                 for y in range(min_cell_y, max_cell_y):
    #                     self._supergrid =
    #
    #
    #                     if self._grid[x][y] < discrete_inflated_polygon[i][j]:
    #                         self._grid[x][y] = discrete_inflated_polygon[i][j]
    #
    #                     if discrete_inflated_polygon[i][j] == self.dd.cost_lethal:
    #                         entity_discrete_cells_set.add((i, j))
    #                     elif self.dd.cost_possibly_nonfree < discrete_inflated_polygon[i][j] < self.dd.cost_lethal:
    #                         entity_discrete_inflated_cells_set.add((i, j))
    #                     j = j + 1
    #                 i = i + 1
    #             entity.set_discrete_cells_set(entity_discrete_cells_set)
    #             entity.set_discrete_inflated_cells_set(entity_discrete_inflated_cells_set)
