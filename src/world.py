import yaml
import numpy as np
from math import ceil, floor

from robot import Robot
from obstacle import Obstacle
from taboo import Taboo

from custom_exceptions import EntityPlacementException

from discretization_data import DiscretizationData

from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from shapely.errors import TopologicalError


class World:

    # Constructor
    def __init__(self, entities=None, taboos=None, robot_uid=None, whitelist=None, push_only_list=None,
                 force_pushes_only=True, dd=None, grid=None):

        self.entities = entities if entities is not None else dict()
        self.taboos = taboos if taboos is not None else dict()

        self.robot_uid = robot_uid

        self.whitelist = whitelist if whitelist is not None else []
        self.push_only_list = push_only_list if push_only_list is not None else []
        self.force_pushes_only = force_pushes_only

        self.dd = dd
        self._grid = grid
        self._is_grid_valid = False

    def load_from_yaml(self, path_to_file):
        config = yaml.load(open(path_to_file))

        # Get map discretization parameters
        self.dd = DiscretizationData(res=config["discretization_data"]["res"],
                                     inflation_radius=config["discretization_data"]["inflation_radius"],
                                     cost_lethal=config["discretization_data"]["cost_lethal"],
                                     cost_inscribed=config["discretization_data"]["cost_inscribed"],
                                     cost_circumscribed=config["discretization_data"]["cost_circumscribed"],
                                     cost_possibly_nonfree=config["discretization_data"]["cost_possibly_nonfree"]
                                     )

        # Get whitelist of movable objects
        self.whitelist = config["whitelist"]

        # Get list of objects that are only pushable
        self.push_only_list = config["push_only_list"]

        # Get whether we force pushes only for all objects in this world
        self.force_pushes_only = config["force_pushes_only"]

        # Get objects
        for object_config in config["objects"]:
            new_object = Obstacle(name=object_config["name"],
                                  polygon=Polygon(object_config["polygon"]),
                                  type_in=object_config["type"],
                                  dd=self.dd,
                                  full_geometry_acquired=True,
                                  pushes_only=True if config["force_pushes_only"] else (
                                    True if object_config["type"] in config["push_only_list"] else False),
                                  movability="movable" if object_config["type"] in config["whitelist"] else "unmovable")

            self.add_entity(new_object)

        # Get robot
        new_robot = Robot(name=config["robot"]["name"],
                          dd=self.dd,
                          full_geometry_acquired=True,
                          radius=config["robot"]["radius"],
                          initial_pose=config["robot"]["initial_pose"],
                          g_fov_max_radius=config["robot"]["g_fov_max_radius"],
                          g_fov_min_radius=config["robot"]["g_fov_min_radius"],
                          g_fov_opening_angle=config["robot"]["g_fov_opening_angle"],
                          s_fov_max_radius=config["robot"]["s_fov_max_radius"],
                          s_fov_min_radius=config["robot"]["s_fov_min_radius"],
                          s_fov_opening_angle=config["robot"]["s_fov_opening_angle"])

        self.add_entity(new_robot)

        # Get Taboo zones
        if isinstance(config["taboos"], list):
            for taboo_config in config["taboos"]:
                new_taboo = Taboo(name=taboo_config["name"],
                                  polygon=Polygon(taboo_config["polygon"]),
                                  cost=taboo_config["cost"],
                                  dd=self.dd)
                self.taboos[new_taboo.uid] = new_taboo

        self._is_grid_valid = False

    def add_entity(self, new_entity):
        for obj in self.entities.values():
            is_within = new_entity.within(obj)
            if is_within:
                raise EntityPlacementException("Entity {} would be within entity {}. Cannot load world.".format(
                    new_entity.name, obj.name))
        self.entities[new_entity.uid] = new_entity

        if isinstance(new_entity, Robot):
            self.robot_uid = new_entity.uid

        self._is_grid_valid = False

    def translate_entity(self, entity_uid, translation):
        self.entities[entity_uid].translate(translation[0], translation[1], self.dd)
        if entity_uid != self.robot_uid:
            self._is_grid_valid = False

    def rotate_entity(self, entity_uid, rotation):
        self.entities[entity_uid].rotate(rotation)
        if entity_uid != self.robot_uid:
            self._is_grid_valid = False

    def update_from_g_fov(self, entities):
        for entity_uid, entity in entities.items():
            if isinstance(entity, Obstacle):
                # If entity is already registered, update it
                try:
                    self_entity = self.entities[entity_uid]
                    # If self entity full geometry has not been acquired, update it
                    if not self_entity.full_geometry_acquired:
                        if entity.full_geometry_acquired:
                            self_entity.set_polygon(entity.polygon, self.dd)
                            self_entity.full_geometry_acquired = True
                        else:
                            self_entity.set_polygon(
                                cascaded_union([self_entity.polygon, entity.polygon]).convex_hull, self.dd)
                    # If it is already known, only translate/rotate the polygon appropriately
                    else:
                        if entity.full_geometry_acquired and \
                                self_entity.movability != "unmovable":
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
        self._is_grid_valid = False

    def update_from_s_fov(self, entities):
        for entity_uid, entity in entities.items():
            if isinstance(entity, Obstacle):
                # If entity is already registered, update it
                try:
                    self.entities[entity_uid].name = entity.name
                    self.entities[entity_uid].type = entity.type
                    if self.entities[entity_uid].movability != "unmovable":
                        self.entities[entity_uid].movability = entity.movability

                # If entity is not registered yet, create it
                except KeyError as e:
                    raise(e, "update_from_s_fov should never need to create an object !")
        self._is_grid_valid = False

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
                                                                 dd=self.dd,
                                                                 full_geometry_acquired=full_geometry_acquired,
                                                                 pushes_only=True if self.force_pushes_only else False,
                                                                 type_in="unknown",
                                                                 movability="unknown",
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
                                                             dd=self.dd,
                                                             full_geometry_acquired=True,
                                                             pushes_only=entity.pushes_only,
                                                             type_in=entity.type,
                                                             movability=entity.movability,
                                                             uid=entity_uid)
        return entities_in_s_fov

    def get_grid(self):
        if not self._is_grid_valid:
            self._update_grid()
            self._is_grid_valid = True
        return self._grid

    def _get_map_bounds(self):
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), -float("inf"), -float("inf")
        for entity_uid, entity in self.entities.items():
            cur_min_x, cur_min_y, cur_max_x, cur_max_y = entity.get_inflated_polygon(self.dd).bounds
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

        d_width, d_height = int(ceil(width / self.dd.res)), int(ceil(height / self.dd.res))
        self.dd.d_width, self.dd.d_height = d_width, d_height

        grid = np.zeros((d_width, d_height))

        for entity_uid, entity in self.entities.items():
            if entity_uid != self.robot_uid:
                e_min_x, e_min_y, e_max_x, e_max_y = entity.get_inflated_polygon(self.dd).bounds

                min_cell_x = int(round((e_min_x - min_x) / self.dd.res))
                min_cell_y = int(round((e_min_y - min_y) / self.dd.res))
                max_cell_x = min_cell_x + entity.get_discrete_polygon(self.dd).shape[0]
                max_cell_y = min_cell_y + entity.get_discrete_polygon(self.dd).shape[1]

                i = 0
                for x in range(min_cell_x, max_cell_x):
                    j = 0
                    for y in range(min_cell_y, max_cell_y):
                        if grid[x][y] < entity.get_discrete_polygon(self.dd)[i][j]:
                            grid[x][y] = entity.get_discrete_polygon(self.dd)[i][j]
                        j = j + 1
                    i = i + 1

        self._grid = grid

    # WIP: Main idea would be:
    # - iterate over see-through entities and create list of points
    # - cast ray from robot center to each point, and associate obstacles that intersection by order of closest euc dist
    # - if a ray passes within an obstacle it is stopped. add intersection point to obstacle point list
    # - if it only intersects with the border, it continues
    # - remove all points that could not be reached by a ray from obstacle's point list
    # def get_entities_in_g_fov(self, robot_uid):
    #     robot = self.entities[robot_uid]
    #
    #     entities_in_g_fov_seethrough = self.get_entities_in_g_fov_seethrough(robot_uid)
    #
    #     entities_in_g_fov = dict()
    #
    #     for entity_uid, entity in entities_in_g_fov_seethrough.items():
    #         for point in entity.polygon.exterior.coords:
