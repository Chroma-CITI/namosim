import yaml
import numpy as np
from math import ceil

from robot import Robot
from obstacle import Obstacle
from taboo import Taboo

from custom_exceptions import EntityPlacementException

from discretization_data import DiscretizationData

from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
from shapely.ops import cascaded_union


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
                                  pushes_only=True if config["force_pushes_only"] else (
                                    True if object_config["type"] in config["push_only_list"] else False),
                                  movability="movable" if object_config["type"] in config["whitelist"] else "unmovable")

            self.add_entity(new_object)

        # Get robot
        robot_polygon = Point((config["robot"]["initial_pose"][0],
                               config["robot"]["initial_pose"][1])).buffer(config["robot"]["radius"])

        robot_g_fov_polygon = World._create_fov(robot_init_pose=config["robot"]["initial_pose"],
                                                fov_max_radius=config["robot"]["g_fov_max_radius"],
                                                fov_min_radius=config["robot"]["g_fov_min_radius"],
                                                fov_opening_angle=config["robot"]["g_fov_opening_angle"])

        robot_s_fov_polygon = World._create_fov(robot_init_pose=config["robot"]["initial_pose"],
                                                fov_max_radius=config["robot"]["s_fov_max_radius"],
                                                fov_min_radius=config["robot"]["s_fov_min_radius"],
                                                fov_opening_angle=config["robot"]["s_fov_opening_angle"])

        new_robot = Robot(name=config["robot"]["name"],
                          polygon=robot_polygon,
                          dd=self.dd,
                          radius=config["robot"]["radius"],
                          g_fov_polygon=robot_g_fov_polygon,
                          s_fov_polygon=robot_s_fov_polygon,
                          initial_pose=config["robot"]["initial_pose"]
                          )

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
        self._is_grid_valid = False

    def update_from_g_fov(self, entities):
        for entity_uid, entity in entities.items():
            if isinstance(entity, Obstacle):
                # If entity is already registered, update it
                try:
                    self.entities[entity_uid].set_polygon(
                        cascaded_union([self.entities[entity_uid].polygon, entity.polygon]).convex_hull, self.dd)
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
                    self.entities[entity_uid].set_polygon(entity.polygon, self.dd)
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
                    else:
                        entity_visible_polygon = entity.polygon.difference(
                            entity.polygon.difference(robot.g_fov_polygon))

                    if isinstance(entity_visible_polygon, Polygon):
                        entities_in_g_fov[entity_uid] = Obstacle(name="unknown",
                                                                 polygon=entity_visible_polygon,
                                                                 dd=self.dd,
                                                                 pushes_only=True if self.force_pushes_only else False,
                                                                 type_in="unknown",
                                                                 movability="movable",  # TODO Change to "unknown"
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
                                                             pushes_only=entity.pushes_only,
                                                             type_in=entity.type,
                                                             movability="movable",  # TODO Change to entity.movability
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
            cur_min_x, cur_min_y, cur_max_x, cur_max_y = entity.inflated_polygon.bounds
            min_x = cur_min_x if cur_min_x < min_x else min_x
            min_y = cur_min_y if cur_min_y < min_y else min_y
            max_x = cur_max_x if cur_max_x > max_x else max_x
            max_y = cur_max_y if cur_max_y > max_y else max_y
        return min_x, min_y, max_x, max_y

    def _update_grid(self):
        # Don't create grid if we don't have dd data
        if self.dd is None:
            return

        min_x, min_y, max_x, max_y = self._get_map_bounds()

        # No grid can be created without entities to populate it...
        if min_x == float("inf") or min_y == float("inf") or max_x == -float("inf") or max_y == -float("inf"):
            return

        width, height = max_x - min_x, max_y - min_y
        self.dd.grid_pose = [-width/2, -height/2, 0.0]

        d_width, d_height = int(ceil(width / self.dd.res)), int(ceil(height / self.dd.res))

        grid = np.zeros((d_width, d_height))

        for entity_uid, entity in self.entities.items():
            if entity_uid != self.robot_uid:
                e_min_x, e_min_y, e_max_x, e_max_y = entity.inflated_polygon.bounds

                min_cell_x = int(round((e_min_x - min_x) / self.dd.res))
                min_cell_y = int(round((e_min_y - min_y) / self.dd.res))
                max_cell_x = min_cell_x + entity.discrete_polygon.shape[0]
                max_cell_y = min_cell_y + entity.discrete_polygon.shape[1]

                i = 0
                for x in range(min_cell_x, max_cell_x):
                    j = 0
                    for y in range(min_cell_y, max_cell_y):
                        if grid[x][y] < entity.discrete_polygon[i][j]:
                            grid[x][y] = entity.discrete_polygon[i][j]
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

    @staticmethod
    def _create_shapely_arc(robot_init_pose, radius, opening_angle, numsegments=5):
        start_angle, end_angle = opening_angle * -0.5, opening_angle * 0.5  # In degrees

        # The coordinates of the arc
        theta = np.radians(np.linspace(start_angle, end_angle, numsegments))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Transform in shapely
        arc_in_robot_ref = LineString(np.column_stack([x, y]))
        arc_after_trans = affinity.translate(arc_in_robot_ref, robot_init_pose[0], robot_init_pose[1])
        arc_after_trans_rot = affinity.rotate(arc_after_trans, robot_init_pose[2],
                                              Point(robot_init_pose[0], robot_init_pose[1]))

        return arc_after_trans_rot

    @staticmethod
    def _create_fov(robot_init_pose, fov_max_radius, fov_min_radius, fov_opening_angle):
        g_fov_outer_arc = World._create_shapely_arc(robot_init_pose, fov_max_radius, fov_opening_angle)

        g_fov_inner_arc = World._create_shapely_arc(robot_init_pose, fov_min_radius, fov_opening_angle)

        # limit_line_length = fov_max_radius - fov_min_radius

        # g_fov_left_limit_line_in_robot_ref = LineString([[0.0, 0.0],
        #                                                  [limit_line_length * math.cos(fov_opening_angle * 0.5),
        #                                                   limit_line_length * math.sin(fov_opening_angle * 0.5)]])
        # g_fov_left_limit_line_after_translation = affinity.translate(g_fov_left_limit_line_in_robot_ref,
        #                                                              robot_init_pose[0],
        #                                                              robot_init_pose[1])
        # g_fov_left_limit_line = affinity.rotate(g_fov_left_limit_line_after_translation, robot_init_pose[2])
        #
        # g_fov_right_limit_line_in_robot_ref = LineString([[0.0, 0.0],
        #                                                   [limit_line_length * math.cos(fov_opening_angle * -0.5),
        #                                                    limit_line_length * math.sin(fov_opening_angle * -0.5)]])
        #
        # g_fov_right_limit_line_after_translation = affinity.translate(g_fov_right_limit_line_in_robot_ref,
        #                                                               robot_init_pose[0],
        #                                                               robot_init_pose[1])
        # g_fov_right_limit_line = affinity.rotate(g_fov_right_limit_line_after_translation, robot_init_pose[2])

        coords_outer = list(g_fov_outer_arc.coords)
        coords_inner = list(g_fov_inner_arc.coords)
        points = coords_inner + list(reversed(coords_outer))

        return Polygon(points)
