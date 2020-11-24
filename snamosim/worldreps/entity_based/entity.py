import math
import numpy as np
import copy
import shapely.affinity as affinity
from shapely.geometry import Polygon

from snamosim.utils import utils
from custom_exceptions import IntersectionError

from PIL import Image, ImageDraw


class Entity:
    last_id = 1

    # Constructor
    def __init__(self, name, polygon, pose, full_geometry_acquired, uid=0):
        if uid == 0:
            self.uid = Entity.last_id
            Entity.last_id = Entity.last_id + 1
        else:
            self.uid = uid
        self.name = name
        self.polygon = polygon
        self.pose = pose
        self.full_geometry_acquired = full_geometry_acquired
        self.is_being_manipulated = False

        self.inflation_radius = 0.
        self.inflated_polygon = None
        self._is_inflated_polygon_valid = False

        self.discrete_polygon = None
        self._is_discrete_polygon_valid = False

        self.discrete_inflated_polygon = None
        self._is_discrete_inflated_polygon_valid = False

        self.discrete_cells_set = None
        self._is_discrete_cell_set_valid = False

        self.discrete_inflated_cells_set = None
        self._is_discrete_inflated_cell_set_valid = False

    def within(self, other_entity):
        return self.polygon.within(other_entity.polygon)

    def get_inflated_polygon(self, inflation_radius, res):
        if not self._is_inflated_polygon_valid:
            self._inflate_polygon(inflation_radius, res)
        return self.inflated_polygon

    def get_discrete_polygon(self, inflation_radius, res, cost_lethal, cost_inscribed):
        if not self._is_discrete_polygon_valid:
            self._discretize_to_grid(inflation_radius, res, cost_lethal, cost_inscribed)
        return self.discrete_polygon

    def get_discrete_inflated_polygon(self, inflation_radius, res, cost_lethal, cost_inscribed):
        if not self._is_discrete_inflated_polygon_valid:
            self._discretize_to_grid(inflation_radius, res, cost_lethal, cost_inscribed)
        return self.discrete_inflated_polygon

    def get_discrete_cells_set(self, inflation_radius, res, grid_pose, grid_d_width, grid_d_height):
        if not self._is_discrete_cell_set_valid:
            self._discretize_to_cell_sets(inflation_radius, res, grid_pose, grid_d_width, grid_d_height)
        return self.discrete_cells_set

    def get_discrete_inflated_cells_set(self, inflation_radius, res, grid_pose, grid_d_width, grid_d_height):
        if not self._is_discrete_inflated_cell_set_valid or inflation_radius != self.inflation_radius:
            self._discretize_to_cell_sets(inflation_radius, res, grid_pose, grid_d_width, grid_d_height)
        return self.discrete_inflated_cells_set

    def set_polygon(self, polygon):
        self.polygon = polygon
        self.pose = [list(self.polygon.centroid.coords)[0][0],
                     list(self.polygon.centroid.coords)[0][1],
                     self.pose[2]]

        self._is_inflated_polygon_valid = False
        self._is_discrete_polygon_valid = False
        self._is_discrete_inflated_polygon_valid = False
        self._is_discrete_cell_set_valid = False
        self._is_discrete_inflated_cell_set_valid = False
        return self

    def rotate(self, angle, rot_center='centroid', other_entities=None, angular_res=5., ignore_collisions=False):
        # May be improved for cases with modulo 90-degrees rotations with specific update of discrete_polygon.
        new_polygon = affinity.rotate(self.polygon, angle, origin=rot_center)
        polygon_center = list(new_polygon.centroid.coords)[0]
        new_pose = (polygon_center[0], polygon_center[1], (self.pose[2] + angle) % 360)

        if other_entities is None:
            # If collision detection with other entities is not required
            self.polygon = new_polygon
            self.pose = new_pose
        else:
            rotation_steps_to_check = int(abs(angle) / angular_res)
            sign = -1. if angle < 0. else 1.
            collision_polygons = [affinity.rotate(self.polygon, sign * float(i) * angular_res, origin=rot_center)
                                  for i in range(rotation_steps_to_check)]
            for entity in other_entities:
                for collision_polygon in collision_polygons:
                    if collision_polygon.intersects(entity.polygon):
                        # from snamosim.display.ros_publisher import RosPublisher
                        # RosPublisher().publish_sim(collision_polygon, entity.polygon, "/collision")
                        if not ignore_collisions:
                            raise IntersectionError({self.uid, entity.uid},
                                ("Entity {self_name} would intersect with entity {other_name} " +
                                 "if rotation of angle ({angle}) at rotation center {rot_center} were to occur").format(
                                    self_name=self.name, other_name=entity.name, angle=angle, rot_center=str(rot_center)
                                ))
                if new_polygon.intersects(entity.polygon):
                    # from snamosim.display.ros_publisher import RosPublisher
                    # RosPublisher().publish_sim(new_polygon, entity.polygon, "/collision")
                    if not ignore_collisions:
                        raise IntersectionError({self.uid, entity.uid},
                            ("Entity {self_name} would intersect with entity {other_name} " +
                             "if rotation of angle ({angle}) at rotation center {rot_center} were to occur").format(
                                self_name=self.name, other_name=entity.name, angle=angle, rot_center=str(rot_center)
                            ))

            self.polygon = new_polygon
            self.pose = new_pose

        self._is_inflated_polygon_valid = False
        self._is_discrete_polygon_valid = False
        self._is_discrete_inflated_polygon_valid = False
        self._is_discrete_cell_set_valid = False
        self._is_discrete_inflated_cell_set_valid = False
        return self

    def translate(self, xoff, yoff, res=0.05, other_entities=None, ignore_collisions=False):
        if all(np.isclose([xoff, yoff], [0., 0.], atol=1e-8)):
            return self

        # May be improved for cases where the translation is equal to a multiple of the resolution
        new_polygon = affinity.translate(self.polygon, xoff, yoff)
        polygon_center = list(new_polygon.centroid.coords)[0]
        new_pose = (polygon_center[0], polygon_center[1], self.pose[2])

        if other_entities is None:
            # If collision detection with other entities is not required
            self.polygon = new_polygon
            self.pose = new_pose
        else:
            translation_length = math.sqrt(xoff ** 2 + yoff ** 2)
            translation_steps_to_check = int(math.ceil(translation_length / res))
            xoff_normed, yoff_normed = xoff / float(translation_steps_to_check), yoff / float(translation_steps_to_check)

            collision_polygons = [affinity.translate(self.polygon, xoff_normed * float(i), yoff_normed * float(i))
                                  for i in range(translation_steps_to_check)]
            for entity in other_entities:
                for collision_polygon in collision_polygons:
                    if collision_polygon.intersects(entity.polygon):
                        # from snamosim.display.ros_publisher import RosPublisher
                        # RosPublisher().publish_sim(collision_polygon, entity.polygon, "/collision")
                        if not ignore_collisions:
                            raise IntersectionError({self.uid, entity.uid},
                                ("Entity {self_name} would intersect with entity {other_name} " +
                                 "if translation of vector ({xoff}, {yoff}) were to occur").format(
                                    self_name=self.name, other_name=entity.name, xoff=xoff, yoff=yoff
                                ))
                if new_polygon.intersects(entity.polygon):
                    # from snamosim.display.ros_publisher import RosPublisher
                    # RosPublisher().publish_sim(new_polygon, entity.polygon, "/collision")
                    if not ignore_collisions:
                        raise IntersectionError({self.uid, entity.uid},
                            ("Entity {self_name} would intersect with entity {other_name} " +
                             "if translation of vector ({xoff}, {yoff}) were to occur").format(
                                self_name=self.name, other_name=entity.name, xoff=xoff, yoff=yoff
                            ))

            self.polygon = new_polygon
            self.pose = new_pose

        self._is_inflated_polygon_valid = False
        if (xoff / res != 0.0) or (yoff / res != 0.0):
            self._is_discrete_polygon_valid = False
            self._is_discrete_inflated_polygon_valid = False
        self._is_discrete_cell_set_valid = False
        self._is_discrete_inflated_cell_set_valid = False
        return self

    def _inflate_polygon(self, inflation_radius, res):
        self.inflation_radius = inflation_radius

        if inflation_radius == 0.:
            self.inflated_polygon = self.polygon
        else:
            self.inflated_polygon = self.polygon.buffer(inflation_radius)
        self._is_inflated_polygon_valid = True

    def _discretize_to_grid(self, inflation_radius, res, cost_lethal, cost_inscribed):
        if inflation_radius != self.inflation_radius:
            self._is_inflated_polygon_valid = False
            self._is_discrete_inflated_polygon_valid = False
            self._is_discrete_inflated_cell_set_valid = False

        inflated_polygon = self.get_inflated_polygon(inflation_radius, res)
        min_x, min_y, max_x, max_y = inflated_polygon.bounds

        width, height = max_x - min_x, max_y - min_y

        d_width, d_height = int(math.ceil(width / res)), int(math.ceil(height / res))

        discrete_polygon_grid = np.zeros((d_width, d_height))
        discrete_inflated_polygon_grid = np.zeros((d_width, d_height))

        for i in range(d_width):
            for j in range(d_height):
                cell = Polygon([(min_x + float(i) * res, min_y + float(j) * res),
                                (min_x + float(i+1) * res, min_y + float(j) * res),
                                (min_x + float(i+1) * res, min_y + float(j+1) * res),
                                (min_x + float(i) * res, min_y + float(j+1) * res)])
                if cell.intersects(self.polygon):
                    discrete_polygon_grid[i][j] = cost_lethal
                    discrete_inflated_polygon_grid[i][j] = cost_lethal
                elif cell.intersects(inflated_polygon):
                    discrete_inflated_polygon_grid[i][j] = cost_inscribed

        self.discrete_polygon = discrete_polygon_grid
        self.discrete_inflated_polygon = discrete_inflated_polygon_grid
        self._is_discrete_polygon_valid = True
        self._is_discrete_inflated_polygon_valid = True

    def _discretize_to_cell_sets(self, inflation_radius, res, grid_pose, grid_d_width, grid_d_height):
        if inflation_radius != self.inflation_radius:
            self._is_inflated_polygon_valid = False
            self._is_discrete_inflated_polygon_valid = False
            self._is_discrete_inflated_cell_set_valid = False

        inflated_polygon = self.get_inflated_polygon(inflation_radius, res)
        extra_inflated_polygon = self.polygon.buffer(inflation_radius * 2.)

        min_x, min_y, max_x, max_y = extra_inflated_polygon.bounds

        width, height = max_x - min_x, max_y - min_y

        d_width, d_height = int(round(width / res)), int(round(height / res))

        d_min_x, d_min_y = utils.real_to_grid(min_x, min_y, res, grid_pose)

        poly_coordinates_in_subgrid = [
            ((x - min_x) / res, (y - min_y) / res) for x, y in self.polygon.exterior.coords]

        img = Image.new('L', (d_width, d_height), 0)
        ImageDraw.Draw(img).polygon(poly_coordinates_in_subgrid, outline=1, fill=1)
        mask = np.flipud(np.rot90(np.array(img)))
        # np.fliplr(np.rot90(world_grid.get_grid(), 3)).flatten().astype(np.int8).tolist()
        x_coords, y_coords = np.where(mask == 1)
        x_coords += d_min_x
        y_coords += d_min_y
        unchecked_cells = zip(x_coords, y_coords)
        self.discrete_cells_set = {
            cell for cell in unchecked_cells if utils.is_in_matrix(cell, grid_d_width, grid_d_height)}

        poly_coordinates_in_subgrid = [
            ((x - min_x) / res, (y - min_y) / res) for x, y in inflated_polygon.exterior.coords]

        img = Image.new('L', (d_width, d_height), 0)
        ImageDraw.Draw(img).polygon(poly_coordinates_in_subgrid, outline=1, fill=1)
        mask = np.flipud(np.rot90(np.array(img)))
        x_coords, y_coords = np.where(mask == 1)
        x_coords += d_min_x
        y_coords += d_min_y
        unchecked_cells = zip(x_coords, y_coords)
        self.discrete_inflated_cells_set = {
            cell for cell in unchecked_cells if utils.is_in_matrix(cell, grid_d_width, grid_d_height)}

        self._is_discrete_cell_set_valid = True
        self._is_discrete_inflated_cell_set_valid = True

    def intersects(self, entities):
        for entity in entities:
            if self.polygon.intersects(entity.polygon):
                return True
        return False

    def light_copy(self):
        return Entity(name=self.name, polygon=copy.deepcopy(self.polygon), pose=self.pose,
                      full_geometry_acquired=self.full_geometry_acquired, uid=self.uid)

    def to_json(self):
        return {
            "name": self.name,
            "type": self.get_type(),
            "geometry": {
                "from": "file",
                "id": self.name
            }
        }