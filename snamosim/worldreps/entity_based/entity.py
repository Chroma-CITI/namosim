import math
import numpy as np
import copy
import shapely.affinity as affinity
from shapely.geometry import Polygon

from snamosim.utils import utils
from .custom_exceptions import IntersectionError

from PIL import Image, ImageDraw


class Entity:
    last_id = 1

    # Constructor
    def __init__(self, name, polygon, pose, full_geometry_acquired, movability = "unknown", uid=0):
        if uid == 0:
            self.uid = Entity.last_id
            Entity.last_id = Entity.last_id + 1
        else:
            self.uid = uid
        self.name = name
        self.polygon = polygon
        self.pose = tuple(pose)
        self.full_geometry_acquired = full_geometry_acquired
        self.is_being_manipulated = False
        self.movability = movability

    def within(self, other_entity):
        return self.polygon.within(other_entity.polygon)

    def set_polygon(self, polygon):
        self.polygon = polygon
        self.pose = [list(self.polygon.centroid.coords)[0][0],
                     list(self.polygon.centroid.coords)[0][1],
                     self.pose[2]]
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

        return self

    def light_copy(self):
        return Entity(name=self.name, polygon=copy.deepcopy(self.polygon), pose=self.pose,
                      full_geometry_acquired=self.full_geometry_acquired, uid=self.uid)

    def to_json(self):
        return {
            "name": self.name,
            "type": self.type,
            "geometry": {
                "from": "file",
                "id": self.name
            }
        }
