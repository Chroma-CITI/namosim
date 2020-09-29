from src.worldreps.entity_based.entity import Entity
from src.utils import utils

from math import sqrt
import numpy as np
import copy
from shapely.geometry import LineString


class Robot(Entity):

    def __init__(self, name, full_geometry_acquired, polygon, pose, sensors,
                 push_only_list, force_pushes_only, movable_whitelist, uid=0):
        polygon = polygon
        Entity.__init__(self, name, polygon, pose, full_geometry_acquired, uid=uid)

        self.sensors = sensors
        for sensor in sensors:
            sensor.parent_uid = self.uid

        self.push_only_list = push_only_list
        self.force_pushes_only = force_pushes_only
        self.movable_whitelist = movable_whitelist
        self.type = 'robot'

        self.min_inflation_radius = self.compute_inflation_radius()
        self.dist_between_robot_front_and_center = self.compute_dist_between_robot_front_and_center()

    def rotate(self, angle, rot_center='centroid', other_entities=None, angular_res=5., ignore_collisions=False):
        Entity.rotate(self, angle, rot_center, other_entities, angular_res, ignore_collisions)
        for sensor in self.sensors:
            pass
            sensor.rotate(angle, rot_center=(self.pose[0], self.pose[1]))
        return self

    def translate(self, xoff, yoff, res=0.05, other_entities=None, ignore_collisions=False):
        Entity.translate(self, xoff, yoff, res, other_entities, ignore_collisions)
        for sensor in self.sensors:
            sensor.translate(xoff, yoff)
        return self

    def update_world_from_sensors(self, reference_world, target_world):
        # Update robot pose in target world
        ref_robot = reference_world.entities[self.uid]
        trans = [ref_robot.pose[0] - self.pose[0], ref_robot.pose[1] - self.pose[1]]
        rot = (ref_robot.pose[2] - self.pose[2]) % 360.
        target_world.translate_entity(self.uid, trans)
        target_world.rotate_entity(self.uid, rot)

        # Update other entities in target world
        for sensor in self.sensors:
            sensor.update_from_fov(reference_world, target_world)

    def deduce_movability(self, obstacle_type):
        if obstacle_type == "unknown":
            return "unknown"
        elif obstacle_type in self.movable_whitelist:
            return "movable"
        else:
            return "unmovable"

    def deduce_push_only(self, obstacle_type):
        if self.force_pushes_only or obstacle_type in self.push_only_list:
            return True
        else:
            return False

    def compute_inflation_radius(self):
        return utils.get_circumscribed_radius(self.polygon)

    def compute_dist_between_robot_front_and_center(self):
        """
        Computes and returns the distance between the robot's center and its front-facing side. For that, we look for
        the intersection between the ray starting from its center with the same direction as its yaw, and finally
        compute the norm between this intersection and the center.
        :return: the distance between the robot's center and its front-facing side.
        :rtype: float
        """
        direction = np.array(utils.direction_from_yaw(self.pose[2]))
        bounds = self.polygon.bounds
        englobing_circle_diameter = sqrt((bounds[2] - bounds[0]) ** 2 + (bounds[3] - bounds[1]) ** 2)
        center = np.array([self.pose[0], self.pose[1]])
        direction_line = LineString([center,
                                     center + direction * englobing_circle_diameter])

        # DEBUG Lines to help you understand why the world hates you
        # import matplotlib.pyplot as plt; plt.plot(*self.polygon.exterior.xy);
        # plt.plot(*direction_line.xy); plt.axis('equal'); plt.show()

        side_intersection = direction_line.intersection(self.polygon).coords[1]
        return sqrt((side_intersection[0] - center[0]) ** 2 + (side_intersection[1] - center[1]) ** 2)

    def light_copy(self):
        return Robot(name=self.name,
                     polygon=copy.deepcopy(self.polygon),
                     pose=self.pose,
                     full_geometry_acquired=self.full_geometry_acquired,
                     sensors=copy.deepcopy(self.sensors),
                     push_only_list=copy.copy(self.push_only_list),
                     force_pushes_only=self.force_pushes_only,
                     movable_whitelist=copy.copy(self.movable_whitelist),
                     uid=self.uid)

    def get_type(self):
        return "robot"

    def to_json(self):
        json_data = Entity.to_json(self)
        json_data["geometry"]["orientation_id"] = self.name + "_dir"
        json_data["movable_whitelist"] = self.movable_whitelist
        json_data["push_only_list"] = self.push_only_list
        json_data["force_pushes_only"] = self.force_pushes_only
        json_data["sensors"] = []
        for sensor in self.sensors:
            json_data["sensors"].append(sensor.to_json())
        return json_data