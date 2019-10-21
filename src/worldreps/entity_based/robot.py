from src.worldreps.entity_based.entity import Entity
from src.worldreps.entity_based.sensors.g_fov_sensor import GFOVSensor
from src.worldreps.entity_based.sensors.s_fov_sensor import SFOVSensor

from shapely import affinity
from math import sqrt


class Robot(Entity):

    def __init__(self, name, full_geometry_acquired, polygon, pose,
                 g_fov_max_radius, g_fov_min_radius, g_fov_opening_angle,
                 s_fov_max_radius, s_fov_min_radius, s_fov_opening_angle,
                 push_only_list, force_pushes_only, movable_whitelist):
        polygon = polygon
        Entity.__init__(self, name, polygon, pose, full_geometry_acquired)

        self.g_fov_sensor = GFOVSensor(g_fov_max_radius, g_fov_min_radius, g_fov_opening_angle, self.pose)
        self.s_fov_sensor = SFOVSensor(s_fov_max_radius, s_fov_min_radius, s_fov_opening_angle, self.pose)

        self.push_only_list = push_only_list
        self.force_pushes_only = force_pushes_only
        self.movable_whitelist = movable_whitelist

        self.min_inflation_radius = self.compute_inflation_radius()

    def rotate(self, angle):
        Entity.rotate(self, angle)
        self.g_fov_sensor.rotate(angle, self.pose)
        self.s_fov_sensor.rotate(angle, self.pose)

    def translate(self, xoff, yoff, dd):
        Entity.translate(self, xoff, yoff, dd)
        self.g_fov_sensor.translate(xoff, yoff)
        self.s_fov_sensor.translate(xoff, yoff)

    def update_world_from_sensors(self, reference_world, target_world):
        # Update robot pose in target world
        ref_robot = reference_world.entities[self.uid]
        trans = [ref_robot.pose[0] - self.pose[0], ref_robot.pose[1] - self.pose[1]]
        rot = (ref_robot.pose[2] - self.pose[2]) % 360.
        target_world.translate_entity(self.uid, trans)
        target_world.rotate_entity(self.uid, rot)

        # Update other entities in target world
        self.g_fov_sensor.update_from_fov(reference_world, target_world)
        self.s_fov_sensor.update_from_fov(reference_world, target_world)

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
        robot_rect_envelope_pts = list(self.polygon.minimum_rotated_rectangle.exterior.coords)
        robot_polygon_radius = 0.
        for i in range(len(robot_rect_envelope_pts) - 1):
            point_a, point_b = robot_rect_envelope_pts[i], robot_rect_envelope_pts[i + 1]
            side_length = sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)
            if side_length > robot_polygon_radius:
                robot_polygon_radius = side_length
        return robot_polygon_radius
