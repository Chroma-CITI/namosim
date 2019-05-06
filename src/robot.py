from entity import Entity
import shapely.affinity as affinity
from shapely.geometry import Point, LineString, Polygon
import numpy as np


class Robot(Entity):

    def __init__(self, name, dd, full_geometry_acquired, radius, initial_pose,
                 g_fov_max_radius, g_fov_min_radius, g_fov_opening_angle,
                 s_fov_max_radius, s_fov_min_radius, s_fov_opening_angle):
        polygon = Point(initial_pose[0], initial_pose[1]).buffer(radius)
        Entity.__init__(self, name, polygon, dd, full_geometry_acquired)
        self.radius = radius
        self.pose = initial_pose
        self.g_fov_max_radius = g_fov_max_radius
        self.g_fov_min_radius = g_fov_min_radius
        self.g_fov_opening_angle = g_fov_opening_angle
        self.s_fov_max_radius = s_fov_max_radius
        self.s_fov_min_radius = s_fov_min_radius
        self.s_fov_opening_angle = s_fov_opening_angle

        self.g_fov_polygon = self._create_fov(self.g_fov_max_radius, self.g_fov_min_radius, g_fov_opening_angle)
        self.s_fov_polygon = self._create_fov(self.s_fov_max_radius, self.s_fov_min_radius, s_fov_opening_angle)

    def rotate(self, angle):
        Entity.rotate(self, angle)
        self.g_fov_polygon = affinity.rotate(self.g_fov_polygon, angle, (self.pose[0], self.pose[1]))
        self.s_fov_polygon = affinity.rotate(self.s_fov_polygon, angle, (self.pose[0], self.pose[1]))

    def translate(self, xoff, yoff, dd):
        Entity.translate(self, xoff, yoff, dd)
        self.g_fov_polygon = affinity.translate(self.g_fov_polygon, xoff, yoff)
        self.s_fov_polygon = affinity.translate(self.s_fov_polygon, xoff, yoff)

    def _create_fov(self, fov_max_radius, fov_min_radius, fov_opening_angle):
        fov_outer_arc = self._create_shapely_arc(self.pose, fov_max_radius, fov_opening_angle)

        fov_inner_arc = self._create_shapely_arc(self.pose, fov_min_radius, fov_opening_angle)

        coords_outer = list(fov_outer_arc.coords)
        coords_inner = list(fov_inner_arc.coords)
        points = coords_inner + list(reversed(coords_outer))

        return Polygon(points)

    @staticmethod
    def _create_shapely_arc(robot_init_pose, radius, opening_angle, numsegments=15):
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
