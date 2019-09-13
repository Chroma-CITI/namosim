from thing import Thing
import shapely.affinity as affinity
from shapely.geometry import Point, LineString, Polygon
import numpy as np
from math import sqrt


class Robot(Thing):

    def __init__(self, name, full_geometry_acquired, polygon, pose,
                 g_fov_max_radius, g_fov_min_radius, g_fov_opening_angle,
                 s_fov_max_radius, s_fov_min_radius, s_fov_opening_angle,
                 push_only_list, force_pushes_only, movable_whitelist):
        polygon = polygon
        Thing.__init__(self, name, polygon, pose, full_geometry_acquired)
        self.g_fov_max_radius = g_fov_max_radius
        self.g_fov_min_radius = g_fov_min_radius
        self.g_fov_opening_angle = g_fov_opening_angle
        self.s_fov_max_radius = s_fov_max_radius
        self.s_fov_min_radius = s_fov_min_radius
        self.s_fov_opening_angle = s_fov_opening_angle

        self.g_fov_polygon = self._create_fov(self.g_fov_max_radius, self.g_fov_min_radius, g_fov_opening_angle)
        self.s_fov_polygon = self._create_fov(self.s_fov_max_radius, self.s_fov_min_radius, s_fov_opening_angle)

        self.push_only_list = push_only_list
        self.force_pushes_only = force_pushes_only
        self.movable_whitelist = movable_whitelist

        self.min_inflation_radius = self.compute_inflation_radius()

    def rotate(self, angle):
        Thing.rotate(self, angle)
        self.g_fov_polygon = affinity.rotate(self.g_fov_polygon, angle, (self.pose[0], self.pose[1]))
        self.s_fov_polygon = affinity.rotate(self.s_fov_polygon, angle, (self.pose[0], self.pose[1]))

    def translate(self, xoff, yoff, dd):
        Thing.translate(self, xoff, yoff, dd)
        self.g_fov_polygon = affinity.translate(self.g_fov_polygon, xoff, yoff)
        self.s_fov_polygon = affinity.translate(self.s_fov_polygon, xoff, yoff)

    def _create_fov(self, fov_max_radius, fov_min_radius, fov_opening_angle):
        fov_outer_arc = self._create_shapely_arc(self.pose, fov_max_radius, fov_opening_angle)

        fov_inner_arc = self._create_shapely_arc(self.pose, fov_min_radius, fov_opening_angle)

        coords_outer = list(fov_outer_arc.coords)
        coords_inner = list(fov_inner_arc.coords)
        points = coords_inner + list(reversed(coords_outer))

        return Polygon(points)

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

    def compute_inflation_radius(self):
        robot_rect_envelope_pts = list(self.polygon.minimum_rotated_rectangle.exterior.coords)
        robot_polygon_radius = 0.
        for i in range(len(robot_rect_envelope_pts) - 1):
            point_a, point_b = robot_rect_envelope_pts[i], robot_rect_envelope_pts[i + 1]
            side_length = sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)
            if side_length > robot_polygon_radius:
                robot_polygon_radius = side_length
        return robot_polygon_radius
