from entity import Entity
import conversion
import numpy as np


class Obstacle(Entity):

    def __init__(self, name, polygon, dd, type_in, movability, pushes_only=False, uid=0):
        Entity.__init__(self, name, polygon, dd, uid)
        self.type = type_in
        self.movability = movability

        self.pushes_only = pushes_only
        self.actions = self._compute_possible_actions(dd.inflation_radius, dd.res)

    def set_polygon(self, polygon, dd):
        self.polygon = polygon
        self._make_inflated_polygon(dd)
        self.discrete_polygon = self._discretize(dd)
        self.actions = self._compute_possible_actions(dd.inflation_radius, dd.res)

    def translate(self, xoff, yoff, dd):
        Entity.translate(self, xoff, yoff, dd)
        self.actions = self._compute_possible_actions(dd.inflation_radius, dd.res)

    def rotate(self, angle, dd):
        Entity.rotate(self, angle, dd)
        self.actions = self._compute_possible_actions(dd.inflation_radius, dd.res)

    @staticmethod
    def _isclose(a, b, abs_tol=1e-06):
        return abs(a - b) <= abs_tol

    def _compute_possible_actions(self, dist_from_border, manip_unit_length):
        actions = dict()

        # ALTERNATIVE METHOD BY CHANGING CARTESIAN REFERENTIAL
        poly_center = self.polygon.centroid.coords[0]
        for i in range(len(self.polygon.exterior.coords) - 1):
            d = dist_from_border
            x_a, y_a = self.polygon.exterior.coords[i]
            x_b, y_b = self.polygon.exterior.coords[i + 1]
            x_m, y_m = ((x_a + x_b) / 2.0, (y_a + y_b) / 2.0)
            norm_a_b = np.linalg.norm([x_b - x_a, y_b - y_a])
            points = [(x_m + d * (y_b - y_a) / norm_a_b, y_m + d * (x_b - x_a) / norm_a_b),
                      (x_m + d * (y_b - y_a) / norm_a_b, y_m - d * (x_b - x_a) / norm_a_b),
                      (x_m - d * (y_b - y_a) / norm_a_b, y_m + d * (x_b - x_a) / norm_a_b),
                      (x_m - d * (y_b - y_a) / norm_a_b, y_m - d * (x_b - x_a) / norm_a_b)]
            manip_point = (0.0, 1.0)
            max_dist = 0.0
            for x_r, y_r in points:
                scalar_product = (x_b - x_a) * (x_r - x_m) + (y_b - y_a) * (y_r - y_m)
                if Obstacle._isclose(scalar_product, 0.0):
                    norm_r_poly_center = np.linalg.norm([poly_center[0] - x_r, poly_center[1] - y_r])
                    if norm_r_poly_center > max_dist:
                        manip_point = (x_r, y_r)
                        max_dist = norm_r_poly_center

            unit_translation = (x_m - manip_point[0], y_m - manip_point[1])
            unit_translation = (unit_translation / np.linalg.norm(unit_translation)) * manip_unit_length
            manip_pose = (manip_point[0], manip_point[1], conversion.yaw_from_direction(unit_translation))

            actions[tuple(unit_translation)] = manip_pose
            if not self.pushes_only:
                actions[tuple(-1.0 * unit_translation)] = manip_pose
        return actions
