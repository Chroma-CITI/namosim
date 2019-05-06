from entity import Entity
import utils
import numpy as np

import math
from math import floor, ceil
from shapely.geometry import Point, box


class Obstacle(Entity):

    def __init__(self, name, polygon, dd, full_geometry_acquired, type_in, movability, pushes_only=False, uid=0):
        Entity.__init__(self, name, polygon, dd, full_geometry_acquired, uid)
        self.type = type_in
        self.movability = movability

        self.pushes_only = pushes_only

        self.actions = self._compute_possible_actions(dd.inflation_radius, dd.res)
        self._is_actions_valid = False

        self.q_l = []
        self._is_q_l_valid = False

    def set_polygon(self, polygon, dd):
        self.polygon = polygon
        self.pose = [list(self.polygon.centroid.coords)[0][0],
                     list(self.polygon.centroid.coords)[0][1],
                     self.pose[2]]
        self._make_inflated_polygon(dd)
        self._discretize(dd)
        if self.movability != "unmovable":
            self._is_actions_valid = False

    def get_actions(self, dd):
        if not self._is_actions_valid:
            self.actions = self._compute_possible_actions(dd.inflation_radius, dd.res)
            self._is_actions_valid = True
        return self.actions

    def get_q_l(self, world, rp):
        if not self._is_q_l_valid:
            self.q_l = self._compute_q_l(world, rp)
            self._is_q_l_valid = True
        return self.q_l

    def translate(self, xoff, yoff, dd):
        Entity.translate(self, xoff, yoff, dd)
        if self.movability != "unmovable":
            self._is_actions_valid = False

    def rotate(self, angle):
        Entity.rotate(self, angle)
        if self.movability != "unmovable":
            self._is_actions_valid = False

    @staticmethod
    def _isclose(a, b, abs_tol=1e-06):
        return abs(a - b) <= abs_tol

    def _compute_possible_actions(self, dist_from_border, manip_unit_length):
        actions = dict()

        polygon = self.polygon
        # polygon = box(self.polygon.bounds[0], self.polygon.bounds[1], self.polygon.bounds[2], self.polygon.bounds[3])

        # ALTERNATIVE METHOD BY CHANGING CARTESIAN REFERENTIAL
        poly_center = polygon.centroid.coords[0]
        for i in range(len(polygon.exterior.coords) - 1):
            d = dist_from_border
            x_a, y_a = polygon.exterior.coords[i]
            x_b, y_b = polygon.exterior.coords[i + 1]
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
            manip_pose = (manip_point[0], manip_point[1], utils.yaw_from_direction(unit_translation))

            actions[tuple(unit_translation)] = manip_pose
            if not self.pushes_only:
                actions[tuple(-1.0 * unit_translation)] = manip_pose
        return actions

    def _compute_q_l(self, world, rp):
        robot = world.entities[world.robot_uid]
        fov_min_r, fov_max_r, fov_angle = robot.s_fov_min_radius, robot.s_fov_max_radius, robot.s_fov_opening_angle

        min_inflated_polygon = self.polygon.buffer(fov_min_r)
        max_inflated_polygon = self.polygon.buffer(fov_max_r)

        rp.publish_min_max_inflated(min_inflated_polygon, max_inflated_polygon)

        map_min_x, map_min_y = world.dd.grid_pose[0], world.dd.grid_pose[1]

        min_x, min_y, max_x, max_y = max_inflated_polygon.bounds

        width, height = max_x - min_x, max_y - min_y
        d_width, d_height = int(ceil(width / world.dd.res)), int(ceil(height / world.dd.res))

        min_cell_x = int(floor((min_x - map_min_x) / world.dd.res))
        min_cell_x = min_cell_x if min_cell_x >= 0 else 0
        min_cell_y = int(floor((min_y - map_min_y) / world.dd.res))
        min_cell_y = min_cell_y if min_cell_y >= 0 else 0
        max_cell_x = min_cell_x + d_width
        max_cell_x = max_cell_x if max_cell_x <= world.get_grid().shape[0] else world.get_grid().shape[0]
        max_cell_y = min_cell_y + d_height
        max_cell_y = max_cell_y if max_cell_y <= world.get_grid().shape[1] else world.get_grid().shape[1]

        q_look = dict()

        for i in range(min_cell_x, max_cell_x):
            for j in range(min_cell_y, max_cell_y):
                # cell_poly = Polygon([(min_x + float(i) * world.dd.res, min_y + float(j) * world.dd.res),
                #                      (min_x + float(i+1) * world.dd.res, min_y + float(j) * world.dd.res),
                #                      (min_x + float(i+1) * world.dd.res, min_y + float(j+1) * world.dd.res),
                #                      (min_x + float(i) * world.dd.res, min_y + float(j+1) * world.dd.res)])
                # if cell_poly.intersects(max_inflated_polygon) and not cell_poly.intersects(min_inflated_polygon):

                observation_position = (map_min_x + float(i) * world.dd.res + 0.5 * world.dd.res,
                                        map_min_y + float(j) * world.dd.res + 0.5 * world.dd.res)
                obs_pos_point = Point(observation_position[0], observation_position[1])

                # Check if point is in a reasonable approximation of the configuration space to allow observation
                if (obs_pos_point.within(max_inflated_polygon)
                        and not obs_pos_point.within(min_inflated_polygon)):

                    # Iterate over world's entities' inflated polygons and check if Point(obs_pose[0], obs_pose[1])
                    # does not intersect with polygon !
                    # point_not_in_any_inflated_obstacle = True
                    # for entity in world.entities.values():
                    #     if entity.uid != self.uid and entity.uid != world.robot_uid:
                    #         if obs_pos_point.within(entity.get_inflated_polygon(world.dd)):
                    #             point_not_in_any_inflated_obstacle = False
                    #             break
                    # if point_not_in_any_inflated_obstacle:
                    if world.get_grid()[i][j] < world.dd.cost_possibly_nonfree:

                        # Check if the obstacle can be seen from this point, and get best angle to view it
                        best_angle = self._best_q_angle(observation_position, fov_min_r, fov_max_r, fov_angle)
                        if best_angle is not None:
                            q_look[(i, j)] =(observation_position[0], observation_position[1], best_angle)

        rp.publish_q_l_poses(q_look.values())
        rp.publish_q_l_cells(q_look.keys(), world.dd)
        return q_look

    def _best_q_angle(self, point_c, fov_min_r, fov_max_r, fov_angle):
        # point_c is the center point of the observer

        nearest_distance = float("inf")

        furthest_distance = -1.0  # Just to be sure that the comparisons will always work

        largest_angle_point_pair = [(), ()]
        largest_angle = -1.0

        for point_a in self.polygon.exterior.coords:
            vector_c_a = [point_a[0] - point_c[0], point_a[1] - point_c[1]]
            norm_c_a = np.linalg.norm(vector_c_a)
            for point_b in self.polygon.exterior.coords:
                if point_a != point_b:
                    vector_c_b = [point_b[0] - point_c[0], point_b[1] - point_c[1]]
                    norm_c_b = np.linalg.norm(vector_c_b)
                    scalar_product_c_a_c_b = vector_c_a[0] * vector_c_b[0] + vector_c_a[1] * vector_c_b[1]
                    cosine_a_c_b = scalar_product_c_a_c_b / (norm_c_a * norm_c_b)
                    angle_a_c_b = 180.0 if cosine_a_c_b <= -1.0 else (
                        0.0 if cosine_a_c_b >= 1.0 else math.degrees(math.acos(cosine_a_c_b)))  # In degrees !
                    if angle_a_c_b > largest_angle:
                        largest_angle = angle_a_c_b
                        largest_angle_point_pair = [(point_a, norm_c_a), (point_b, norm_c_b)]
            if norm_c_a > furthest_distance:
                furthest_distance = norm_c_a
            if norm_c_a < nearest_distance:
                nearest_distance = norm_c_a

        if (largest_angle <= fov_angle
                and fov_min_r <= nearest_distance <= fov_max_r
                and fov_min_r <= furthest_distance <= fov_max_r
                and fov_min_r <= largest_angle_point_pair[0][1] <= fov_max_r
                and fov_min_r <= largest_angle_point_pair[1][1] <= fov_max_r):
            point_a, point_b = largest_angle_point_pair[0][0], largest_angle_point_pair[1][0]
            middle_point = [(point_a[0] + point_b[0]) * 0.5, (point_a[1] + point_b[1]) * 0.5]
            direction = [middle_point[0] - point_c[0], middle_point[1] - point_c[1]]
            return utils.yaw_from_direction(direction)
        else:
            return None
