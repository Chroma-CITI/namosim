from src.worldreps.entity_based.entity import Entity
from src.utils import utils
import numpy as np
import copy

import math
from math import floor, ceil
from shapely.geometry import Point


class Obstacle(Entity):

    def __init__(self, name, polygon, pose, full_geometry_acquired, type_in, uid=0):
        Entity.__init__(self, name, polygon, pose, full_geometry_acquired, uid=uid)
        self.type = type_in

        self.actions = dict()
        self._is_actions_valid = False

        self.q_l = []
        self._is_q_l_valid = False

    def set_polygon(self, polygon):
        Entity.set_polygon(self, polygon)
        self._is_actions_valid = False
        return self

    def get_actions(self, inflation_radius, res, pushes_only):
        if not self._is_actions_valid:
            self.actions = self._compute_possible_actions(inflation_radius, res, pushes_only)
            self._is_actions_valid = True
        return self.actions

    def get_q_l(self, world, ns):
        if not self._is_q_l_valid:
            self.q_l = self._compute_q_l(world, ns=ns)
            self._is_q_l_valid = True
        return self.q_l

    def translate(self, xoff, yoff, res=0.05, other_entities=None, ignore_collisions=False):
        Entity.translate(self, xoff, yoff, res, other_entities, ignore_collisions)
        self._is_actions_valid = False
        return self

    def rotate(self, angle, rot_center='centroid', other_entities=None, angular_res=5., ignore_collisions=False):
        Entity.rotate(self, angle, rot_center, other_entities, angular_res, ignore_collisions)
        self._is_actions_valid = False
        return self

    @staticmethod
    def _isclose(a, b, abs_tol=1e-06):
        return abs(a - b) <= abs_tol

    def get_middle_of_sides_manipulation_poses(self, dist_from_sides):
        """
        Computes and returns the manipulation poses that are at a distance dist_from_border from the sides,
        and facing their middle.
        :param dist_from_sides: distance from the obstacle's sides at which the manipulation poses are computed [m]
        :type dist_from_sides: float
        :return: list of manipulation poses
        :rtype: list(tuple(float, float, float))
        """
        poses = []
        polygon = self.polygon

        # METHOD BY CHANGING CARTESIAN REFERENTIAL
        poly_center = polygon.centroid.coords[0]
        for i in range(len(polygon.exterior.coords) - 1):
            d = dist_from_sides
            x_a, y_a = polygon.exterior.coords[i]  # First side segment point
            x_b, y_b = polygon.exterior.coords[i + 1]  # Second side segment point
            x_m, y_m = ((x_a + x_b) / 2.0, (y_a + y_b) / 2.0)  # Middle of side segment
            norm_a_b = np.linalg.norm([x_b - x_a, y_b - y_a])  # Side segment length
            if norm_a_b != 0.:
                # Compute candidate manip points obtained by cartesian referential change
                points = [(x_m + d * (y_b - y_a) / norm_a_b, y_m + d * (x_b - x_a) / norm_a_b),
                          (x_m + d * (y_b - y_a) / norm_a_b, y_m - d * (x_b - x_a) / norm_a_b),
                          (x_m - d * (y_b - y_a) / norm_a_b, y_m + d * (x_b - x_a) / norm_a_b),
                          (x_m - d * (y_b - y_a) / norm_a_b, y_m - d * (x_b - x_a) / norm_a_b)]
                manip_point = (0., 0.)
                max_dist = 0.0
                # Iterate over candidate manip points to select only the closest one orthogonal to side segment
                for x_r, y_r in points:
                    scalar_product = (x_b - x_a) * (x_r - x_m) + (y_b - y_a) * (y_r - y_m)
                    if Obstacle._isclose(scalar_product, 0.0):
                        norm_r_poly_center = np.linalg.norm([poly_center[0] - x_r, poly_center[1] - y_r])
                        if norm_r_poly_center > max_dist:
                            manip_point = (x_r, y_r)
                            max_dist = norm_r_poly_center

                # Save selected manip point in returned list
                direction = (x_m - manip_point[0], y_m - manip_point[1])
                manip_pose = (manip_point[0], manip_point[1], utils.yaw_from_direction(direction))
                poses.append(manip_pose)

        return poses

    def get_regularly_sampled_manipulation_poses_along_buffered_poly(self, dist_from_border, sample_distance):
        """
        Compute buffered polygon at distance dist_from_border from original polygon sides, then sample manipulation
        poses along this buffered polygon separated by a distance sample_distance each. Could also compute the
        sample_distance from a nb_sampled_poses parameter ?
        :param dist_from_border:
        :param sample_distance:
        :return: list of manipulation poses
        :rtype: list(tuple(float, float, float))
        """
        raise NotImplementedError()
        # Basic idea would be

        # 1 - Compute sample_distance as sum_of_sides_lengths / nb_sampled_poses if not provided

        # 2 - While nb_sampled_poses not reached, iterate over polygon segments

        # 2.a - If distance from last sampled pose is < distance_left_to_go (successively updated from sample_distance)
        # then add new pose on current segment
        # else get to next segment and distance_left_to_go -= distance from last sampled pose to current segment point

    def get_manipulation_poses_along_parallel_side_lines(self, dist_from_border, sample_distance):
        """
        Compute parallel segments at distance dist_from_border from original polygon sides, then sample manipulation
        poses along these segments separated by a distance sample_distance each, and also preventing the footprint of
        the robot to overflow from the sides. Could also compute the sample_distance from a nb_sampled_poses parameter ?
        :param dist_from_border:
        :param sample_distance:
        :return: list of manipulation poses
        :rtype: list(tuple(float, float, float))
        """
        raise NotImplementedError()
        # Get buffered polygon at dist_from_border
        # Only keep sides aligned to existing sides
        # Make them as long as original sides (giving a list of LineStrings)
        # Then sample poses along these LineStrings by iterating over them in a similar way as in the
        # get_regularly_sampled_manipulation_poses_along_buffered_poly method

    def _compute_possible_actions(self, dist_from_border, manip_unit_length, pushes_only):
        """
        Computes and returns unit translation vectors corresponding to the manipulation poses that are at a distance
        dist_from_border from the sides, and facing their middle. If pushes_only, only translation vectors orientated
        towards the obstacle are returned.
        :param dist_from_border: distance from the obstacle's sides at which the manipulation poses are computed [m]
        :type dist_from_border: float
        :param manip_unit_length: norm of the unit translation vectors [m]
        :type manip_unit_length: float
        :param pushes_only: forces translation vectors to be orientated towards the obstacle if True
        :return: dictionnary with translation vectors as keys, corresponding manipulation poses as values
        :rtype: dict(tuple(float, float): tuple(float, float, float))
        """
        manip_poses = self.get_middle_of_sides_manipulation_poses(dist_from_border)
        actions = dict()
        for manip_pose in manip_poses:
            unit_translation = np.array(utils.direction_from_yaw(manip_pose[2])) * manip_unit_length
            actions[tuple(unit_translation)] = manip_pose
            if not pushes_only:
                actions[tuple(-1.0 * unit_translation)] = manip_pose
        return actions

    def _compute_q_l(self, world, ns):
        robot = world.entities[world.robot_uid]
        fov_min_r, fov_max_r, fov_angle = robot.s_fov_min_radius, robot.s_fov_max_radius, robot.s_fov_opening_angle

        min_inflated_polygon = self.polygon.buffer(fov_min_r)
        max_inflated_polygon = self.polygon.buffer(fov_max_r)

        from src.display.ros_publisher import RosPublisher
        RosPublisher().publish_min_max_inflated(min_inflated_polygon, max_inflated_polygon, ns=ns)

        map_min_x, map_min_y = world.dd.grid_pose[0], world.dd.grid_pose[1]

        min_x, min_y, max_x, max_y = max_inflated_polygon.bounds

        width, height = max_x - min_x, max_y - min_y
        d_width, d_height = int(ceil(width / world.dd.res)), int(ceil(height / world.dd.res))

        min_cell_x = int(floor((min_x - map_min_x) / world.dd.res))
        min_cell_x = min_cell_x if min_cell_x >= 0 else 0
        min_cell_y = int(floor((min_y - map_min_y) / world.dd.res))
        min_cell_y = min_cell_y if min_cell_y >= 0 else 0
        max_cell_x = min_cell_x + d_width
        max_cell_x = max_cell_x if max_cell_x <= world.dd.d_width else world.dd.d_width
        max_cell_y = min_cell_y + d_height
        max_cell_y = max_cell_y if max_cell_y <= world.dd.d_height else world.dd.d_height

        q_look = dict()

        grid = world.get_grid()

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
                    if grid[i][j] < world.dd.cost_possibly_nonfree:

                        # Check if the obstacle can be seen from this point, and get best angle to view it
                        best_angle = self._best_q_angle(observation_position, fov_min_r, fov_max_r, fov_angle)
                        if best_angle is not None:
                            q_look[(i, j)] = (observation_position[0], observation_position[1], best_angle)
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

    def light_copy(self, copy_polygon=True):
        return Obstacle(self.name, None if not copy_polygon else copy.deepcopy(self.polygon),
                        self.pose, self.full_geometry_acquired,
                        type_in=self.type, uid=self.uid)

    def get_type(self):
        return self.type