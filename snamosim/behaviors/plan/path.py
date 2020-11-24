import numpy as np
from snamosim.utils import utils
from shapely.geometry import LineString, Point, MultiPolygon
from shapely import affinity
from shapely.ops import cascaded_union


class Path:
    def __init__(self, path, weight=1.0, is_transfer=False, is_observation=False,
                 o_uid=None, translation=None, phys_cost=None, social_cost=0.0, collision_geometry=None):
        self.path = path
        self.weight = weight
        self.is_transfer = is_transfer
        self.is_observation = is_observation
        self.obstacle_uid = o_uid
        self.collision_geometry = collision_geometry
        self.translation = translation
        self.phys_cost = phys_cost if phys_cost is not None else self.__sum_of_euclidean_distances() * weight
        self.social_cost = social_cost
        self.total_cost = self.phys_cost + self.social_cost

    @classmethod
    def line_path(cls, start_pose, goal_pose, weigth, unit_translation,
                  is_transfer=False, is_observation=False, o_uid=None, social_cost=0.0, social_cost_weight=0.0):
        path = [start_pose]
        unit_dist = np.linalg.norm(unit_translation)
        dist_to_go = np.linalg.norm([goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1]])
        phys_cost = dist_to_go * weigth
        yaw = utils.yaw_from_direction(unit_translation)
        prev_pose = start_pose
        dist_to_go = dist_to_go - unit_dist
        while dist_to_go > 0.0:
            intermediate_pose = [prev_pose[0] + unit_translation[0],
                                 prev_pose[1] + unit_translation[1],
                                 yaw]
            path.append(intermediate_pose)
            dist_to_go = dist_to_go - unit_dist
            prev_pose = intermediate_pose
        path.append(goal_pose)

        return cls(path, weigth, is_transfer, False, o_uid, unit_translation, phys_cost, social_cost)

    def has_infinite_cost(self):
        return True if self.total_cost == float("inf") else False

    def is_not_empty(self):
        return bool(self.path)

    def is_valid(self, world, robot_uid, blocked_obstacles):
        # HACK
        return True


        if self.has_infinite_cost():
            return False

        # TODO: Should be the only method in the future to check for validity
        # TODO: Change it so that as the path is executed, the list of polygons to check shortens
        if self.collision_geometry is not None:
            # First, check if the obstacle is still movable,
            robot = world.entities[robot_uid]
            obstacle = world.entities[self.obstacle_uid]
            if robot.deduce_movability(obstacle.type) == "unmovable" or obstacle.uid in blocked_obstacles:
                return False

            # Then check for collisions
            other_entities = {entity_uid: entity for entity_uid, entity in world.entities.items()
                              if entity_uid != robot_uid and entity_uid != self.obstacle_uid}
            for uid, entity in other_entities.items():
                if any(entity.polygon.intersects(polygon) for polygon in self.collision_geometry):
                    # from snamosim.display.ros_publisher import RosPublisher
                    # _rp = RosPublisher()
                    # _rp.publish_debug_polygons(self.collision_geometry, ns=ns)
                    return False

            return True
        # ENDTODO

        # Compute bounding polygon for path.
        positions_array = []
        bounding_polygon = None
        for pose in self.path:
            positions_array.append([pose[0], pose[1]])
        if len(positions_array) == 1:
            bounding_polygon = Point(positions_array[0]).buffer(world.dd.inflation_radius)
        else:
            bounding_polygon = LineString(positions_array).buffer(world.dd.inflation_radius)

        # If transfer path:
        if self.is_transfer:
            # First, check if the obstacle is still movable,
            robot = world.entities[robot_uid]
            obstacle = world.entities[self.obstacle_uid]
            if robot.deduce_movability(obstacle.type) == "unmovable" or obstacle.uid in blocked_obstacles:
                return False

            # Third, add obstacle swept area to bounding polygon, so that this collision is checked too.
            # Compute obstacle swept area, works under the assumption of a convex obstacle and of a translation action.
            polygon_at_start = world.entities[self.obstacle_uid].polygon
            if len(self.path) >= 2:
                translation = [self.path[len(self.path)-1][0] - self.path[0][0],
                               self.path[len(self.path)-1][1] - self.path[0][1]]
                polygon_at_end = affinity.translate(polygon_at_start,
                                                    xoff=translation[0],
                                                    yoff=translation[1])
            else:
                polygon_at_end = polygon_at_start
            obstacle_swept_area = cascaded_union([polygon_at_start, polygon_at_end]).convex_hull
            bounding_polygon = cascaded_union([bounding_polygon, obstacle_swept_area])

        # If path is in collision with entities other than the robot or manipulated obstacle
        for entity_uid, entity in world.entities.items():
            if entity_uid != robot_uid and (
                    True if self.obstacle_uid is None else entity_uid != self.obstacle_uid):
                if bounding_polygon.intersects(entity.polygon):
                    return False

        # If every check is in order:
        return True

    def pop_next_step(self):
        # If there are steps left to execute in self.path, pop and return the first
        if self.path:
            return self.path.pop(0)
        else:
            # Return None otherwise
            return None

    def __sum_of_euclidean_distances(self):
        if len(self.path) == 0:
            return float("inf")
        elif len(self.path) == 1:
            return 0.0

        total = 0.0
        prev_pose = self.path[0]
        for cur_pose in self.path[1:len(self.path)]:
            total = total + np.linalg.norm([cur_pose[0] - prev_pose[0], cur_pose[1] - prev_pose[1]])
            prev_pose = cur_pose

        return total

        # TODO: To add invalidation of c_0, if therre no remaining pose to execute in the c_0 path component, then the
        #  path is invalid. Since there may be valid observation poses in c_1 after compute_c0_c1, get-last-look-q
        #  and split-at-pose should be reexcuted after ? Careful, because of this, c_1 can end up being empty ! In which
        #  case only one Path should remain with attributes is_transfer == False and is_observation == True.
