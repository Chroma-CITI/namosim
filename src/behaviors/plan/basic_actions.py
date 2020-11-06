from shapely import affinity
from shapely.geometry import Point, LineString

from src.utils import utils, collision


class GoalResult:
    def __init__(self, goal):
        self.goal = goal


class GoalsFinished:
    def __init__(self):
        pass


class GoalSuccess(GoalResult):
    def __init__(self, goal):
        GoalResult.__init__(self, goal)

    def __str__(self):
        return "success"


class GoalFailed(GoalResult):
    def __init__(self, goal):
        GoalResult.__init__(self, goal)

    def __str__(self):
        return "failure"


class GoToPose:
    def __init__(self, pose):
        self.pose = pose


class Wait:
    def __init__(self):
        pass


class Rotation:

    def __init__(self, angle):
        self.angle = angle

    def apply(self, polygon, pose):
        return affinity.rotate(geom=polygon, angle=self.angle, origin=(pose[0], pose[1]), use_radians=False)

    def predict_pose(self, pose, center):
        new_point = affinity.rotate(
            geom=Point((pose[0], pose[1])), angle=self.angle, origin=center, use_radians=False
        ).coords[0]
        orientation = (pose[2] + self.angle) % 360.
        orientation = orientation if orientation >= 0. else orientation + 360.
        return (
            new_point[0], new_point[1], orientation

        )


class Translation:

    def __init__(self, translation_vector):
        self.translation_vector = translation_vector
        self.translation_length = utils.euclidean_distance((0., 0.), translation_vector)
        self.translation_linestring = LineString([(0., 0.), self.translation_vector])

    def apply(self, polygon, pose):
        rotated_linestring = affinity.rotate(self.translation_linestring, pose[2], origin=(0., 0.))
        translation_vector = rotated_linestring.coords[1]
        return affinity.translate(geom=polygon, xoff=translation_vector[0], yoff=translation_vector[1], zoff=0.)

    def predict_pose(self, pose):
        rotated_linestring = affinity.rotate(self.translation_linestring, pose[2], origin=(0., 0.))
        translation_vector = rotated_linestring.coords[1]
        new_point = affinity.translate(
            geom=Point((pose[0], pose[1])), xoff=translation_vector[0], yoff=translation_vector[1], zoff=0.
        ).coords[0]
        return new_point[0], new_point[1], pose[2]


class Grab(Translation):
    def __init__(self, translation_vector, entity_uid):
        Translation.__init__(self, translation_vector)
        self.entity_uid = entity_uid


class Release(Translation):
    def __init__(self, translation_vector, entity_uid):
        Translation.__init__(self, translation_vector)
        self.entity_uid = entity_uid


def convert_action(action, robot_pose):
    if isinstance(action, Translation):
        translation_linestring = LineString([(0., 0.), action.translation_vector])
        rotated_linestring = affinity.rotate(translation_linestring, robot_pose[2], origin=(0., 0.))
        translation_vector = rotated_linestring.coords[1]
        return collision.Translation(translation_vector)
    elif isinstance(action, Rotation):
        return collision.Rotation(action.angle, (robot_pose[0], robot_pose[1]))
