from shapely import affinity
from shapely.geometry import Point, LineString

from snamosim.utils import utils


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


class Wait:
    def __init__(self):
        pass


class Rotation:
    def __init__(self, angle):
        self.angle = angle

    def apply(self, polygon, pose):
        return affinity.rotate(
            geom=polygon, angle=self.angle, origin=(pose[0], pose[1]), use_radians=False
        )

    def predict_pose(self, pose, center):
        new_point = affinity.rotate(
            geom=Point((pose[0], pose[1])),
            angle=self.angle,
            origin=center,
            use_radians=False,
        ).coords[0]
        orientation = (pose[2] + self.angle) % 360.0
        orientation = orientation if orientation >= 0.0 else orientation + 360.0
        return (new_point[0], new_point[1], orientation)


class Translation:
    def __init__(self, translation_vector):
        self.translation_vector = translation_vector
        self.translation_length = utils.euclidean_distance(
            (0.0, 0.0), translation_vector
        )
        self.translation_linestring = LineString([(0.0, 0.0), self.translation_vector])

    @classmethod
    def from_absolute_translation_vector(cls, absolute_translation_vector):
        translation_length = utils.euclidean_distance(
            (0.0, 0.0), absolute_translation_vector
        )
        translation_vector = (translation_length, 0.0)
        return cls(translation_vector)

    def compute_translation_vector(self, angle):
        # TODO Replace by call to utils.direction_from_yaw(angle) multiplying self.translation_vector ?
        rotated_linestring = affinity.rotate(
            self.translation_linestring, angle, origin=(0.0, 0.0)
        )
        translation_vector = rotated_linestring.coords[1]
        return translation_vector

    def apply(self, polygon, pose):
        translation_vector = self.compute_translation_vector(pose[2])
        return affinity.translate(
            geom=polygon,
            xoff=translation_vector[0],
            yoff=translation_vector[1],
            zoff=0.0,
        )

    def predict_pose(self, pose, direction_angle):
        rotated_linestring = affinity.rotate(
            self.translation_linestring, direction_angle, origin=(0.0, 0.0)
        )
        translation_vector = rotated_linestring.coords[1]
        new_point = affinity.translate(
            geom=Point((pose[0], pose[1])),
            xoff=translation_vector[0],
            yoff=translation_vector[1],
            zoff=0.0,
        ).coords[0]
        return new_point[0], new_point[1], pose[2]


class AbsoluteTranslation(Translation):
    def __init__(self, translation_vector):
        Translation.__init__(self, translation_vector)

    def compute_translation_vector(self, angle):
        return self.translation_vector

    def apply(self, polygon, pose):
        return affinity.translate(
            geom=polygon,
            xoff=self.translation_vector[0],
            yoff=self.translation_vector[1],
            zoff=0.0,
        )

    def predict_pose(self, pose, direction_angle):
        new_point = affinity.translate(
            geom=Point((pose[0], pose[1])),
            xoff=self.translation_vector[0],
            yoff=self.translation_vector[1],
            zoff=0.0,
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
