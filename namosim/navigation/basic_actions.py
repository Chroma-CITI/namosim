import typing as t
from abc import ABC

from shapely import Polygon, affinity
from shapely.geometry import LineString, Point

from namosim.data_models_v2 import PoseModel
from namosim.utils import utils


class BasicAction(ABC):
    pass

    # def apply(self, polygon: Polygon, pose: PoseModel) -> Polygon:
    #     raise NotImplementedError()


class GoalResult(BasicAction):
    def __init__(self, goal: PoseModel):
        self.goal = goal


class GoalsFinished(BasicAction):
    def __init__(self):
        pass


class GoalSuccess(GoalResult):
    def __init__(self, goal: PoseModel):
        GoalResult.__init__(self, goal)

    def __str__(self):
        return "success"


class GoalFailed(GoalResult):
    def __init__(self, goal: PoseModel):
        GoalResult.__init__(self, goal)

    def __str__(self):
        return "failure"


class Wait(BasicAction):
    def __init__(self):
        pass


class Rotation(BasicAction):
    def __init__(self, angle: float):
        self.angle = angle

    def apply(self, polygon: Polygon, pose: PoseModel) -> Polygon:
        return t.cast(
            Polygon,
            affinity.rotate(
                geom=polygon,
                angle=self.angle,
                origin=(pose[0], pose[1]),  # type: ignore
                use_radians=False,
            ),
        )

    def predict_pose(self, pose: PoseModel, center: t.Tuple[float, float]) -> PoseModel:
        new_point = affinity.rotate(
            geom=Point((pose[0], pose[1])),
            angle=self.angle,
            origin=center,  # type: ignore
            use_radians=False,
        ).coords[0]
        orientation = (pose[2] + self.angle) % 360.0
        orientation = orientation if orientation >= 0.0 else orientation + 360.0
        return (new_point[0], new_point[1], orientation)


class Translation(BasicAction):
    def __init__(self, translation_vector: t.Tuple[float, float]):
        self.translation_vector = translation_vector
        self.translation_length = utils.euclidean_distance(
            (0.0, 0.0), translation_vector
        )
        self.translation_linestring = LineString([(0.0, 0.0), self.translation_vector])

    @classmethod
    def from_absolute_translation_vector(
        cls, absolute_translation_vector: t.Tuple[float, float]
    ):
        translation_length = utils.euclidean_distance(
            (0.0, 0.0), absolute_translation_vector
        )
        translation_vector = (translation_length, 0.0)
        return cls(translation_vector)

    def compute_translation_vector(self, angle: float) -> t.Tuple[float, float]:
        # TODO Replace by call to utils.direction_from_yaw(angle) multiplying self.translation_vector ?
        rotated_linestring = affinity.rotate(
            self.translation_linestring,
            angle,
            origin=(0.0, 0.0),  # type: ignore
        )
        translation_vector = rotated_linestring.coords[1]
        return translation_vector  # type: ignore

    def apply(self, polygon: Polygon, pose: PoseModel) -> Polygon:
        translation_vector = self.compute_translation_vector(pose[2])
        return affinity.translate(
            geom=polygon,
            xoff=translation_vector[0],
            yoff=translation_vector[1],
            zoff=0.0,
        )

    def predict_pose(self, pose: PoseModel, direction_angle: float) -> PoseModel:
        rotated_linestring = affinity.rotate(
            self.translation_linestring,
            direction_angle,
            origin=(0.0, 0.0),  # type: ignore
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
    def __init__(self, translation_vector: t.Tuple[float, float]):
        Translation.__init__(self, translation_vector)

    def compute_translation_vector(self, angle: float):
        return self.translation_vector

    def apply(self, polygon: Polygon, pose: PoseModel) -> Polygon:
        return affinity.translate(
            geom=polygon,
            xoff=self.translation_vector[0],
            yoff=self.translation_vector[1],
            zoff=0.0,
        )

    def predict_pose(self, pose: PoseModel, direction_angle: float) -> PoseModel:
        new_point = affinity.translate(
            geom=Point((pose[0], pose[1])),
            xoff=self.translation_vector[0],
            yoff=self.translation_vector[1],
            zoff=0.0,
        ).coords[0]
        return new_point[0], new_point[1], pose[2]


class Grab(Translation):
    def __init__(self, translation_vector: t.Tuple[float, float], entity_uid: int):
        Translation.__init__(self, translation_vector)
        self.entity_uid = entity_uid


class Release(Translation):
    def __init__(self, translation_vector: t.Tuple[float, float], entity_uid: int):
        Translation.__init__(self, translation_vector)
        self.entity_uid = entity_uid
        self.entity_uid = entity_uid
