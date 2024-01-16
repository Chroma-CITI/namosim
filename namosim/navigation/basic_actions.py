import typing as t
from abc import ABC

import numpy as np
from shapely import Polygon, affinity
from shapely.geometry import LineString, Point

from namosim.data_models import UID, PoseModel


class AbsoluteAction(ABC):
    """`Absolute` actions are actions that can be applied without knowing the robot's current pose."""

    def apply(self, polygon: Polygon) -> Polygon:
        raise NotImplementedError()

    def to_absolute(self, pose: PoseModel):
        return self

    def predict_pose(self, pose: PoseModel) -> PoseModel:
        raise NotImplementedError()


class RelativeAction(ABC):
    """`Relative` actions are always applied relative to robot's current pose."""

    def to_absolute(self, pose: PoseModel) -> AbsoluteAction:
        raise NotImplementedError()


Action = t.Union[AbsoluteAction, RelativeAction]


class GoalResult(RelativeAction):
    def __init__(self, goal: PoseModel):
        self.goal = goal


class GoalsFinished(RelativeAction):
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


class Wait(RelativeAction):
    def __init__(self):
        pass


class AbsoluteRotation(AbsoluteAction):
    """This action represents an rotation about a given point, regardless of the robots current pose."""

    def __init__(self, angle: float, center: t.Tuple[float, float]):
        self.angle = angle
        self.center = center

    def __str__(self):
        return f"AbsoluteRotation(angle={self.angle}, center={self.center})"

    def apply(self, polygon: Polygon) -> Polygon:
        return t.cast(
            Polygon,
            affinity.rotate(
                geom=polygon,
                angle=self.angle,
                origin=self.center,  # type: ignore
                use_radians=False,
            ),
        )


class Rotation(RelativeAction):
    """This action represents a rotation relative to the robots current pose."""

    def __init__(self, angle: float):
        self.angle = angle

    def __str__(self):
        return f"Rotation(angle={self.angle})"

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

    def to_absolute(self, pose: PoseModel) -> AbsoluteRotation:
        return AbsoluteRotation(angle=self.angle, center=(pose[0], pose[1]))


class AbsoluteTranslation(AbsoluteAction):
    """This action represents an arbitrary translation regardless of the robot's current orientation. This applies mainly to holonomic robots."""

    def __init__(self, v: t.Tuple[float, float]):
        self.v = v
        self.length = np.linalg.norm(v)

    def apply(self, polygon: Polygon) -> Polygon:
        return affinity.translate(
            geom=polygon,
            xoff=self.v[0],
            yoff=self.v[1],
            zoff=0.0,
        )

    def predict_pose(self, pose: PoseModel) -> PoseModel:
        new_point = affinity.translate(
            geom=Point((pose[0], pose[1])),
            xoff=self.v[0],
            yoff=self.v[1],
            zoff=0.0,
        ).coords[0]
        return new_point[0], new_point[1], pose[2]


class Advance(RelativeAction):
    """This action represents a translation along the robots current directional axis. It may be positive or negative."""

    def __init__(self, distance: float):
        self.distance = distance

    def __str__(self):
        return f"Advance(distance={self.distance})"

    def compute_translation_vector(self, angle: float) -> t.Tuple[float, float]:
        # TODO Replace by call to utils.direction_from_yaw(angle) multiplying self.translation_vector ?
        rotated_linestring = affinity.rotate(
            LineString([(0.0, 0.0), (self.distance, 0.0)]),
            angle,
            origin=(0.0, 0.0),  # type: ignore
        )
        translation_vector = rotated_linestring.coords[1]
        return translation_vector  # type: ignore

    def apply(self, polygon: Polygon, pose: PoseModel) -> Polygon:
        v = self.compute_translation_vector(pose[2])
        return affinity.translate(
            geom=polygon,
            xoff=v[0],
            yoff=v[1],
            zoff=0.0,
        )

    def predict_pose(self, pose: PoseModel, direction_angle: float) -> PoseModel:
        v = self.compute_translation_vector(direction_angle)
        new_point = affinity.translate(
            geom=Point((pose[0], pose[1])),
            xoff=v[0],
            yoff=v[1],
            zoff=0.0,
        ).coords[0]
        return new_point[0], new_point[1], pose[2]

    def to_absolute(self, pose: PoseModel) -> AbsoluteTranslation:
        v = self.compute_translation_vector(pose[2])
        return AbsoluteTranslation(v)


class Grab(Advance):
    def __init__(self, distance: float, entity_uid: UID):
        Advance.__init__(self, distance)
        self.entity_uid = entity_uid

    def __str__(self):
        return f"Grab(distance={self.distance})"


class Release(Advance):
    def __init__(self, distance: float, entity_uid: UID):
        Advance.__init__(self, distance)
        self.entity_uid = entity_uid

    def __str__(self):
        return f"Release(distance={self.distance})"
        return f"Release(distance={self.distance})"
