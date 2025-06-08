import typing as t
from abc import ABC

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import LineString, Point
from shapely import affinity
from namosim.data_models import Pose2D
from namosim.utils import utils


class BaseAction(ABC):
    """`Absolute` actions are actions that can be applied without knowing the robot's current pose."""

    def apply(self, robot_pose: Pose2D, polygon: Polygon) -> t.Tuple[Pose2D, Polygon]:
        return self.predict_pose(
            robot_pose=robot_pose, pose=robot_pose
        ), self.predict_polygon(robot_pose=robot_pose, polygon=polygon)

    def predict_polygon(self, robot_pose: Pose2D, polygon: Polygon) -> Polygon:
        raise NotImplementedError()

    def predict_pose(self, robot_pose: Pose2D, pose: Pose2D) -> Pose2D:
        raise NotImplementedError()


class GoalsFinished(BaseAction):
    def __init__(self):
        pass


class GoalSuccess(BaseAction):
    def __init__(self, goal: Pose2D):
        self.goal = goal

    def __str__(self):
        return "success"


class GoalFailed(BaseAction):
    def __init__(self, goal: Pose2D, is_timeout: bool = False):
        self.goal = goal
        self.is_timeout = is_timeout

    def __str__(self):
        return "failure"


class Wait(BaseAction):
    def __init__(self):
        pass


class Rotation(BaseAction):
    """This action represents a rotation relative to the robots current pose."""

    def __init__(self, angle: float):
        self.angle = angle

    def __str__(self):
        return f"Rotation(angle={self.angle})"

    def predict_polygon(self, robot_pose: Pose2D, polygon: Polygon) -> Polygon:
        next_polygon = t.cast(
            Polygon,
            affinity.rotate(
                geom=polygon,
                angle=self.angle,
                origin=(robot_pose[0], robot_pose[1]),  # type: ignore
                use_radians=False,
            ),
        )
        return next_polygon

    def predict_pose(self, robot_pose: Pose2D, pose: Pose2D) -> Pose2D:
        next_position = affinity.rotate(
            geom=Point((pose[0], pose[1])),
            angle=self.angle,
            origin=(robot_pose[0], robot_pose[1]),  # type: ignore
            use_radians=False,
        ).coords[0]
        orientation = utils.normalize_angle_degrees(pose[2] + self.angle)
        return Pose2D(next_position[0], next_position[1], orientation)

    def apply_to_point(
        self, center: t.Tuple[float, float], point: t.Tuple[float, float]
    ) -> t.Tuple[float, float]:
        rotated = t.cast(
            Point,
            affinity.rotate(
                geom=Point(point),
                angle=self.angle,
                origin=center,  # type: ignore
                use_radians=False,
            ),
        )
        return (rotated.x, rotated.y)


class Translation(BaseAction):
    """This action represents an arbitrary translation regardless of the robot's current orientation. This applies mainly to holonomic robots."""

    def __init__(self, v: t.Tuple[float, float]):
        self.v = v
        self.length = np.linalg.norm(v)

    def predict_polygon(self, robot_pose: Pose2D, polygon: Polygon) -> Polygon:
        next_polygon = affinity.translate(
            geom=polygon,
            xoff=self.v[0],
            yoff=self.v[1],
            zoff=0.0,
        )
        return next_polygon

    def predict_pose(self, robot_pose: Pose2D, pose: Pose2D) -> Pose2D:
        next_position = affinity.translate(
            geom=Point((pose[0], pose[1])),
            xoff=self.v[0],
            yoff=self.v[1],
            zoff=0.0,
        ).coords[0]
        next_pose = Pose2D(next_position[0], next_position[1], pose[2])
        return next_pose


class Advance(BaseAction):
    """This action represents a translation along the robots current directional axis. It may be positive or negative."""

    def __init__(self, distance: float):
        self.distance = distance

    def __str__(self):
        return f"Advance(distance={self.distance})"

    def to_translation(self, angle: float) -> Translation:
        # TODO Replace by call to utils.direction_from_yaw(angle) multiplying self.translation_vector ?
        rotated_linestring = affinity.rotate(
            LineString([(0.0, 0.0), (self.distance, 0.0)]),
            angle,
            origin=(0.0, 0.0),  # type: ignore
        )
        translation_vector: t.Tuple[float, float] = rotated_linestring.coords[1]  # type: ignore
        return Translation(v=translation_vector)

    def predict_polygon(self, robot_pose: Pose2D, polygon: Polygon) -> Polygon:
        return self.to_translation(angle=robot_pose[2]).predict_polygon(
            robot_pose, polygon
        )

    def predict_pose(self, robot_pose: Pose2D, pose: Pose2D) -> Pose2D:
        return self.to_translation(angle=robot_pose[2]).predict_pose(robot_pose, pose)


class Grab(Advance):
    def __init__(self, entity_uid: str, distance: float = 0.0):
        Advance.__init__(self, distance=distance)
        self.entity_uid = entity_uid

    def __str__(self):
        return f"Grab(entity={self.entity_uid})"


class Release(Advance):
    def __init__(self, entity_uid: str, distance: float):
        Advance.__init__(self, distance=distance)
        self.entity_uid = entity_uid

    def __str__(self):
        return f"Release(entity={self.entity_uid}, distance={self.distance})"


Action = t.Union[
    GoalsFinished,
    GoalSuccess,
    GoalFailed,
    Wait,
    Rotation,
    Translation,
    Advance,
    Grab,
    Release,
]
