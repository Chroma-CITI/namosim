import copy
import re
import typing as t
from enum import Enum

from shapely.geometry import Polygon
from typing_extensions import Self

import namosim.utils.utils as utils
from namosim.data_models import PoseModel
from shapely import affinity


class Movability(Enum):
    UNKNOWN = 1
    MOVABLE = 2
    STATIC = 3
    UNMOVABLE = 4


class Style:
    def __init__(
        self,
        fill: str = "",
        fill_opacity: str = "",
        stroke: str = "",
        stroke_width: str = "",
        stroke_opacity: str = "",
        **_,
    ):
        self.fill = fill
        self.fill_opacity: float = float(fill_opacity) if fill_opacity else 1.0
        self.stroke = stroke
        try:
            self.stroke_width = float(
                re.findall(r"[-+]?(?:\d*\.*\d+)", stroke_width)[0]
            )
        except IndexError:
            self.stroke_width = 0.0
        self.stroke_opacity = float(stroke_opacity) if stroke_opacity else 1.0

    # noinspection PyTypeChecker
    @classmethod
    def from_string(cls, style: str):
        d: t.Dict[str, str] = dict(
            [a.strip().replace("-", "_") for a in attribute.split(":", 1)]
            for attribute in style.split(";")
            if attribute
        )
        return cls(**d)

    def to_string(self):
        style_str = ""
        if self.fill:
            style_str += f"fill:{self.fill};"
        if self.fill_opacity:
            style_str += f"fill-opacity:{self.fill_opacity};"
        if self.stroke_width:
            style_str += f"stroke-width:{self.stroke_width};"
        if self.stroke:
            style_str += f"stroke:{self.stroke};"
        if self.stroke_opacity:
            style_str += f"stroke-opacity:{self.stroke_opacity};"
        return style_str


class Entity:
    # Constructor
    def __init__(
        self,
        type_: str,
        uid: str,
        polygon: Polygon,
        pose: PoseModel,
        full_geometry_acquired: bool,
        movability: Movability = Movability.UNKNOWN,
    ):
        self.uid = uid
        self.polygon = polygon
        self.pose = pose
        self.full_geometry_acquired = full_geometry_acquired
        self.is_being_manipulated = False
        self.movability = movability
        self.type_ = type_
        self.circumscribed_radius = utils.get_circumscribed_radius(polygon=polygon)

    def within(self, other_entity: Self) -> bool:
        return self.polygon.within(other_entity.polygon)

    def copy(self):
        return Entity(
            uid=self.uid,
            type_=self.type_,
            polygon=copy.deepcopy(self.polygon),
            pose=self.pose,
            full_geometry_acquired=self.full_geometry_acquired,
        )

    def move_to_pose(self, new_pose: PoseModel):
        dx, dy, dtheta = (
            new_pose[0] - self.pose[0],
            new_pose[1] - self.pose[1],
            new_pose[2] - self.pose[2],
        )
        new_polygon = affinity.translate(self.polygon, xoff=dx, yoff=dy)
        new_polygon = affinity.rotate(new_polygon, angle=dtheta)
        self.polygon = new_polygon
        self.pose = (new_pose[0], new_pose[1], new_pose[2])
