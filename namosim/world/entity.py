import copy
import re
import typing as t
from enum import Enum

from shapely import Polygon
from typing_extensions import Self

import namosim.utils.utils as utils
from namosim.data_models import UID, PoseModel


class Movability(Enum):
    UNKNOWN = 1
    MOVABLE = 2
    STATIC = 3
    UNMOVABLE = 4


class Style:
    def __init__(
        self,
        fill: str = "#000000",
        fill_opacity: str = "1",
        stroke: str = "#000000",
        stroke_width: str = "1",
        stroke_opacity: str = "1",
        **_,
    ):
        self.fill = fill
        self.fill_opacity = float(fill_opacity)
        self.stroke = stroke
        try:
            self.stroke_width = float(
                re.findall(r"[-+]?(?:\d*\.*\d+)", stroke_width)[0]
            )
        except IndexError:
            self.stroke_width = 1.0
        self.stroke_opacity = float(stroke_opacity)

    # noinspection PyTypeChecker
    @classmethod
    def from_string(cls, style: str):
        d: t.Dict[str, str] = dict(
            [a.strip().replace("-", "_") for a in attribute.split(":", 1)]
            for attribute in style.split(";")
            if attribute
        )
        return cls(**d)


class Entity:
    last_id = 1

    # Constructor
    def __init__(
        self,
        type_: str,
        name: str,
        polygon: Polygon,
        pose: PoseModel,
        full_geometry_acquired: bool,
        style: Style,
        movability: Movability = Movability.UNKNOWN,
        uid: UID = 0,
    ):
        if uid == 0:
            self.uid = Entity.last_id
            Entity.last_id = Entity.last_id + 1
        else:
            self.uid = uid
        self.name = name
        self.polygon = polygon
        self.pose = pose
        self.full_geometry_acquired = full_geometry_acquired
        self.is_being_manipulated = False
        self.movability = movability
        self.style = style
        self.type_ = type_
        self.circumscribed_radius = utils.get_circumscribed_radius(polygon=polygon)

    def within(self, other_entity: Self) -> bool:
        return self.polygon.within(other_entity.polygon)

    def copy(self):
        return Entity(
            name=self.name,
            type_=self.type_,
            polygon=copy.deepcopy(self.polygon),
            pose=self.pose,
            full_geometry_acquired=self.full_geometry_acquired,
            uid=self.uid,
            style=self.style,
        )
