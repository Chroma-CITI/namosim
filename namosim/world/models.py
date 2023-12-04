import typing as t

from pydantic import BaseModel, Field
from typing_extensions import Literal

from namosim.data_models import PoseModel


class DiscretizationDataModel(BaseModel):
    """
    The discretization data contains parameters that related to discretizing or "rasterizing" the world into a rectangular grid of cells.
    """

    res: float
    """
    This "resolution" parameter is equal to the size of a grid cell in the occcupancy grid representations of the world. The units are equal to those
    of the svg viewbox.
    """


class WorldFilesModel(BaseModel):
    geometry_file: str


class GeometryModel(BaseModel):
    from_: str = Field(..., alias="from")
    id: str
    orientation_id: t.Optional[str] = None


class OmniscientSensorModel(BaseModel):
    type_: Literal["omniscient"] = Field(..., alias="type")


class GFovSensorModel(BaseModel):
    type_: Literal["perfect_g_fov"] = Field(..., alias="type")
    max_radius: float
    min_radius: float
    opening_angle: float


class SFovSensorModel(BaseModel):
    type_: Literal["perfect_s_fov"] = Field(..., alias="type")
    max_radius: float
    min_radius: float
    opening_angle: float


class RobotEntityModel(BaseModel):
    type_: Literal["robot"] = Field(..., alias="type")
    force_pushes_only: bool
    geometry: GeometryModel
    movable_whitelist: t.List[str]
    name: str
    push_only_list: t.List[str]
    sensors: t.List[t.Union[OmniscientSensorModel, GFovSensorModel, SFovSensorModel]]


class ObstacleEntityModel(BaseModel):
    type_: Literal["wall", "box", "stool", "pillar", "table"] = Field(..., alias="type")
    name: str
    geometry: GeometryModel


class GoalModel(BaseModel):
    geometry: t.Optional[GeometryModel] = None
    pose: t.Optional[PoseModel] = None
    name: str


class TabooModel(BaseModel):
    geometry: GeometryModel
    name: str


class ZonesModel(BaseModel):
    goals: t.Optional[t.List[GoalModel]] = None
    taboos: t.Optional[t.List[TabooModel]] = None


class WorldThings(BaseModel):
    entities: t.List[t.Union[RobotEntityModel, ObstacleEntityModel]]
    zones: t.Optional[ZonesModel] = None


class WorldModel(BaseModel):
    discretization_data: DiscretizationDataModel
    files: WorldFilesModel
    geometry_scale: float
    things: WorldThings
    no_scaling_workaround: t.Optional[bool] = None
