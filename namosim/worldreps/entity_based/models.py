import typing as t

from pydantic import BaseModel, Field
from typing_extensions import Literal


class DiscretizationDataModel(BaseModel):
    cost_circumscribed: int
    cost_inscribed: int
    cost_lethal: int
    cost_possibly_nonfree: int
    inflation_radius: float
    res: float


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
    pose: t.Optional[t.Tuple[float, float, float]] = None
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
