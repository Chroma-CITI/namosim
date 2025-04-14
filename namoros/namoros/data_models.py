from enum import Enum
from pydantic import BaseModel
from typing import List, Union, Literal
import yaml


class ShapeEnum(str, Enum):
    rectangle = "rectangle"
    circle = "circle"


class Obstacle(BaseModel):
    marker_id: str
    name: str


class Rectangle(Obstacle):
    shape: Literal[ShapeEnum.rectangle]
    length: float
    width: float


class Circle(Obstacle):
    shape: Literal[ShapeEnum.circle]
    radius: float


class NamoRosConfig(BaseModel):
    obstacles: List[Union[Rectangle, Circle]]


def load_namoros_config(file_path: str) -> NamoRosConfig:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return NamoRosConfig(**data)
