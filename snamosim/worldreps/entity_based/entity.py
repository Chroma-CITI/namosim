import math
import numpy as np
import copy
import shapely.affinity as affinity
from .custom_exceptions import IntersectionError


class Entity:
    last_id = 1

    # Constructor
    def __init__(self, name, polygon, pose, full_geometry_acquired, movability="unknown", uid=0):
        if uid == 0:
            self.uid = Entity.last_id
            Entity.last_id = Entity.last_id + 1
        else:
            self.uid = uid
        self.name = name
        self.polygon = polygon
        self.pose = tuple(pose)
        self.full_geometry_acquired = full_geometry_acquired
        self.is_being_manipulated = False
        self.movability = movability

    def within(self, other_entity):
        return self.polygon.within(other_entity.polygon)

    def light_copy(self):
        return Entity(name=self.name, polygon=copy.deepcopy(self.polygon), pose=self.pose,
                      full_geometry_acquired=self.full_geometry_acquired, uid=self.uid)

    def to_json(self):
        return {
            "name": self.name,
            "type": self.type,
            "geometry": {
                "from": "file",
                "id": self.name
            }
        }
