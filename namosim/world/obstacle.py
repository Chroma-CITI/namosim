import copy

from shapely.geometry import Polygon

from namosim.data_models import PoseModel
from namosim.world.entity import Entity, Movability, Style


class Obstacle(Entity):
    def __init__(
        self,
        uid: str,
        polygon: Polygon,
        pose: PoseModel,
        full_geometry_acquired: bool,
        type_: str,
        movability: Movability,
        style: Style,
    ):
        Entity.__init__(
            self,
            uid=uid,
            type_=type_,
            polygon=polygon,
            pose=pose,
            full_geometry_acquired=full_geometry_acquired,
            movability=movability,
        )
        self.style = style
        self.actions = dict()
        self._is_actions_valid = False
        self.q_l = []
        self._is_q_l_valid = False

    def copy(self, copy_polygon: bool = True):
        return Obstacle(
            uid=self.uid,
            polygon=None if not copy_polygon else copy.deepcopy(self.polygon),  # type: ignore
            pose=self.pose,
            full_geometry_acquired=self.full_geometry_acquired,
            type_=self.type_,
            style=self.style,
            movability=self.movability,
        )

    def get_type(self):
        return self.type_
