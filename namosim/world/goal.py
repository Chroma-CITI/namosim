from shapely import Polygon

from namosim.data_models_v2 import PoseModel


class Goal:
    last_id = 1

    def __init__(
        self,
        name: str,
        polygon: Polygon,
        pose: PoseModel,
        uid: int = 0,
    ):
        if uid == 0:
            self.uid = Goal.last_id
            Goal.last_id = Goal.last_id + 1
        else:
            self.uid = uid

        self.name = name
        self.polygon = polygon
        self.pose = pose

    def to_json(self):
        return {
            "name": self.name,
            "geometry": {
                "from": "file",
                "id": self.name,
                "orientation_id": self.name + "_dir",
            },
        }
