import typing as t

from shapely.geometry import Polygon

from namosim import svg_styles
from namosim.data_models import Pose2D


class Goal:
    def __init__(
        self,
        uid: str,
        polygon: Polygon,
        pose: Pose2D,
        svg_style: svg_styles.AgentStyle | None = None,
    ):
        self.uid = uid
        self.pose = pose
        self.polygon = polygon
        if svg_style:
            self.svg_style = svg_style
        else:
            self.svg_style = svg_styles.AgentStyle(
                shape=svg_styles.DEFAULT_GOAL_SHAPE_STYLE,
                orientation=svg_styles.DEFAULT_GOAL_ORIENTATION_STYLE,
            )

    def __key(self):
        return self.pose

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other: t.Any):
        if isinstance(other, Goal):
            return self.__key() == other.__key()
        return NotImplemented
