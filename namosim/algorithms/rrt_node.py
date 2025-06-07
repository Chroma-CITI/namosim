from dataclasses import dataclass
from typing import Optional
from namosim.data_models import Pose2D


@dataclass
class RRTNode:
    pose: Pose2D
    parent: Optional["RRTNode"] = None
    cost: float = 0.0
