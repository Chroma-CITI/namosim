from dataclasses import dataclass
from typing import Optional
from namosim.data_models import PoseModel


@dataclass
class RRTNode:
    pose: PoseModel
    parent: Optional["RRTNode"] = None
    cost: float = 0.0
