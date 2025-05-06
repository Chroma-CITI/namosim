DEFAULT_ROBOT_SHAPE_STYLE = "fill:#0000ff;stroke-width:3.0;"
DEFAULT_ROBOT_ORIENTATION_STYLE = "fill:#00ffff;stroke:none;"
DEFAULT_GOAL_SHAPE_STYLE = "fill:none;stroke:#0000ff;stroke-width:3.0;"
DEFAULT_GOAL_ORIENTATION_STYLE = "fill:#0000ff;stroke:none;stroke-width:2.66547;"
OBSTACE_TRACE_STYLE = "fill:#000000;fill-opacity:0.05231688;fill-rule:evenodd;stroke:#f1c232;stroke-width:1;stroke-linecap:square;stroke-miterlimit:10;stroke-opacity:1"
UNKNOWN_ENTITY_STYLE = "fill:#674ea7;fill-rule:evenodd"
DEFAULT_MOVABLE_ENTITY_STYLE = "fill:#f1c232;fill-rule:evenodd"
FIXED_ENTITY_STYLE = "fill:#000000;fill-rule:evenodd"
ROBOT_ENTITY_STYLE = "fill:#6d9eeb;fill-opacity:1;stroke:none;stroke-opacity:1"


class AgentStyle:
    def __init__(
        self,
        shape: str = DEFAULT_ROBOT_SHAPE_STYLE,
        orientation: str = DEFAULT_ROBOT_ORIENTATION_STYLE,
    ):
        self.shape = shape or DEFAULT_ROBOT_SHAPE_STYLE
        self.orientation = orientation or DEFAULT_ROBOT_ORIENTATION_STYLE
