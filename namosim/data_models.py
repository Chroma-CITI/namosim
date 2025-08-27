import typing as t

from pydantic_xml import BaseXmlModel, attr, element
import yaml
from pydantic import BaseModel, Field


class Pose2D(t.NamedTuple):
    x: float
    y: float
    degrees: float


FixedPrecisionPose2D = t.Tuple[int, int, int]
GridCellModel = t.Tuple[int, int]
GridCellSet = t.Set[GridCellModel]
VertexModel = t.Tuple[float, float]


class GoalConfigModel(BaseXmlModel, tag="goal"):
    goal_id: str = attr()


class BaseBehaviorConfigModel(BaseXmlModel, tag="behavior"):
    pass


class NavigationOnlyBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["navigation_only_behavior"] = attr()


class RRTAgentConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["rrt"] = attr()
    use_kd_tree: bool = attr(default=True)


class TeleopBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["teleop_behavior"] = attr()


class BaseRLAgentConfigModel(BaseBehaviorConfigModel):
    mode: t.Literal["learn", "execute"] = attr(default="learn")
    checkpoint: t.Optional[str] = attr(default=None)


class StilmanBehaviorParametersModel(BaseXmlModel, tag="parameters"):
    check_new_local_opening_before_global: bool = attr(default=True)
    activate_grids_logging: bool = attr(default=False)
    push_only: bool = attr(default=False)
    robot_rotation_unit_angle: float = attr(default=15)
    manip_search_bound_percentage: float = attr(default=0.1)
    use_social_cost: bool = attr(default=True)
    resolve_conflicts: bool = attr(default=True)
    resolve_deadlocks: bool = attr(default=True)
    deadlock_strategy: t.Literal["SOCIAL", "DISTANCE", ""] = attr(default="")
    drive_type: t.Literal["holonomic", "differential"] = attr(default="differential")
    grab_start_distance: float | None = attr(default=None)
    grab_end_distance: float | None = attr(default=None)
    conflict_horizon: int = attr(default=40)


class StilmanBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["stilman_2005_behavior"] = attr()
    parameters: StilmanBehaviorParametersModel = element()


class StilmanRRTBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["stilman_rrt"] = attr()
    parameters: StilmanBehaviorParametersModel = element()


class StilmanRRTStarBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["stilman_rrt_star_behavior"] = attr()
    parameters: StilmanBehaviorParametersModel = element()


AgentBehaviorXmlConfig = t.Union[
    NavigationOnlyBehaviorConfigModel,
    RRTAgentConfigModel,
    TeleopBehaviorConfigModel,
    StilmanBehaviorConfigModel,
    StilmanRRTStarBehaviorConfigModel,
]


class AgentConfigXmlModel(BaseXmlModel, tag="agent_config"):
    agent_id: str = attr()
    goals: t.List[GoalConfigModel] = element(tag="goal", default=[])
    behavior: AgentBehaviorXmlConfig = element(tag="behavior")


class NamoConfigXmlModel(BaseXmlModel, tag="namo_config"):
    cell_size_cm: float = attr()
    collision_margin_cm: float | None = attr(default=None)
    random_seed: int = attr(default=10)
    generate_report: bool = attr(default=True)
    agents: t.List[AgentConfigXmlModel] = element("agent", default=[])


# YAML MODELS


class BaseBehaviorConfigYamlModel(BaseModel):
    pass  # Abstract base class, no fields defined here


class StilmanBehaviorParametersYamlModel(BaseModel):
    check_new_local_opening_before_global: bool = True
    activate_grids_logging: bool = False
    push_only: bool = False
    robot_rotation_unit_angle: float = 15.0
    manip_search_bound_percentage: float = 0.1
    use_social_cost: bool = True
    resolve_conflicts: bool = True
    resolve_deadlocks: bool = True
    deadlock_strategy: t.Literal["SOCIAL", "DISTANCE", ""] = ""
    drive_type: t.Literal["holonomic", "differential"] = "differential"
    grab_start_distance: t.Optional[float] = None
    grab_end_distance: t.Optional[float] = None
    conflict_horizon: int = 40


class StilmanBehaviorConfigYamlModel(BaseBehaviorConfigYamlModel):
    type: t.Literal["stilman_2005_behavior"] = "stilman_2005_behavior"
    parameters: StilmanBehaviorParametersYamlModel = (
        StilmanBehaviorParametersYamlModel()
    )


AgentBehaviorYamlConfig = t.Union[StilmanBehaviorConfigYamlModel,]


class GoalYamlModel(BaseModel):
    id: str
    pose: t.List[float]


class AgentConfigYamlModel(BaseModel):
    id: str
    initial_pose: t.List[float] | None = None
    radius: float
    behavior: AgentBehaviorYamlConfig


class NamoConfigYamlModel(BaseModel):
    map_yaml: str
    svg_file: str | None = None
    agents: t.List[AgentConfigYamlModel] = []
    collision_margin_cm: float | None = None
    random_seed: int = 10
    generate_report: bool = False


class MapYamlConfigModel(BaseModel):
    image: str
    mode: str
    resolution: float
    origin: t.List[float]
    negate: int | bool
    occupied_thresh: float
    free_thresh: float


def namo_config_from_yaml(file_path: str) -> NamoConfigYamlModel:
    with open(file_path, "r") as stream:
        config = yaml.safe_load(stream)
    return NamoConfigYamlModel(**config)
