import typing as t

from pydantic_xml import BaseXmlModel, attr, element
import yaml
from pydantic import BaseModel


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


class WuLevihnBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["wu_levihn_2014_behavior"] = attr()
    check_new_opening_activated: bool = attr()
    manip_weight: float = attr()
    reset_knowledge_activated: bool = attr()
    social_movability_evaluation_activated: bool = attr()
    social_placement_choice_activated: bool = attr()
    use_social_layer: bool = attr()


AgentBehaviorConfig = t.Union[
    WuLevihnBehaviorConfigModel,
    NavigationOnlyBehaviorConfigModel,
    RRTAgentConfigModel,
    TeleopBehaviorConfigModel,
    StilmanBehaviorConfigModel,
    StilmanRRTStarBehaviorConfigModel,
]


class AgentConfigModel(BaseXmlModel, tag="agent_config"):
    agent_id: str = attr()
    goals: t.List[GoalConfigModel] = element(tag="goal", default=[])
    behavior: AgentBehaviorConfig = element(tag="behavior")


class NamoConfigModel(BaseXmlModel, tag="namo_config"):
    cell_size_cm: float = attr()
    collision_margin_cm: float | None = attr(default=None)
    random_seed: int = attr(default=10)
    generate_report: bool = attr(default=True)
    agents: t.List[AgentConfigModel] = element("agent", default=[])


class GoalYamlModel(BaseModel):
    id: str
    pose: t.List[float]


class NamoAgentYamlModel(BaseModel):
    id: str
    initial_pose: t.List[float] | None = None
    radius: float
    push_only: bool = False
    grab_start_distance: float | None = None
    grab_end_distance: float | None = None


class NamoConfigYamlModel(BaseModel):
    map_yaml: str
    svg_file: str | None = None
    agents: t.List[NamoAgentYamlModel] = []
    collision_margin: float | None = None


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
