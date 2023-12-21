import typing as t

from pydantic_xml import BaseXmlModel, attr, element

PoseModel = t.Tuple[float, float, float]
UID = t.Union[str, int]
FixedPrecisionPoseModel = t.Tuple[int, int, int]
GridCellModel = t.Tuple[int, int]
GridCellSet = t.Set[GridCellModel]
VertexModel = t.Tuple[float, float]


class GoalConfigModel(BaseXmlModel, tag="goal"):
    goal_id: str = attr()


class BaseBehaviorConfigModel(BaseXmlModel, tag="behavior"):
    pass


class StilmanOnlyParametersModel(BaseXmlModel):
    use_social_cost: bool = attr(default=False)
    robot_translation_unit_length: float = attr()
    robot_rotation_unit_angle: float = attr(default=30)


class StilmanOnlyBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["stilman_only_behavior"] = attr()
    parameters: StilmanOnlyParametersModel = element()


class NavigationOnlyBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["navigation_only_behavior"] = attr()


class StilmanBehaviorParametersModel(BaseXmlModel, tag="parameters"):
    alpha_for_obstacle_choice_heur: float = attr(default=0.5)
    basic_rotation_moment: float = attr(default=2.0)
    basic_translation_force: float = attr(default=2.0)
    check_new_local_opening_before_global: bool = attr(default=True)
    collision_check_angular_res: float = attr(default=5.0)
    activate_grids_logging: bool = attr(default=False)
    forbid_rotations: bool = attr(default=False)
    heuristic_cost_for_traversing_obstacle_in_choice_heur: float = attr(default=2.0)
    manipulation_search_procedure: t.Literal["BFS", "DFS"] = attr(default="BFS")
    neighborhood_for_obstacle_choice_heur: t.Literal["TAXI"] = attr(default="TAXI")
    robot_rotation_unit_angle: float = attr(default=30)
    robot_translation_unit_length: float = attr()
    solution_interval_bound_percentage: float = attr(default=0.01)
    use_social_cost: bool = attr(default=True)


class StilmanBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["stilman_2005_behavior"] = attr()
    parameters: StilmanBehaviorParametersModel = element()


class WuLevihnBehaviorConfigModel(BaseBehaviorConfigModel):
    type: t.Literal["wu_levihn_2014_behavior"] = attr()
    check_new_opening_activated: bool = attr()
    manip_weight: float = attr()
    reset_knowledge_activated: bool = attr()
    social_movability_evaluation_activated: bool = attr()
    social_placement_choice_activated: bool = attr()
    use_social_layer: bool = attr()


class AgentConfigModel(BaseXmlModel, tag="agent_config"):
    agent_id: str = attr()
    goals: t.List[GoalConfigModel] = element(tag="goal")
    behavior: t.Union[
        StilmanOnlyBehaviorConfigModel,
        WuLevihnBehaviorConfigModel,
        NavigationOnlyBehaviorConfigModel,
        StilmanBehaviorConfigModel,
    ] = element(tag="behavior")


class NamosimConfigModel(BaseXmlModel, tag="namo_config"):
    cell_size: float = attr()
    random_seed: int = attr(default=10)
    agents: t.List[AgentConfigModel] = element("agent")
