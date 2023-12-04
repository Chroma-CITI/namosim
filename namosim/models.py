import typing as t

from pydantic import BaseModel

PoseModel = t.Tuple[float, float, float]
FixedPrecisionPoseModel = t.Tuple[int, int, int]
GridCellModel = t.Tuple[int, int]
GridCellSet = t.Set[GridCellModel]
VertexModel = t.Tuple[float, float]


class NavigationGoalModel(BaseModel):
    name: str


class StilmanBehaviorParametersModel(BaseModel):
    alpha_for_obstacle_choice_heur: float
    basic_rotation_moment: float
    basic_translation_force: float
    check_new_local_opening_before_global: bool
    collision_check_angular_res: float
    deactivate_grids_logging: bool
    forbid_rotations: bool
    heuristic_cost_for_traversing_obstacle_in_choice_heur: float
    manipulation_search_procedure: t.Literal["BFS", "DFS"]
    neighborhood_for_obstacle_choice_heur: t.Literal["TAXI"]
    robot_rotation_unit_angle: float
    robot_translation_unit_length: float
    solution_interval_bound_percentage: float
    use_social_cost: bool


class WuLevihnBehaviorParametersModel(BaseModel):
    check_new_opening_activated: bool
    manip_weight: float
    reset_knowledge_activated: bool
    social_movability_evaluation_activated: bool
    social_placement_choice_activated: bool
    use_social_layer: bool


class BaseBehaviorConfigModel(BaseModel):
    pass


class WuLevihnBehaviorConfigModel(BaseBehaviorConfigModel):
    name: t.Literal["wu_levihn_2014_behavior"]
    navigation_goals: t.List[NavigationGoalModel]
    parameters: WuLevihnBehaviorParametersModel


class NavigationOnlyBehaviorConfigModel(BaseBehaviorConfigModel):
    name: t.Literal["navigation_only_behavior"]
    navigation_goals: t.List[NavigationGoalModel]


class StilmanOnlyBehaviorConfigModel(BaseBehaviorConfigModel):
    name: t.Literal["navigation_only_behavior"]
    navigation_goals: t.List[NavigationGoalModel]
    use_social_cost: bool = False
    robot_translation_unit_length: float


class StilmanBehaviorConfigModel(BaseBehaviorConfigModel):
    name: t.Literal["stilman_2005_behavior"]
    navigation_goals: t.Optional[t.List[NavigationGoalModel]] = None
    parameters: StilmanBehaviorParametersModel


class AgentBehaviorModel(BaseModel):
    agent_name: str
    behavior: t.Union[
        StilmanBehaviorConfigModel,
        WuLevihnBehaviorConfigModel,
        NavigationOnlyBehaviorConfigModel,
    ]


class SimulationFilesModel(BaseModel):
    world_file: str


class SimulationModel(BaseModel):
    agents_behaviors: t.List[AgentBehaviorModel]
    display_sim_knowledge_only_once: bool
    provide_walls: bool
    files: SimulationFilesModel
    random_seed: t.Optional[int] = None
