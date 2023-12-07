import copy
import time
import typing as t

import numpy as np
import numpy.typing as npt
from aabbtree import AABBTree
from shapely import Polygon

import namosim.navigation.basic_actions as ba
import namosim.world.social_topological_occupation_cost_grid as stocg
from namosim.algorithms import graph_search
from namosim.algorithms.new_local_opening_check import check_new_local_opening
from namosim.behaviors.baseline_behavior import BaselineBehavior, ThinkResult
from namosim.behaviors.stilman_configurations import (
    BaseConfiguration,
    RCHConfiguration,
    RobotConfiguration,
    RobotObstacleConfiguration,
)
from namosim.data_models import (
    GridCellModel,
    PoseModel,
    StilmanOnlyBehaviorParametersModel,
)
from namosim.display.ros2_publisher import RosPublisher
from namosim.navigation.navigation_path import Path, TransferPath, TransitPath
from namosim.navigation.navigation_plan import Plan
from namosim.utils import collision, connectivity, utils
from namosim.world.binary_occupancy_grid import (
    BinaryInflatedOccupancyGrid,
    BinaryOccupancyGrid,
)
from namosim.world.obstacle import Obstacle
from namosim.world.robot import Robot
from namosim.world.world import World


class StilmanOnlyBehavior(BaselineBehavior):
    """Implements basic Stilman behavior for dealing with movable obstacles but does not handle conflicts with dynamic obstacles or other agents."""

    def __init__(
        self,
        initial_world: World,
        robot_uid: int,
        navigation_goals: t.List[PoseModel],
        params: StilmanOnlyBehaviorParametersModel,
        logs_dir: str,
    ):
        BaselineBehavior.__init__(
            self,
            initial_world=initial_world,
            robot_uid=robot_uid,
            navigation_goals=navigation_goals,
            name="stilman_only_behavior",
            logs_dir=logs_dir,
        )
        self.params = params
        self._social_costmap: npt.NDArray[t.Any] | None = None
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD
        self.robot_max_inflation_radius = utils.get_circumscribed_radius(
            self._robot.polygon
        )
        all_entities_polygons = {
            uid: e.polygon for uid, e in self.world.entities.items()
        }
        static_obs_polygons = {
            uid: entity.polygon
            for uid, entity in self.world.entities.items()
            if (
                isinstance(entity, Obstacle)
                and entity.movability == "unmovable"
                or entity.movability == "static"
            )
        }
        self.robot_max_inflation_radius = utils.get_circumscribed_radius(
            self._robot.polygon
        )
        self.static_obs_inf_grid = BinaryInflatedOccupancyGrid(
            static_obs_polygons,
            self.world.discretization_data.res,
            self.robot_max_inflation_radius,
            neighborhood=self.neighborhood,
        )
        self.static_obs_grid = BinaryOccupancyGrid(
            static_obs_polygons,
            self.world.discretization_data.res,
            neighborhood=self.neighborhood,
            params=self.static_obs_inf_grid.params,
        )

        all_entities_polygons = {
            uid: e.polygon for uid, e in self.world.entities.items()
        }

        self.inflated_grid_by_robot = BinaryInflatedOccupancyGrid(
            all_entities_polygons,
            self.world.discretization_data.res,
            self.robot_max_inflation_radius,
            neighborhood=self.neighborhood,
            params=self.static_obs_inf_grid.params,
        )

        # TODO Make sure static and generalist grid share same width and height (occurs naturally if map borders are static, but not otherwise)
        self.inflated_grid_by_robot.deactivate_entities({self._robot.uid})

        # Robot action space parameters
        self.translation_unit_cost = 1.0
        self.rotation_unit_cost = 1.0
        self.transfer_coefficient = 2.0  # Note: MUST ALWAYS BE > 1 !
        self.angular_res = 5.0
        self.rotation_unit_angle = 60.0
        self.translation_unit_length = self.params.robot_translation_unit_length
        self.translation_factor = (
            self.translation_unit_cost / self.translation_unit_length
        )
        self.rotation_factor = self.rotation_unit_cost / self.rotation_unit_angle
        self.trans_mult = 1.0 / self.world.discretization_data.res * 10.0
        self.rot_mult = 1.0

        # holonomic
        self._trans_vectors = np.array(
            [
                (self.translation_unit_length, 0.0),
                (-self.translation_unit_length, 0.0),
                (0.0, self.translation_unit_length),
                (0.0, -self.translation_unit_length),
            ]
        )
        self._rot_angles = np.array(
            [self.rotation_unit_angle, -self.rotation_unit_angle]
        )

        self._new_actions = []
        for trans_vector in self._trans_vectors:
            self._new_actions.append(ba.AbsoluteTranslation(trans_vector))
        for rot_angle in self._rot_angles:
            self._new_actions.append(ba.Rotation(rot_angle))

    def think(self, ros_publisher: RosPublisher):
        if self._q_goal is None:
            if self._navigation_goals:
                self._q_goal = self._navigation_goals.pop(0)
                self._p_opt = Plan([], self._q_goal)
            else:
                return ThinkResult(
                    next_action=ba.GoalsFinished(),
                    did_replan=False,
                    robot_name=self._robot_name,
                    has_conflicts=False,
                )

        if self._p_opt is None:
            raise Exception("No plan")

        # If current robot pose is close enough to goal, return Success
        if self.is_goal_reached(
            self.world.entities[self._robot_uid].pose, self._q_goal
        ):
            result = ThinkResult(
                next_action=ba.GoalSuccess(self._q_goal),
                did_replan=False,
                robot_name=self._robot_name,
                has_conflicts=False,
            )
            self._q_goal = None
            return result

        if not self._p_opt.is_empty():
            return ThinkResult(
                next_action=self._p_opt.pop_next_action(),
                did_replan=False,
                robot_name=self._robot_name,
                has_conflicts=False,
            )

        self._p_opt = self.compute_stilman_plan(
            w_t=self.world,
            static_obs_inf_grid=self.static_obs_inf_grid,
            inflated_grid_by_robot_max=self.inflated_grid_by_robot,
            r_f=self._q_goal,
            trans_mult=self.trans_mult,
            rot_mult=self.rot_mult,
            ros_publisher=ros_publisher,
            ccs_data=None,
            prev_list=set(),
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
            action_space_reduction="only_r_acc_then_c_1_x",
        )

        if self._p_opt.is_empty():
            result = ThinkResult(
                next_action=ba.GoalFailed(self._q_goal),
                did_replan=True,
                robot_name=self._robot_name,
                has_conflicts=False,
            )
            self._q_goal = None
            return result

        self.goal_to_plans[self._q_goal] = self._p_opt

        return ThinkResult(
            next_action=self._p_opt.pop_next_action(),
            did_replan=True,
            robot_name=self._robot_name,
            has_conflicts=False,
        )

    def init_social_costmap(self, ros_publisher: RosPublisher):
        # Initialize social occupation costmap
        if self.params.use_social_cost and self._social_costmap is None:
            self._social_costmap = stocg.compute_social_costmap(
                self.static_obs_grid.grid,
                self.world.discretization_data.res,
                ros_publisher=ros_publisher,
                log_costmaps=False,
                logs_dir=self.logs_dir,
                ns=self._robot_name,
            )
            ros_publisher.publish_social_grid_map(
                self._social_costmap,
                self.world.discretization_data.res,
                ns=self._robot_name,
            )

    def compute_stilman_plan(
        self,
        w_t: World,
        static_obs_inf_grid: BinaryInflatedOccupancyGrid,
        inflated_grid_by_robot_max: BinaryInflatedOccupancyGrid,
        r_f: PoseModel,
        trans_mult: float,
        rot_mult: float,
        ros_publisher: RosPublisher,
        ccs_data: connectivity.CCSData | None = None,
        prev_list: set[int] = set(),
        neighborhood: t.Sequence[GridCellModel] = utils.CHESSBOARD_NEIGHBORHOOD,
        action_space_reduction: str = "only_r_acc_then_c_1_x",
    ):
        """
        This is the entry-point to the Stilman-based high-level navigation planner planner. It computes the full-navigation plan
        consisting of a sequence of transit and transfer paths ultimately ending at the goal pose.

        It works by first computing a sub-plan for the first obstacle to move and component to fuse (if any),
        and then it recursively computes a plan for the rest of the path. Recursive sub-plans are concatenated into the final
        overall plan.

        It makes use of _rch and _manip_search in a greedy heuristic search with backtracking.
        It backtracks locally when the object selected by _rch cannot be moved to merge the selected c_1 in c_free.
        It backtracks globally when all the paths identified by _rch from c_1 are unsuccessful.
        SC calls _find_path to determine a transit path from r_t to a contact point, r_t_plus_1 . The existence of the
        path is guaranteed by the choice of contacts in Manip-Search.

        :param w_t: state of the world at time t
        :type w_t: World
        :param static_obs_inf_grid: _description_
        :type static_obs_inf_grid: BinaryInflatedOccupancyGrid
        :param inflated_grid_by_robot_max: _description_
        :type inflated_grid_by_robot_max: BinaryInflatedOccupancyGrid
        :param r_f: goal robot configuration [x, y, theta] in {m, m, degrees}
        :type r_f: PoseModel
        :param trans_mult: _description_
        :type trans_mult: float
        :param rot_mult: _description_
        :type rot_mult: float
        :param ros_publisher: _description_
        :type ros_publisher: RosPublisher
        :param ccs_data: _description_, defaults to None
        :type ccs_data: connectivity.CCSData | None, optional
        :param prev_list: _description_, defaults to set()
        :type prev_list: set[int], optional
        :param neighborhood: _description_, defaults to utils.CHESSBOARD_NEIGHBORHOOD
        :type neighborhood: t.Sequence[GridCellModel], optional
        :param action_space_reduction: _description_, defaults to "only_r_acc_then_c_1_x"
        :type action_space_reduction: str, optional
        :raises ValueError: _description_
        :raises ValueError: _description_
        :return: _description_
        :rtype: _type_
        """
        robot = w_t.entities[self._robot_uid]
        r_t = robot.pose

        avoid_list: t.Set[GridCellModel] = set()

        robot_cell = utils.real_to_grid(
            r_t[0], r_t[1], static_obs_inf_grid.res, static_obs_inf_grid.grid_pose
        )
        goal_cell = utils.real_to_grid(
            r_f[0], r_f[1], static_obs_inf_grid.res, static_obs_inf_grid.grid_pose
        )

        simple_path_to_goal = self.find_path(
            robot_pose=r_t,
            goal_pose=r_f,
            robot_inflated_grid=inflated_grid_by_robot_max,
            robot_polygon=robot.polygon,
        )
        if simple_path_to_goal:
            # If the goal is in the same free space component as the robot in simulated w_t
            ros_publisher.cleanup_robot_sim(ns=self._robot_name)
            return Plan([simple_path_to_goal], r_f, self._robot_uid)

        if ccs_data is None:
            ccs_data = connectivity.init_ccs_for_grid(
                inflated_grid_by_robot_max.grid,
                inflated_grid_by_robot_max.d_width,
                inflated_grid_by_robot_max.d_height,
                neighborhood,
            )
        connected_components_grid = ccs_data.grid
        ros_publisher.publish_connected_components_grid(
            connected_components_grid, w_t.discretization_data.res, ns=robot.name
        )

        c_0 = ccs_data.grid[robot_cell[0]][robot_cell[1]]
        prev_list = prev_list if c_0 == 0 else prev_list.union({c_0})

        # Gather set of all cells currently accessible to the robot
        accessible_cells = (
            set()
            if inflated_grid_by_robot_max.grid[robot_cell[0]][robot_cell[1]] > 0
            else connectivity.bfs_init(
                inflated_grid_by_robot_max.grid,
                inflated_grid_by_robot_max.d_width,
                inflated_grid_by_robot_max.d_height,
                robot_cell,
                neighborhood,
            ).visited
        )

        if inflated_grid_by_robot_max.cell_to_obstacle_id(robot_cell) == -1:
            return Plan(plan_error="start_cell_in_several_movable_obstacles_error")

        if (
            static_obs_inf_grid.grid[robot_cell[0]][robot_cell[1]] > 0
            or static_obs_inf_grid.grid[goal_cell[0]][goal_cell[1]] > 0
        ):
            return Plan(plan_error="start_or_goal_cell_in_static_obstacle_error")

        # if inflated_grid_by_robot_max.grid[goal_cell[0]][goal_cell[1]] > 1: Should not be necessary thanks to first check
        #     return Plan(plan_error="goal_cell_in_more_than_one_movable_obstacle_error")

        other_robot_uids = {  # Dynamic obstacles are forbidden !
            uid
            for uid, entity in w_t.entities.items()
            if (
                (isinstance(entity, Robot) and uid != self._robot.uid)
                or (
                    uid in w_t.entity_to_agent
                    and w_t.entity_to_agent[uid] != self._robot.uid
                )
            )
        }

        o_1, c_1 = self.rch(
            start_cell=robot_cell,
            goal_cell=goal_cell,
            static_obs_grid=static_obs_inf_grid,
            connected_components_grid=connected_components_grid,
            inflated_robot_grid=inflated_grid_by_robot_max,
            avoid_list=avoid_list,
            prev_list=prev_list,
            forbidden_obstacles=other_robot_uids,
            ros_publisher=ros_publisher,
            neighborhood=neighborhood,
        )

        while o_1 != 0:
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: select_connect: selected entity {} for manipulation search to reach component {}.".format(
                        robot.name, w_t.entities[o_1].name, c_1
                    ),
                    self._step_count,
                )
            )
            if action_space_reduction == "none":
                w_t_plus_2, tho_m = self.manip_search(
                    w_t=w_t,
                    o_1=o_1,
                    c_1=c_1,
                    ccs_data=ccs_data,
                    r_acc_cells=accessible_cells,
                    r_f=r_f,
                    inflated_grid_by_robot_max=inflated_grid_by_robot_max,
                    trans_mult=trans_mult,
                    rot_mult=rot_mult,
                    ros_publisher=ros_publisher,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=True,
                )
            elif action_space_reduction == "only_r_acc":
                w_t_plus_2, tho_m = self.manip_search(
                    w_t=w_t,
                    o_1=o_1,
                    c_1=c_1,
                    ccs_data=ccs_data,
                    r_acc_cells=accessible_cells,
                    r_f=r_f,
                    inflated_grid_by_robot_max=inflated_grid_by_robot_max,
                    trans_mult=trans_mult,
                    rot_mult=rot_mult,
                    ros_publisher=ros_publisher,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=False,
                )
            elif action_space_reduction == "only_r_acc_then_c_1_x":
                w_t_plus_2, tho_m = self.manip_search(
                    w_t=w_t,
                    o_1=o_1,
                    c_1=c_1,
                    ccs_data=ccs_data,
                    r_acc_cells=accessible_cells,
                    r_f=r_f,
                    inflated_grid_by_robot_max=inflated_grid_by_robot_max,
                    trans_mult=trans_mult,
                    rot_mult=rot_mult,
                    ros_publisher=ros_publisher,
                    obstacle_can_intrude_r_acc=True,
                    obstacle_can_intrude_c_1_x=False,
                )
                if tho_m is None:
                    w_t_plus_2, tho_m = self.manip_search(
                        w_t=w_t,
                        o_1=o_1,
                        c_1=c_1,
                        ccs_data=ccs_data,
                        r_acc_cells=accessible_cells,
                        r_f=r_f,
                        inflated_grid_by_robot_max=inflated_grid_by_robot_max,
                        trans_mult=trans_mult,
                        rot_mult=rot_mult,
                        ros_publisher=ros_publisher,
                        obstacle_can_intrude_r_acc=False,
                        obstacle_can_intrude_c_1_x=True,
                    )
            else:
                raise ValueError(
                    "action_space_reduction variable value is {}, but it should be one of {}".format(
                        action_space_reduction,
                        ["none", "only_r_acc", "only_r_acc_then_c_1_x"],
                    )
                )

            if tho_m is not None:
                self.simulation_log.append(
                    utils.BasicLog(
                        "Agent {}: select_connect: found partial plan manipulating entity {} to reach component {}.".format(
                            robot.name, w_t.entities[o_1].name, c_1
                        ),
                        self._step_count,
                    )
                )

                prev_cells_sets = inflated_grid_by_robot_max.update(
                    {o_1: w_t_plus_2.entities[o_1].polygon}
                )

                future_plan = self.compute_stilman_plan(
                    w_t=w_t_plus_2,
                    static_obs_inf_grid=static_obs_inf_grid,
                    inflated_grid_by_robot_max=inflated_grid_by_robot_max,
                    r_f=r_f,
                    trans_mult=trans_mult,
                    rot_mult=rot_mult,
                    ros_publisher=ros_publisher,
                    ccs_data=ccs_data,
                    prev_list=(prev_list if c_1 == 0 else prev_list.union({c_1})),
                    neighborhood=neighborhood,
                    action_space_reduction=action_space_reduction,
                )

                inflated_grid_by_robot_max.cells_sets_update(prev_cells_sets)

                if not future_plan.plan_error:
                    tho_n = self.find_path(
                        robot_pose=r_t,
                        goal_pose=tho_m.robot_path.poses[0],
                        robot_inflated_grid=inflated_grid_by_robot_max,
                        robot_polygon=robot.polygon,
                    )
                    if not tho_n:
                        raise ValueError(
                            "It should not be possible not to find a transit path when the transfer path is found."
                        )
                    plan_components: t.List[TransitPath | TransferPath] = (
                        [tho_n, tho_m] if tho_n.actions else [tho_m]
                    )
                    return Plan(plan_components, r_f, self._robot_uid).append(
                        future_plan
                    )

            # Extra check for when the goal is in a movable obstacle that we could not find how to move
            if c_1 == 0:
                self.simulation_log.append(
                    utils.BasicLog(
                        "Agent {}: select_connect: did not find a reachable component if manipulating {}.".format(
                            robot.name, w_t.entities[o_1].name
                        ),
                        self._step_count,
                    )
                )
                break

            avoid_list.add((o_1, c_1))

            o_1, c_1 = self.rch(
                start_cell=robot_cell,
                goal_cell=goal_cell,
                static_obs_grid=static_obs_inf_grid,
                connected_components_grid=connected_components_grid,
                inflated_robot_grid=inflated_grid_by_robot_max,
                avoid_list=avoid_list,
                prev_list=prev_list,
                forbidden_obstacles=other_robot_uids,
                ros_publisher=ros_publisher,
                neighborhood=neighborhood,
            )

        # ros_publisher.cleanup_robot_sim(ns=self._robot_name)
        return Plan(plan_error="no_plan_found_error")

    def rch(
        self,
        start_cell: GridCellModel,
        goal_cell: GridCellModel,
        static_obs_grid: BinaryInflatedOccupancyGrid,
        connected_components_grid: npt.NDArray[np.int_],
        inflated_robot_grid: BinaryInflatedOccupancyGrid,
        avoid_list: t.Set[GridCellModel],
        prev_list: t.Set[int],
        forbidden_obstacles: t.Set[int],
        ros_publisher: RosPublisher,
        neighborhood: t.Sequence[GridCellModel] = utils.TAXI_NEIGHBORHOOD,
    ):
        """This is the obstacle selection subroutine. It performs an A* search over (obstacle, component) pairs to find
        a sequence of obstacles to move to make the goal reachable. It returns the ids of the first obstacle, component pair in the sequence.

        :param start_cell: _description_
        :type start_cell: GridCellModel
        :param goal_cell: _description_
        :type goal_cell: GridCellModel
        :param static_obs_grid: _description_
        :type static_obs_grid: BinaryInflatedOccupancyGrid
        :param connected_components_grid: _description_
        :type connected_components_grid: npt.NDArray[np.int_]
        :param inflated_robot_grid: _description_
        :type inflated_robot_grid: BinaryInflatedOccupancyGrid
        :param avoid_list: _description_
        :type avoid_list: t.Set[GridCellModel]
        :param prev_list: _description_
        :type prev_list: t.Set[int]
        :param forbidden_obstacles: _description_
        :type forbidden_obstacles: t.Set[int]
        :param ros_publisher: _description_
        :type ros_publisher: RosPublisher
        :param neighborhood: _description_, defaults to utils.TAXI_NEIGHBORHOOD
        :type neighborhood: t.Sequence[GridCellModel], optional
        :raises ValueError: _description_
        :return: _description_
        :rtype: _type_
        """
        if static_obs_grid.grid[start_cell[0]][start_cell[1]] > 0:
            obstacle_names = {
                self.world.entities[uid].name
                for uid in static_obs_grid.obstacles_uids_in_cell(start_cell)
            }
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot start cell {} in a rch call must always be outside of static obstacles, here: {}.".format(
                        self._robot_name, start_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        if static_obs_grid.grid[goal_cell[0]][goal_cell[1]] > 0:
            obstacle_names = {
                self.world.entities[uid].name
                for uid in static_obs_grid.obstacles_uids_in_cell(goal_cell)
            }
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot goal cell {} in a rch call must always be outside of static obstacles, here: {}.".format(
                        self._robot_name, goal_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        start_obstacle_uid = inflated_robot_grid.cell_to_obstacle_id(start_cell)
        if start_obstacle_uid == -1 or start_obstacle_uid in forbidden_obstacles:
            obstacle_names = {
                self.world.entities[uid].name
                for uid in inflated_robot_grid.obstacles_uids_in_cell(start_cell)
            }
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot start cell {} in a rch call must always be at most in one obstacle and not a forbidden one, here: {}.".format(
                        self._robot_name, start_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        if inflated_robot_grid.grid[goal_cell[0]][goal_cell[1]] > 1:
            obstacle_names = {
                self.world.entities[uid].name
                for uid in inflated_robot_grid.obstacles_uids_in_cell(goal_cell)
            }
            self.simulation_log.append(
                utils.BasicLog(
                    "Agent {}: rch: The robot goal cell {} in a rch call must be at most within one movable obstacle, here: {}.".format(
                        self._robot_name, goal_cell, obstacle_names
                    ),
                    self._step_count,
                )
            )
            return 0, 0

        # TODO Create custom exceptions for above

        sqrt_of_2_times_res = utils.SQRT_OF_2 * inflated_robot_grid.res
        goal_real = utils.grid_to_real(
            goal_cell[0],
            goal_cell[1],
            inflated_robot_grid.res,
            inflated_robot_grid.grid_pose,
        )

        def g_function(
            current: RCHConfiguration,
            neighbor: RCHConfiguration,
            is_transfer: bool = False,
        ) -> float:
            dist = (
                sqrt_of_2_times_res
                if neighbor.cell
                in [
                    (current.cell[0] + i, current.cell[1] + j)
                    for i, j in utils.CHESSBOARD_NEIGHBORHOOD_EXTRAS
                ]
                else inflated_robot_grid.res
            )
            translation_cost = self.translation_factor * dist
            return translation_cost * (
                1.0 if not is_transfer else self.transfer_coefficient
            )

        def h_function(_c: RCHConfiguration, _g: RCHConfiguration) -> float:
            translation_cost = self.translation_factor * utils.euclidean_distance(
                utils.grid_to_real(
                    _c.cell[0],
                    _c.cell[1],
                    inflated_robot_grid.res,
                    inflated_robot_grid.grid_pose,
                ),
                goal_real,
            )
            return translation_cost

        traversed_obstacles_ids = utils.OrderedSet()

        def rch_get_neighbors_instance(
            current: RCHConfiguration,
            gscore: t.Dict[RCHConfiguration, float],
            close_set: t.Set[RCHConfiguration],
            open_queue: graph_search.PriorityQueue,
            came_from: t.Dict[RCHConfiguration, RCHConfiguration],
        ):
            return self.rch_get_neighbors(
                current,
                gscore,
                close_set,
                open_queue,
                came_from,
                static_obs_grid,
                connected_components_grid,
                inflated_robot_grid,
                avoid_list,
                prev_list,
                g_function,
                traversed_obstacles_ids,
                forbidden_obstacles,
                ros_publisher,
                neighborhood,
            )

        def exit_condition(_current: RCHConfiguration, _goal: RCHConfiguration) -> bool:
            return _current.cell == _goal.cell

        start = RCHConfiguration(
            start_cell, start_obstacle_uid if start_obstacle_uid > 0 else 0, 0
        )
        goal = RCHConfiguration(
            goal_cell, 0, 0
        )  # Note the zeroes are never used, this line is just for coherence

        end_config: RCHConfiguration
        path_found, end_config, _, _, _, _ = graph_search.new_generic_a_star(
            start, goal, exit_condition, rch_get_neighbors_instance, h_function
        )  # type: ignore

        if path_found:
            if end_config.first_obstacle_uid == 0:
                raise ValueError(
                    "Rch found a path where no obstacle needed to be traversed."
                )
            return end_config.first_obstacle_uid, end_config.first_component_uid
        else:
            return 0, 0

    def rch_get_neighbors(
        self,
        current: RCHConfiguration,
        gscore: t.Dict[RCHConfiguration, float],
        close_set: t.Set[RCHConfiguration],
        open_queue: graph_search.PriorityQueue,
        came_from: t.Dict[RCHConfiguration, RCHConfiguration],
        static_obs_grid: BinaryInflatedOccupancyGrid,
        connected_components_grid: npt.NDArray[np.int_],
        inflated_robot_grid: BinaryInflatedOccupancyGrid,
        avoid_list: t.Set[GridCellModel],
        prev_list: t.Set[int],
        g_function: t.Callable[[RCHConfiguration, RCHConfiguration, bool], float],
        traversed_obstacles_ids: utils.OrderedSet,
        forbidden_obstacles: t.Set[int],
        ros_publisher: RosPublisher,
        neighborhood: t.Sequence[GridCellModel] = utils.TAXI_NEIGHBORHOOD,
    ) -> t.Tuple[t.List[RCHConfiguration], t.List[float]]:
        """
        Combined formulation from Stilman's thesis and his article.
        """
        neighbors, tentative_gscores = [], []
        current_gscore = gscore[current]
        path_has_traversed_first_disconnected_comp = current.first_component_uid != 0
        path_has_traversed_first_obstacle = current.first_obstacle_uid != 0

        # Filter out cells that are not in the map, and in static obstacles
        candidate_neighbor_cells = utils.get_neighbors_no_coll(
            current.cell,
            static_obs_grid.grid,
            static_obs_grid.d_width,
            static_obs_grid.d_height,
            neighborhood,
        )

        for neighbor_cell in candidate_neighbor_cells:
            neighbor = None
            if path_has_traversed_first_disconnected_comp:
                # Note: This validation was added according to the description in the article about not allowing
                # transitions between two different obstacles or to a cell with several obstacles, though it was not
                # explicit in the pseudocode formulation in Stilman's thesis.
                cur_cell_obs_uid = inflated_robot_grid.cell_to_obstacle_id(current.cell)
                neighbor_cell_obs_uid = inflated_robot_grid.cell_to_obstacle_id(
                    neighbor_cell
                )

                cur_and_neighbor_not_in_mult_obs = (
                    cur_cell_obs_uid != -1 and neighbor_cell_obs_uid != -1
                )
                current_or_neighbor_in_free_space = (
                    cur_cell_obs_uid == 0 or neighbor_cell_obs_uid == 0
                )
                transition_is_valid = (
                    cur_and_neighbor_not_in_mult_obs
                    and (
                        current_or_neighbor_in_free_space
                        or cur_cell_obs_uid == neighbor_cell_obs_uid
                    )
                    and neighbor_cell_obs_uid != current.first_obstacle_uid
                )
                if transition_is_valid:
                    neighbor = RCHConfiguration(
                        neighbor_cell,
                        current.first_obstacle_uid,
                        current.first_component_uid,
                    )
            else:
                neighbor_cell_component_uid: int = t.cast(
                    int, connected_components_grid[neighbor_cell[0]][neighbor_cell[1]]
                )

                neighbor_cell_in_free_space = (
                    inflated_robot_grid.grid[neighbor_cell[0]][neighbor_cell[1]] == 0
                )

                if path_has_traversed_first_obstacle:
                    if neighbor_cell_in_free_space:
                        neighbor_cell_not_in_prev_component_nor_avoid_list_nor_in_init_obstacle = (
                            neighbor_cell_component_uid not in prev_list
                            and (
                                current.first_obstacle_uid,
                                neighbor_cell_component_uid,
                            )
                            not in avoid_list
                            and neighbor_cell_component_uid != 0
                        )
                        if neighbor_cell_not_in_prev_component_nor_avoid_list_nor_in_init_obstacle:
                            neighbor = RCHConfiguration(
                                cell=neighbor_cell,
                                first_obstacle_uid=current.first_obstacle_uid,
                                first_component_uid=neighbor_cell_component_uid,
                            )
                        else:
                            # Either the neighbor tries to go back to robot acc. space, or in a (obs., comp.)
                            # combination that has already been explored and for which no manip. could be found
                            pass

                    else:
                        neighbor_cell_obs_uid = inflated_robot_grid.cell_to_obstacle_id(
                            neighbor_cell
                        )
                        if neighbor_cell_obs_uid == current.first_obstacle_uid:
                            neighbor = RCHConfiguration(
                                neighbor_cell, current.first_obstacle_uid, 0
                            )
                        else:
                            # Either the neighbor is in another obstacle, or in multiple, which is forbidden
                            pass
                else:
                    if neighbor_cell_in_free_space:
                        # If no obstacle has been traversed, we are still in the robot acc. space
                        neighbor = RCHConfiguration(neighbor_cell, 0, 0)
                    else:
                        neighbor_cell_obstacle_uid = (
                            inflated_robot_grid.cell_to_obstacle_id(neighbor_cell)
                        )
                        if neighbor_cell_obstacle_uid > 0:
                            neighbor = RCHConfiguration(
                                neighbor_cell, neighbor_cell_obstacle_uid, 0
                            )
                        else:
                            # The neighbor is in multiple obstacles, which is forbidden
                            pass
            if (
                neighbor is not None
                and neighbor not in close_set
                and neighbor.first_obstacle_uid not in forbidden_obstacles
            ):
                neighbors.append(neighbor)
                tentative_gscores.append(
                    current_gscore
                    + g_function(
                        current,
                        neighbor,
                        inflated_robot_grid.grid[neighbor.cell[0]][neighbor.cell[1]]
                        > 0,
                    )
                )
                traversed_obstacles_ids.add(neighbor.first_obstacle_uid)

        ros_publisher.publish_rch_data(
            current=current,
            came_from=came_from,
            neighbors=neighbors,
            traversed_obstacles_ids=traversed_obstacles_ids,
            res=inflated_robot_grid.res,
            grid_pose=inflated_robot_grid.grid_pose,
            ns=self._robot_name,
        )

        return neighbors, tentative_gscores

    def manip_search(
        self,
        w_t: World,
        o_1: int,
        c_1: int,
        ccs_data: connectivity.CCSData,
        r_acc_cells: t.Set[GridCellModel],
        r_f: PoseModel,
        inflated_grid_by_robot_max: BinaryInflatedOccupancyGrid,
        trans_mult: float,
        rot_mult: float,
        ros_publisher: RosPublisher,
        check_new_local_opening_before_global: bool = True,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ) -> t.Tuple[World, TransferPath | None]:
        # Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = copy.deepcopy(w_t)

        ros_publisher.publish_robot_sim_world(w_t_plus_2, self._robot_uid)

        c_1_cells_set = set() if c_1 == 0 else ccs_data.ccs[c_1].visited

        res = w_t_plus_2.discretization_data.res

        other_entities = [
            entity
            for entity in w_t_plus_2.entities.values()
            if entity.uid != self._robot.uid and entity.uid != o_1
        ]
        other_entities_polygons = {
            entity.uid: entity.polygon for entity in other_entities
        }
        other_entities_aabb_tree = collision.polygons_to_aabb_tree(
            other_entities_polygons
        )

        robot = w_t_plus_2.entities[self._robot.uid]
        robot_uid, robot_pose, robot_polygon, robot_name = (
            robot.uid,
            robot.pose,
            robot.polygon,
            robot.name,
        )

        robot_min_inflation_radius = utils.get_inscribed_radius(robot_polygon)
        # robot_max_inflation_radius = utils.get_circumscribed_radius(robot_polygon)

        obstacle = w_t_plus_2.entities[o_1]
        obstacle_uid, obstacle_pose, obstacle_polygon = (
            obstacle.uid,
            obstacle.pose,
            obstacle.polygon,
        )
        obstacle_min_inflation_radius = utils.get_inscribed_radius(obstacle_polygon)

        inf_robot, inf_obstacle = copy.deepcopy(robot), copy.deepcopy(obstacle)
        inf_robot.polygon, inf_obstacle.polygon = (
            robot.polygon.buffer(res, join_style="mitre"),
            obstacle.polygon.buffer(res, join_style="mitre"),
        )

        goal_pose, goal_cell = (
            r_f,
            utils.real_to_grid(
                r_f[0], r_f[1], res, inflated_grid_by_robot_max.grid_pose
            ),
        )

        # Get accessible sampled navigation points around obstacle
        (
            transfer_start_configs_to_cost,
            transfer_start_to_prev_transit_end,
        ) = self.get_transfer_start_to_transit_end_and_cost(
            robot_polygon,
            robot_pose,
            robot_uid,
            obstacle_uid,
            other_entities_polygons,
            other_entities_aabb_tree,
            inflated_grid_by_robot_max,
            ccs_data,
            r_acc_cells,
            obstacle_pose,
            obstacle_polygon,
            trans_mult,
            rot_mult,
            ros_publisher=ros_publisher,
        )

        if not transfer_start_configs_to_cost:
            # If there are no attainable manipulation configurations, exit early
            ros_publisher.cleanup_q_manips_for_obs(ns=self._robot_name)
            return w_t_plus_2, None

        # CAREFUL : We inflate by inscribed radius MINUS sqrt(2)*res to make sure occupied cells are really where the
        # entity's center should NEVER be to avoid collisions.
        # Poses in free cells of this grid may sometimes be colliding.
        inflated_grid_by_robot_min = BinaryInflatedOccupancyGrid(
            other_entities_polygons,
            res,
            max(robot_min_inflation_radius - utils.SQRT_OF_2 * res, 0.0),
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
        )
        inflated_grid_by_obstacle = BinaryInflatedOccupancyGrid(
            other_entities_polygons,
            res,
            obstacle_min_inflation_radius - utils.SQRT_OF_2 * res,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
            params=inflated_grid_by_robot_max.params,
        )
        # Only deactivate obstacle cells once transit end and transfer start are computed (grab action)
        inflated_grid_by_robot_max.deactivate_entities([obstacle_uid])

        # Use Dijkstra algorithm to compute a transfer path that allows for an opening to be created
        (
            path_found,
            transfer_end_configuration,
            came_from,
            _close_set,
            gscore,
            _,
        ) = self.dijkstra_for_manip_search(
            transfer_start_configs_to_cost,
            robot_uid,
            robot_name,
            obstacle_uid,
            obstacle_polygon,
            other_entities_polygons,
            other_entities_aabb_tree,
            inflated_grid_by_robot_min,
            inflated_grid_by_robot_max,
            inflated_grid_by_obstacle,
            r_acc_cells,
            c_1_cells_set,
            ccs_data,
            trans_mult,
            rot_mult,
            check_new_local_opening_before_global,
            goal_pose,
            goal_cell,
            ros_publisher,
            obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
            obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
        )

        tho_m: TransferPath | None = None

        if path_found:
            # ros_publisher.publish_sim(
            #     transfer_end_configuration.robot.polygon, transfer_end_configuration.obstacle.polygon,
            #     "/target", ns=self._robot_name
            # )
            raw_path: t.List[
                RobotObstacleConfiguration
            ] = graph_search.reconstruct_path(came_from, transfer_end_configuration)  # type: ignore

            prev_transit_end_configuration: RCHConfiguration = (
                transfer_start_to_prev_transit_end[raw_path[0]]
            )
            next_transit_start_configuration = (
                self.get_next_transit_start_configuration(
                    inflated_grid_by_robot_max,
                    raw_path[-1].robot.floating_point_pose,
                    raw_path[-1].robot.polygon,
                    robot_uid,
                    obstacle_uid,
                    raw_path[-1].obstacle.floating_point_pose,
                    other_entities_polygons,
                    other_entities_aabb_tree,
                    trans_mult,
                    rot_mult,
                )
            )
            tho_m_phys_cost = gscore[transfer_end_configuration] + self.g(
                transfer_end_configuration.robot.floating_point_pose,
                next_transit_start_configuration.floating_point_pose,
                is_transfer=True,
            )
            tho_m = self.get_transfer_path_from_config(
                prev_transit_end_configuration,
                next_transit_start_configuration,
                raw_path,
                obstacle_uid,
                tho_m_phys_cost,
            )
        else:
            # If after exhausting all possible configurations, none opens a path to the connected component,
            # return None
            tho_m = None

        # Don't forget to update w_t_plus_2 with transfer end state
        if tho_m:
            robot.pose, robot.polygon = (
                tho_m.robot_path.poses[-1],
                tho_m.robot_path.polygons[-1],
            )
            obstacle.pose, obstacle.polygon = (
                tho_m.obstacle_path.poses[-1],
                tho_m.obstacle_path.polygons[-1],
            )

        ros_publisher.publish_robot_sim_world(w_t_plus_2, self._robot_uid)
        ros_publisher.cleanup_robot_sim(ns=self._robot_name)
        ros_publisher.cleanup_q_manips_for_obs(ns=self._robot_name)

        inflated_grid_by_robot_max.activate_entities([obstacle_uid])

        return w_t_plus_2, tho_m

    def get_transfer_start_to_transit_end_and_cost(
        self,
        robot_polygon: Polygon,
        robot_pose: PoseModel,
        robot_uid: int,
        obstacle_uid: int,
        other_entities_polygons: t.Dict[int, Polygon],
        other_entities_aabb_tree: AABBTree,
        inflated_grid_by_robot_max: BinaryInflatedOccupancyGrid,
        ccs_data: connectivity.CCSData,
        r_acc_cells: t.Set[GridCellModel],
        obstacle_pose: PoseModel,
        obstacle_polygon: Polygon,
        trans_mult: float,
        rot_mult: float,
        ros_publisher: RosPublisher,
    ) -> t.Tuple[
        t.Dict[RobotObstacleConfiguration, float],
        t.Dict[RobotObstacleConfiguration, BaseConfiguration | None],
    ]:
        transfer_start_configs_to_cost: t.Dict[RobotObstacleConfiguration, float]
        transfer_start_to_prev_transit_end: t.Dict[
            RobotObstacleConfiguration, BaseConfiguration | None
        ]

        robot_cell = utils.real_to_grid(
            robot_pose[0],
            robot_pose[1],
            inflated_grid_by_robot_max.res,
            inflated_grid_by_robot_max.grid_pose,
        )
        cell_in_manip_obs = (
            inflated_grid_by_robot_max.cell_to_obstacle_id(robot_cell) == obstacle_uid
        )

        if cell_in_manip_obs:
            # If we are in the case where the robot starts from within the inflation of the manipulated obstacle,
            # exit early with only the start transfer configuration
            transfer_start_configuration = RobotObstacleConfiguration(
                robot_floating_point_pose=robot_pose,
                robot_polygon=robot_polygon,
                robot_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                    robot_pose, trans_mult, rot_mult
                ),
                robot_cell_in_grid=utils.real_to_grid(
                    robot_pose[0],
                    robot_pose[1],
                    inflated_grid_by_robot_max.res,
                    inflated_grid_by_robot_max.grid_pose,
                ),
                obstacle_floating_point_pose=obstacle_pose,
                obstacle_polygon=obstacle_polygon,
                obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                    obstacle_pose, trans_mult, rot_mult
                ),
                obstacle_cell_in_grid=utils.real_to_grid(
                    obstacle_pose[0],
                    obstacle_pose[1],
                    inflated_grid_by_robot_max.res,
                    inflated_grid_by_robot_max.grid_pose,
                ),
                manip_pose_id=0,
            )

            transfer_start_configs_to_cost = {transfer_start_configuration: 0.0}
            transfer_start_to_prev_transit_end = {transfer_start_configuration: None}

            return transfer_start_configs_to_cost, transfer_start_to_prev_transit_end

        # General case otherwise
        (
            transit_end_robot_poses,
            transfer_start_robot_poses,
        ) = self.get_transit_end_and_transfer_start_poses(
            obstacle_polygon, inflated_grid_by_robot_max, ros_publisher=ros_publisher
        )

        transfer_start_to_transit_end_robot_pose = {
            manip_pose: nav_pose
            for nav_pose, manip_pose in zip(
                transit_end_robot_poses, transfer_start_robot_poses
            )
        }

        transfer_start_configs_to_cost = {}
        transfer_start_to_prev_transit_end = {}
        for manip_pose_id, (transfer_start_pose, transit_end_pose) in enumerate(
            transfer_start_to_transit_end_robot_pose.items()
        ):
            transit_end_cell = utils.real_to_grid(
                transit_end_pose[0],
                transit_end_pose[1],
                inflated_grid_by_robot_max.res,
                inflated_grid_by_robot_max.grid_pose,
            )

            if transit_end_cell not in r_acc_cells:
                continue

            prev_transit_end_robot_polygon = utils.set_polygon_pose(
                robot_polygon, robot_pose, transit_end_pose
            )

            grab_action = ba.Grab(
                translation_vector=(
                    utils.euclidean_distance(transfer_start_pose, transit_end_pose),
                    0.0,
                ),
                entity_uid=obstacle_uid,
            )
            transfer_start_robot_polygon = grab_action.apply(
                prev_transit_end_robot_polygon, transit_end_pose
            )

            (
                _,
                collides_with,
                _,
                csv_polygons,
                _,
                bb_vertices,
            ) = collision.csv_check_collisions(
                main_uid=robot_uid,
                other_polygons=other_entities_polygons,
                polygon_sequence=[
                    prev_transit_end_robot_polygon,
                    transfer_start_robot_polygon,
                ],
                action_sequence=[
                    collision.convert_action(grab_action, transit_end_pose)
                ],
                bb_type="minimum_rotated_rectangle",
                aabb_tree=other_entities_aabb_tree,
            )

            if not collides_with:
                prev_transit_end_configuration = RobotConfiguration(
                    floating_point_pose=transit_end_pose,
                    polygon=prev_transit_end_robot_polygon,
                    cell_in_grid=utils.real_to_grid(
                        transit_end_pose[0],
                        transit_end_pose[1],
                        inflated_grid_by_robot_max.res,
                        inflated_grid_by_robot_max.grid_pose,
                    ),
                    fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                        transit_end_pose, trans_mult, rot_mult
                    ),
                    action=None,
                    csv_polygon=prev_transit_end_robot_polygon,
                )
                temp_transfer_start_configuration = RobotObstacleConfiguration(
                    robot_floating_point_pose=transfer_start_pose,
                    robot_polygon=utils.set_polygon_pose(
                        robot_polygon, robot_pose, transfer_start_pose
                    ),
                    robot_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                        transfer_start_pose, trans_mult, rot_mult
                    ),
                    robot_cell_in_grid=utils.real_to_grid(
                        transfer_start_pose[0],
                        transfer_start_pose[1],
                        inflated_grid_by_robot_max.res,
                        inflated_grid_by_robot_max.grid_pose,
                    ),
                    obstacle_floating_point_pose=obstacle_pose,
                    obstacle_polygon=obstacle_polygon,
                    obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                        obstacle_pose, trans_mult, rot_mult
                    ),
                    obstacle_cell_in_grid=utils.real_to_grid(
                        obstacle_pose[0],
                        obstacle_pose[1],
                        inflated_grid_by_robot_max.res,
                        inflated_grid_by_robot_max.grid_pose,
                    ),
                    manip_pose_id=manip_pose_id,
                    action=grab_action,
                    robot_csv_polygon=csv_polygons[(0,)],
                    obstacle_csv_polygon=obstacle_polygon,
                )
                transfer_start_configs_to_cost[
                    temp_transfer_start_configuration
                ] = self.g(transit_end_pose, transfer_start_pose, is_transfer=True)
                transfer_start_to_prev_transit_end[
                    temp_transfer_start_configuration
                ] = prev_transit_end_configuration

        return transfer_start_configs_to_cost, transfer_start_to_prev_transit_end

    def dijkstra_for_manip_search(
        self,
        start: t.Dict[RobotObstacleConfiguration, float],
        robot_uid: int,
        robot_name: str,
        obstacle_uid: int,
        obstacle_polygon: Polygon,
        other_entities_polygons: t.Dict[int, Polygon],
        other_entities_aabb_tree: AABBTree,
        inflated_grid_by_robot_min: BinaryInflatedOccupancyGrid,
        inflated_grid_by_robot_max: BinaryInflatedOccupancyGrid,
        inflated_grid_by_obstacle: BinaryInflatedOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        c_1_cells_set: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        trans_mult: float,
        rot_mult: float,
        check_new_local_opening_before_global: bool,
        overall_goal_pose: PoseModel,
        overall_goal_cell: GridCellModel,
        ros_publisher: RosPublisher,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ):
        def get_neighbors(_current, _gscore, _close_set, _open_queue, _came_from):
            return self.get_neighbors(
                _current,
                _gscore,
                _close_set,
                _open_queue,
                _came_from,
                start,
                inflated_grid_by_robot_min,
                inflated_grid_by_robot_max,
                inflated_grid_by_obstacle,
                r_acc_cells,
                ccs_data,
                robot_uid,
                obstacle_uid,
                trans_mult,
                rot_mult,
                other_entities_polygons,
                other_entities_aabb_tree,
                ros_publisher,
                obstacle_can_intrude_r_acc=obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x=obstacle_can_intrude_c_1_x,
            )

        def exit_condition(_current):
            next_transit_start_configuration = (
                self.get_next_transit_start_configuration(
                    inflated_grid_by_robot_max,
                    _current.robot.floating_point_pose,
                    _current.robot.polygon,
                    robot_uid,
                    obstacle_uid,
                    _current.obstacle.floating_point_pose,
                    other_entities_polygons,
                    other_entities_aabb_tree,
                    trans_mult,
                    rot_mult,
                )
            )
            if next_transit_start_configuration:
                #   3. ... and creates a global opening to c1
                has_new_global_opening, _, _ = self.is_there_opening_to_c_1(
                    check_new_local_opening_before_global=check_new_local_opening_before_global,
                    robot_name=robot_name,
                    robot_cell=next_transit_start_configuration.cell_in_grid,
                    obstacle_uid=obstacle_uid,
                    old_obstacle_polygon=obstacle_polygon,
                    new_obstacle_polygon=_current.obstacle.polygon,
                    other_entities_polygons=other_entities_polygons,
                    other_entities_aabb_tree=other_entities_aabb_tree,
                    inflated_grid_by_robot_max=inflated_grid_by_robot_max,
                    c_1_cells_set=c_1_cells_set,
                    goal_pose=overall_goal_pose,
                    goal_cell=overall_goal_cell,
                    ros_publisher=ros_publisher,
                    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                    init_blocking_areas=None,
                    init_entity_inflated_polygon=None,
                )
                if has_new_global_opening:
                    return True
            return False

        return graph_search.new_generic_dijkstra(
            start, exit_condition=exit_condition, get_neighbors=get_neighbors
        )

    def is_there_opening_to_c_1(
        self,
        check_new_local_opening_before_global: bool,
        robot_name: str,
        robot_cell: GridCellModel,
        obstacle_uid: int,
        old_obstacle_polygon: Polygon,
        new_obstacle_polygon: Polygon,
        other_entities_polygons,
        other_entities_aabb_tree,
        inflated_grid_by_robot_max,
        c_1_cells_set,
        goal_pose: PoseModel,
        goal_cell: GridCellModel,
        ros_publisher: RosPublisher,
        neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
        init_blocking_areas: t.List[Polygon] | None = None,
        init_entity_inflated_polygon: Polygon | None = None,
    ):
        """
        Checks if there is a path between robot_cell and a random cell in c_1_cells_set that is not covered by an
        obstacle (especially the one considered for manipulation).
        :return: True if a path is found, False otherwise
        TODO: Add proper return of init_blocking_areas and init_entity_inflated_polygon and save them in caller methods
        """
        if check_new_local_opening_before_global:
            (
                has_new_local_opening,
                init_blocking_areas,
                init_entity_inflated_polygon,
            ) = check_new_local_opening(
                init_entity_polygon=old_obstacle_polygon,
                target_entity_polygon=new_obstacle_polygon,
                other_entities_polygons=other_entities_polygons,
                other_entities_aabb_tree=other_entities_aabb_tree,
                inflation_radius=inflated_grid_by_robot_max.inflation_radius,
                goal_pose=goal_pose,
                ros_publisher=ros_publisher,
                init_blocking_areas=init_blocking_areas,
                init_entity_inflated_polygon=init_entity_inflated_polygon,
                ns=robot_name,
            )
        else:
            has_new_local_opening = True

        if has_new_local_opening:
            obstacle_initially_deactivated = (
                obstacle_uid
                in inflated_grid_by_robot_max.deactivated_entities_cells_sets
            )
            if obstacle_initially_deactivated:
                inflated_grid_by_robot_max.activate_entities({obstacle_uid})
            previous_cells_sets = inflated_grid_by_robot_max.update(
                new_or_updated_polygons={obstacle_uid: new_obstacle_polygon}
            )

            if not c_1_cells_set or (c_1_cells_set and goal_cell in c_1_cells_set):
                cell_in_c_1 = goal_cell
            else:
                c_1_cells_set_iterator = iter(c_1_cells_set)
                cell_in_c_1 = next(c_1_cells_set_iterator)
                while (
                    inflated_grid_by_robot_max.grid[cell_in_c_1[0]][cell_in_c_1[1]] != 0
                ):
                    # While selected cell not in free space after manipulation, try another cell
                    try:
                        cell_in_c_1 = next(c_1_cells_set_iterator)
                    except StopIteration:
                        # Note: using the the exception detection is the pythonic way it seems (no has_next)
                        # No opening because c_1_cells_set is entirely inaccessible to the robot after manipulation
                        inflated_grid_by_robot_max.cells_sets_update(
                            new_or_updated_cells_sets=previous_cells_sets
                        )
                        if obstacle_initially_deactivated:
                            inflated_grid_by_robot_max.deactivate_entities(
                                {obstacle_uid}
                            )
                        has_new_global_opening, skipped_global_opening_check = (
                            False,
                            False,
                        )
                        return (
                            has_new_global_opening,
                            has_new_local_opening,
                            skipped_global_opening_check,
                        )

            # TODO Evaluate the performance change (particularly compared to Dijkstra search) if A* star had an
            #  unadmissible heuristic to hasten path discovery (or write Best-FS based solely on heuristic)
            has_new_global_opening, _, _, _, _, _ = graph_search.grid_search_a_star(
                robot_cell,
                cell_in_c_1,
                inflated_grid_by_robot_max.grid,
                inflated_grid_by_robot_max.d_width,
                inflated_grid_by_robot_max.d_height,
                neighborhood,
                check_diag_neighbors=False,
            )

            inflated_grid_by_robot_max.cells_sets_update(
                new_or_updated_cells_sets=previous_cells_sets
            )
            if obstacle_initially_deactivated:
                inflated_grid_by_robot_max.deactivate_entities({obstacle_uid})
            skipped_global_opening_check = False

            return (
                has_new_global_opening,
                has_new_local_opening,
                skipped_global_opening_check,
            )
        else:
            has_new_global_opening, skipped_global_opening_check = False, True
            return (
                has_new_global_opening,
                has_new_local_opening,
                skipped_global_opening_check,
            )

    def h(self, r_i: t.Tuple[float, float, float], r_j: t.Tuple[float, float, float]):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        return translation_cost

    def g(
        self,
        r_i: t.Tuple[float, float, float],
        r_j: t.Tuple[float, float, float],
        is_transfer: bool = False,
    ):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        rotation_cost = self.rotation_factor * abs(r_j[2] - r_i[2])
        return (translation_cost + rotation_cost) * (
            1.0 if not is_transfer else self.transfer_coefficient
        )

    @staticmethod
    def get_next_transit_start_configuration(
        grid: BinaryInflatedOccupancyGrid,
        robot_pose: PoseModel,
        robot_polygon: Polygon,
        robot_uid: int,
        obstacle_uid: int,
        obstacle_pose: PoseModel,
        other_entities_polygons: t.Dict[int, Polygon],
        other_entities_aabb_tree: AABBTree,
        trans_mult: float,
        rot_mult: float,
    ):
        release_action = ba.Release(
            translation_vector=(-1.0 * (grid.inflation_radius + 1.5 * grid.res), 0.0),
            entity_uid=obstacle_uid,
        )
        new_robot_pose = release_action.predict_pose(robot_pose, robot_pose[2])

        cell = utils.real_to_grid(
            new_robot_pose[0], new_robot_pose[1], grid.res, grid.grid_pose
        )

        if utils.is_in_matrix(cell, grid.d_width, grid.d_height):
            if grid.grid[cell[0]][cell[1]] > 0:
                # If the robot cell after release is in an obstacle in the grid, return False
                return None
        else:
            # If robot cell outside of grid, return False
            return None

        new_robot_polygon = release_action.apply(robot_polygon, robot_pose)

        # Check if robot is still within map bounds
        if not new_robot_polygon.within(grid.aabb_polygon):
            return None

        # Finally, we check dynamic collisions (between init configuration and after-action configuration)
        (
            _,
            collides_with,
            _,
            csv_polygons,
            intersections,
            bb_vertices,
        ) = collision.csv_check_collisions(
            main_uid=robot_uid,
            other_polygons=other_entities_polygons,
            polygon_sequence=[robot_polygon, new_robot_polygon],
            action_sequence=[collision.convert_action(release_action, robot_pose)],
            bb_type="minimum_rotated_rectangle",
            aabb_tree=other_entities_aabb_tree,
        )

        if not collides_with:
            new_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
                new_robot_pose, trans_mult, rot_mult
            )
            next_transit_start_configuration = RobotConfiguration(
                floating_point_pose=new_robot_pose,
                polygon=new_robot_polygon,
                cell_in_grid=cell,
                fixed_precision_pose=new_fixed_precision_pose,
                action=release_action,
                csv_polygon=csv_polygons[(0,)],
            )
            return next_transit_start_configuration
        else:
            return None

    def get_transfer_path_from_config(
        self,
        prev_transit_end_configuration: RobotConfiguration,
        next_transit_start_configuration: RobotConfiguration,
        transfer_configurations: t.List[RobotObstacleConfiguration],
        obstacle_uid: int,
        phys_cost: t.Optional[float] = None,
        social_cost: float = 0.0,
        weight: float = 1.0,
    ) -> TransferPath | None:
        if len(transfer_configurations) == 0:
            return None

        manip_pose_id: int = transfer_configurations[0].manip_pose_id  # type: ignore

        actions = [
            configuration.action
            for configuration in transfer_configurations
            if configuration.action
        ]
        grab_action: ba.Grab = actions[0] if prev_transit_end_configuration else None  # type: ignore
        release_action: ba.Release = next_transit_start_configuration.action
        actions.append(release_action)

        robot_poses = [
            configuration.robot.floating_point_pose
            for configuration in transfer_configurations
        ]
        robot_poses.append(next_transit_start_configuration.floating_point_pose)
        robot_polygons = [
            configuration.robot.polygon for configuration in transfer_configurations
        ]
        robot_polygons.append(next_transit_start_configuration.polygon)
        robot_csv_polygons = {
            (i + 1,): config.robot.csv_polygon
            for i, config in enumerate(transfer_configurations)
        }
        robot_csv_polygons[
            (len(transfer_configurations),)
        ] = next_transit_start_configuration.csv_polygon
        robot_bb_vertices = [
            config.robot.bb_vertices
            for config in transfer_configurations
            if config.robot.bb_vertices
        ]
        robot_bb_vertices.append(next_transit_start_configuration.bb_vertices)
        if prev_transit_end_configuration:
            robot_poses.insert(0, prev_transit_end_configuration.floating_point_pose)
            robot_polygons.insert(0, prev_transit_end_configuration.polygon)
            robot_csv_polygons[(0,)] = prev_transit_end_configuration.csv_polygon
            robot_bb_vertices.insert(0, prev_transit_end_configuration.bb_vertices)

        robot_path = Path(
            poses=robot_poses,
            polygons=robot_polygons,
            csv_polygons=robot_csv_polygons,
            bb_vertices=robot_bb_vertices,
        )

        obstacle_path = Path(
            poses=[
                configuration.obstacle.floating_point_pose
                for configuration in transfer_configurations
            ],
            polygons=[
                configuration.obstacle.polygon
                for configuration in transfer_configurations
            ],
            csv_polygons={
                (i + 1,): config.obstacle.csv_polygon
                for i, config in enumerate(transfer_configurations)
            },
            bb_vertices=[
                config.obstacle.bb_vertices for config in transfer_configurations
            ],
        )
        obstacle_path.poses.append(obstacle_path.poses[-1])
        obstacle_path.polygons.append(obstacle_path.polygons[-1])
        obstacle_path.bb_vertices.append([])
        if prev_transit_end_configuration:
            obstacle_path.poses.insert(0, obstacle_path.poses[0])
            obstacle_path.polygons.insert(0, obstacle_path.polygons[0])
            obstacle_path.bb_vertices.insert(0, [])

        return TransferPath(
            robot_path=robot_path,
            obstacle_path=obstacle_path,
            actions=actions,
            grab_action=grab_action,
            release_action=release_action,
            obstacle_uid=obstacle_uid,
            manip_pose_id=manip_pose_id,
            phys_cost=phys_cost,
            social_cost=social_cost,
            weight=weight,
        )

    def get_neighbors(
        self,
        current_configuration: RobotObstacleConfiguration,
        gscore: t.Dict[RobotObstacleConfiguration, float],
        close_set: t.Set[RobotObstacleConfiguration],
        open_queue: graph_search.PriorityQueue,
        came_from: t.Dict[RobotObstacleConfiguration, RobotObstacleConfiguration],
        start: t.Dict[RobotObstacleConfiguration, float],
        inflated_grid_by_robot_min: BinaryInflatedOccupancyGrid,
        inflated_grid_by_robot_max: BinaryInflatedOccupancyGrid,
        inflated_grid_by_obstacle: BinaryInflatedOccupancyGrid,
        r_acc_cells: t.Set[GridCellModel],
        ccs_data: connectivity.CCSData,
        robot_uid: int,
        obstacle_uid: int,
        trans_mult: float,
        rot_mult: float,
        other_entities_polygons: t.Dict[int, Polygon],
        other_entities_aabb_tree: AABBTree,
        ros_publisher: RosPublisher,
        obstacle_can_intrude_r_acc: bool = True,
        obstacle_can_intrude_c_1_x: bool = True,
    ) -> t.List[RobotObstacleConfiguration]:
        """
        Creates list of neighbors that are not in close set, do not collide dynamically nor statically
        """
        # TODO Add debug display option for intersections, be it on grid(s) or in between polygons
        neighbors: t.List[RobotObstacleConfiguration] = []
        tentative_g_scores = []

        for action in self._new_actions:
            if isinstance(action, ba.Rotation):
                neighbor_action_opposes_prev_action = (
                    isinstance(current_configuration.action, ba.Rotation)
                    and action.angle == -1.0 * current_configuration.action.angle
                )
                if neighbor_action_opposes_prev_action:
                    continue

                robot_center = (
                    current_configuration.robot.floating_point_pose[0],
                    current_configuration.robot.floating_point_pose[1],
                )
                new_robot_pose = action.predict_pose(
                    current_configuration.robot.floating_point_pose, robot_center
                )
                new_obstacle_pose = action.predict_pose(
                    current_configuration.obstacle.floating_point_pose, robot_center
                )
                extra_g_cost = self.rotation_unit_cost
            elif isinstance(action, ba.Translation):
                neighbor_action_opposes_prev_action = (
                    isinstance(current_configuration.action, ba.Translation)
                    and action.translation_vector[0]
                    == -1.0 * current_configuration.action.translation_vector[0]
                    and action.translation_vector[1]
                    == -1.0 * current_configuration.action.translation_vector[1]
                )
                if neighbor_action_opposes_prev_action:
                    continue

                new_robot_pose = action.predict_pose(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.robot.floating_point_pose[2],
                )
                new_obstacle_pose = action.predict_pose(
                    current_configuration.obstacle.floating_point_pose,
                    current_configuration.robot.floating_point_pose[2],
                )
                extra_g_cost = self.translation_unit_cost
            else:
                raise TypeError(
                    "action must either be of type NewRotation or NewTranslation"
                )

            # First, check whether the new configuration is in close set, if it is, ignore it
            robot_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
                new_robot_pose, trans_mult, rot_mult
            )
            obstacle_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(
                new_obstacle_pose, trans_mult, rot_mult
            )

            if (robot_fixed_precision_pose, obstacle_fixed_precision_pose) in close_set:
                continue

            # Then check for collisions, starting at a grid level
            robot_cell_in_grid = utils.real_to_grid(
                new_robot_pose[0],
                new_robot_pose[1],
                inflated_grid_by_robot_min.res,
                inflated_grid_by_robot_min.grid_pose,
            )
            obstacle_cell_in_grid = utils.real_to_grid(
                new_obstacle_pose[0],
                new_obstacle_pose[1],
                inflated_grid_by_obstacle.res,
                inflated_grid_by_obstacle.grid_pose,
            )

            is_no_longer_in_grid = not (
                utils.is_in_matrix(
                    robot_cell_in_grid,
                    inflated_grid_by_robot_min.d_width,
                    inflated_grid_by_robot_min.d_height,
                )
                and utils.is_in_matrix(
                    obstacle_cell_in_grid,
                    inflated_grid_by_obstacle.d_width,
                    inflated_grid_by_obstacle.d_height,
                )
            )
            if is_no_longer_in_grid:
                continue
            if (
                inflated_grid_by_robot_min.grid[robot_cell_in_grid[0]][
                    robot_cell_in_grid[1]
                ]
                != 0
            ):
                continue
            if (
                inflated_grid_by_obstacle.grid[obstacle_cell_in_grid[0]][
                    obstacle_cell_in_grid[1]
                ]
                != 0
            ):
                continue

            # Continue at static polygon level, check if still in map
            new_robot_polygon = action.apply(
                current_configuration.robot.polygon,
                current_configuration.robot.floating_point_pose,
            )

            # Check if robot is still within map bounds
            if not new_robot_polygon.within(inflated_grid_by_robot_min.aabb_polygon):
                continue

            new_obstacle_polygon = action.apply(
                current_configuration.obstacle.polygon,
                current_configuration.robot.floating_point_pose,
            )

            # Check if obstacle is still within map bounds
            if not new_obstacle_polygon.within(inflated_grid_by_obstacle.aabb_polygon):
                continue

            # Finally, we check dynamic collisions (between init configuration and after-action configuration)
            (
                _,
                collides_with,
                _,
                robot_csv_polygons,
                _,
                robot_bb_vertices,
            ) = collision.csv_check_collisions(
                main_uid=robot_uid,
                other_polygons=other_entities_polygons,
                polygon_sequence=[
                    current_configuration.robot.polygon,
                    new_robot_polygon,
                ],
                action_sequence=[
                    collision.convert_action(
                        action, current_configuration.robot.floating_point_pose
                    )
                ],
                bb_type="minimum_rotated_rectangle",
                aabb_tree=other_entities_aabb_tree,
            )
            if collides_with:
                continue
            # TODO Refactor collision.csv_check_collisions to check for any number of attached polygons or make new function
            (
                _,
                collides_with,
                _,
                obstacle_csv_polygons,
                _,
                obstacle_bb_vertices,
            ) = collision.csv_check_collisions(
                main_uid=obstacle_uid,
                other_polygons=other_entities_polygons,
                polygon_sequence=[
                    current_configuration.obstacle.polygon,
                    new_obstacle_polygon,
                ],
                action_sequence=[
                    collision.convert_action(
                        action, current_configuration.obstacle.floating_point_pose
                    )
                ],
                bb_type="minimum_rotated_rectangle",
                aabb_tree=other_entities_aabb_tree,
            )
            if collides_with:
                continue

            # If option is activated, check that obstacle intruded the appropriate component(s)
            intrudes = self.polygon_intrudes_components(
                new_obstacle_polygon,
                inflated_grid_by_robot_max,
                r_acc_cells,
                ccs_data,
                obstacle_can_intrude_r_acc,
                obstacle_can_intrude_c_1_x,
            )
            if intrudes:
                continue

            # If we are here, then this newly computed neighbor configuration is valid and we must save it
            neighbor_configuration = RobotObstacleConfiguration(
                robot_floating_point_pose=new_robot_pose,
                robot_polygon=new_robot_polygon,
                robot_fixed_precision_pose=robot_fixed_precision_pose,
                robot_cell_in_grid=robot_cell_in_grid,
                obstacle_floating_point_pose=new_obstacle_pose,
                obstacle_polygon=new_obstacle_polygon,
                obstacle_fixed_precision_pose=obstacle_fixed_precision_pose,
                obstacle_cell_in_grid=obstacle_cell_in_grid,
                action=action,
                manip_pose_id=current_configuration.manip_pose_id,
                robot_csv_polygon=robot_csv_polygons[(0,)],
                obstacle_csv_polygon=obstacle_csv_polygons[(0,)],
            )

            neighbors.append(neighbor_configuration)
            tentative_g_scores.append(gscore[current_configuration] + extra_g_cost)

        manip_poses_ids = [c.manip_pose_id for c in start.keys()]

        ros_publisher.publish_manip_search_data(
            current_manip_pose_id=current_configuration.manip_pose_id,  # type: ignore
            manip_poses_ids=manip_poses_ids,
            robot_pose=current_configuration.robot.floating_point_pose,
            robot_fixed_precision_pos=current_configuration.robot.fixed_precision_pose,
            robot_polygon=current_configuration.robot.polygon,
            obstacle_polygon=current_configuration.obstacle.polygon,
            obstacle_pose=current_configuration.obstacle.floating_point_pose,
            res=inflated_grid_by_robot_min.res,
            neighbor_poses=[n.robot.floating_point_pose for n in neighbors],
            ns=self._robot_name,
        )

        return neighbors, tentative_g_scores

    def get_transit_end_and_transfer_start_poses(
        self,
        obstacle_polygon: Polygon,
        inflated_grid_by_robot_max: BinaryInflatedOccupancyGrid,
        ros_publisher: RosPublisher,
    ) -> t.Tuple[t.List[PoseModel], t.List[PoseModel]]:
        """
        For the given obstacle polygon, computes the valid transit end poses and
        corresponding valid transfer start poses:
            - Transfer start poses are at a robot inflation radius distance from the sides, and facing their middle.
            - Transit end poses are a one and a half times the grid resolution away from the obstacle's sides, so that
                their corresponding cell is **always** outside of the inflated obstacle's cells set.
                They also have the same orientation as their corresponding transfer start pose, to make the
                initialization step of the transfer path as safe as possible (the robot only has to drive a bit forward
                to touch the obstacle's side).

        TODO Add two other sampling strategies:
            - points sampled along buffered polygon
            - points sampled along lines parallel to sides, s.t. we have at least a half robot width from endpoints
        :param obstacle_polygon:
        :type obstacle_polygon:
        :param inflated_grid_by_robot_max:
        :type inflated_grid_by_robot_max:
        :return: the lists of valid transit end poses and corresponding valid transfer start poses
        :rtype: tuple(list(tuple(float, float, float)), list(tuple(float, float, float)))
        """
        candidate_transfer_start_poses = utils.sample_poses_at_middle_of_inflated_sides(
            obstacle_polygon, inflated_grid_by_robot_max.inflation_radius
        )
        candidate_transit_end_poses = utils.sample_poses_at_middle_of_inflated_sides(
            obstacle_polygon,
            inflated_grid_by_robot_max.inflation_radius
            + 1.5 * inflated_grid_by_robot_max.res,
        )

        valid_transit_end_poses, valid_transfer_start_poses = [], []
        for transit_end_pose, transfer_start_pose in zip(
            candidate_transit_end_poses, candidate_transfer_start_poses
        ):
            valid_transit_end_poses.append(transit_end_pose)
            valid_transfer_start_poses.append(transfer_start_pose)

        ros_publisher.cleanup_q_manips_for_obs(ns=self._robot_name)
        ros_publisher.publish_q_manips_for_obs(
            valid_transfer_start_poses, ns=self._robot_name
        )

        return valid_transit_end_poses, valid_transfer_start_poses

    @staticmethod
    def polygon_intrudes_components(
        new_obstacle_polygon: Polygon,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        r_acc_cells,
        ccs_data,
        obstacle_can_intrude_r_acc,
        obstacle_can_intrude_c_1_x,
    ):
        if obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            return False
        elif obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
            new_obstacle_exterior_cells = utils.accurate_rasterize_in_grid(
                new_obstacle_polygon.buffer(inflated_grid_by_robot.inflation_radius),
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
                inflated_grid_by_robot.d_width,
                inflated_grid_by_robot.d_height,
                fill=False,
            )
            for cell in new_obstacle_exterior_cells:
                if ccs_data.grid[cell[0]][cell[1]] > 0 and cell not in r_acc_cells:
                    return True
        elif not obstacle_can_intrude_r_acc and obstacle_can_intrude_c_1_x:
            new_obstacle_exterior_cells = utils.accurate_rasterize_in_grid(
                new_obstacle_polygon.buffer(inflated_grid_by_robot.inflation_radius),
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
                inflated_grid_by_robot.d_width,
                inflated_grid_by_robot.d_height,
                fill=False,
            )
            for cell in new_obstacle_exterior_cells:
                if cell in r_acc_cells:
                    return True
        elif not obstacle_can_intrude_r_acc and not obstacle_can_intrude_c_1_x:
            return True

        return False

    def log_grids(
        self,
        inflated_grid_by_obstacle,
        acc_cells_for_obs,
        normalized_social_cost,
        normalized_distance_cost,
        sorted_cell_to_combined_cost,
        normalized_distance_to_goal=None,
    ):
        stocg.display_or_log(
            grid=np.invert(inflated_grid_by_obstacle.grid.astype(bool)),
            suffix="-obs_inf_grid",
            start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            debug_display=False,
            log_costmaps=True,
            logs_dir=self.logs_dir,
        )

        normalized_social_cost_costmap = np.zeros(
            (inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height)
        )
        normalized_distance_from_obs_costmap = np.zeros(
            (inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height)
        )
        normalized_distance_from_goal_costmap = np.zeros(
            (inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height)
        )

        for i in range(len(acc_cells_for_obs)):
            cell = acc_cells_for_obs[i]
            normalized_social_cost_costmap[cell[0]][cell[1]] = normalized_social_cost[i]
            normalized_distance_from_obs_costmap[cell[0]][
                cell[1]
            ] = normalized_distance_cost[i]
            if normalized_distance_to_goal is not None:
                normalized_distance_from_goal_costmap[cell[0]][
                    cell[1]
                ] = normalized_distance_to_goal[i]

        stocg.display_or_log(
            grid=normalized_social_cost_costmap,
            suffix="-n_social_costmap",
            start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            debug_display=False,
            log_costmaps=True,
            logs_dir=self.logs_dir,
        )
        stocg.display_or_log(
            grid=normalized_distance_from_obs_costmap,
            suffix="-n_d_to_obs_costmap",
            start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            debug_display=False,
            log_costmaps=True,
            logs_dir=self.logs_dir,
        )
        if normalized_distance_to_goal is not None:
            stocg.display_or_log(
                grid=normalized_distance_from_goal_costmap,
                suffix="-n_d_to_goal_costmap",
                start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False,
                log_costmaps=True,
                logs_dir=self.logs_dir,
            )

        combined_costmap = np.zeros(
            (inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height)
        )
        for cell, combined_cost in sorted_cell_to_combined_cost.items():
            combined_costmap[cell[0]][cell[1]] = combined_cost
        stocg.display_or_log(
            grid=combined_costmap,
            suffix="-combined_costmap",
            start_time_str=time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            debug_display=False,
            log_costmaps=True,
            logs_dir=self.logs_dir,
        )
