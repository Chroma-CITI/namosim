import copy
import heapq
import numpy as np
import time
from collections import OrderedDict
from shapely.geometry import Point
import random

from .baseline_behavior import BaselineBehavior
from snamosim.behaviors.algorithms import graph_search
from snamosim.utils import utils
from snamosim.worldreps.entity_based.obstacle import Obstacle
from snamosim.worldreps.entity_based.robot import Robot
from snamosim.behaviors.algorithms.new_local_opening_check import check_new_local_opening
import snamosim.behaviors.plan.basic_actions as ba
import snamosim.behaviors.plan.action_result as ar
from snamosim.worldreps.occupation_based.binary_occupancy_grid import BinaryOccupancyGrid, BinaryInflatedOccupancyGrid
import snamosim.worldreps.occupation_based.social_topological_occupation_cost_grid as stocg
import snamosim.utils.b2_collision as b2_collision
import snamosim.utils.connectivity as connectivity


class Stilman2005Behavior(BaselineBehavior):
    def __init__(self, initial_world, robot_uid, navigation_goals, behavior_config, abs_path_to_logs_dir):
        BaselineBehavior.__init__(
            self, initial_world, robot_uid, navigation_goals, behavior_config, abs_path_to_logs_dir)

        # Configuration parameters
        parameters = behavior_config["parameters"]
        # - Original Stilman method configuration parameters
        self.alpha = parameters["alpha_for_obstacle_choice_heur"]
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD  # default if bad parameter
        # self.heur_w = parameters["heuristic_cost_for_traversing_obstacle_in_choice_heur"]
        # self.basic_trans_force = parameters["basic_translation_force"]
        # self.basic_rot_moment = parameters["basic_rotation_moment"]
        self.translation_unit_cost = 1.
        self.rotation_unit_cost = 1.
        self.transfer_coefficient = 2.  # Note: MUST ALWAYS BE > 1 !
        # - Robot action space parameters
        self.angular_res = parameters["collision_check_angular_res"]
        self.rotation_unit_angle = 60.  # parameters["robot_rotation_unit_angle"]
        self.translation_unit_length = parameters["robot_translation_unit_length"]
        self.forbid_rotations = parameters["forbid_rotations"]
        self.translation_factor = self.translation_unit_cost / self.translation_unit_length
        self.rotation_factor = self.rotation_unit_cost / self.rotation_unit_angle

        # - S-NAMO parameters
        self.use_social_cost = parameters["use_social_cost"]
        self.bound_percentage = parameters["solution_interval_bound_percentage"]
        if parameters["manipulation_search_procedure"] == "DFS":
            if self.use_social_cost:
                self.manip_search_procedure = self.focused_manip_search
            else:
                raise ValueError("Focused manipulation search requires the use_social_cost variable to be True !")
        elif parameters["manipulation_search_procedure"] == "BFS":
            self.manip_search_procedure = self.manip_search
        self.w_social, self.w_obs, self.w_goal = 15., 10., 2.
        self.w_sum = self.w_social + self.w_obs + self.w_goal
        self.distance_to_obs_cost_is_realistic = True

        # - Extra performance parameters
        self.check_new_local_opening_before_global = parameters["check_new_local_opening_before_global"]
        self.activate_grids_logging = False  # not parameters["deactivate_grids_logging"]

        self._trans_vectors = np.array([(self.translation_unit_length, 0.), (-self.translation_unit_length, 0.)])
        if self.forbid_rotations:
            self._rot_angles = np.array([])
        else:
            self._rot_angles = np.array([self.rotation_unit_angle, -self.rotation_unit_angle])
        self._all_rot_angles = self.rotation_unit_angle * np.array(range(1, 360 // int(self.rotation_unit_angle)))
        self._nb_possible_angles = len(self._all_rot_angles)

        self._new_actions = []
        for trans_vector in self._trans_vectors:
            self._new_actions.append(ba.Translation(trans_vector))
        for rot_angle in self._rot_angles:
            self._new_actions.append(ba.Rotation(rot_angle))

        self._social_costmap = None

        self.is_first_transfer_step = False

        self.check_horizon = 10

        self.min_nb_steps_to_wait = 5
        self.max_nb_steps_to_wait = 20
        self.wait_steps = -1

        self.grabbed_obstacles = set()

        # Initialize movability status of obstacles
        for entity in self._world.entities.values():
            entity.movability = self._robot.deduce_movability(entity.type)

        # Initialize overall Box2D world simulation
        self.b2_sim = b2_collision.B2Sim(self._world.entities)

        # Initialize static obstacles occupation grid, since it is not supposed to change
        static_obs_polygons = {
            uid: entity.polygon for uid, entity in self._world.entities.items()
            if (
                (isinstance(entity, Robot) and entity.uid != self._robot_uid)
                or (isinstance(entity, Obstacle) and self._robot.deduce_movability(entity.type) == "unmovable")
            )
        }
        robot_max_inflation_radius = utils.get_circumscribed_radius(self._robot.polygon)
        self.static_obs_inf_grid = BinaryInflatedOccupancyGrid(
            static_obs_polygons, self._world.dd.res, robot_max_inflation_radius, neighborhood=self.neighborhood
        )

        # Initialize social occupation costmap
        if self.use_social_cost and self._social_costmap is None:
            static_obs_grid = BinaryOccupancyGrid(
                static_obs_polygons, self._world.dd.res, neighborhood=self.neighborhood,
                params=self.static_obs_inf_grid.params
            )
            self._social_costmap = stocg.compute_social_costmap(
                static_obs_grid.grid, self._world.dd.res, log_costmaps=self.activate_grids_logging,
                abs_path_to_logs_dir=self.abs_path_to_logs_dir, ns=self._robot_name)
            self._rp.publish_grid_map(self._social_costmap, self._world.dd.res, ns=self._robot_name)

        # Initialize robot max radius inflated costmap used to check for transit path validity
        other_entities_polygons = {
            uid: e.polygon for uid, e in self._world.entities.items() if e.uid != self._robot.uid
        }
        self.inflated_grid_by_robot = BinaryInflatedOccupancyGrid(
            other_entities_polygons, self._world.dd.res, robot_max_inflation_radius, neighborhood=self.neighborhood,
            params=self.static_obs_inf_grid.params
        )

        self.replan_count = 10

    def are_all_goals_finished(self):
        return not self._navigation_goals and self._q_goal is None

    def is_goal_success(self, q_r):
        # TODO Extract abs_tol constant and make it a parameter for each goal
        return all(np.isclose(q_r, self._q_goal, rtol=1e-5))

    def get_current_goal(self):
        return self._q_goal

    def sense(self, ref_world, last_action_result, step_count):
        # Update baseline world representation (polygons)
        BaselineBehavior.sense(self, ref_world, last_action_result, step_count)

        # Update Box2D-based world representation
        self.b2_sim.add_entities({uid: self._world.entities[uid] for uid in self._added_uids})
        self.b2_sim.update_entities({uid: self._world.entities[uid] for uid in self._updated_uids})
        self.b2_sim.remove_entities(self._removed_uids)

        # Update grid(s)
        self.inflated_grid_by_robot.update(
            new_polygons={
                uid: self._world.entities[uid].polygon
                for uid in self._added_uids.union(self._updated_uids) if uid != self._robot_uid
            },
            removed_polygons=self._removed_uids
        )

    def think(self):
        # TODO Try to rewrite this more cleanly
        if isinstance(self._last_action_result, ar.ActionSuccess):
            if isinstance(self._last_action_result.action, ba.Grab):
                self.grabbed_obstacles.add(self._last_action_result.action.entity_uid)
            elif isinstance(self._last_action_result.action, ba.Release):
                if self._last_action_result.action.entity_uid in self.grabbed_obstacles:
                    self.grabbed_obstacles.remove(self._last_action_result.action.entity_uid)

        if self.are_all_goals_finished():
            # Exit early if there are no goals for the behavior to reach
            self.replan_count = 0
            return ba.GoalsFinished()

        if self.wait_steps > 0:
            # Exit early if the behavior must still wait
            self.simulation_log.append(utils.BasicLog(
                "Agent '{}' is waiting for {} steps.".format(self._robot.name, self.wait_steps),
                self._step_count)
            )
            self.wait_steps -= 1
            return ba.Wait()

        if self._q_goal is None:
            self.replan_count = 0
            self._q_goal = self._navigation_goals.pop(0)
            self._p_opt = Plan([], self._q_goal, self._robot_uid)

        q_r = self._robot.pose

        if self.is_goal_success(q_r):
            self.replan_count = 0
            action = ba.GoalSuccess(self._q_goal)
            self._q_goal = None
            return action

        replan = False
        if isinstance(self._last_action_result, ar.ActionFailure):
            action_failed_because_of_other_agent = isinstance(
                self._last_action_result,
                (ar.SimultaneousGrabFailure, ar.DynamicCollisionFailure, ar.GrabbedByOtherFailure)
            )
            if action_failed_because_of_other_agent:
                self.wait_steps = random.randint(self.min_nb_steps_to_wait, self.max_nb_steps_to_wait)
                self.wait_steps -= 1
                self.simulation_log.append(utils.BasicLog(
                    str(action_failed_because_of_other_agent),
                    self._step_count)
                )
                self.simulation_log.append(utils.BasicLog(
                    "Agent '{}' starts waiting for {} steps.".format(self._robot.name, self.wait_steps),
                    self._step_count)
                )
                return ba.Wait()
            else:
                replan = True
        else:
            # If last action was a success, check if plan is valid (at the fixed horizon if given)
            all_entities_poses = {uid: entity.pose for uid, entity in self._world.entities.items()}
            try:
                p_opt_is_valid = self._p_opt.is_valid(
                    all_entities_poses, self.b2_sim, self.inflated_grid_by_robot, check_horizon=self.check_horizon
                )
            except PlanValidityError as e:
                p_opt_is_valid = False
                if isinstance(e, DynamicCollisionError):
                    colliding_entities = []
                    for uid_1, uid_2 in e.colliding_entities:
                        colliding_entities.append(self._world.entities[uid_1])
                        colliding_entities.append(self._world.entities[uid_2])
                    all_colliding_entities_are_robots = all([isinstance(e, Robot) for e in colliding_entities])
                    if all_colliding_entities_are_robots and self.wait_steps == -1:
                        self.wait_steps = random.randint(self.min_nb_steps_to_wait, self.max_nb_steps_to_wait)
                        self.wait_steps -= 1
                        self.simulation_log.append(utils.BasicLog(
                            "Plan of agent '{}' has been invalidated because collides with other agent(s) {}.".format(
                                self._robot.name, str([e.name for e in colliding_entities])),
                            self._step_count)
                        )
                        self.simulation_log.append(utils.BasicLog(
                            "Agent '{}' starts waiting for {} steps.".format(self._robot.name, self.wait_steps),
                            self._step_count)
                        )
                        return ba.Wait()

            if not p_opt_is_valid:
                # If the plan is not valid, try replanning
                replan = True
                self.wait_steps = -1
                self.simulation_log.append(utils.BasicLog(
                    "Plan of agent '{}' is invalid.".format(self._robot.name),
                    self._step_count)
                )
                if self._p_opt.executed_only_first_step():
                    self.simulation_log.append(utils.BasicLog(
                        "Goal failure because of plan validity check incoherence.",
                        self._step_count)
                    )
                    gf_action = ba.GoalFailed(self._q_goal)
                    self.replan_count = 0
                    self._q_goal = None
                    return gf_action
        if replan:
            self.simulation_log.append(utils.BasicLog(
                "Agent '{}' tries replanning.".format(self._robot.name),
                self._step_count)
            )

            if self.grabbed_obstacles:
                self.simulation_log.append(utils.BasicLog(
                    "Agent '{}' must first release obstacle before replanning.".format(self._robot.name),
                    self._step_count)
                )
                self._p_opt = Plan([], self._q_goal, self._robot_uid)

                other_entities_polygons = {
                    uid: e.polygon for uid, e in self._world.entities.items() if e.uid != self._robot.uid
                }
                inflated_grid_by_robot_max = BinaryInflatedOccupancyGrid(
                    other_entities_polygons, self._world.dd.res, self.static_obs_inf_grid.inflation_radius,
                    self.neighborhood, self.static_obs_inf_grid.params
                )

                can_release, _ = self.can_robot_walk_back_to_next_transit_pose(
                    inflated_grid_by_robot_max, self._robot.pose, self._robot.polygon, self._robot.uid, self.b2_sim
                )
                if can_release:
                    obstacle_uid = self.grabbed_obstacles.pop()
                    new_configuration = self.get_robot_walk_back_to_next_transit_configuration(
                        self._robot.pose, self._robot.polygon, self.static_obs_inf_grid.inflation_radius,
                        self._world.dd.res, inflated_grid_by_robot_max.grid_pose, obstacle_uid, 0.01, 1.
                    )
                    return new_configuration.action
                else:
                    gf_action = ba.GoalFailed(self._q_goal)
                    self.replan_count = 0
                    self._q_goal = None
                    return gf_action

            self._p_opt = self.select_connect(
                self._world, self.static_obs_inf_grid, self.inflated_grid_by_robot, self.b2_sim, self._q_goal,
                neighborhood=self.neighborhood
            )
            self.replan_count += 1

        if not self._p_opt:
            # If no plan for the goal was found
            self.simulation_log.append(utils.BasicLog(
                "FAILURE: Agent '{name}' has failed to reach pose {nav_goal}.".format(
                    name=self._robot.name, nav_goal=str(self._q_goal)
                ),
                self._step_count)
            )
            gf_action = ba.GoalFailed(self._q_goal)
            self.replan_count = 0
            self._q_goal = None
            return gf_action
        else:
            plan_has_steps_left = not self._p_opt.is_empty()
            if plan_has_steps_left:
                next_step = self._p_opt.pop_next_step()
                return next_step

    def select_connect(self, w_t, static_obs_inf_grid, inflated_grid_by_robot, b2_sim, r_f, ccs_data=None,
                       neighborhood=utils.CHESSBOARD_NEIGHBORHOOD, nb_self_calls=0, nb_rch_calls=0):
        """
        High Level Planner _select_connect (SC).
        It makes use of _rch and _manip_search in a greedy heuristic search with backtracking.
        It backtracks locally when the object selected by _rch cannot be moved to merge the selected c_1 in c_free.
        It backtracks globally when all the paths identified by _rch from c_1 are unsuccessful.
        SC calls _find_path to determine a transit path from r_t to a contact point, r_t_plus_1 . The existence of the
        path is guaranteed by the choice of contacts in Manip-Search.
        # :param w_t: state of the world at time t
        # :param r_f: goal robot configuration [x, y, theta] in {m, m, degrees}
        # :return: None to backtrack, current partial plan otherwise.
        """
        if nb_self_calls > 20 or nb_rch_calls > 20:
            self.simulation_log.append(utils.BasicLog(
                "Too much recursion, there must be a problem, failing goal.", self._step_count)
            )
            return None

        robot_at_t = w_t.entities[self._robot.uid]
        r_t = robot_at_t.pose

        avoid_list = set()

        robot_cell = utils.real_to_grid(r_t[0], r_t[1], static_obs_inf_grid.res, static_obs_inf_grid.grid_pose)
        goal_cell = utils.real_to_grid(r_f[0], r_f[1], static_obs_inf_grid.res, static_obs_inf_grid.grid_pose)

        simple_path_to_goal = self.find_path(r_t, r_f, inflated_grid_by_robot, robot_at_t.polygon)
        if simple_path_to_goal:
            # If the goal is in the same free space component as the robot in simulated w_t
            # Orig. condition in pseudo-code is : x^f in C^acc_R(W)
            # TODO FIX COST COMPUTATION TO FIT SAME MODEL AS MANIP SEARCH !
            return Plan([simple_path_to_goal], self._q_goal, self._robot_uid)

        if ccs_data is None:
            ccs_data = connectivity.init_ccs_for_grid(
                inflated_grid_by_robot.grid, inflated_grid_by_robot.d_width,
                inflated_grid_by_robot.d_height, neighborhood
            )
        else:
            ccs_data = connectivity.update_ccs_and_grid(
                ccs_data, inflated_grid_by_robot.grid, inflated_grid_by_robot.d_width,
                inflated_grid_by_robot.d_height, neighborhood
            )
        connected_components_grid = ccs_data.ccs_grid
        self._rp.publish_connected_components_grid(connected_components_grid, w_t.dd, ns=self._robot_name)

        nb_rch_calls += 1
        o_1, c_1 = self.rch(
            robot_cell, goal_cell, static_obs_inf_grid, connected_components_grid,
            inflated_grid_by_robot, avoid_list, neighborhood
        )
        while o_1 != 0:
            self.simulation_log.append(utils.BasicLog(
                "Agent {} selected entity {} for manipulation search to reach component {}.".format(
                    self._robot_name, w_t.entities[o_1].name, c_1
                ),
                self._step_count)
            )
            r_acc_cells_set = ccs_data.ccs[ccs_data.ccs_grid[robot_cell[0]][robot_cell[1]]].visited
            c_1_cells_set = set() if c_1 == 0 else ccs_data.ccs[c_1].visited
            w_t_plus_2, tho_m = self.manip_search_procedure(w_t, o_1, r_acc_cells_set, c_1_cells_set, r_f, b2_sim)

            if tho_m is not None:
                self.simulation_log.append(utils.BasicLog(
                    "Agent {} found partial plan manipulating entity {} to reach component {}.".format(
                        self._robot_name, w_t.entities[o_1].name, c_1
                    ),
                    self._step_count)
                )
                future_plan = self.select_connect(
                    w_t_plus_2, static_obs_inf_grid, inflated_grid_by_robot, self.b2_sim, r_f, ccs_data, neighborhood,
                    nb_self_calls=nb_self_calls+1, nb_rch_calls=nb_rch_calls
                )
                if future_plan is not None:
                    tho_n = self.find_path(r_t, tho_m.robot_path.poses[0], inflated_grid_by_robot, robot_at_t.polygon)
                    return Plan([tho_n, tho_m], self._q_goal, self._robot_uid).append(future_plan)

            # Extra check for when the goal is in a movable obstacle that we could not find how to move
            if c_1 == 0:
                self.simulation_log.append(utils.BasicLog(
                    "Agent {} did not find a reachable component if manipulating {}.".format(
                        self._robot_name, w_t.entities[o_1].name
                    ),
                    self._step_count)
                )
                break

            avoid_list.add((o_1, c_1))

            nb_rch_calls += 1
            o_1, c_1 = self.rch(
                robot_cell, goal_cell, static_obs_inf_grid, connected_components_grid,
                inflated_grid_by_robot, avoid_list, neighborhood
            )

        return None

    def rch_get_neighbors(self, current, gscore, close_set, open_queue, came_from,
                          static_obs_grid, connected_components_grid, inflated_robot_grid,
                          avoid_list, init_robot_component_uid, g_function, traversed_obstacles_ids,
                          neighborhood=utils.TAXI_NEIGHBORHOOD):
        """
        Combined formulation from Stilman's thesis and his article. The prevlist parameter was not used because not
        in the article and a priori not helpful. Not only that, it is not properly defined, and actually does not
        algorithmically make sense, because the C1 component gets merged with the robot accessible space as the robot
        moves obstacles around.
        """
        neighbors, tentative_gscores = [], []
        current_gscore = gscore[current]
        path_has_traversed_first_disconnected_comp = current.first_component_uid != 0
        path_has_traversed_first_obstacle = current.first_obstacle_uid != 0

        # Filter out cells that are not in the map, and in static obstacles
        candidate_neighbor_cells = utils.get_neighbors_no_coll(
            current.cell, static_obs_grid.grid, inflated_robot_grid.d_width, inflated_robot_grid.d_height, neighborhood
        )

        for neighbor_cell in candidate_neighbor_cells:
            neighbor = None
            if path_has_traversed_first_disconnected_comp:
                # Note: This validation was added according to the description in the article about not allowing
                # transitions between two different obstacles or to a cell with several obstacles, though it was not
                # explicit in the pseudocode formulation in Stilman's thesis.
                cur_cell_obs_uid = inflated_robot_grid.only_obstacle_uid_in_cell(current.cell)
                neighbor_cell_obs_uid = inflated_robot_grid.only_obstacle_uid_in_cell(neighbor_cell)

                cur_and_neighbor_not_in_mult_obs = cur_cell_obs_uid != -1 and neighbor_cell_obs_uid != -1
                current_or_neighbor_in_free_space = cur_cell_obs_uid == 0 or neighbor_cell_obs_uid == 0
                transition_is_valid = (
                    cur_and_neighbor_not_in_mult_obs
                    and (current_or_neighbor_in_free_space or cur_cell_obs_uid == neighbor_cell_obs_uid)
                    and neighbor_cell_obs_uid != current.first_obstacle_uid
                )
                if transition_is_valid:
                    neighbor = RCHConfiguration(neighbor_cell, current.first_obstacle_uid, current.first_component_uid)
            else:
                neighbor_cell_component_uid = connected_components_grid[neighbor_cell[0]][neighbor_cell[1]]
                neighbor_cell_in_free_space = neighbor_cell_component_uid > 0
                if path_has_traversed_first_obstacle:
                    if neighbor_cell_in_free_space:
                        neighbor_cell_not_in_robot_component_nor_avoid_list = (
                            neighbor_cell_component_uid != init_robot_component_uid
                            and (current.first_obstacle_uid, neighbor_cell_component_uid) not in avoid_list
                        )
                        if neighbor_cell_not_in_robot_component_nor_avoid_list:
                            neighbor = RCHConfiguration(
                                neighbor_cell, current.first_obstacle_uid, neighbor_cell_component_uid
                            )
                        else:
                            # Either the neighbor tries to go back to robot acc. space, or in a (obs., comp.)
                            # combination that has already been explored and for which no manip. could be found
                            pass

                    else:
                        neighbor_cell_obs_uid = inflated_robot_grid.only_obstacle_uid_in_cell(neighbor_cell)
                        if neighbor_cell_obs_uid == current.first_obstacle_uid:
                            neighbor = RCHConfiguration(neighbor_cell, current.first_obstacle_uid, 0)
                        else:
                            # Either the neighbor is in another obstacle, or in multiple, which is forbidden
                            pass
                else:
                    if neighbor_cell_in_free_space:
                        # If no obstacle has been traversed, we are still in the robot acc. space
                        neighbor = RCHConfiguration(neighbor_cell, 0, 0)
                    else:
                        neighbor_cell_obstacle_uid = inflated_robot_grid.only_obstacle_uid_in_cell(neighbor_cell)
                        if neighbor_cell_obstacle_uid > 0:
                            neighbor = RCHConfiguration(neighbor_cell, neighbor_cell_obstacle_uid, 0)
                        else:
                            # The neighbor is in multiple obstacles, which is forbidden
                            pass
            if neighbor is not None and neighbor not in close_set:
                neighbors.append(neighbor)
                tentative_gscores.append(
                    current_gscore + g_function(
                        current, neighbor, is_transfer=inflated_robot_grid.grid[neighbor.cell[0]][neighbor.cell[1]] > 0
                    )
                )
                traversed_obstacles_ids.add(neighbor.first_obstacle_uid)

        self._rp.publish_rch_data(
            current, gscore, close_set, open_queue, came_from, neighbors, traversed_obstacles_ids,
            inflated_robot_grid.res, inflated_robot_grid.grid_pose, ns=self._robot_name
        )

        return neighbors, tentative_gscores

    def rch(self, start_cell, goal_cell,
            static_obs_grid, connected_components_grid, inflated_robot_grid, avoid_list,
            neighborhood=utils.TAXI_NEIGHBORHOOD):

        if inflated_robot_grid.grid[start_cell[0]][start_cell[1]] > 0:
            self.simulation_log.append(utils.BasicLog(
                'The robot start cell in a rch call must always be outside of any obstacles.', self._step_count)
            )
            return 0, 0

        if static_obs_grid.grid[goal_cell[0]][goal_cell[1]] > 0:
            self.simulation_log.append(utils.BasicLog(
                'The robot goal cell in a rch call mush always be outside of static obstacles.', self._step_count)
            )
            return 0, 0

        if inflated_robot_grid.grid[goal_cell[0]][goal_cell[1]] > 1:
            self.simulation_log.append(utils.BasicLog(
                'The robot goal cell in a rch call must be at most within one movable obstacle.', self._step_count)
            )
            return 0, 0

        # TODO Create custom exceptions for above

        init_robot_component_uid = connected_components_grid[start_cell[0]][start_cell[1]]
        sqrt_of_2_times_res = utils.SQRT_OF_2 * inflated_robot_grid.res
        goal_real = utils.grid_to_real(
            goal_cell[0], goal_cell[1], inflated_robot_grid.res, inflated_robot_grid.grid_pose
        )

        def g_function(current, neighbor, is_transfer=False):
            dist = (
                sqrt_of_2_times_res
                if neighbor.cell in [
                    (current.cell[0] + i, current.cell[1] + j) for i, j in utils.CHESSBOARD_NEIGHBORHOOD_EXTRAS
                ]
                else inflated_robot_grid.res
            )
            translation_cost = self.translation_factor * dist
            return translation_cost * (1. if not is_transfer else self.transfer_coefficient)

        def h_function(_c, _g):
            translation_cost = self.translation_factor * utils.euclidean_distance(
                utils.grid_to_real(_c.cell[0], _c.cell[1], inflated_robot_grid.res, inflated_robot_grid.grid_pose),
                goal_real
            )
            return translation_cost

        traversed_obstacles_ids = utils.OrderedSet()

        def rch_get_neighbors_instance(current, gscore, close_set, open_queue, came_from):
            return self.rch_get_neighbors(
                current, gscore, close_set, open_queue, came_from,
                static_obs_grid, connected_components_grid, inflated_robot_grid,
                avoid_list, init_robot_component_uid, g_function, traversed_obstacles_ids, neighborhood
            )

        def exit_condition(_current, _goal):
            return _current.cell == _goal.cell

        start = RCHConfiguration(start_cell, 0, 0)
        goal = RCHConfiguration(goal_cell, 0, 0)  # Note the zeroes are never used, this line is just for coherence

        path_found, end_config, _, _, _, _ = graph_search.new_generic_a_star(
            start, goal, exit_condition, rch_get_neighbors_instance, h_function
        )
        if path_found:
            if end_config.first_obstacle_uid == 0:
                raise ValueError('Rch found a path where no obstacle needed to be traversed.')
            return end_config.first_obstacle_uid, end_config.first_component_uid
        else:
            return 0, 0

    def manip_search(self, w_t, o_1, r_acc_cells_set, c_1_cells_set, r_f, b2_sim,
                     check_new_local_opening_before_global=True):
        # Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = copy.deepcopy(w_t)
        self._rp.publish_robot_sim_world(w_t_plus_2, self._robot_uid, ns=self._robot_name)

        res = w_t_plus_2.dd.res

        other_entities = [entity for entity in w_t_plus_2.entities.values()
                          if entity.uid != self._robot.uid and entity.uid != o_1]
        other_entities_polygons = {entity.uid: entity.polygon for entity in other_entities}

        robot = w_t_plus_2.entities[self._robot.uid]
        robot_uid, robot_pose, robot_polygon, robot_name = robot.uid, robot.pose, robot.polygon, robot.name
        robot_min_inflation_radius = utils.get_inscribed_radius(robot_polygon)
        robot_max_inflation_radius = utils.get_circumscribed_radius(robot_polygon)

        obstacle = w_t_plus_2.entities[o_1]
        obstacle_uid, obstacle_pose, obstacle_polygon = obstacle.uid, obstacle.pose, obstacle.polygon
        obstacle_min_inflation_radius = utils.get_inscribed_radius(obstacle_polygon)

        # CAREFUL : We inflate by inscribed radius MINUS sqrt(2)*res to make sure occupied cells are really where the
        # entity's center should NEVER be to avoid collisions.
        # Poses in free cells of this grid may sometimes be colliding.
        inflated_grid_by_robot_min = BinaryInflatedOccupancyGrid(
            other_entities_polygons, res, robot_min_inflation_radius - utils.SQRT_OF_2 * res,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD
        )
        inflated_grid_by_robot_max = BinaryInflatedOccupancyGrid(
            other_entities_polygons, res, robot_max_inflation_radius,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD, params=inflated_grid_by_robot_min.params
        )
        inflated_grid_by_obstacle = BinaryInflatedOccupancyGrid(
            other_entities_polygons, res, obstacle_min_inflation_radius - utils.SQRT_OF_2 * res,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD, params=inflated_grid_by_robot_min.params
        )

        goal_pose, goal_cell = r_f, utils.real_to_grid(r_f[0], r_f[1], res, inflated_grid_by_robot_min.grid_pose)

        trans_mult = 1. / res * 10.
        rot_mult = 1.

        # Get accessible sampled navigation points around obstacle and paths to them
        transit_end_robot_poses, transfer_start_robot_poses = self.get_transit_end_and_transfer_start_poses(
            obstacle_polygon, robot_max_inflation_radius, r_acc_cells_set, res, inflated_grid_by_robot_min.grid_pose
        )

        if not transfer_start_robot_poses:
            # If there are no attainable manipulation configurations, exit early
            return w_t_plus_2, None

        transfer_start_to_transit_end_robot_pose = {
            manip_pose: nav_pose for nav_pose, manip_pose in zip(transit_end_robot_poses, transfer_start_robot_poses)
        }

        transfer_start_configurations = {
            RobotObstacleConfiguration(
                robot_floating_point_pose=manip_pose,
                robot_polygon=utils.set_polygon_pose(robot_polygon, robot_pose, manip_pose),
                robot_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(manip_pose, trans_mult, rot_mult),
                robot_cell_in_grid=utils.real_to_grid(
                    manip_pose[0], manip_pose[1],
                    res, inflated_grid_by_robot_min.grid_pose
                ),
                obstacle_floating_point_pose=obstacle_pose,
                obstacle_polygon=obstacle_polygon,
                obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                    obstacle_pose, trans_mult, rot_mult
                ),
                obstacle_cell_in_grid=utils.real_to_grid(
                    obstacle_pose[0], obstacle_pose[1],
                    res, inflated_grid_by_obstacle.grid_pose
                ),
                manip_pose_id=manip_pose_id
            ): self.g(nav_pose, manip_pose, is_transfer=True)
            for manip_pose_id, (manip_pose, nav_pose) in enumerate(transfer_start_to_transit_end_robot_pose.items())
        }

        # Use Dijkstra algorithm to compute a transfer path that allows for an opening to be created
        path_found, transfer_end_configuration, came_from, close_set, gscore, _ = self.dijkstra_for_manip_search(
            transfer_start_configurations, robot_uid, robot_name, obstacle_uid, obstacle_polygon,
            other_entities_polygons, inflated_grid_by_robot_min, inflated_grid_by_robot_max, inflated_grid_by_obstacle,
            c_1_cells_set, trans_mult, rot_mult, b2_sim, check_new_local_opening_before_global, goal_pose, goal_cell
        )
        if path_found:
            self._rp.publish_sim(
                transfer_end_configuration.robot.polygon, transfer_end_configuration.obstacle.polygon,
                "/target", ns=self._robot_name
            )
            raw_path = graph_search.reconstruct_path(came_from, transfer_end_configuration)
            prev_transit_end_configuration, next_transit_start_configuration = self.transit_paths_end_start(
                transfer_end_configuration, raw_path, transfer_start_to_transit_end_robot_pose,
                robot_polygon, robot_pose, inflated_grid_by_robot_max, obstacle_uid, trans_mult, rot_mult
            )
            tho_m_phys_cost = gscore[transfer_end_configuration] + self.g(
                transfer_end_configuration.robot.floating_point_pose,
                next_transit_start_configuration.floating_point_pose,
                is_transfer=True
            )
            tho_m = TransferPath.from_configurations(
                prev_transit_end_configuration, next_transit_start_configuration,
                raw_path, obstacle_uid, tho_m_phys_cost
            )
        else:
            # If after exhausting all possible configurations, none opens a path to the connected component,
            # return None
            tho_m = None

        # Don't forget to update w_t_plus_2 with transfer end state
        if tho_m:
            robot.pose, robot.polygon = tho_m.robot_path.poses[-1], tho_m.robot_path.polygons[-1]
            obstacle.pose, obstacle.polygon = tho_m.obstacle_path.poses[-1], tho_m.obstacle_path.polygons[-1]

        self._rp.publish_robot_sim_world(w_t_plus_2, self._robot_uid, ns=self._robot_name)
        self._rp.cleanup_robot_sim(ns=self._robot_name)
        self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)

        return w_t_plus_2, tho_m

    def focused_manip_search(self, w_t, o_1, r_acc_cells_set, c_1_cells_set, r_f, b2_sim,
                             check_new_local_opening_before_global=True):
        # Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = copy.deepcopy(w_t)
        self._rp.publish_robot_sim_world(w_t_plus_2, self._robot_uid, ns=self._robot_name)

        res = w_t_plus_2.dd.res

        other_entities = [entity for entity in w_t_plus_2.entities.values()
                          if entity.uid != self._robot.uid and entity.uid != o_1]
        other_entities_polygons = {entity.uid: entity.polygon for entity in other_entities}

        robot = w_t_plus_2.entities[self._robot.uid]
        robot_uid, robot_pose, robot_polygon, robot_name = robot.uid, robot.pose, robot.polygon, robot.name
        robot_min_inflation_radius = utils.get_inscribed_radius(robot_polygon)
        robot_max_inflation_radius = utils.get_circumscribed_radius(robot_polygon)

        obstacle = w_t_plus_2.entities[o_1]
        obstacle_uid, obstacle_pose, obstacle_polygon = obstacle.uid, obstacle.pose, obstacle.polygon
        obstacle_min_inflation_radius = utils.get_inscribed_radius(obstacle_polygon)

        # CAREFUL : We inflate by inscribed radius MINUS sqrt(2)*res to make sure occupied cells are really where the
        # entity's center should NEVER be to avoid collisions.
        # Poses in free cells of this grid may sometimes be colliding.
        inflated_grid_by_robot_min = BinaryInflatedOccupancyGrid(
            other_entities_polygons, res, robot_min_inflation_radius - utils.SQRT_OF_2 * res,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD
        )
        inflated_grid_by_robot_max = BinaryInflatedOccupancyGrid(
            other_entities_polygons, res, robot_max_inflation_radius,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD, params=inflated_grid_by_robot_min.params
        )
        inflated_grid_by_obstacle = BinaryInflatedOccupancyGrid(
            other_entities_polygons, res, obstacle_min_inflation_radius - utils.SQRT_OF_2 * res,
            neighborhood=utils.CHESSBOARD_NEIGHBORHOOD, params=inflated_grid_by_robot_min.params
        )

        goal_pose, goal_cell = r_f, utils.real_to_grid(r_f[0], r_f[1], res, inflated_grid_by_robot_min.grid_pose)

        trans_mult = 1. / res * 10.
        rot_mult = 1.

        # Get accessible sampled navigation points around obstacle and paths to them
        transit_end_robot_poses, transfer_start_robot_poses = self.get_transit_end_and_transfer_start_poses(
            obstacle_polygon, robot_max_inflation_radius, r_acc_cells_set, res, inflated_grid_by_robot_min.grid_pose
        )

        if not transfer_start_robot_poses:
            # If there are no attainable manipulation configurations, exit early
            return w_t_plus_2, None

        transfer_start_to_transit_end_robot_pose = {
            manip_pose: nav_pose for nav_pose, manip_pose in zip(transit_end_robot_poses, transfer_start_robot_poses)
        }

        transfer_start_configurations = {
            RobotObstacleConfiguration(
                robot_floating_point_pose=manip_pose,
                robot_polygon=utils.set_polygon_pose(robot_polygon, robot_pose, manip_pose),
                robot_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(manip_pose, trans_mult, rot_mult),
                robot_cell_in_grid=utils.real_to_grid(
                    manip_pose[0], manip_pose[1],
                    res, inflated_grid_by_robot_min.grid_pose
                ),
                obstacle_floating_point_pose=obstacle_pose,
                obstacle_polygon=obstacle_polygon,
                obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                    obstacle_pose, trans_mult, rot_mult
                ),
                obstacle_cell_in_grid=utils.real_to_grid(
                    obstacle_pose[0], obstacle_pose[1],
                    res, inflated_grid_by_obstacle.grid_pose
                ),
                manip_pose_id=manip_pose_id
            ): self.g(nav_pose, manip_pose, is_transfer=True)
            for manip_pose_id, (manip_pose, nav_pose) in enumerate(transfer_start_to_transit_end_robot_pose.items())
        }

        # Get potentially accessible cells for obstacle ordered by associated combined costs
        cells_sorted_by_combined_cost, sorted_cell_to_combined_cost = self.new_sorted_cells_by_combined_cost(
            inflated_grid_by_obstacle, robot_polygon, robot_pose, obstacle_pose, goal_pose
        )
        bound_quantile_index = int(round(len(cells_sorted_by_combined_cost) * (1. - self.bound_percentage))) - 1
        bound_quantile_index = 0 if bound_quantile_index < 0 else bound_quantile_index
        bound_quantile = sorted_cell_to_combined_cost[cells_sorted_by_combined_cost[bound_quantile_index]]

        # 1. Find the best obstacle transfer end configuration, that is, the one with the best compromise cost
        best_transfer_end_configuration = self.find_best_transfer_end_configuration(
            robot_pose, robot_polygon, robot_name, robot_uid,
            obstacle_uid, obstacle_pose, obstacle_polygon,
            goal_pose, goal_cell, other_entities_polygons, b2_sim,
            inflated_grid_by_robot_max, cells_sorted_by_combined_cost, c_1_cells_set, transfer_start_robot_poses,
            trans_mult, rot_mult, gscore=None, close_set=None,
            check_new_local_opening_before_global=check_new_local_opening_before_global
        )
        if best_transfer_end_configuration is not None:
            self._rp.publish_sim(
                best_transfer_end_configuration.robot.polygon, best_transfer_end_configuration.obstacle.polygon,
                "/target", ns=self._robot_name
            )

            # 2. If a best obstacle transfer end configuration has been found, use A Star to find a path toward it
            path_found, transfer_end_configuration, came_from, close_set, gscore, _ = self.a_star_for_manip_search(
                transfer_start_configurations, best_transfer_end_configuration,
                robot_uid, robot_name, obstacle_uid, obstacle_polygon,
                other_entities_polygons,
                inflated_grid_by_robot_min, inflated_grid_by_robot_max, inflated_grid_by_obstacle, c_1_cells_set,
                trans_mult, rot_mult,
                sorted_cell_to_combined_cost, bound_quantile, b2_sim, check_new_local_opening_before_global,
                goal_pose, goal_cell
            )
            if path_found:
                # 3. If a path is found, return it
                self._rp.publish_sim(
                    transfer_end_configuration.robot.polygon, transfer_end_configuration.obstacle.polygon,
                    "/target", ns=self._robot_name
                )
                raw_path = graph_search.reconstruct_path(came_from, transfer_end_configuration)
                prev_transit_end_configuration, next_transit_start_configuration = self.transit_paths_end_start(
                    transfer_end_configuration, raw_path, transfer_start_to_transit_end_robot_pose,
                    robot_polygon, robot_pose, inflated_grid_by_robot_max, obstacle_uid, trans_mult, rot_mult
                )
                tho_m_phys_cost = gscore[transfer_end_configuration] + self.g(
                    transfer_end_configuration.robot.floating_point_pose,
                    next_transit_start_configuration.floating_point_pose,
                    is_transfer=True
                )
                tho_m = TransferPath.from_configurations(
                    prev_transit_end_configuration, next_transit_start_configuration,
                    raw_path, obstacle_uid, tho_m_phys_cost
                )
            else:
                # 4. If no path is found on the first, try finding a best configuration that has a path towards it
                #   (because we assume the A Star search to have completed, giving us the paths to ALL reachable
                #   configurations.
                best_transfer_end_configuration = self.find_best_transfer_end_configuration(
                    robot_pose, robot_polygon, robot_name, robot_uid,
                    obstacle_uid, obstacle_pose, obstacle_polygon,
                    goal_pose, goal_cell, other_entities_polygons, b2_sim,
                    inflated_grid_by_robot_max, cells_sorted_by_combined_cost, c_1_cells_set,
                    transfer_start_robot_poses, trans_mult, rot_mult, gscore=gscore, close_set=close_set,
                    check_new_local_opening_before_global=check_new_local_opening_before_global
                )
                if best_transfer_end_configuration is not None:
                    self._rp.publish_sim(
                        best_transfer_end_configuration.robot.polygon, best_transfer_end_configuration.obstacle.polygon,
                        "/target", ns=self._robot_name
                    )
                    raw_path = graph_search.reconstruct_path(came_from, best_transfer_end_configuration)
                    prev_transit_end_configuration, next_transit_start_configuration = self.transit_paths_end_start(
                        best_transfer_end_configuration, raw_path, transfer_start_to_transit_end_robot_pose,
                        robot_polygon, robot_pose, inflated_grid_by_robot_max, obstacle_uid, trans_mult, rot_mult
                    )
                    tho_m_phys_cost = gscore[transfer_end_configuration] + self.g(
                        best_transfer_end_configuration.robot.floating_point_pose,
                        next_transit_start_configuration.floating_point_pose,
                        is_transfer=True
                    )
                    tho_m = TransferPath.from_configurations(
                        prev_transit_end_configuration, next_transit_start_configuration,
                        raw_path, obstacle_uid, tho_m_phys_cost
                    )
                else:
                    # If after exhausting all possible configurations, none opens a path to the connected component,
                    # return None
                    tho_m = None
        else:
            # If after exhausting all possible configurations, none opens a path to the connected component,
            # return None
            tho_m = None

        # Don't forget to update w_t_plus_2 with transfer end state
        if tho_m:
            robot.pose, robot.polygon = tho_m.robot_path.poses[-1], tho_m.robot_path.polygons[-1]
            obstacle.pose, obstacle.polygon = tho_m.obstacle_path.poses[-1], tho_m.obstacle_path.polygons[-1]

        self._rp.publish_robot_sim_world(w_t_plus_2, self._robot_uid, ns=self._robot_name)
        self._rp.cleanup_robot_sim(ns=self._robot_name)
        self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)

        return w_t_plus_2, tho_m

    def dijkstra_for_manip_search(
            self, start, robot_uid, robot_name, obstacle_uid, obstacle_polygon,
            other_entities_polygons,
            inflated_grid_by_robot_min, inflated_grid_by_robot_max, inflated_grid_by_obstacle, c_1_cells_set,
            trans_mult, rot_mult, b2_sim, check_new_local_opening_before_global,
            overall_goal_pose, overall_goal_cell):

        def get_neighbors(_current, _gscore, _close_set, _open_queue, _came_from):
            return self.get_neighbors(
                _current, _gscore, _close_set, _open_queue, _came_from,
                start, inflated_grid_by_robot_min, inflated_grid_by_obstacle, robot_uid, obstacle_uid,
                trans_mult, rot_mult, b2_sim
            )

        def exit_condition(_current, _goal):
            can_walk_back, robot_cell_after = self.can_robot_walk_back_to_next_transit_pose(
                inflated_grid_by_robot_max, _current.robot.floating_point_pose,
                _current.robot.polygon, robot_uid, b2_sim
            )
            if can_walk_back:
                #   3. ... and creates a global opening to c1
                has_new_global_opening, _, _ = self.is_there_opening_to_c_1(
                    check_new_local_opening_before_global,
                    robot_name, robot_cell_after,
                    obstacle_uid, obstacle_polygon, _current.obstacle.polygon,
                    other_entities_polygons, b2_sim,
                    inflated_grid_by_robot_max, c_1_cells_set,
                    overall_goal_pose, overall_goal_cell,
                    neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                    init_blocking_areas=None, init_entity_inflated_polygon=None
                )
                if has_new_global_opening:
                    return True
            return False

        return graph_search.new_generic_dijkstra(
            start, goal=None, exit_condition=exit_condition, get_neighbors=get_neighbors
        )

    def a_star_for_manip_search(self, start, goal,
                                robot_uid, robot_name, obstacle_uid, obstacle_polygon,
                                other_entities_polygons,
                                inflated_grid_by_robot, inflated_grid_by_robot_max,
                                inflated_grid_by_obstacle, c_1_cells_set,
                                trans_mult, rot_mult,
                                sorted_cell_to_combined_cost, bound_quantile, b2_sim,
                                check_new_local_opening_before_global, overall_goal_pose, overall_goal_cell):

        def get_neighbors(_current, _gscore, _close_set, _open_queue, _came_from):
            return self.get_neighbors(
                _current, _gscore, _close_set, _open_queue, _came_from,
                start, inflated_grid_by_robot, inflated_grid_by_obstacle, robot_uid, obstacle_uid,
                trans_mult, rot_mult, b2_sim
            )

        def heuristic(_neighbor, _goal):
            return self.h(_neighbor.robot.floating_point_pose, _goal.robot.floating_point_pose)

        def flexible_exit_condition(_current, _goal):
            goal_cell_reached = _current.obstacle.cell_in_grid == _goal.obstacle.cell_in_grid
            if goal_cell_reached:
                return True

            if _current.obstacle.cell_in_grid not in sorted_cell_to_combined_cost:
                # TODO Remove this TEMPORARY condition caused by sometimes missing cell in sorted_cell_to_combined_cost
                return False

            current_cell_cc_within_bound = sorted_cell_to_combined_cost[_current.obstacle.cell_in_grid] < bound_quantile

            if current_cell_cc_within_bound:
                can_walk_back, robot_cell_after = self.can_robot_walk_back_to_next_transit_pose(
                    inflated_grid_by_robot_max, _current.robot.floating_point_pose,
                    _current.robot.polygon, robot_uid, b2_sim
                )
                if can_walk_back:
                    #   3. ... and creates a global opening to c1
                    has_new_global_opening, _, _ = self.is_there_opening_to_c_1(
                        check_new_local_opening_before_global,
                        robot_name, robot_cell_after,
                        obstacle_uid, obstacle_polygon, _current.obstacle.polygon,
                        other_entities_polygons, b2_sim,
                        inflated_grid_by_robot_max, c_1_cells_set,
                        overall_goal_pose, overall_goal_cell,
                        neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                        init_blocking_areas=None, init_entity_inflated_polygon=None
                    )
                    if has_new_global_opening:
                        return True
            return False

        return graph_search.new_generic_a_star(
            start, goal, exit_condition=flexible_exit_condition, get_neighbors=get_neighbors, heuristic=heuristic
        )

    def transit_paths_end_start(self, transfer_end_configuration, raw_path,
                                transfer_start_to_transit_end_robot_pose, robot_polygon,
                                robot_pose, inflated_grid_by_robot_max, obstacle_uid, trans_mult, rot_mult):
        transfer_path_start_pose = raw_path[0].robot.floating_point_pose
        init_transit_start_pose = transfer_start_to_transit_end_robot_pose[transfer_path_start_pose]
        grab_action = ba.Grab(
            translation_vector=(utils.euclidean_distance(transfer_path_start_pose, init_transit_start_pose), 0.),
            entity_uid=obstacle_uid
        )
        init_transit_start_configuration = Configuration(
            init_transit_start_pose,
            utils.set_polygon_pose(robot_polygon, robot_pose, init_transit_start_pose),
            utils.real_to_grid(
                init_transit_start_pose[0], init_transit_start_pose[1],
                inflated_grid_by_robot_max.res, inflated_grid_by_robot_max.grid_pose
            ),
            utils.real_pose_to_grid_pose(
                init_transit_start_pose, inflated_grid_by_robot_max.res,
                inflated_grid_by_robot_max.grid_pose, self.rotation_unit_angle
            ),
            grab_action
        )

        next_transit_start_configuration = self.get_robot_walk_back_to_next_transit_configuration(
            transfer_end_configuration.robot.floating_point_pose,
            transfer_end_configuration.robot.polygon,
            inflated_grid_by_robot_max.inflation_radius,
            inflated_grid_by_robot_max.res, inflated_grid_by_robot_max.grid_pose, obstacle_uid,
            trans_mult, rot_mult
        )
        return init_transit_start_configuration, next_transit_start_configuration

    def find_best_transfer_end_configuration(self, robot_pose, robot_polygon, robot_name, robot_uid,
                                             obstacle_uid, obstacle_pose, obstacle_polygon,
                                             goal_pose, goal_cell,
                                             other_entities_polygons, b2_sim,
                                             inflated_grid_by_robot_max, ordered_cells_by_cost, c_1_cells_set,
                                             init_robot_manip_poses, trans_mult, rot_mult, gscore=None, close_set=None,
                                             check_new_local_opening_before_global=True):
        if close_set:
            # If all reachable configurations have been explored, index them by obstacle cell
            obs_cell_to_reachable_configurations = {}
            for c in close_set:
                if c.obstacle.cell_in_grid in obs_cell_to_reachable_configurations:
                    obs_cell_to_reachable_configurations[c.obstacle.cell_in_grid].append(c)
                else:
                    obs_cell_to_reachable_configurations[c.obstacle.cell_in_grid] = [c]

            # Then iterate over ordered_cells_by_cost until we find a configuration that:
            while ordered_cells_by_cost:
                current_best_cell = ordered_cells_by_cost.pop()
                if current_best_cell in obs_cell_to_reachable_configurations:
                    #   1. Is reachable, ...
                    possible_configurations = sorted(
                        obs_cell_to_reachable_configurations[current_best_cell], key=lambda x: gscore[x]
                    )
                    for configuration in possible_configurations:
                        #   2. ... allows sufficient space for the robot to release the object, ...
                        can_walk_back, robot_cell_after = self.can_robot_walk_back_to_next_transit_pose(
                            inflated_grid_by_robot_max, configuration.robot.floating_point_pose,
                            configuration.robot.polygon, robot_uid, b2_sim
                        )
                        if can_walk_back:
                            #   3. ... and creates a global opening to c1
                            has_new_global_opening, _, _ = self.is_there_opening_to_c_1(
                                check_new_local_opening_before_global,
                                robot_name, robot_cell_after,
                                obstacle_uid, obstacle_polygon, configuration.obstacle.polygon,
                                other_entities_polygons, b2_sim,
                                inflated_grid_by_robot_max, c_1_cells_set,
                                goal_pose, goal_cell, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                                init_blocking_areas=None, init_entity_inflated_polygon=None
                            )
                            if has_new_global_opening:
                                return configuration
        else:
            # If we do not already have the set of reachable configurations, close_set...
            while ordered_cells_by_cost:
                # We iterate over the cells ordered by combined cost until we find a valid transfer end configuration
                current_cell = ordered_cells_by_cost[-1]

                # For that, we:
                for rot in [0.] + self._all_rot_angles:
                    # Iterate over the possible obstacle rotations in this cell
                    obstacle_pose_at_transfer_end = utils.grid_pose_to_real_pose(
                        list(current_cell) + [rot], inflated_grid_by_robot_max.res, inflated_grid_by_robot_max.grid_pose
                    )

                    # If the obstacle collides at this pose, don't consider checking further
                    obstacle_collision_pairs = b2_sim.check_teleportation_with_ghost(
                        obstacle_uid, {obstacle_uid: obstacle_polygon}, {obstacle_uid: obstacle_pose}, obstacle_pose_at_transfer_end
                    )
                    if obstacle_collision_pairs:
                        continue

                    for init_robot_manip_pose in init_robot_manip_poses:
                        # Iterate over the possible robot poses corresponding to each obstacle pose
                        robot_pose_at_transfer_end = self.deduce_robot_goal_pose(
                            init_robot_manip_pose, obstacle_pose, obstacle_pose_at_transfer_end
                        )

                        # For this (robot, obstacle) configuration, check if:
                        #   1. there are no static collisions for robot too, ...
                        robot_collision_pairs = b2_sim.check_teleportation_with_ghost(
                            robot_uid, {robot_uid: robot_polygon}, {robot_uid: robot_pose}, robot_pose_at_transfer_end
                        )
                        if robot_collision_pairs:
                            continue

                        #   2. ... the configuration allows sufficient space for the robot to release the object, ...
                        robot_transfer_end_poly = utils.set_polygon_pose(
                            robot_polygon, robot_pose, robot_pose_at_transfer_end
                        )
                        can_walk_back, robot_cell_after = self.can_robot_walk_back_to_next_transit_pose(
                            inflated_grid_by_robot_max, robot_pose_at_transfer_end,
                            robot_transfer_end_poly, robot_uid, b2_sim,
                        )
                        if can_walk_back:
                            #   3. ... and creates a global opening to c1
                            obstacle_transfer_end_poly = utils.set_polygon_pose(
                                obstacle_polygon, obstacle_pose, obstacle_pose_at_transfer_end
                            )
                            has_new_global_opening, _, _ = self.is_there_opening_to_c_1(
                                check_new_local_opening_before_global,
                                robot_name, robot_cell_after,
                                obstacle_uid, obstacle_polygon, obstacle_transfer_end_poly,
                                other_entities_polygons, b2_sim,
                                inflated_grid_by_robot_max, c_1_cells_set,
                                goal_pose, goal_cell, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                                init_blocking_areas=None, init_entity_inflated_polygon=None
                            )
                            if has_new_global_opening:
                                return RobotObstacleConfiguration(
                                    robot_floating_point_pose=robot_pose_at_transfer_end,
                                    robot_polygon=robot_transfer_end_poly,
                                    robot_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                                        robot_pose_at_transfer_end, trans_mult, rot_mult
                                    ),
                                    robot_cell_in_grid=utils.real_to_grid(
                                        robot_pose_at_transfer_end[0], robot_pose_at_transfer_end[1],
                                        inflated_grid_by_robot_max.res, inflated_grid_by_robot_max.grid_pose
                                    ),
                                    obstacle_floating_point_pose=obstacle_pose_at_transfer_end,
                                    obstacle_polygon=obstacle_transfer_end_poly,
                                    obstacle_fixed_precision_pose=utils.real_pose_to_fixed_precision_pose(
                                        obstacle_pose_at_transfer_end, trans_mult, rot_mult
                                    ),
                                    obstacle_cell_in_grid=utils.real_to_grid(
                                        obstacle_pose_at_transfer_end[0], obstacle_pose_at_transfer_end[1],
                                        inflated_grid_by_robot_max.res, inflated_grid_by_robot_max.grid_pose
                                    )
                                )
                ordered_cells_by_cost.pop()
        return None  # If no valid configuration could be found...

    @staticmethod
    def can_robot_walk_back_to_next_transit_pose(grid, robot_pose, robot_polygon, robot_uid, b2_sim):
        release_translation = ba.Translation(
            translation_vector=(
                -1. * (grid.inflation_radius + 1.5 * grid.res), 0.
            )
        )
        new_robot_pose = release_translation.predict_pose(robot_pose, robot_pose[2])
        cell = utils.real_to_grid(new_robot_pose[0], new_robot_pose[1], grid.res, grid.grid_pose)

        if utils.is_in_matrix(cell, grid.d_width, grid.d_height):
            if grid.grid[cell[0]][cell[1]] > 0:
                # If the robot cell after release is in an obstacle in the grid, return False
                return False, cell
        else:
            # If robot cell outside of grid, return False
            return False, cell

        new_robot_polygon = release_translation.apply(robot_polygon, robot_pose)

        # Check if robot is still within map bounds
        if not new_robot_polygon.within(grid.aabb_polygon):
            return False, cell

        # Finally, we check dynamic collisions (between init configuration and after-action configuration)
        collision_pairs = b2_sim.check_actions_with_ghost(
            key=robot_uid, entities_polygons={robot_uid: robot_polygon}, entities_poses={robot_uid: robot_pose},
            actions=[release_translation], main_pose=robot_pose
        )

        return not collision_pairs, cell

    @staticmethod
    def get_robot_walk_back_to_next_transit_configuration(robot_pose, robot_polygon, robot_max_inflation_radius,
                                                          res, grid_pose, obstacle_uid, trans_mult, rot_mult):
        release_action = ba.Release(
            translation_vector=(-1. * (robot_max_inflation_radius + 1.5 * res), 0.),
            entity_uid=obstacle_uid
        )
        new_robot_polygon = release_action.apply(robot_polygon, robot_pose)
        new_robot_pose = release_action.predict_pose(robot_pose, robot_pose[2])
        new_cell_in_grid = utils.real_to_grid(new_robot_pose[0], new_robot_pose[1], res, grid_pose)
        new_fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(new_robot_pose, trans_mult, rot_mult)
        return Configuration(
            new_robot_pose, new_robot_polygon, new_cell_in_grid, new_fixed_precision_pose,
            release_action
        )

    def is_there_opening_to_c_1(self, check_new_local_opening_before_global,
                                robot_name, robot_cell,
                                obstacle_uid, old_obstacle_polygon, new_obstacle_polygon,
                                other_entities_polygons, b2_sim,
                                inflated_grid_by_robot_max, c_1_cells_set,
                                goal_pose, goal_cell, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
                                init_blocking_areas=None, init_entity_inflated_polygon=None):
        """
        Checks if there is a path between robot_cell and a random cell in c_1_cells_set that is not covered by an
        obstacle (especially the one considered for manipulation).
        :return: True if a path is found, False otherwise
        TODO: Add proper return of init_blocking_areas and init_entity_inflated_polygon and save them in caller methods
        """
        if check_new_local_opening_before_global:
            has_new_local_opening, init_blocking_areas, init_entity_inflated_polygon = check_new_local_opening(
                old_obstacle_polygon, new_obstacle_polygon,
                other_entities_polygons, b2_sim,
                inflated_grid_by_robot_max.inflation_radius, goal_pose,
                init_blocking_areas, init_entity_inflated_polygon, robot_name
            )
        else:
            has_new_local_opening = True

        if has_new_local_opening:
            inflated_grid_by_robot_max.update(new_polygons={obstacle_uid: new_obstacle_polygon})

            if not c_1_cells_set or (c_1_cells_set and goal_cell in c_1_cells_set):
                cell_in_c_1 = goal_cell
            else:
                c_1_cells_set_iterator = iter(c_1_cells_set)
                cell_in_c_1 = next(c_1_cells_set_iterator)
                while inflated_grid_by_robot_max.grid[cell_in_c_1[0]][cell_in_c_1[1]] != 0:
                    # While selected cell not in free space after manipulation, try another cell
                    try:
                        cell_in_c_1 = next(c_1_cells_set_iterator)
                    except StopIteration:
                        # Note: using the the exception detection is the pythonic way it seems (no has_next)
                        # No opening because c_1_cells_set is entirely inaccessible to the robot after manipulation
                        inflated_grid_by_robot_max.update(removed_polygons={obstacle_uid})
                        has_new_global_opening, skipped_global_opening_check = False, False
                        return has_new_global_opening, has_new_local_opening, skipped_global_opening_check

            # TODO Evaluate the performance change (particularly compared to Dijkstra search) if A* star had an
            #  unadmissible heuristic to hasten path discovery (or write Best-FS based solely on heuristic)
            has_new_global_opening, _, _, _, _, _ = graph_search.grid_search_a_star(
                robot_cell, cell_in_c_1, inflated_grid_by_robot_max.grid,
                inflated_grid_by_robot_max.d_width, inflated_grid_by_robot_max.d_height,
                neighborhood, check_diag_neighbors=False
            )

            inflated_grid_by_robot_max.update(removed_polygons={obstacle_uid})

            self._rp.cleanup_a_star_close_set(ns=robot_name)
            self._rp.cleanup_diameter_inflated_polygons(ns=robot_name)
            self._rp.cleanup_blocking_areas(ns=robot_name)

            skipped_global_opening_check = False

            return has_new_global_opening, has_new_local_opening, skipped_global_opening_check
        else:
            has_new_global_opening, skipped_global_opening_check = False, True
            return has_new_global_opening, has_new_local_opening, skipped_global_opening_check

    def get_neighbors(self, current_configuration, gscore, close_set, open_queue, came_from,
                      start, inflated_grid_by_robot, inflated_grid_by_obstacle, robot_uid, obstacle_uid,
                      trans_mult, rot_mult, b2_sim):
        """
        Creates list of neighbors that are not in close set, do not collide dynamically nor statically
        """
        neighbors = []
        tentative_g_scores = []

        for action in self._new_actions:
            if isinstance(action, ba.Rotation):
                neighbor_action_opposes_prev_action = (
                    isinstance(current_configuration.action, ba.Rotation)
                    and action.angle == -1. * current_configuration.action.angle
                )
                if neighbor_action_opposes_prev_action:
                    continue

                robot_center = (
                    current_configuration.robot.floating_point_pose[0],
                    current_configuration.robot.floating_point_pose[1]
                )
                new_robot_pose = action.predict_pose(current_configuration.robot.floating_point_pose, robot_center)
                new_obstacle_pose = action.predict_pose(
                    current_configuration.obstacle.floating_point_pose, robot_center)
                extra_g_cost = self.rotation_unit_cost
            elif isinstance(action, ba.Translation):
                neighbor_action_opposes_prev_action = (
                    isinstance(current_configuration.action, ba.Translation)
                    and action.translation_vector[0] == -1. * current_configuration.action.translation_vector[0]
                    and action.translation_vector[1] == -1. * current_configuration.action.translation_vector[1]
                )
                if neighbor_action_opposes_prev_action:
                    continue

                new_robot_pose = action.predict_pose(
                    current_configuration.robot.floating_point_pose,
                    current_configuration.robot.floating_point_pose[2]
                )
                new_obstacle_pose = action.predict_pose(
                    current_configuration.obstacle.floating_point_pose,
                    current_configuration.robot.floating_point_pose[2]
                )
                extra_g_cost = self.translation_unit_cost
            else:
                raise TypeError('action must either be of type NewRotation or NewTranslation')

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
                new_robot_pose[0], new_robot_pose[1],
                inflated_grid_by_robot.res, inflated_grid_by_robot.grid_pose
            )
            obstacle_cell_in_grid = utils.real_to_grid(
                new_obstacle_pose[0], new_obstacle_pose[1],
                inflated_grid_by_obstacle.res, inflated_grid_by_obstacle.grid_pose
            )

            is_no_longer_in_grid = not (
                    utils.is_in_matrix(
                        robot_cell_in_grid, inflated_grid_by_robot.d_width, inflated_grid_by_robot.d_height
                    )
                    and utils.is_in_matrix(
                        obstacle_cell_in_grid, inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height
                    )
            )
            if is_no_longer_in_grid:
                continue
            if inflated_grid_by_robot.grid[robot_cell_in_grid[0]][robot_cell_in_grid[1]] != 0:
                continue
            if inflated_grid_by_obstacle.grid[obstacle_cell_in_grid[0]][obstacle_cell_in_grid[1]] != 0:
                continue

            # Continue at static polygon level, check if still in map
            new_robot_polygon = action.apply(
                current_configuration.robot.polygon, current_configuration.robot.floating_point_pose)

            # Check if robot is still within map bounds
            if not new_robot_polygon.within(inflated_grid_by_robot.aabb_polygon):
                continue

            new_obstacle_polygon = action.apply(
                current_configuration.obstacle.polygon, current_configuration.robot.floating_point_pose)

            # Check if obstacle is still within map bounds
            if not new_obstacle_polygon.within(inflated_grid_by_obstacle.aabb_polygon):
                continue

            # Finally, we check dynamic collisions (between init configuration and after-action configuration)
            collision_pairs = b2_sim.check_actions_with_ghost(
                key=(robot_uid, obstacle_uid, current_configuration.manip_pose_id),
                entities_polygons={
                    robot_uid: current_configuration.robot.polygon,
                    obstacle_uid: current_configuration.obstacle.polygon
                },
                entities_poses={
                    robot_uid: current_configuration.robot.floating_point_pose,
                    obstacle_uid: current_configuration.obstacle.floating_point_pose
                },
                actions=[action], main_pose=current_configuration.robot.floating_point_pose
            )
            if collision_pairs:
                continue

            # If we are here, then this newly computed neighbor configuration is valid and we must save it
            neighbor_configuration = RobotObstacleConfiguration(
                robot_floating_point_pose=new_robot_pose, robot_polygon=new_robot_polygon,
                robot_fixed_precision_pose=robot_fixed_precision_pose, robot_cell_in_grid=robot_cell_in_grid,
                obstacle_floating_point_pose=new_obstacle_pose, obstacle_polygon=new_obstacle_polygon,
                obstacle_fixed_precision_pose=obstacle_fixed_precision_pose,
                obstacle_cell_in_grid=obstacle_cell_in_grid, action=action,
                manip_pose_id=current_configuration.manip_pose_id
            )

            neighbors.append(neighbor_configuration)
            tentative_g_scores.append(
                gscore[current_configuration] + extra_g_cost
            )

        self._rp.publish_manip_search_data(
            current_configuration, gscore, close_set, open_queue, came_from, neighbors, start,
            inflated_grid_by_robot.res, inflated_grid_by_robot.grid_pose, ns=self._robot_name
        )

        return neighbors, tentative_g_scores

    def get_transit_end_and_transfer_start_poses(self, obstacle_polygon, robot_max_inflation_radius,
                                                 robot_accessible_cells, res, grid_pose):
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
        :param robot_max_inflation_radius:
        :type robot_max_inflation_radius:
        :param robot_accessible_cells:
        :type robot_accessible_cells:
        :param res:
        :type res:
        :param grid_pose:
        :type grid_pose:
        :return: the lists of valid transit end poses and corresponding valid transfer start poses
        :rtype: tuple(list(tuple(float, float, float)), list(tuple(float, float, float)))
        """
        candidate_transfer_start_poses = utils.sample_poses_at_middle_of_inflated_sides(
            obstacle_polygon, robot_max_inflation_radius)
        candidate_transit_end_poses = utils.sample_poses_at_middle_of_inflated_sides(
            obstacle_polygon, robot_max_inflation_radius + 1.5 * res)

        valid_transit_end_poses, valid_transfer_start_poses = [], []
        for transit_end_pose, transfer_start_pose in zip(candidate_transit_end_poses, candidate_transfer_start_poses):
            transit_end_cell = utils.real_to_grid(transit_end_pose[0], transit_end_pose[1], res, grid_pose)
            if transit_end_cell in robot_accessible_cells:
                valid_transit_end_poses.append(transit_end_pose)
                valid_transfer_start_poses.append(transfer_start_pose)

        self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)
        self._rp.publish_q_manips_for_obs(valid_transit_end_poses + valid_transfer_start_poses, ns=self._robot_name)

        return valid_transit_end_poses, valid_transfer_start_poses

    def find_path(self, robot_pose, goal_pose, inflated_grid_by_robot, robot_polygon):
        raw_path = graph_search.real_to_grid_search_a_star(robot_pose, goal_pose, inflated_grid_by_robot)
        if raw_path:
            phys_cost = 0.
            raw_path_iterator = iter(raw_path)
            prev_step = next(raw_path_iterator)
            for cur_step in raw_path_iterator:
                phys_cost += self.g(prev_step, cur_step, is_transfer=False)
            return TransitPath.from_poses(raw_path, robot_polygon, robot_pose, phys_cost)
        else:
            return None

    @staticmethod
    def deduce_robot_goal_pose(robot_manip_pose, obs_init_pose, obs_goal_pose):
        translation, rotation = utils.get_translation_and_rotation(obs_init_pose, obs_goal_pose)
        robot_goal_point = list(utils.rotate_then_translate_polygon(
                Point((robot_manip_pose[0], robot_manip_pose[1])), translation, rotation,
                (obs_init_pose[0], obs_init_pose[1])
            ).coords[0])
        orientation = (robot_manip_pose[2] + rotation) % 360.
        orientation = orientation if orientation >= 0. else orientation + 360.
        return robot_goal_point[0], robot_goal_point[1], orientation

    @staticmethod
    def dijkstra_cc_and_cost(start_cell, grid, res, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD):
        straight_dist = res
        diag_dist = res * utils.SQRT_OF_2
        width, height = grid.shape

        frontier = []
        heapq.heappush(frontier, (0., start_cell))
        cost_so_far = {start_cell: 0.}

        while frontier:
            current = heapq.heappop(frontier)[1]
            for neighbor in utils.get_neighbors_no_coll(current, grid, width, height, neighborhood):
                extra_cost = straight_dist if current[0] == neighbor[0] or current[1] == neighbor[1] else diag_dist
                new_cost = cost_so_far[current] + extra_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor))

        return cost_so_far

    def new_sorted_cells_by_combined_cost(self, inflated_grid_by_obstacle,
                                          robot_polygon, robot_pose,
                                          obstacle_pose, goal_pose):
        # Initialize some need variables
        obstacle_cell = utils.real_to_grid(
            obstacle_pose[0], obstacle_pose[1],
            inflated_grid_by_obstacle.res, inflated_grid_by_obstacle.grid_pose
        )

        robot_poly_at_goal = utils.set_polygon_pose(
            robot_polygon, robot_pose, goal_pose
        )

        robot_cells_at_goal = utils.accurate_rasterize_in_grid(
            robot_poly_at_goal, inflated_grid_by_obstacle.res,
            inflated_grid_by_obstacle.grid_pose,
            inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height, fill=True
        )

        # Compute set of potentially reachable cells for obstacle and a heuristic cost to join them
        cell_to_cost = self.dijkstra_cc_and_cost(
            obstacle_cell, inflated_grid_by_obstacle.grid,
            inflated_grid_by_obstacle.res, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD
        )
        for cell in robot_cells_at_goal:
            if cell in cell_to_cost:
                del cell_to_cost[cell]

        # Filter cells where social == -1.
        for cell in cell_to_cost.keys():
            if self._social_costmap[cell[0]][cell[1]] == -1.:
                del cell_to_cost[cell]

        acc_cells_for_obs, distance_cost = list(cell_to_cost.keys()), np.array(list(cell_to_cost.values()))

        social_cost = np.array([
            self._social_costmap[cell[0]][cell[1]]
            for cell in acc_cells_for_obs])

        if not self.distance_to_obs_cost_is_realistic:
            distance_cost = np.array([
                utils.euclidean_distance(
                    utils.grid_to_real(
                        cell[0], cell[1], inflated_grid_by_obstacle.res, inflated_grid_by_obstacle.grid_pose
                    ), obstacle_pose
                )
                for cell in acc_cells_for_obs])

        distance_to_goal = np.array([
            utils.euclidean_distance(
                utils.grid_to_real(
                    cell[0], cell[1], inflated_grid_by_obstacle.res, inflated_grid_by_obstacle.grid_pose
                ), goal_pose
            )
            for cell in acc_cells_for_obs])

        normalized_social_cost = (social_cost - np.min(social_cost)) / np.ptp(social_cost)
        normalized_distance_cost = (distance_cost - np.min(distance_cost)) / np.ptp(distance_cost)
        normalized_distance_to_goal = (distance_to_goal - np.min(distance_to_goal)) / np.ptp(distance_to_goal)

        combined_cost = (self.w_social * normalized_social_cost
                         + self.w_obs * normalized_distance_cost
                         + self.w_goal * normalized_distance_to_goal) / self.w_sum

        sorted_cell_to_combined_cost = OrderedDict(
            sorted(zip(acc_cells_for_obs, combined_cost), key=lambda t: t[1], reverse=True)
        )

        cells_sorted_by_combined_cost = list(sorted_cell_to_combined_cost.keys())

        self._rp.cleanup_multigoal_a_star_close_set(ns=self._robot_name)
        self._rp.cleanup_grid_map(ns=self._robot_name)

        # TODO Rewrite display functions to only display what's relevant
        # self._rp.publish_combined_costmap(sorted_cell_to_combined_cost, dd, ns=self._robot_name)

        if self.activate_grids_logging:
            stocg.display_or_log(
                np.invert(
                    inflated_grid_by_obstacle.grid.astype(np.bool)
                ),
                "-obs_inf_grid", time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False, log_costmaps=True, abs_path_to_logs_dir=self.abs_path_to_logs_dir)

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
                normalized_distance_from_obs_costmap[cell[0]][cell[1]] = normalized_distance_cost[i]
                normalized_distance_from_goal_costmap[cell[0]][cell[1]] = normalized_distance_to_goal[i]

            stocg.display_or_log(
                normalized_social_cost_costmap, "-n_social_costmap", time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False, log_costmaps=True, abs_path_to_logs_dir=self.abs_path_to_logs_dir)
            stocg.display_or_log(
                normalized_distance_from_obs_costmap, "-n_d_to_obs_costmap", time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False, log_costmaps=True, abs_path_to_logs_dir=self.abs_path_to_logs_dir)
            stocg.display_or_log(
                normalized_distance_from_goal_costmap, "-n_d_to_goal_costmap", time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False, log_costmaps=True, abs_path_to_logs_dir=self.abs_path_to_logs_dir)

            combined_costmap = np.zeros((inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height))
            for cell, combined_cost in sorted_cell_to_combined_cost.items():
                combined_costmap[cell[0]][cell[1]] = combined_cost
            stocg.display_or_log(
                combined_costmap, "-combined_costmap", time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False, log_costmaps=True, abs_path_to_logs_dir=self.abs_path_to_logs_dir)

        return cells_sorted_by_combined_cost, sorted_cell_to_combined_cost

    def h(self, r_i, r_j):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        # rotation_cost = self.rotation_factor * (abs(r_j[2] - r_i[2]) % 180.)
        return translation_cost  # + rotation_cost

    def g(self, r_i, r_j, is_transfer=False):
        translation_cost = self.translation_factor * utils.euclidean_distance(r_j, r_i)
        rotation_cost = self.rotation_factor * abs(r_j[2] - r_i[2])
        return (translation_cost + rotation_cost) * (1. if not is_transfer else self.transfer_coefficient)


class RCHConfiguration:
    def __init__(self, cell, first_obstacle_uid, first_component_uid):
        self.cell = cell
        self.first_obstacle_uid = first_obstacle_uid
        self.first_component_uid = first_component_uid

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.cell == other
        else:
            return (
                    self.cell == other.cell
                    and self.first_obstacle_uid == other.first_obstacle_uid
                    and self.first_component_uid == other.first_component_uid
            )

    def __hash__(self):
        return hash((self.cell, self.first_obstacle_uid, self.first_component_uid))


class Configuration:
    def __init__(self, floating_point_pose, polygon, cell_in_grid, fixed_precision_pose,
                 action):
        self.floating_point_pose = floating_point_pose
        self.polygon = polygon
        self.cell_in_grid = cell_in_grid
        self.fixed_precision_pose = fixed_precision_pose
        self.action = action

    def __eq__(self, other):
        if isinstance(other, graph_search.HeapNode):
            return self.fixed_precision_pose == other.element.fixed_precision_pose
        elif isinstance(other, tuple):
            return self.fixed_precision_pose == other
        else:
            return self.fixed_precision_pose == other.fixed_precision_pose

    def __hash__(self):
        return hash(self.fixed_precision_pose)


class RobotObstacleConfiguration:
    def __init__(self, robot_floating_point_pose, robot_polygon, robot_cell_in_grid, robot_fixed_precision_pose,
                 obstacle_floating_point_pose, obstacle_polygon, obstacle_cell_in_grid, obstacle_fixed_precision_pose,
                 action=None, manip_pose_id=None):
        self.robot = Configuration(
            robot_floating_point_pose, robot_polygon, robot_cell_in_grid, robot_fixed_precision_pose,
            action=action
        )
        self.obstacle = Configuration(
            obstacle_floating_point_pose, obstacle_polygon, obstacle_cell_in_grid, obstacle_fixed_precision_pose,
            action=action
        )
        self.action = action
        self.manip_pose_id = manip_pose_id

    def __eq__(self, other):
        if isinstance(other, graph_search.HeapNode):
            return (
                    self.robot.fixed_precision_pose == other.element.robot.fixed_precision_pose
                    and self.obstacle.fixed_precision_pose == other.element.obstacle.fixed_precision_pose
            )
        elif isinstance(other, tuple):
            return (
                    self.robot.fixed_precision_pose == other[0]
                    and self.obstacle.fixed_precision_pose == other[1]
            )
        else:
            return (
                    self.robot.fixed_precision_pose == other.robot.fixed_precision_pose
                    and self.obstacle.fixed_precision_pose == other.obstacle.fixed_precision_pose
            )

    def __hash__(self):
        return hash((self.robot.fixed_precision_pose, self.obstacle.fixed_precision_pose))


class PlanValidityError(Exception):
    def __init__(self, *args):
        Exception(args)


class DynamicCollisionError(PlanValidityError):
    def __init__(self, collision_data, *args):
        self.colliding_entities = collision_data
        PlanValidityError(args)


class ObstacleNotAtStartPoseError(PlanValidityError):
    def __init__(self, obstacle_uid, *args):
        self.obstacle_uid = obstacle_uid
        PlanValidityError(args)


class HasInfiniteCostError(PlanValidityError):
    def __init__(self, *args):
        PlanValidityError(args)


class HasNoPathComponents(PlanValidityError):
    def __init__(self, *args):
        PlanValidityError(args)


class Path:
    def __init__(self, poses, polygons, actions, indexes=None):
        self.poses = poses
        self.polygons = polygons
        self.actions = actions
        self.indexes = indexes
        if self.indexes is None:
            self.reset_indexes()

    def is_empty(self):
        return not bool(self.indexes)

    def pop_next_step(self):
        # If there are steps left to execute in self.path, pop and return the first
        if self.indexes:
            return self.actions[self.indexes.pop(0)]
        else:
            # Return None otherwise
            return None

    def reset_indexes(self):
        self.indexes = [i for i in range(len(self.actions))]

    def is_path_started(self):
        return len(self.indexes) != len(self.actions)

    def is_path_at_last(self):
        return len(self.indexes) == 1

    # TODO Have these trans and rot precision values be passed from calling functions !
    def is_start_pose(self, pose, trans_mult=100., rot_mult=1.):
        fixed_precision_pose = utils.real_pose_to_fixed_precision_pose(pose, trans_mult, rot_mult)
        fixed_precision_self_pose = utils.real_pose_to_fixed_precision_pose(self.poses[0], trans_mult, rot_mult)
        return fixed_precision_pose == fixed_precision_self_pose

    def get_length(self):
        return len(self.indexes)


class TransferPath:
    def __init__(self, robot_path, obstacle_path, obstacle_uid, manip_pose_id,
                 phys_cost=None, social_cost=0., weight=1.):
        self.robot_path = robot_path
        self.obstacle_path = obstacle_path
        self.obstacle_uid = obstacle_uid
        self.manip_pose_id = manip_pose_id
        self.phys_cost = (
            phys_cost if phys_cost is not None
            else utils.sum_of_euclidean_distances(self.robot_path.poses) * weight
        )
        self.social_cost = social_cost
        self.total_cost = self.phys_cost + self.social_cost
        # TODO Remove this attribute that is currentlty kept to avoid circular dependency with ros_conversion.py
        self.is_transfer = True

    @classmethod
    def from_configurations(cls, prev_transit_end_configuration, next_transit_start_configuration,
                            transfer_configurations, obstacle_uid, phys_cost=None, social_cost=0., weight=1.):
        if not transfer_configurations:
            return None

        configurations_min_start = transfer_configurations[1:]
        robot_path = Path(
            poses=(
                [prev_transit_end_configuration.floating_point_pose]
                + [configuration.robot.floating_point_pose for configuration in transfer_configurations]
                + [next_transit_start_configuration.floating_point_pose]
            ),
            polygons=(
                [prev_transit_end_configuration.polygon]
                + [configuration.robot.polygon for configuration in transfer_configurations]
                + [next_transit_start_configuration.polygon]
            ),
            actions=(
                [prev_transit_end_configuration.action]
                + [configuration.action for configuration in configurations_min_start]
                + [next_transit_start_configuration.action]
            )
        )
        obstacle_path = Path(
            poses=[configuration.obstacle.floating_point_pose for configuration in transfer_configurations],
            polygons=[configuration.obstacle.polygon for configuration in transfer_configurations],
            actions=[configuration.action for configuration in configurations_min_start]
        )
        manip_pose_id = transfer_configurations[0].manip_pose_id
        return cls(robot_path, obstacle_path, obstacle_uid, manip_pose_id, phys_cost, social_cost, weight)

    def has_infinite_cost(self):
        return True if self.total_cost == float("inf") else False

    def is_empty(self):
        return self.robot_path.is_empty()

    def is_valid(self, all_entities_poses, robot_uid, b2_sim, check_horizon=None, check_start_pose=True):
        obstacle_pose, robot_pose = all_entities_poses[self.obstacle_uid], all_entities_poses[robot_uid]

        # Check that the obstacle that is to transferred is actually in its initial place
        # (only relevant if this transfer path has not already be started)
        if check_start_pose and not self.robot_path.is_path_started():
            obstacle_at_start_pose = self.obstacle_path.is_start_pose(obstacle_pose)
            if not obstacle_at_start_pose:
                raise ObstacleNotAtStartPoseError(self.obstacle_uid)

        # If the path is fully executed, it's valid
        if self.robot_path.is_empty() and self.obstacle_path.is_empty():
            return True

        # Check potential collisions caused by actions within horizon if one is specified
        if check_horizon is None:
            actions_to_check = [self.robot_path.actions[index] for index in self.robot_path.indexes]
        elif check_horizon == 0:
            return True
        else:
            actions_to_check = [self.robot_path.actions[index] for index in self.robot_path.indexes[:check_horizon]]

        collision_pairs = b2_sim.check_actions_with_ghost(
            key=(robot_uid, self.obstacle_uid, self.manip_pose_id),
            entities_polygons={
                robot_uid: self.robot_path.polygons[1],
                self.obstacle_uid: self.obstacle_path.polygons[1]
            },
            entities_poses={
                robot_uid: self.robot_path.poses[1],
                self.obstacle_uid: self.obstacle_path.poses[1]
            },
            actions=actions_to_check, main_pose=robot_pose
        )
        if collision_pairs:
            raise DynamicCollisionError(collision_pairs)

        return True

    def pop_next_step(self):
        if self.robot_path.is_path_started():
            # Only start popping obstacle_path when the robot_path has lost its first step (transit to transfer pose)
            self.obstacle_path.pop_next_step()
        return self.robot_path.pop_next_step()

    def get_length(self):
        return self.robot_path.get_length()

    def is_path_started(self):
        return self.robot_path.is_path_started()

    def is_path_at_last(self):
        return len(self.robot_path.indexes)


class TransitPath:
    def __init__(self, robot_path, phys_cost=None, social_cost=0., weight=1.):
        self.robot_path = robot_path
        self.phys_cost = (
            phys_cost if phys_cost is not None
            else utils.sum_of_euclidean_distances(self.robot_path.poses) * weight
        )
        self.social_cost = social_cost
        self.total_cost = self.phys_cost + self.social_cost
        # TODO Remove this attribute that is currentlty kept to avoid circular dependency with ros_conversion.py
        self.is_transfer = False

    @classmethod
    def from_poses(cls, poses, robot_polygon, robot_pose, phys_cost=None, social_cost=0., weight=1.):
        polygons = [utils.set_polygon_pose(robot_polygon, robot_pose, pose) for pose in poses]
        robot_path = Path(poses, polygons, [], indexes=[i for i in range(len(poses))])
        return cls(robot_path, phys_cost, social_cost, weight)

    def has_infinite_cost(self):
        return True if self.total_cost == float("inf") else False

    def is_empty(self):
        return self.robot_path.is_empty()

    def is_valid(self, inflated_grid_by_robot, check_horizon=None):
        if check_horizon is not None:
            if check_horizon and self.robot_path.indexes:
                # TODO: Remove this conversion by also directly saving the cell sequence in the object
                poses_to_check = [self.robot_path.poses[index] for index in self.robot_path.indexes[:check_horizon]]
            else:
                poses_to_check = []
        else:
            poses_to_check = [self.robot_path.poses[index] for index in self.robot_path.indexes]

        for pose in poses_to_check:
            cell = utils.real_to_grid(
                pose[0], pose[1], inflated_grid_by_robot.res, inflated_grid_by_robot.grid_pose
            )
            if inflated_grid_by_robot.grid[cell[0]][cell[1]] != 0:
                raise DynamicCollisionError(inflated_grid_by_robot.obstacles_uids_in_cell(cell))

        return True

    def pop_next_step(self):
        # If there are steps left to execute in self.path, pop and return the first
        if self.robot_path.indexes:
            return self.robot_path.poses[self.robot_path.indexes.pop(0)]
        else:
            # Return None otherwise
            return None

    def get_length(self):
        return self.robot_path.get_length()


class Plan:
    def __init__(self, path_components, goal, robot_uid):
        self.path_components = path_components
        self.goal = goal
        self.robot_uid = robot_uid
        self.phys_cost = 0.0
        self.social_cost = 0.0
        self.total_cost = 0.0
        if path_components:
            for path in path_components:
                self.phys_cost = self.phys_cost + path.phys_cost
                self.social_cost = self.social_cost + path.social_cost
                self.total_cost = self.total_cost + path.total_cost
        else:
            self.phys_cost = float("inf")
            self.social_cost = float("inf")
            self.total_cost = float("inf")

    def append(self, future_plan):
        self.path_components += future_plan.path_components
        self.phys_cost += future_plan.phys_cost
        self.social_cost += future_plan.social_cost
        self.total_cost += future_plan.total_cost
        return self

    def has_infinite_cost(self):
        return True if self.total_cost == float("inf") else False

    def is_empty(self):
        if self.path_components:
            for path_component in self.path_components:
                if not path_component.is_empty():
                    return False
            return True
        else:
            return True

    def is_valid(self, all_entities_poses, b2_sim, inflated_grid_by_robot, check_horizon=None):
        # Basic checks
        if self.has_infinite_cost():
            raise HasInfiniteCostError()
        if not self.path_components:
            raise HasNoPathComponents()

        # Check validity of each component
        shared_horizon = check_horizon
        previously_moved_entities_uids = set()
        for counter, path in enumerate(self.path_components):
            if shared_horizon is None or shared_horizon > 0:
                if isinstance(path, TransitPath):
                    path.is_valid(inflated_grid_by_robot, check_horizon=shared_horizon)
                elif isinstance(path, TransferPath):
                    # If the previously checked path components are valid, we assume it leaves any manipulated
                    # obstacles in the right place so we don't check again:
                    # - We simply deactivate collisions with them from the world representation
                    # - or if another path component needs to move them (check_start_pose)
                    previously_moved_entities_uids.add(path.obstacle_uid)
                    inflated_grid_by_robot.deactivate_entities([path.obstacle_uid])
                    b2_sim.deactivate_entities([path.obstacle_uid])
                    check_start_pose = path.obstacle_uid not in previously_moved_entities_uids

                    path.is_valid(
                        all_entities_poses, self.robot_uid, b2_sim, shared_horizon, check_start_pose
                    )
                else:
                    raise TypeError('Expected TransitPath or TransferPath instance.')

                if shared_horizon:
                    shared_horizon = 0 if path.get_length() >= shared_horizon else shared_horizon - path.get_length()
            else:
                if shared_horizon <= 0:
                    break

        # Reactivate entities that had been deactivated during checks
        inflated_grid_by_robot.activate_entities(previously_moved_entities_uids)
        b2_sim.activate_entities(previously_moved_entities_uids)

        return True

    def pop_next_step(self):
        """
        Get the next plan step to execute
        :return: the action object to be executed if there is one, None if the plan is empty
        :rtype: action or None
        """
        if self.is_empty():
            return None

        first_component = self.path_components[0]
        if not self.path_components[0].is_empty():
            # If the current first component is not empty, use it
            current_component = first_component
        else:
            # Else, if it is empty, it's finished, pop it,
            self.path_components.pop(0)
            # use the next component if there is one
            if self.path_components:
                current_component = self.path_components[0]
            else:
                return None

        if isinstance(current_component, TransitPath):
            return ba.GoToPose(pose=current_component.pop_next_step())
        elif isinstance(current_component, TransferPath):
            next_step = current_component.pop_next_step()
            if next_step:
                return next_step
            else:
                return ba.Wait()
        else:
            raise TypeError('A plan can only pop steps from TransitPath or TransferPath object.')

    def executed_only_first_step(self):
        """
        Temporary badly written function to prevent infinite looping because of incoherence between plan computation
        validity checks (based on b2_sim) and plan verification validity checks (based on custom convex approach)
        :return:
        :rtype:
        """
        if not self.is_empty():
            path = self.path_components[0]
            if len(path.robot_path.poses) - 1 == len(path.robot_path.indexes):
                return True
        return False
