import copy
import heapq
import math
import numpy as np
import time
from shapely.geometry import LineString, Point, box
from shapely import affinity
from shapely.ops import cascaded_union

from baseline_behavior import BaselineBehavior
from src.behaviors.algorithms.a_star import a_star_real_path
from src.behaviors.plan.path import Path
from src.behaviors.plan.plan import Plan
from src.behaviors.algorithms.multi_goal_a_star import multi_goal_a_star_real_path
from src.utils import utils
from src.worldreps.entity_based.obstacle import Obstacle
from src.worldreps.entity_based.robot import Robot
from src.behaviors.algorithms.new_local_opening_check import check_new_local_opening  # , is_move_passing_over_pose
from plan.basic_actions import ActionGoalFailure, ActionGoalsFinished, ActionGoalSuccess
from src.worldreps.entity_based.custom_exceptions import IntersectionError
from src.worldreps.occupation_based.binary_occupancy_grid import BinaryOccupancyGrid
from src.worldreps.occupation_based.binary_inflated_occupancy_grid import BinaryInflatedOccupancyGrid
import src.worldreps.occupation_based.social_topological_occupation_cost_grid as stocg


class Stilman2005Behavior(BaselineBehavior):
    def __init__(self, initial_world, robot_uid, navigation_goals, behavior_config, abs_path_to_logs_dir):
        BaselineBehavior.__init__(
            self, initial_world, robot_uid, navigation_goals, behavior_config, abs_path_to_logs_dir)

        # Configuration parameters
        parameters = behavior_config["parameters"]
        # - Original Stilman method configuration parameters
        self.alpha = parameters["alpha_for_obstacle_choice_heur"]
        if parameters["neighborhood_for_obstacle_choice_heur"] == "TAXI":
            self.neighborhood = utils.TAXI_NEIGHBORHOOD
        elif parameters["neighborhood_for_obstacle_choice_heur"] == "CHESSBOARD":
            self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD
        else:
            self.neighborhood = utils.TAXI_NEIGHBORHOOD  # default if bad parameter
        self.heur_w = parameters["heuristic_cost_for_traversing_obstacle_in_choice_heur"]
        self.basic_trans_force = parameters["basic_translation_force"]
        self.basic_rot_moment = parameters["basic_rotation_moment"]
        # - Robot action space parameters
        self.angular_res = parameters["collision_check_angular_res"]
        self.rotation_unit_angle = parameters["robot_rotation_unit_angle"]
        self.translation_unit_length = parameters["robot_translation_unit_length"]
        self.forbid_rotations = parameters["forbid_rotations"]

        # - S-NAMO parameters
        self.use_social_cost = parameters["use_social_cost"]
        self.bound_percentage = parameters["solution_interval_bound_percentage"]
        if parameters["manipulation_search_procedure"] == "DFS":
            if self.use_social_cost:
                self.manip_search_procedure = self._focused_manip_search
            else:
                raise ValueError("Focused manipulation search requires the use_social_cost variable to be True !")
        elif parameters["manipulation_search_procedure"] == "BFS":
            self.manip_search_procedure = self._manip_search
        self.w_social, self.w_obs, self.w_goal = 15., 10., 2.
        self.w_sum = self.w_social + self.w_obs + self.w_goal
        self.distance_to_obs_cost_is_realistic = True

        # - Extra performance parameters
        self.check_new_local_opening_before_global = parameters["check_new_local_opening_before_global"]
        self.activate_grids_logging = not parameters["deactivate_grids_logging"]

        self._trans_vectors = np.array([(self.translation_unit_length, 0.), (-self.translation_unit_length, 0.)])
        if self.forbid_rotations:
            self._rot_angles = np.array([])
        else:
            self._rot_angles = np.array([self.rotation_unit_angle, -self.rotation_unit_angle])
        self._all_rot_angles = self.rotation_unit_angle * np.array(range(1, 360 // int(self.rotation_unit_angle)))
        self._nb_possible_angles = len(self._all_rot_angles)

        self._actions = []
        for trans_vector in self._trans_vectors:
            self._actions.append(Translation(trans_vector, self.translation_unit_length))
        for rot_angle in self._rot_angles:
            self._actions.append(Rotation(rot_angle))

        self._social_costmap = None

        self.is_first_transfer_step = False

    def think(self):
        if self._navigation_goals or self._q_goal is not None:
            if self._q_goal is None:
                self._q_goal = self._navigation_goals.pop(0)
                self._p_opt = Plan([], self._q_goal)

            q_r = self._robot.pose

            # TODO Extract abs_tol constant and make it a parameter for each goal
            is_close_enough_to_goal = all(np.isclose(q_r, self._q_goal, rtol=1e-5))
            if is_close_enough_to_goal:
                print("SUCCESS: Agent '{name}' has successfully reached pose {nav_goal}.".format(
                    name=self._robot.name, nav_goal=str(self._q_goal)))
                action = ActionGoalSuccess(self._q_goal)
                self._q_goal = None
                return action

            if not self._p_opt.is_valid(self._world, self._robot_uid):
                if self.use_social_cost and self._social_costmap is None:
                    movable_entities_uids = [uid for uid, entity in self._world.entities.items()
                                             if isinstance(entity, Robot) or (isinstance(entity, Obstacle)
                                             and self._robot.deduce_movability(entity.type) != "unmovable")]
                    static_occ_grid = BinaryOccupancyGrid(
                        self._world.dd.d_width, self._world.dd.d_height, self._world.dd.res, self._world.dd.grid_pose,
                        self._world.dd.inflation_radius, self._world.entities,
                        entities_to_ignore=movable_entities_uids + [self._robot_uid]).get_grid()
                    self._social_costmap = stocg.compute_social_costmap(
                        static_occ_grid, self._world.dd.res, log_costmaps=self.activate_grids_logging,
                        abs_path_to_logs_dir=self.abs_path_to_logs_dir, ns=self._robot_name)
                    self._rp.publish_grid_map(self._social_costmap, self._world.dd.res, ns=self._robot_name)
                    # time.sleep(3.)

                # top_1 = time.time()
                self._p_opt = self._select_connect(self._world, set(), self._q_goal)
                # top_2 = time.time()
                # duration = top_2 - top_1
                # print(duration)
                self.is_first_transfer_step = False

            if self._p_opt is not None and not self._p_opt.is_empty():
                next_step = self._p_opt.pop_next_step()
                if not self.is_first_transfer_step and next_step.is_transfer:
                    self.is_first_transfer_step = True
                    next_step.is_transfer = False  # HACK so that the first step does not cause manipulation of object
                elif self.is_first_transfer_step and not next_step.is_transfer:
                    self.is_first_transfer_step = False
                return next_step
            elif self._p_opt is None or self._p_opt.has_infinite_cost():
                print("FAILURE: Agent '{name}' has failed to reach pose {nav_goal}.".format(
                    name=self._robot.name, nav_goal=str(self._q_goal)))
                action = ActionGoalFailure(self._q_goal)
                self._q_goal = None
                return action

        else:
            print("FINISH: Agent '{name}' has finished trying to reach its goals !".format(name=self._robot.name))
            return ActionGoalsFinished()

    def _select_connect(self, w_t, prev_list, r_f, busy_obstacles=tuple()):
        """
        High Level Planner _select_connect (SC).
        It makes use of _rch and _manip_search in a greedy heuristic search with backtracking.
        It backtracks locally when the object selected by _rch cannot be moved to merge the selected c_1 \in c_free.
        It backtracks globally when all the paths identified by _rch from c_1 are unsuccessful.
        SC calls _find_path to determine a transit path from r_t to a contact point, r_t_plus_1 . The existence of the
        path is guaranteed by the choice of contacts in Manip-Search.
        :param w_t: state of the world at time t
        :param prev_list: list of previously visited free space components c_j
        :param r_f: goal robot configuration [x, y, theta] in {m, m, degrees}
        :return: None to backtrack, current partial plan otherwise.
        """
        r_t = w_t.entities[self._robot.uid].pose

        avoid_list = set()

        simple_path_to_goal = self._find_path(w_t, r_t, r_f)
        if simple_path_to_goal:
            # If the goal is in the same free space component as the robot in simulated w_t
            # Orig. condition in pseudo-code is : x^f in C^acc_R(W)
            return Plan([Path(simple_path_to_goal)], self._q_goal)

        o_1, c_1 = self._rch(w_t, avoid_list, prev_list, r_f, busy_obstacles)
        while (o_1, c_1) != (None, None):
            c_1_cells_set = w_t.get_connected_components_grid((self._robot_uid,)).get_components()[c_1]
            w_t_plus_2, tho_n, tho_m, cost = self.manip_search_procedure(w_t, o_1, c_1_cells_set, r_f)

            if tho_m is not None:
                prev_list.add(c_1)
                future_plan = self._select_connect(w_t_plus_2, prev_list, r_f, busy_obstacles)
                if future_plan is not None:
                    # Following line comes from original algorithm, but does not make sense ?
                    # tho_n = self._find_path(w_t, x_t, tho_m[0])
                    future_plan.path_components[0].obstacle_uid = o_1  #
                    return Plan([tho_n, tho_m], self._q_goal).append(future_plan)

            avoid_list.add((o_1, c_1))

            o_1, c_1 = self._rch(w_t, avoid_list, prev_list, r_f, busy_obstacles)

        return None

    def _rch(self, w_t, avoid_list, prev_list, r_f, busy_obstacles=tuple()):
        """
        Relaxed Constraint Heuristic (RCH)
        It is a navigation planner derived from A* that allows collisions with movable obstacles, with the following
        restrictions :
        - Rule #1: The robot can not enter cells occupied by obstacles it cannot move
        - Rule #2: The robot cannot transition from a cell occupied by one obstacle to a cell occupied by another
        - Rule #3: For any pair (Oi, Cj) in avoid_list, the robot cannot transition consecutively from acc_r_com
          to cells occupied by Oi and then cells occupied by Cj
        Ultimately, it returns a pair of obstacle + disconnected configuration space component that are the most likely
        to provide a good plan for select_connect.
        :param w_t: state of the world at time t
        :type w_t: World
        :param avoid_list: set of obstacle+free space component (o_i , c_i) pairs to avoid
        :type avoid_list: set(tuple(int, int))
        :param prev_list: set of previously visited free space components c_j
        :type prev_list: set(int)
        :param r_f: goal robot configuration [x, y, theta] in {m, m, degrees}
        :type r_f: tuple(float, float, float)
        :return: the pair (o_1 , c_1) of the first obstacle in the path, and the first component id of free space.
        Returns (None, None) if no path exists.
        :rtype: tuple(int, int) or tuple(None, None)
        """
        connected_components_obj = w_t.get_connected_components_grid((self._robot_uid,))
        connected_components_grid = connected_components_obj.get_grid()
        self._rp.publish_connected_components_grid(connected_components_grid, w_t.dd, ns=self._robot_name)

        movable_entities_uids = [
            uid for uid, entity in w_t.entities.items()
            if uid == self._robot_uid or self._robot.deduce_movability(entity.type) != "unmovable"]
        static_obs_grid = w_t.get_binary_inflated_occupancy_grid(tuple(movable_entities_uids)).get_grid()

        robot_pose = w_t.entities[self._robot_uid].pose
        start_cell = utils.real_to_grid(robot_pose[0], robot_pose[1], w_t.dd.res, w_t.dd.grid_pose)
        goal_cell = utils.real_to_grid(r_f[0], r_f[1], w_t.dd.res, w_t.dd.grid_pose)

        # The set of nodes already evaluated
        close_set = set()

        # The dictionary that remembers for each node, the cost of getting from the start node to that node.
        # The cost of going from start to start is zero.
        gscore = {start_cell: 0}

        # The set of currently discovered nodes that are not evaluated yet.
        open_queue = []
        # Initially, only the start node is known.
        heapq.heappush(
            open_queue, HeapQueueElement(
                f=self.__h(start_cell, goal_cell), cell=start_cell, obs_id=0, comp_id=0))

        self._rp.cleanup_a_star_close_set(ns=self._robot_name)
        # self._rp.publish_rch_open_queue(open_queue, w_t.dd.res, w_t.dd.grid_pose, ns=self._robot_name)

        # While open_heap is not empty == While there are discovered nodes that have not been evaluated
        while open_queue:

            # The node in open_heap having the lowest fScore[] value
            current = heapq.heappop(open_queue)
            # self._rp.publish_rch_open_queue(open_queue, w_t.dd.res, w_t.dd.grid_pose, ns=self._robot_name)
            self._rp.publish_current_cell(current.cell, w_t.dd.res, w_t.dd.grid_pose, ns=self._robot_name)

            # Exit early if goal is reached
            if current.cell == goal_cell:
                self._rp.publish_rch_closed_set(close_set, w_t.dd.res, w_t.dd.grid_pose, ns=self._robot_name)
                # self._rp.cleanup_rch_closed_set()
                return current.obs_id, current.comp_id

            close_set.add(current.cell)
            self._rp.publish_rch_closed_set(close_set, w_t.dd.res, w_t.dd.grid_pose, ns=self._robot_name)

            # For each neighbor of current node in the defined neighborhood
            for i, j in self.neighborhood:
                neighbor_cell = current.cell[0] + i, current.cell[1] + j
                # self._rp.publish_current_neighbor(neighbor_cell, w_t.dd.res, w_t.dd.grid_pose, ns=self._robot_name)

                # If neighbor's g score has not been computed yet, assign +inf
                if neighbor_cell not in gscore:
                    gscore[neighbor_cell] = float("inf")

                # Check that neighbor exists within the map, has not already been evaluated, and verify Rule #1
                # (don't consider passing through unmovable obstacles)
                if (utils.is_in_matrix(neighbor_cell, w_t.dd.d_width, w_t.dd.d_height)
                        and neighbor_cell not in close_set
                        and static_obs_grid[neighbor_cell[0]][neighbor_cell[1]] == 0):

                    # The cost from start to a neighbor.
                    tentative_g_score = self.__g(
                        current.cell, neighbor_cell, gscore[current.cell], connected_components_grid)
                    # WAS BEFORE : tentative_g_score =  gscore[current.cell] + dist_between(current.cell, neighbor_cell)

                    # Discover a new node or update info about known one :
                    if tentative_g_score < gscore[neighbor_cell] or neighbor_cell not in [i.cell for i in open_queue]:
                        # This path is the best until now. Record it!
                        gscore[neighbor_cell] = tentative_g_score
                        fscore_neighbor_cell = tentative_g_score + self.__h(neighbor_cell, goal_cell)

                        path_has_traversed_first_disconnected_comp = current.comp_id != 0

                        if path_has_traversed_first_disconnected_comp:
                            # After the path has traversed its first disconnected component, keep remembering it and the
                            # traversed obstacle to get to it as the firsts !
                            self.__enqueue(open_queue, neighbor_cell, fscore_neighbor_cell, current.obs_id, current.comp_id)
                        else:
                            # If no other disconnected component has been traversed yet, take a look at whether we have
                            # traversed an obstacle or not
                            path_has_traversed_first_obstacle = current.obs_id != 0

                            neighbor_component = connected_components_grid[neighbor_cell[0]][neighbor_cell[1]]
                            is_neighbor_in_c_r_free = neighbor_component > 0

                            if path_has_traversed_first_obstacle:
                                # If the path has already traversed its first obstacle...

                                if is_neighbor_in_c_r_free:
                                    # ...and the neighbor cell is in the free space...
                                    if (neighbor_component not in prev_list
                                            and (current.obs_id, neighbor_component) not in avoid_list):
                                        # ...and the component associated with this cell is not in any blacklist, then
                                        # associate the neighbor cell with the same obstacle and the new component
                                        # --> Application of Rule #3
                                        self.__enqueue(open_queue, neighbor_cell, fscore_neighbor_cell,
                                                       current.obs_id, neighbor_component)
                                    else:
                                        print("Path has not traversed its first disconnected component but has"
                                              "traversed its first obstacle, and neighbor is in c_r_free, but "
                                              "its component is in prev_list or avoid_list.")
                                else:
                                    # ...and the neighbor cell is in an obstacle...
                                    o_uid = self.__robot_exc_contained_in_obs(neighbor_cell, w_t)
                                    is_obs_traversed_by_neighbor_cell_same_as_current = o_uid == current.obs_id
                                    if is_obs_traversed_by_neighbor_cell_same_as_current:
                                        # ...and this obstacle is the same as the current one so we associate the
                                        # neighbor cell with it too. Application of Rule #2
                                        self.__enqueue(open_queue, neighbor_cell, fscore_neighbor_cell,
                                                       current.obs_id, 0)
                            else:
                                # If the path has not traversed its first obstacle yet...

                                if is_neighbor_in_c_r_free:
                                    # ...and the neighbor cell is in the free space (in the initial component),
                                    # associate the cell with no obstacle and with the initial component
                                    self.__enqueue(open_queue, neighbor_cell, fscore_neighbor_cell, 0, 0)
                                    continue

                                o_uid = self.__robot_exc_contained_in_obs(neighbor_cell, w_t)
                                is_valid_obstacle_uid = o_uid > 0 and o_uid not in busy_obstacles
                                if is_valid_obstacle_uid:
                                    # If the neighbor cell is not in the free space but in an obstacle, and if it is
                                    # exclusively in this obstacle, then associate the cell with this obstacle and the
                                    # initial component. Application of Rule #2
                                    self.__enqueue(open_queue, neighbor_cell, fscore_neighbor_cell, o_uid, 0)

                        # self._rp.publish_rch_open_queue(open_queue, w_t.dd.res, w_t.dd.grid_pose, ns=self._robot_name)
        return None, None

    def _manip_search(self, w_t, o_1, c_1_cells_set, r_f):
        # 1 - Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = copy.deepcopy(w_t)
        self._rp.publish_robot_sim_world(w_t_plus_2, self._robot_uid, ns=self._robot_name)
        obstacle = w_t_plus_2.entities[o_1]
        robot = w_t_plus_2.entities[self._robot.uid]
        binary_inflated_occupancy_grid = w_t_plus_2.get_binary_inflated_occupancy_grid((self._robot.uid,))
        dd = w_t_plus_2.dd
        other_entities = [entity for entity in w_t_plus_2.entities.values()
                          if entity.uid != robot.uid and entity.uid != obstacle.uid]
        map_box = box(*w_t_plus_2.get_map_bounds())

        goal_cell = utils.real_to_grid(r_f[0], r_f[1], dd.res, dd.grid_pose)

        # 2 - Get accessible sampled navigation points around obstacle and paths to them
        nav_poses, manip_poses, cost_and_paths = self.get_manip_poses_and_paths(
            obstacle, robot, binary_inflated_occupancy_grid, dd.res, dd.grid_pose)

        # 3 - Explore robot action space
        # 3.a - Initialize list of nodes to be explored with action leaves starting at each valid manipulation pose.
        # Ordered by min phys cost.
        init_action_leaves_to_explore = self.get_init_action_leaves_to_explore(
            nav_poses, manip_poses, cost_and_paths, robot, obstacle, other_entities)

        # 3.b - Iterate over action trees and extract the best action leaf out of each
        nb_states_explored = 0
        nb_states_evaluated_for_collisions = 0
        nb_states_evaluated_for_local_opening = 0
        nb_states_evaluated_for_global_opening = 0
        nb_states_with_global_opening = 0

        best_action_leaf_for_tree = []
        for init_action_leaf_heap_node in init_action_leaves_to_explore:
            # Action tree nodes to currently explore. Ordered by min phys cost.
            cur_action_leaves_to_explore = [init_action_leaf_heap_node]
            # Action tree nodes to explore at next Breadth-FS iteration. Ordered by min phys cost.
            next_action_leaves_to_explore = []
            # Action tree nodes for which there is a successful plan. Ordered by total min cost.
            successful_action_tree_nodes = []

            best_in_cur_action_leaves_to_explore = cur_action_leaves_to_explore[0]
            best_in_successful_action_tree_nodes = None

            init_action_leaf = init_action_leaf_heap_node.action_tree_node
            self._rp.publish_sim(init_action_leaf.robot.polygon, init_action_leaf.obstacle.polygon, "/init", ns=self._robot_name)

            manip_robot_d_pose = utils.real_pose_to_grid_pose(
                init_action_leaf.robot.pose, dd.res, dd.grid_pose, self.rotation_unit_angle)
            orig_obs_d_pose = utils.real_pose_to_grid_pose(
                init_action_leaf.obstacle.pose, dd.res, dd.grid_pose, self.rotation_unit_angle)
            evaluated_configurations = {(manip_robot_d_pose, orig_obs_d_pose)}

            # As long as the ascending physical cost of the currently explored action leaf is lower than the best
            # successful action leaf, and there are leaves to be explored,
            # we explore the action tree (this bounds the exploration).
            while ((best_in_successful_action_tree_nodes is None or
                    best_in_cur_action_leaves_to_explore is None or
                    (int(best_in_cur_action_leaves_to_explore.phys_cost * self.rounder)
                     < int(best_in_successful_action_tree_nodes.phys_cost * self.rounder)))
                   and cur_action_leaves_to_explore):
                for action_heap_node in cur_action_leaves_to_explore:
                    leaf = action_heap_node.action_tree_node
                    for action in self._actions:
                        nb_states_explored += 1

                        old_robot, old_obstacle = leaf.robot, leaf.obstacle

                        # I - Check that the robot+obstacle has not been evaluated yet (avoids cycles in search)
                        pred_robot_pose, pred_obstacle_pose = action.predicted_pose(old_robot, old_obstacle)
                        pred_robot_d_pose = utils.real_pose_to_grid_pose(pred_robot_pose, dd.res, dd.grid_pose, self.rotation_unit_angle)
                        pred_obstacle_d_pose = utils.real_pose_to_grid_pose(pred_obstacle_pose, dd.res, dd.grid_pose, self.rotation_unit_angle)
                        predicted_d_config = (pred_robot_d_pose, pred_obstacle_d_pose)
                        if predicted_d_config in evaluated_configurations:
                            continue
                        # Add configuration to evaluated configurations list
                        evaluated_configurations.add(predicted_d_config)

                        # II - Check whether the robot or the obstacle enter in collision because of the action
                        nb_states_evaluated_for_collisions += 1

                        new_robot = old_robot.light_copy()
                        new_obstacle = old_obstacle.light_copy()
                        try:
                            new_robot, new_obstacle = action.apply(new_robot, new_obstacle, other_entities)
                            if not new_robot.polygon.within(map_box):
                                raise IntersectionError({robot.uid}, "Out of map bounds !")
                            if not new_obstacle.polygon.within(map_box):
                                raise IntersectionError({obstacle.uid}, "Out of map bounds !")
                        except IntersectionError:
                            continue
                        self._rp.publish_sim(new_robot.polygon, new_obstacle.polygon, "/target", ns=self._robot_name)

                        # III - Check if there is a new (local/global) opening
                        has_new_global_op, has_new_local_op, skipped_global_op_check = self._is_there_opening_to_c_1(
                            binary_inflated_occupancy_grid, dd.res, dd.grid_pose, pred_robot_pose,
                            c_1_cells_set, robot, obstacle, old_obstacle, new_obstacle, leaf.has_new_local_opening,
                            goal_cell
                        )
                        nb_states_evaluated_for_local_opening += 1 if self.check_new_local_opening_before_global else 0
                        nb_states_evaluated_for_global_opening += 0 if skipped_global_op_check else 1
                        nb_states_with_global_opening += 1 if has_new_global_op else 0

                        # IV - Always compute physical cost, but don't bother computing social cost if no opening
                        phys_cost = leaf.phys_cost + self.__manip_e(old_robot.pose, new_robot.pose)
                        social_cost = 0. if not (self.use_social_cost and has_new_global_op) else sum(
                            [self._social_costmap[cell[0]][cell[1]]
                             for cell in new_obstacle.get_discrete_cells_set(
                                w_t_plus_2.dd.inflation_radius, w_t_plus_2.dd.res, w_t_plus_2.dd.grid_pose,
                                w_t_plus_2.dd.d_width, w_t_plus_2.dd.d_height)
                             if self._social_costmap[cell[0]][cell[1]] != -1.])
                        new_leaf = ActionTreeNode(phys_cost=phys_cost, social_cost=social_cost, comb_cost=0.,
                                                  parent=leaf, action=action,
                                                  robot=new_robot, obstacle=new_obstacle,
                                                  has_new_local_opening=has_new_local_op)
                        heapq.heappush(
                            next_action_leaves_to_explore, ActionHeapNode(new_leaf.phys_cost, new_leaf))

                        if has_new_global_op:
                            heapq.heappush(
                                successful_action_tree_nodes, ActionHeapNode(new_leaf.phys_cost, new_leaf))

                cur_action_leaves_to_explore = next_action_leaves_to_explore
                next_action_leaves_to_explore = []
                best_in_cur_action_leaves_to_explore = (None if not cur_action_leaves_to_explore
                                                        else cur_action_leaves_to_explore[0].action_tree_node)
                best_in_successful_action_tree_nodes = (None if not successful_action_tree_nodes
                                                        else successful_action_tree_nodes[0].action_tree_node)

            if best_in_successful_action_tree_nodes is not None:
                heapq.heappush(
                    best_action_leaf_for_tree, ActionHeapNode(best_in_successful_action_tree_nodes.phys_cost,
                                                              best_in_successful_action_tree_nodes))

        print("nb_states_explored = {}".format(nb_states_explored))
        print("nb_states_evaluated_for_collisions = {}".format(nb_states_evaluated_for_collisions))
        print("nb_states_evaluated_for_local_opening = {}".format(nb_states_evaluated_for_local_opening))
        print("nb_states_evaluated_for_global_opening = {}".format(nb_states_evaluated_for_global_opening))
        print("nb_states_with_global_opening = {}".format(nb_states_with_global_opening))

        # 6. Get best action branch from best action tree leaf and get its corresponding contact pose,
        # transit path to contact pose tho_n, transfer path from contact pose tho_m and overall cost
        best_action_tree_leaf = best_action_leaf_for_tree[0].action_tree_node
        best_action_branch = self.__get_actions_branch(best_action_tree_leaf)
        cost = best_action_branch[-1].phys_cost
        tho_n, tho_m = self.__actions_branch_to_path(best_action_branch)

        # 7. Update w_t_plus_2 to represent with final state that corresponds to best action branch result
        final_robot = best_action_branch[-1].robot
        final_obstacle = best_action_branch[-1].obstacle
        w_t_plus_2.set_entity_polygon(final_robot.uid, final_robot.polygon, final_robot.full_geometry_acquired)
        w_t_plus_2.set_entity_polygon(final_obstacle.uid, final_obstacle.polygon, final_obstacle.full_geometry_acquired)
        self._rp.publish_robot_sim_world(w_t_plus_2, self._robot_uid, ns=self._robot_name)

        # Update displays
        self._rp.cleanup_robot_sim(ns=self._robot_name)
        self._rp.publish_sim(final_robot.polygon, final_obstacle.polygon, "/target", ns=self._robot_name)
        # cc_grid.re_init_grid(binary_inflated_occupancy_grid.get_grid())
        # self._rp.publish_connected_components_grid(cc_grid.get_grid(), dd)
        return w_t_plus_2, tho_n, tho_m, cost

    def _focused_manip_search(self, w_t, o_1, c_1_cells_set, r_f):
        # 1 - Initialize manip search simulation world and some shortcut variables
        start_time = time.time()
        w_t_plus_2 = copy.deepcopy(w_t)
        self._rp.publish_robot_sim_world(w_t_plus_2, self._robot_uid, ns=self._robot_name)
        obstacle = w_t_plus_2.entities[o_1]
        robot = w_t_plus_2.entities[self._robot.uid]
        binary_inflated_occupancy_grid = w_t_plus_2.get_binary_inflated_occupancy_grid((self._robot.uid,))
        dd = w_t_plus_2.dd
        other_entities = [entity for entity in w_t_plus_2.entities.values()
                          if entity.uid != robot.uid and entity.uid != obstacle.uid]
        map_box = box(*w_t_plus_2.get_map_bounds())
        robot_d_pose = utils.real_pose_to_grid_pose(robot.pose, dd.res, dd.grid_pose, self.rotation_unit_angle)
        goal_cell = utils.real_to_grid(r_f[0], r_f[1], dd.res, dd.grid_pose)

        # 2 - Get accessible sampled navigation points around obstacle and paths to them
        nav_poses, manip_poses, cost_and_paths = self.get_manip_poses_and_paths(
            obstacle, robot, binary_inflated_occupancy_grid, dd.res, dd.grid_pose)

        # 2bis Get potentially accessible cells for obstacle ordered by associated combined costs
        sorted_cells = self.sorted_cells_by_combined_cost(w_t_plus_2, obstacle, robot, dd, r_f)

        # 3 - Explore robot action space
        # 3.a - Initialize list of nodes to be explored with action leaves starting at each valid manipulation pose.
        # Ordered by min phys cost.
        init_action_leaves_to_explore = self.get_init_action_leaves_to_explore(
            nav_poses, manip_poses, cost_and_paths, robot, obstacle, other_entities)

        # 3.b - Iterate over action trees and extract the best action leaf out of each
        nb_states_explored = 0
        nb_states_evaluated_for_collisions = 0
        nb_states_evaluated_for_local_opening = 0
        nb_states_evaluated_for_global_opening = 0
        nb_states_with_global_opening = 0

        last_goal_obs_poses = []
        best_leaf = None

        robot_collision_cache = {}
        obstacle_collision_cache = {}

        while best_leaf is None:
            obs_goal_pose, goal_obs_poly = self.find_best_goal(
                sorted_cells, last_goal_obs_poses, dd.res, dd.grid_pose, obstacle, other_entities,
                binary_inflated_occupancy_grid, robot.pose, c_1_cells_set, robot, goal_cell)
            goal_obs_cell = utils.real_to_grid(obs_goal_pose[0], obs_goal_pose[1], dd.res, dd.grid_pose)

            nb_best_cells = int(math.ceil(self.bound_percentage * float(len(sorted_cells))))
            best_cells = sorted_cells[-nb_best_cells:]

            for init_action_leaf_heap_node in init_action_leaves_to_explore:
                if best_leaf is not None:
                    break

                init_action_leaf = init_action_leaf_heap_node.action_tree_node

                manip_robot_pose = init_action_leaf.robot.pose
                manip_robot_d_pose = utils.real_pose_to_grid_pose(
                    manip_robot_pose, dd.res, dd.grid_pose, self.rotation_unit_angle)
                orig_obs_d_pose = utils.real_pose_to_grid_pose(
                    init_action_leaf.obstacle.pose, dd.res, dd.grid_pose, self.rotation_unit_angle)
                evaluated_configurations = {(manip_robot_d_pose, orig_obs_d_pose)}

                self._rp.publish_sim(
                    init_action_leaf.robot.polygon, init_action_leaf.obstacle.polygon, "/init", ns=self._robot_name)

                goal_robot_pose = self.deduce_robot_goal_pose(init_action_leaf.robot.pose, obstacle.pose, obs_goal_pose)
                goal_robot_polygon = affinity.rotate(
                    affinity.translate(
                        robot.polygon, goal_robot_pose[0] - robot.pose[0], goal_robot_pose[1] - robot.pose[1]),
                    goal_robot_pose[2] - robot.pose[2]
                )

                self._rp.publish_sim(goal_robot_polygon, goal_obs_poly, "/target", ns=self._robot_name)

                if self.polygon_collides(goal_robot_polygon, other_entities):
                    break

                open_heap = []
                heapq.heappush(
                    open_heap, ActionHeapNode(
                        self.__h(init_action_leaf.obstacle.pose, obs_goal_pose), init_action_leaf))

                while open_heap:
                    leaf = heapq.heappop(open_heap).action_tree_node
                    old_robot, old_obstacle = leaf.robot, leaf.obstacle

                    obs_position = (leaf.obstacle.pose[0], leaf.obstacle.pose[1])
                    obs_cell = utils.real_to_grid(obs_position[0], obs_position[1], dd.res, dd.grid_pose)

                    if obs_cell == goal_obs_cell or obs_cell in best_cells:
                        if obs_cell != goal_obs_cell:
                            new_obstacle = obstacle.light_copy(copy_polygon=False)
                            goal_obs_poly = affinity.rotate(
                                affinity.translate(
                                    obstacle.polygon, leaf.obstacle.pose[0] - obstacle.pose[0],
                                    leaf.obstacle.pose[1] - obstacle.pose[1]),
                                leaf.obstacle.pose[2] - obstacle.pose[2]
                            )
                            new_obstacle.polygon = goal_obs_poly
                            has_new_global_op, _, _ = self._is_there_opening_to_c_1(
                                binary_inflated_occupancy_grid, dd.res, dd.grid_pose, manip_robot_pose,
                                c_1_cells_set, robot, obstacle, old_obstacle, new_obstacle, False, goal_cell)
                        else:
                            has_new_global_op = True

                        if has_new_global_op:
                            best_leaf = leaf if (best_leaf is None or leaf.phys_cost < best_leaf.phys_cost) else best_leaf
                            break

                    for action in self._actions:
                        nb_states_explored += 1

                        # I - Check that the robot+obstacle has not been evaluated yet (avoids cycles in search)
                        pred_robot_pose, pred_obstacle_pose = action.predicted_pose(old_robot, old_obstacle)
                        pred_robot_d_pose = utils.real_pose_to_grid_pose(pred_robot_pose, dd.res, dd.grid_pose,
                                                                         self.rotation_unit_angle)
                        pred_obstacle_d_pose = utils.real_pose_to_grid_pose(pred_obstacle_pose, dd.res, dd.grid_pose,
                                                                            self.rotation_unit_angle)
                        predicted_d_config = (pred_robot_d_pose, pred_obstacle_d_pose)
                        if predicted_d_config in evaluated_configurations:
                            continue
                        # Add configuration to evaluated configurations list
                        evaluated_configurations.add(predicted_d_config)

                        # II - Check whether the robot or the obstacle enter in collision because of the action
                        nb_states_evaluated_for_collisions += 1

                        new_robot = old_robot.light_copy()
                        new_obstacle = old_obstacle.light_copy()

                        try:
                            new_robot, new_obstacle = action.apply(new_robot, new_obstacle, other_entities)
                            if not new_robot.polygon.within(map_box):
                                raise IntersectionError({robot.uid}, "Out of map bounds !")
                            elif not new_obstacle.polygon.within(map_box):
                                raise IntersectionError({obstacle.uid}, "Out of map bounds !")
                            self.update_collision_cache(pred_robot_d_pose, robot_collision_cache, False)
                            self.update_collision_cache(pred_obstacle_d_pose, obstacle_collision_cache, False)
                        except IntersectionError as e:
                            if robot.uid in e.colliding_entities_uids:
                                self.update_collision_cache(pred_robot_d_pose, robot_collision_cache, True)
                            if obstacle.uid in e.colliding_entities_uids:
                                self.update_collision_cache(pred_obstacle_d_pose, obstacle_collision_cache, True)
                                cell = (pred_obstacle_d_pose[0], pred_obstacle_d_pose[1])
                                collisions = obstacle_collision_cache[cell].values()
                                if len(collisions) == self._nb_possible_angles and all(collisions):
                                    try:
                                        sorted_cells.remove(cell)
                                        nb_best_cells = int(math.ceil(self.bound_percentage * float(len(sorted_cells))))
                                        best_cells = sorted_cells[-nb_best_cells:]
                                    except ValueError:
                                        pass
                            continue
                        self._rp.publish_sim(new_robot.polygon, new_obstacle.polygon, "/intermediate", ns=self._robot_name)

                        # III - Always compute physical cost, but don't bother computing social cost if no opening
                        phys_cost = leaf.phys_cost + self.__manip_e(old_robot.pose, new_robot.pose)
                        new_leaf = ActionTreeNode(phys_cost=phys_cost, social_cost=0., comb_cost=0.,
                                                  parent=leaf, action=action,
                                                  robot=new_robot, obstacle=new_obstacle,
                                                  has_new_local_opening=False)

                        heuristic_cost = self.__manip_e(new_leaf.obstacle.pose, obs_goal_pose)
                        heapq.heappush(
                            open_heap, ActionHeapNode(phys_cost + heuristic_cost, new_leaf))

        print("nb_states_explored = {}".format(nb_states_explored))
        print("nb_states_evaluated_for_collisions = {}".format(nb_states_evaluated_for_collisions))
        print("nb_states_evaluated_for_local_opening = {}".format(nb_states_evaluated_for_local_opening))
        print("nb_states_evaluated_for_global_opening = {}".format(nb_states_evaluated_for_global_opening))
        print("nb_states_with_global_opening = {}".format(nb_states_with_global_opening))

        # 6. Get best action branch from best action tree leaf and get its corresponding contact pose,
        # transit path to contact pose tho_n, transfer path from contact pose tho_m and overall cost
        best_action_branch = self.__get_actions_branch(best_leaf)
        cost = best_action_branch[-1].phys_cost
        tho_n, tho_m = self.__actions_branch_to_path(best_action_branch)

        # 7. Update w_t_plus_2 to represent with final state that corresponds to best action branch result
        final_robot = best_action_branch[-1].robot
        final_obstacle = best_action_branch[-1].obstacle
        w_t_plus_2.set_entity_polygon(final_robot.uid, final_robot.polygon, final_robot.full_geometry_acquired)
        w_t_plus_2.set_entity_polygon(final_obstacle.uid, final_obstacle.polygon, final_obstacle.full_geometry_acquired)
        self._rp.publish_robot_sim_world(w_t_plus_2, self._robot_uid, ns=self._robot_name)

        # Update displays
        self._rp.cleanup_robot_sim(ns=self._robot_name)
        self._rp.publish_sim(final_robot.polygon, final_obstacle.polygon, "/target", ns=self._robot_name)
        # cc_grid.re_init_grid(binary_inflated_occupancy_grid.get_grid())
        # self._rp.publish_connected_components_grid(cc_grid.get_grid(), dd)

        end_time = time.time()
        print(end_time - start_time)

        return w_t_plus_2, tho_n, tho_m, cost

    def get_manip_poses_and_paths(self, obstacle, robot, binary_inflated_occupancy_grid, res, grid_pose):
        # 2 - Get sampled navigation points around obstacle
        # TODO Implement generic method that can have three possibilities:
        #  - points from middle of sides (DONE)
        #  - points sampled along buffered polygon (to create from scratch)
        #  - points sampled along lines parallel to sides, s.t. we have at least a robot width from endpoints (scratch)

        # Diferrentiate between the *navigation pose* at the end of transit path that must be outside of the inflated
        # obstacle and the *manipulation pose* that can be within the inflated obstacle but is part of the transfer path
        nav_poses = obstacle.get_middle_of_sides_manipulation_poses(robot.min_inflation_radius + res)
        manip_poses = obstacle.get_middle_of_sides_manipulation_poses(robot.min_inflation_radius)
        self._rp.publish_q_manips_for_obs(nav_poses, ns=self._robot_name)

        # Find paths to all accessible navigation cells and only keep these
        self._rp.cleanup_a_star_close_set(ns=self._robot_name)
        self._rp.cleanup_rch_closed_set(ns=self._robot_name)
        cost_and_real_paths = multi_goal_a_star_real_path(
            binary_inflated_occupancy_grid.get_grid(), robot.pose, nav_poses, res, grid_pose, ns=self._robot_name)

        self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)

        return nav_poses, manip_poses, cost_and_real_paths

    @staticmethod
    def get_init_action_leaves_to_explore(nav_poses, manip_poses, cost_and_paths, robot, obstacle, other_entities):
        action_root = ActionTreeNode(0., 0., 0., robot=robot, obstacle=obstacle)
        cur_action_leaves_to_explore = []
        for counter, (cost, path) in enumerate(cost_and_paths):
            if cost == float("inf"):
                continue  # Ignore navigation poses that don't a path to them
            nav_pose = nav_poses[counter]
            to_nav_pose_action = FollowTransitPathAction(path)
            at_nav_pose_robot, _ = to_nav_pose_action.apply(robot.light_copy(), obstacle.light_copy(), other_entities)
            at_nav_pose_action_node = ActionTreeNode(
                cost, 0., 0., robot=at_nav_pose_robot, obstacle=obstacle, parent=action_root, action=to_nav_pose_action)
            manip_pose = manip_poses[counter]
            translation_to_manip_pose = (manip_pose[0] - at_nav_pose_robot.pose[0], manip_pose[1] - at_nav_pose_robot.pose[1])
            dist = utils.euclidean_distance(manip_pose, at_nav_pose_robot.pose)
            to_manip_pose_action = Translation(translation_to_manip_pose, dist, rectify_orientation=False)
            at_manip_pose_robot, _ = to_manip_pose_action.apply(at_nav_pose_robot.light_copy(), at_nav_pose_robot.light_copy(), other_entities)
            at_manip_pose_robot.pose = manip_pose
            at_manip_pose_action_node = ActionTreeNode(
                at_nav_pose_action_node.phys_cost + dist, 0., 0., robot=at_manip_pose_robot, obstacle=obstacle,
                parent=at_nav_pose_action_node, action=to_manip_pose_action)
            heapq.heappush(
                cur_action_leaves_to_explore, ActionHeapNode(
                    at_manip_pose_action_node.phys_cost, at_manip_pose_action_node))
        return cur_action_leaves_to_explore

    def _is_there_opening_to_c_1(self, inflated_grid, res, grid_pose, pred_robot_pose, c_1_cells_set, robot, obstacle,
                                 old_obstacle, new_obstacle, leaf_has_new_local_opening, goal_cell):
        """
        Checks if there is a path between robot_cell and a random cell in c_1_cells_set that is not covered by an
        obstacle (especially the one considered for manipulation).
        :param inflated_grid: 2D matrix of occupation data where free is 0 and > 0 is occupied
        :type inflated_grid: numpy.array([[int16]])
        :param res: grid resolution in [m] / cell
        :type res: float
        :param grid_pose: grid pose in real coords
        :type grid_pose: tuple(float, float, float)
        :param robot_d_pose: robot cell coordinates in grid
        :type robot_d_pose: tuple(int, int)
        :param c_1_cells_set: set of coordinates in c_1 free space component
        :type c_1_cells_set: set(tuple(int, int))
        :raises:
            ValueError: if the cells set given for c_1 is empty : either it means that c_1 is fully covered by an
            obstacle, or that not updating c_1 with freed cells was actually a bad idea.
        :return: True if a path is found, False otherwise
        """
        if self.check_new_local_opening_before_global:
            other_entities_polygons = [entity.polygon for entity in self._world.entities.values()
                                       if entity.uid != self._robot_uid
                                       and entity.uid != obstacle.uid]
            if leaf_has_new_local_opening:
                has_new_local_opening = True
            else:
                has_new_local_opening, _ = check_new_local_opening(
                    old_obstacle.polygon, new_obstacle.polygon, other_entities_polygons,
                    robot.min_inflation_radius, ns=self._robot_name)
            # Don't prevent full evaluation of plans when obstacle would pass over the goal
            # moved_polygons = [old_robot.polygon, new_robot.polygon, old_obstacle.polygon,
            #                   new_obstacle.polygon]
            move_passes_over_goal = False  # is_move_passing_over_pose(moved_polygons, r_f)
        else:
            has_new_local_opening, move_passes_over_goal = True, True

        is_it_worth_fully_evaluating = has_new_local_opening or move_passes_over_goal

        if is_it_worth_fully_evaluating:
            if c_1_cells_set:
                inflated_grid.update_buffered_entities({obstacle.uid: obstacle}, {new_obstacle.uid: new_obstacle})
                grid = inflated_grid.get_grid()
                if goal_cell in c_1_cells_set:
                    cell_in_c_1 = goal_cell
                else:
                    c_1_cells_set_iterator = iter(c_1_cells_set)
                    cell_in_c_1 = next(c_1_cells_set_iterator)
                    while grid[cell_in_c_1[0]][cell_in_c_1[1]] != 0:
                        try:
                            cell_in_c_1 = next(c_1_cells_set_iterator)
                        except StopIteration:
                            raise ValueError("_has_created_new_opening_to_c_1 could not find a new opening because"
                                             " c_1_cells_set is entirely inaccessible to the robot")
                cell_pose_in_c_1 = utils.grid_to_real(cell_in_c_1[0], cell_in_c_1[1], res, grid_pose)
                inflated_grid.update_buffered_entities({new_obstacle.uid: new_obstacle}, {obstacle.uid: obstacle})
                self._rp.cleanup_a_star_close_set(ns=self._robot_name)
                has_new_global_opening = bool(
                    a_star_real_path(grid, pred_robot_pose, cell_pose_in_c_1, res, grid_pose,
                                     ns=self._robot_name))  # HACK ?
                skipped_global_opening_check = False
                self._rp.cleanup_blocking_areas(ns=self._robot_name)
                self._rp.cleanup_diameter_inflated_polygons(ns=self._robot_name)
                return has_new_global_opening, has_new_local_opening, skipped_global_opening_check
            else:
                raise ValueError("c_1_cells_set should never be empty !")
        else:
            has_new_global_opening, skipped_global_opening_check = False, True
            return has_new_global_opening, has_new_local_opening, skipped_global_opening_check

    def _find_path(self, w_t, r_t, r_f):
        circum_radius = utils.get_circumscribed_radius(w_t.entities[self._robot.uid].polygon) # + w_t.dd.res / 2.
        grid = BinaryInflatedOccupancyGrid(
            w_t.dd.d_width, w_t.dd.d_height, w_t.dd.res, w_t.dd.grid_pose,
            circum_radius,
            w_t.entities, (self._robot.uid,)).get_grid()
        return a_star_real_path(grid, r_t, r_f, w_t.dd.res, w_t.dd.grid_pose, restrict_4_neighbors=False, ns=self._robot_name)

    def find_best_goal(self, cell_to_cost, goal_obs_poses, res, grid_pose, obstacle, other_entities,
                       grid, manip_robot_pose, c_1_cells_set, robot, goal_cell):
        while goal_obs_poses or cell_to_cost:
            if goal_obs_poses:
                goal_obs_pose = goal_obs_poses.pop()
                new_obstacle = obstacle.light_copy(copy_polygon=False)
                goal_obs_poly = affinity.rotate(
                    affinity.translate(
                        obstacle.polygon, goal_obs_pose[0] - obstacle.pose[0], goal_obs_pose[1] - obstacle.pose[1]),
                    goal_obs_pose[2] - obstacle.pose[2]
                )
                new_obstacle.polygon = goal_obs_poly
                if not self.polygon_collides(goal_obs_poly, other_entities):
                    self._rp.publish_sim(None, goal_obs_poly, "/target", ns=self._robot_name)
                    has_new_global_op, has_new_local_op, skipped_global_op_check = self._is_there_opening_to_c_1(
                        grid, res, grid_pose, manip_robot_pose,
                        c_1_cells_set, robot, obstacle, obstacle, new_obstacle, False, goal_cell)
                    if has_new_global_op:
                        return goal_obs_pose, goal_obs_poly
            elif cell_to_cost:
                    goal_obs_cell = cell_to_cost.pop()
                    goal_obs_poses = [
                        utils.grid_pose_to_real_pose(list(goal_obs_cell) + [rot], res, grid_pose)
                        for rot in self._all_rot_angles]
        return None

    @staticmethod
    def update_collision_cache(entity_d_pose, collision_cache, collides):
            robot_d_position = (entity_d_pose[0], entity_d_pose[1])
            if robot_d_position in collision_cache:
                collision_cache[robot_d_position][entity_d_pose[2]] = collides
            else:
                collision_cache[robot_d_position] = {entity_d_pose[2]: collides}

    @staticmethod
    def deduce_robot_goal_pose(robot_manip_pose, obs_init_pose, obs_goal_pose):
        translation = (obs_goal_pose[0] - obs_init_pose[0], obs_goal_pose[1] - obs_init_pose[1])
        rotation = (obs_goal_pose[2] - obs_init_pose[2]) % 360.
        robot_goal_point = list(affinity.translate(
            affinity.rotate(
                Point((robot_manip_pose[0], robot_manip_pose[1])),
                rotation, origin=(obs_init_pose[0], obs_init_pose[1])),
            translation[0], translation[1]).coords[0])
        return robot_goal_point[0], robot_goal_point[1], (robot_manip_pose[2] + rotation) % 360.

    @staticmethod
    def polygon_collides(polygon, other_entities):
        for entity in other_entities:
            if entity.polygon.intersects(polygon):
                return True
        return False

    @staticmethod
    def __make_priority_queue(cell, f, obs_id, comp_id):
        priority_queue = []
        heapq.heappush(priority_queue, HeapQueueElement(cell, f, obs_id, comp_id))
        return priority_queue

    def __h(self, x_i, x_j):
        return math.sqrt((x_j[0] - x_i[0]) ** 2 + (x_j[1] - x_i[1]) ** 2)

    def __f(self, x_j, g_x_j, x_f):
        return g_x_j + self.__h(x_j, x_f)

    def __g(self, x_i, x_j, g_x_i, cc_grid):
        return g_x_i + (1 - self.alpha) + self.alpha * self.__w_heur(x_i, x_j, cc_grid)

    def __w_heur(self, x_i, x_j, cc_grid):
        if cc_grid[x_j[0]][x_j[1]] > 0:
            return 0.
        else:
            return self.heur_w

    def __manip_e(self, r_i, r_j):
        translation_energy = self.basic_trans_force * np.linalg.norm((r_j[0] - r_i[0], r_j[1] - r_i[1]))
        rotation_energy = 0. if self._rot_angles.size == 0 else self.basic_rot_moment * self._world.dd.res * (
                    abs(r_j[2] - r_i[2]) / abs(self.rotation_unit_angle))
        return translation_energy + rotation_energy

    @staticmethod
    def __remove_first(queue):
        return heapq.heappop(queue)

    @staticmethod
    def __enqueue(queue, cell, f, obs_id, comp_id):
        heapq.heappush(queue, HeapQueueElement(cell, f, obs_id, comp_id))

    @staticmethod
    def __compute_current_robot_cells(x_init, x_cur, init_robot_cells):
        """
        Likely Deprecated
        :param x_init:
        :param x_cur:
        :param init_robot_cells:
        :return:
        """
        discrete_translation = (x_cur[0] - x_init[0], x_cur[1], x_init[1])
        current_robot_cells = set()
        for cell in init_robot_cells:
            current_robot_cells.add((cell[0] + discrete_translation[0], cell[1] + discrete_translation[1]))
        return current_robot_cells

    def __robot_exc_contained_in_obs(self, cell, w_t):
        """
        If cell is contained only by one obstacle o_i, returns o_i.
        If contained by no obstacle, returns 0. If contained by more than one, returns -1.
        :param cell: cell coordinates as integer tuple (x, y)
        :return: obstacle uid or 0 or -1
        """
        grid = w_t.get_binary_inflated_occupancy_grid((self._robot.uid,)).get_grid()
        if grid[cell[0]][cell[1]] == 0:
            return 0
        elif grid[cell[0]][cell[1]] > 1:
            return -1

        for uid, entity in w_t.entities.items():
            if uid != self._robot.uid:
                obs_cells = w_t.entities[uid].get_discrete_inflated_cells_set(
                    w_t.dd.inflation_radius, w_t.dd.res, w_t.dd.grid_pose, w_t.dd.d_width, w_t.dd.d_height)
                if cell in obs_cells:
                    return uid

        raise RuntimeError("__robot_exc_contained_in_obs should never reach this point.")

    @staticmethod
    def __get_actions_branch(action_node):
        branch = [action_node]
        while branch[-1].parent is not None:
            branch.append(branch[-1].parent)
        branch.reverse()
        return branch

    def __actions_branch_to_path(self, actions_branch):
        real_transit_path, real_transfer_path = [], []
        transit_path_cost = 0.
        actions_branch_iter = iter(actions_branch)
        prev_action_node = next(actions_branch_iter)
        # prev_action_node = next(actions_branch_iter)
        collision_polygons = []  # [prev_action_node.robot.polygon, prev_action_node.obstacle.polygon]
        # self._rp.publish_debug_polygons(collision_polygons, ns=self._robot_name)
        for action_node in actions_branch_iter:
            if isinstance(action_node.action, Translation) or isinstance(action_node.action, Rotation):
                real_transfer_path.append(action_node.robot.pose)
                if isinstance(action_node.action, Translation):
                    translation_length = action_node.action.unit_length
                    translation_steps_to_check = int(math.ceil(translation_length / self._world.dd.res))
                    xoff_normed, yoff_normed =(
                        action_node.action.translation_vector[0] / float(translation_steps_to_check),
                        action_node.action.translation_vector[1] / float(translation_steps_to_check)
                    )
                    collision_polygons += [
                        affinity.translate(
                            prev_action_node.robot.polygon, xoff_normed * float(i), yoff_normed * float(i))
                        for i in range(translation_steps_to_check)] + [
                        affinity.translate(
                            prev_action_node.obstacle.polygon, xoff_normed * float(i), yoff_normed * float(i))
                        for i in range(translation_steps_to_check)
                    ]
                    # self._rp.publish_debug_polygons(collision_polygons, ns=self._robot_name)
                if isinstance(action_node.action, Rotation):
                    rotation_steps_to_check = int(abs(action_node.action.angle) / self.angular_res)
                    sign = -1. if action_node.action.angle < 0. else 1.
                    collision_polygons += [
                        affinity.rotate(
                            prev_action_node.robot.polygon, sign * float(i) * self.angular_res, origin="centroid")
                        for i in range(rotation_steps_to_check)] + [
                        affinity.rotate(
                            prev_action_node.obstacle.polygon, sign * float(i) * self.angular_res,
                            origin=prev_action_node.robot.polygon.centroid.coords[0])
                        for i in range(rotation_steps_to_check)
                    ]
                    # self._rp.publish_debug_polygons(collision_polygons, ns=self._robot_name)
            prev_action_node = action_node
            if isinstance(action_node.action, FollowTransitPathAction):
                real_transfer_path.append(action_node.robot.pose)
                real_transit_path = action_node.action.real_path
                transit_path_cost = action_node.phys_cost
            if action_node.parent is None:
                break

        tho_n = Path(real_transit_path, phys_cost=transit_path_cost)

        transfer_path_phys_cost = actions_branch[-1].phys_cost - transit_path_cost
        transfer_path_social_cost = actions_branch[-1].social_cost
        o_uid = actions_branch[-1].obstacle.uid

        tho_m = Path(real_transfer_path, is_transfer=True, o_uid=o_uid, phys_cost=transfer_path_phys_cost,
                     social_cost=transfer_path_social_cost, collision_geometry=collision_polygons)

        # self._rp.publish_debug_polygons(collision_polygons, ns=self._robot_name)

        return tho_n, tho_m

    def __get_min_kernel_social_costmap(self, w_t, o_1):
        o_1_polygon = w_t.entities[o_1].polygon
        dd = w_t.dd

        o_1_inscribed_radius = utils.get_inscribed_radius(o_1_polygon)
        o_1_inflated_occ_grid = BinaryInflatedOccupancyGrid(
            dd.d_width, dd.d_height, dd.res, dd.grid_pose, o_1_inscribed_radius, w_t.entities,
            entities_to_ignore=[o_1, self._robot_uid]).get_grid()
        square_d_length = int(round(utils.get_inscribed_square_sidelength(o_1_inscribed_radius) / dd.res))
        half_square_d_length = square_d_length // 2

        coords_to_update = np.where(o_1_inflated_occ_grid == 0)

        for (x, y) in zip(coords_to_update[0], coords_to_update[1]):
            x_min = x - half_square_d_length
            x_min = 0 if x_min < 0 else x_min
            x_max = x + half_square_d_length + 1
            y_min = y - half_square_d_length
            y_min = 0 if y_min < 0 else y_min
            y_max = y + half_square_d_length + 1
            values = self._social_costmap[x_min: x_max, y_min: y_max]
            total = np.sum(np.extract([values != -1.], values))
            o_1_inflated_occ_grid[x][y] = total

        return o_1_inflated_occ_grid

    @staticmethod
    def display_graphics(names, cost_x, cost_y, x_label, y_label, title):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, wrap=True)

        sc = plt.scatter(cost_x, cost_y, marker="1")

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):

            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}".format(" ".join([str(names[n]) for n in ind["ind"]]))
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()

    @staticmethod
    def is_pareto_efficient(costs, return_mask=True):
        """
        Copied from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

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
            for next in utils.get_neighbors_no_coll(current, grid, width, height, neighborhood):
                extra_cost = straight_dist if current[0] == next[0] or current[1] == next[1] else diag_dist
                new_cost = cost_so_far[current] + extra_cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    heapq.heappush(frontier, (new_cost, next))

        return cost_so_far

    def sorted_cells_by_combined_cost(self, w_t_plus_2, obstacle, robot, dd, r_f):
        # Order a priori accessible cells for obstacle by social cost
        inflation_radius = utils.get_inscribed_radius(obstacle.polygon)
        obs_inflated_grid = BinaryInflatedOccupancyGrid(
            dd.d_width, dd.d_height, dd.res, dd.grid_pose, inflation_radius,
            w_t_plus_2.entities, (self._robot_uid, obstacle.uid)).get_grid()
        obstacle_d_pose = utils.real_pose_to_grid_pose(obstacle.pose, dd.res, dd.grid_pose, self.rotation_unit_angle)
        obstacle_d_cell = (obstacle_d_pose[0], obstacle_d_pose[1])

        robot_poly_at_r_f = affinity.translate(
            affinity.rotate(robot.polygon.buffer(inflation_radius), r_f[2] - robot.pose[2]),
            r_f[0] - robot.pose[0], r_f[1] - robot.pose[1])
        r_f_cells = utils.polygon_to_discrete_cells_set(
            robot_poly_at_r_f, dd.res, dd.grid_pose, dd.d_width, dd.d_height, fill=True)

        cell_to_cost = self.dijkstra_cc_and_cost(
            obstacle_d_cell, obs_inflated_grid, dd.res, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD)
        for cell in r_f_cells:
            if cell in cell_to_cost:
                del cell_to_cost[cell]

        acc_cells_for_obs, distance_cost = cell_to_cost.keys(), np.array(cell_to_cost.values())
        # acc_cells_for_obs_xy = zip(*acc_cells_for_obs)
        social_cost = np.array([
            self._social_costmap[cell[0]][cell[1]]
            for cell in acc_cells_for_obs
            if self._social_costmap[cell[0]][cell[1]] != -1.])

        if not self.distance_to_obs_cost_is_realistic:
            distance_cost = np.array([
                utils.euclidean_distance(utils.grid_to_real(cell[0], cell[1], dd.res, dd.grid_pose), obstacle.pose)
                for cell in acc_cells_for_obs])

        distance_to_goal = np.array([
            utils.euclidean_distance(utils.grid_to_real(cell[0], cell[1], dd.res, dd.grid_pose), r_f)
            for cell in acc_cells_for_obs])

        # self.display_graphics(
        #     acc_cells_for_obs, social_cost, distance_cost, "Social cost [UA]", "Distance cost [m]",
        #   "Euclidian distance from initial to goal obstacle center as a function of the social cost "
        #   "in the corresponding goal cell: D(cell_init, cell_goal) = f(SC_goal)")

        tables_length = (social_cost.shape[0] + distance_cost.shape[0] + distance_to_goal.shape[0]) / 3

        normalized_social_cost = (social_cost - np.min(social_cost)) / np.ptp(social_cost)
        normalized_distance_cost = (distance_cost - np.min(distance_cost)) / np.ptp(distance_cost)
        normalized_distance_to_goal = (distance_to_goal - np.min(distance_to_goal)) / np.ptp(distance_to_goal)

        combined_cost = (self.w_social * normalized_social_cost
                         + self.w_obs * normalized_distance_cost
                         + self.w_goal * normalized_distance_to_goal) / self.w_sum
        sorted_cell_to_combined_cost = sorted(
            zip(acc_cells_for_obs, combined_cost), key=lambda tup: tup[1], reverse=True)
        sorted_cells, _ = zip(*sorted_cell_to_combined_cost)
        sorted_cells = list(sorted_cells)

        self._rp.cleanup_multigoal_a_star_close_set(ns=self._robot_name)
        self._rp.cleanup_grid_map(ns=self._robot_name)
        self._rp.publish_combined_costmap(sorted_cell_to_combined_cost, dd, ns=self._robot_name)
        # time.sleep(3.)

        if self.activate_grids_logging:
            stocg.display_or_log(
                np.invert(obs_inflated_grid.astype(np.bool)), "-obs_inf_grid", time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False, log_costmaps=True, abs_path_to_logs_dir=self.abs_path_to_logs_dir)

            normalized_social_cost_costmap = np.zeros((dd.d_width, dd.d_height))
            normalized_distance_from_obs_costmap = np.zeros((dd.d_width, dd.d_height))
            normalized_distance_from_goal_costmap = np.zeros((dd.d_width, dd.d_height))

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

            combined_costmap = np.zeros((dd.d_width, dd.d_height))
            for cell, combined_cost in sorted_cell_to_combined_cost:
                combined_costmap[cell[0]][cell[1]] = combined_cost
            stocg.display_or_log(
                combined_costmap, "-combined_costmap", time.strftime("%Y-%m-%d-%Hh%Mm%Ss"),
                debug_display=False, log_costmaps=True, abs_path_to_logs_dir=self.abs_path_to_logs_dir)

        # self.display_graphics(acc_cells_for_obs, normalized_social_cost, normalized_distance_cost)
        #
        # pareto_efficient_indices = np.where(self.is_pareto_efficient(np.array(zip(social_cost, distance_cost))))
        # pareto_normalized_social_cost = np.take(normalized_social_cost, pareto_efficient_indices)[0]
        # pareto_normalized_distance_cost = np.take(normalized_distance_cost, pareto_efficient_indices)[0]
        # pareto_acc_cells_for_obs = np.take(acc_cells_for_obs, pareto_efficient_indices, axis=0)[0]
        #
        # self.display_graphics(pareto_acc_cells_for_obs, pareto_normalized_social_cost, pareto_normalized_distance_cost)
        #
        # pareto_social_cost = np.take(social_cost, pareto_efficient_indices)[0]
        # pareto_distance_cost = np.take(distance_cost, pareto_efficient_indices)[0]
        #
        # self.display_graphics(pareto_acc_cells_for_obs, pareto_social_cost, pareto_distance_cost)

        return sorted_cells


class Action:

    def __init__(self):
        pass

    def apply(self, robot, obstacle, other_entities):
        raise NotImplementedError

    def predicted_pose(self, robot, obstacle):
        raise NotImplementedError


class FollowTransitPathAction(Action):
    def __init__(self, real_path):
        self.real_path = real_path

    def apply(self, robot, obstacle, other_entities):
        end_pose = self.real_path[-1]
        translation_to_nav_pose = (end_pose[0] - robot.pose[0], end_pose[1] - robot.pose[1])
        rotation_to_end_pose = end_pose[2] - robot.pose[2]
        robot_state_at_nav_pose = robot.translate(
            translation_to_nav_pose[0], translation_to_nav_pose[1], other_entities=other_entities, ignore_collisions=True
        ).rotate(rotation_to_end_pose, other_entities=other_entities, ignore_collisions=True)
        robot_state_at_nav_pose.pose = end_pose
        return (robot_state_at_nav_pose,
                obstacle)

    def predicted_pose(self, robot, obstacle):
        return self.real_path[-1], obstacle.pose


class Rotation(Action):

    def __init__(self, angle):
        self.angle = angle

    def apply(self, robot, obstacle, other_entities):
        return (robot.rotate(self.angle, other_entities=other_entities),
                obstacle.rotate(self.angle, rot_center=(robot.pose[0], robot.pose[1]), other_entities=other_entities))

    def predicted_pose(self, robot, obstacle):
        position = list(affinity.rotate(
            Point(obstacle.pose[0], obstacle.pose[1]), self.angle, origin=(robot.pose[0], robot.pose[1])).coords)[0]
        return (
            (robot.pose[0], robot.pose[1], (robot.pose[2] + self.angle) % 360.0),
            (position[0], position[1], (obstacle.pose[2] + self.angle) % 360.0)
        )
        # rot_radius = utils.euclidean_distance(robot.pose, obstacle.pose)
        # return (
        #     (robot.pose[0], robot.pose[1], (robot.pose[2] + self.angle) % 360.0),
        #     (obstacle.pose[0] + math.cos(obstacle.pose[2] - self.angle) * rot_radius,
        #      obstacle.pose[1] + math.sin(obstacle.pose[2] - self.angle) * rot_radius,
        #      (obstacle.pose[2] + self.angle) % 360.0)
        # )


class Translation(Action):
    def __init__(self, translation_vector, unit_length, rectify_orientation=True):
        self.translation_vector = translation_vector
        self.unit_length = unit_length
        self.rectify_orientation = rectify_orientation

    def apply(self, robot, obstacle, other_entities):
        translation_vector = self.translation_vector
        if self.rectify_orientation:
            translation_linestring = LineString([(0., 0.), self.translation_vector])
            rotated_linestring = affinity.rotate(translation_linestring, robot.pose[2], origin=(0., 0.))
            translation_vector = rotated_linestring.coords[1]
            return (robot.translate(
                translation_vector[0], translation_vector[1], self.unit_length, other_entities=other_entities),
                    obstacle.translate(
                        translation_vector[0], translation_vector[1], self.unit_length, other_entities=other_entities))
        else:
            return (robot.translate(
                translation_vector[0], translation_vector[1], self.unit_length,
                other_entities=other_entities, ignore_collisions=True),
                    obstacle)

    def predicted_pose(self, robot, obstacle):
        translation_linestring = LineString([(0., 0.), self.translation_vector])
        rotated_linestring = affinity.rotate(translation_linestring, robot.pose[2], origin=(0., 0.))
        rotated_translation_vector = rotated_linestring.coords[1]
        return (
            (robot.pose[0] + rotated_translation_vector[0],
             robot.pose[1] + rotated_translation_vector[1],
             robot.pose[2]),
            (obstacle.pose[0] + rotated_translation_vector[0],
             obstacle.pose[1] + rotated_translation_vector[1],
             obstacle.pose[2])
        )


class HeapQueueElement:
    def __init__(self, cell, f, obs_id, comp_id):
        self.cell = cell
        self.obs_id = obs_id
        self.comp_id = comp_id
        self.f = f

    def __cmp__(self, other):
        return cmp(self.f, other.f)

    def __lt__(self, other):
        return self.f < other.f


class ActionTreeNode:
    def __init__(self, phys_cost, social_cost, comb_cost, parent=None, action=None, robot=None, obstacle=None,
                 has_new_local_opening=False):
        self.parent = parent
        self.action = action
        self.robot = robot
        self.obstacle = obstacle
        self.has_new_local_opening = has_new_local_opening

        self.phys_cost = phys_cost
        self.social_cost = social_cost
        self.comb_cost = comb_cost


class ActionHeapNode:
    def __init__(self, cost, action_tree_node):
        self.cost = cost
        self.action_tree_node = action_tree_node

    def __cmp__(self, other):
        return cmp(self.cost, other.cost)

    def __lt__(self, other):
        return self.cost < other.cost


class CellActionHeapNode:
    def __init__(self, cost, cell, action_tree_node):
        self.cost = cost
        self.cell = cell
        self.action_tree_node = action_tree_node

    def __cmp__(self, other):
        return cmp(self.cost, other.cost)

    def __lt__(self, other):
        return self.cost < other.cost
