import copy
import heapq
import math
import numpy as np
from shapely.geometry import LineString, Point
from shapely import affinity

from baseline_behavior import BaselineBehavior
from src.behaviors.algorithms.a_star import a_star_real_path, astar
from src.behaviors.plan.path import Path
from src.behaviors.plan.plan import Plan
from src.behaviors.algorithms.multi_goal_a_star import multi_goal_a_star_real_path
from src.utils import utils
from src.worldreps.entity_based.obstacle import Obstacle
from src.worldreps.entity_based.robot import Robot
from src.behaviors.algorithms.new_local_opening_check import check_new_local_opening, is_move_passing_over_pose
from plan.basic_actions import ActionGoalFailure, ActionGoalsFinished, ActionGoalSuccess
from src.worldreps.entity_based.custom_exceptions import IntersectionError


class Stilman2005Behavior(BaselineBehavior):
    def __init__(self, ref_world, initial_world, robot_uid, navigation_goals, behavior_config):
        BaselineBehavior.__init__(self, ref_world, initial_world, robot_uid, navigation_goals, behavior_config)

        # Configuration parameters
        self.alpha = 0.5
        self.neighborhood = utils.TAXI_NEIGHBORHOOD
        self.cost_for_obstacle_occupied_cells = 2.
        self.basic_trans_force = 2.
        self.basic_rot_moment = 2.
        self.heur_w = 2.
        self._check_new_local_opening_activated = True
        self.forbid_rotations = False

        self.robot_type = "diff"  # Possible types: ["omni", "diff"]
        self.trans_vectors = []
        self.rot_angles = []
        if self.robot_type == "omni":
            if self.neighborhood == utils.CHESSBOARD_NEIGHBORHOOD:
                self.trans_vectors = np.array(utils.OMNI_ROBOT_CHESSBOARD_TRANS_VECTORS) * self._world.dd.res
                self.rot_angles = np.array(utils.OMNI_ROBOT_CHESSBOARD_ROT_ANGLES)
            elif self.neighborhood == utils.TAXI_NEIGHBORHOOD:
                self.trans_vectors = np.array(utils.OMNI_ROBOT_TAXI_TRANS_VECTORS) * self._world.dd.res
                self.rot_angles = np.array(utils.OMNI_ROBOT_TAXI_ROT_ANGLES)
        elif self.robot_type == "diff":
            if self.neighborhood == utils.CHESSBOARD_NEIGHBORHOOD:
                self.trans_vectors = np.array(utils.DIFF_ROBOT_CHESSBOARD_TRANS_VECTORS) * self._world.dd.res
                self.rot_angles = np.array(utils.DIFF_ROBOT_CHESSBOARD_ROT_ANGLES)
            elif self.neighborhood == utils.TAXI_NEIGHBORHOOD:
                self.trans_vectors = np.array(utils.DIFF_ROBOT_TAXI_TRANS_VECTORS) * self._world.dd.res
                self.rot_angles = np.array(utils.DIFF_ROBOT_TAXI_ROT_ANGLES)

        if self.forbid_rotations:
            self.rot_angles = np.array([])

        self.actions = []
        for trans_vector in self.trans_vectors:
            self.actions.append(Translation(trans_vector, self._world.dd.res))
        for rot_angle in self.rot_angles:
            self.actions.append(Rotation(rot_angle))

    def think(self):
        if self._navigation_goals or self._q_goal is not None:
            if self._q_goal is None:
                self._q_goal = self._navigation_goals.pop(0)
                self._p_opt = Plan([Path([])])

            q_r = self._robot.pose

            # TODO Extract abs_tol constant and make it a parameter for each goal
            is_close_enough_to_goal = all(np.isclose(q_r, self._q_goal, rtol=1e-5))
            if is_close_enough_to_goal:
                print("SUCCESS: Agent '{name}' has successfully reached pose {nav_goal}.".format(
                    name=self._robot.name, nav_goal=str(self._q_goal)))
                self._q_goal = None
                return ActionGoalSuccess()

            if not self._p_opt.is_valid(self._world, self._robot_uid):
                self._p_opt = self._select_connect(self._world, set(), self._q_goal)

            if not self._p_opt.is_empty():
                next_step = self._p_opt.pop_next_step()
                return next_step
            elif self._p_opt.has_infinite_cost():
                print("FAILURE: Agent '{name}' has failed to reach pose {nav_goal}.".format(
                    name=self._robot.name, nav_goal=str(self._q_goal)))
                self._q_goal = None
                return ActionGoalFailure()

        else:
            print("FINISH: Agent '{name}' has finished trying to reach its goals !".format(name=self._robot.name))
            return ActionGoalsFinished()

    def _select_connect(self, w_t, prev_list, r_f):
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
            return Plan([Path(simple_path_to_goal)])

        o_1, c_1 = self._rch(w_t, avoid_list, prev_list, r_f)
        while (o_1, c_1) != (None, None):
            c_1_cells_set = w_t.get_connected_components_grid((self._robot_uid,)).get_components()[c_1]
            w_t_plus_2, tho_n, tho_m, cost = self._manip_search(w_t, o_1, c_1_cells_set, r_f)

            if tho_m is not None:
                prev_list.add(c_1)
                future_plan = self._select_connect(w_t_plus_2, prev_list, r_f)
                if future_plan is not None:
                    # Following line comes from original algorithm, but does not make sense ?
                    # tho_n = self._find_path(w_t, x_t, tho_m[0])
                    future_plan.path_components[0].obstacle_uid = o_1  #
                    return Plan([tho_n, tho_m]).append(future_plan)

            avoid_list.add((o_1, c_1))

            o_1, c_1 = self._rch(w_t, avoid_list, prev_list, r_f)

        return None

    def _rch(self, w_t, avoid_list, prev_list, r_f):
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
        self._rp.publish_connected_components_grid(connected_components_grid, w_t.dd)

        movable_entities_uids = [
            uid for uid, entity in w_t.entities.items()
            if isinstance(entity, Robot) or isinstance(entity, Obstacle) and self._robot.deduce_movability(entity.type) != "unmovable"]
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

        self._rp.publish_rch_open_queue(open_queue, w_t.dd.res, w_t.dd.grid_pose)

        # While open_heap is not empty == While there are discovered nodes that have not been evaluated
        while open_queue:

            # The node in open_heap having the lowest fScore[] value
            current = heapq.heappop(open_queue)
            self._rp.publish_rch_open_queue(open_queue, w_t.dd.res, w_t.dd.grid_pose)
            self._rp.publish_current_cell(current.cell, w_t.dd.res, w_t.dd.grid_pose)

            # Exit early if goal is reached
            if current.cell == goal_cell:
                return current.obs_id, current.comp_id

            close_set.add(current.cell)
            self._rp.publish_rch_closed_set(close_set, w_t.dd.res, w_t.dd.grid_pose)

            # For each neighbor of current node in the defined neighborhood
            for i, j in self.neighborhood:
                neighbor_cell = current.cell[0] + i, current.cell[1] + j
                self._rp.publish_current_neighbor(neighbor_cell, w_t.dd.res, w_t.dd.grid_pose)

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
                                is_valid_obstacle_uid = o_uid > 0
                                if is_valid_obstacle_uid:
                                    # If the neighbor cell is not in the free space but in an obstacle, and if it is
                                    # exclusively in this obstacle, then associate the cell with this obstacle and the
                                    # initial component. Application of Rule #2
                                    self.__enqueue(open_queue, neighbor_cell, fscore_neighbor_cell, o_uid, 0)

                        self._rp.publish_rch_open_queue(open_queue, w_t.dd.res, w_t.dd.grid_pose)
        return None, None

    def _manip_search(self, w_t, o_1, c_1_cells_set, r_f):
        # 1 - Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = copy.deepcopy(w_t)
        obstacle = w_t_plus_2.entities[o_1]
        robot = w_t_plus_2.entities[self._robot.uid]
        binary_inflated_occupancy_grid = w_t_plus_2.get_binary_inflated_occupancy_grid((self._robot.uid,))
        dd = w_t_plus_2.dd
        other_entities = [entity for entity in w_t_plus_2.entities.values()
                          if entity.uid != robot.uid and entity.uid != obstacle.uid]

        # cc_grid = w_t.get_connected_components_grid((self._robot_uid,))
        # self._rp.publish_connected_components_grid(cc_grid.get_grid(), dd)

        # 2 - Get sampled navigation points around obstacle
        # TODO Implement generic method that can have three possibilities:
        #  - points from middle of sides (DONE)
        #  - points sampled along buffered polygon (to create from scratch)
        #  - points sampled along lines parallel to sides, s.t. we have at least a robot width from endpoints (scratch)
        navigation_poses = obstacle.get_middle_of_sides_manipulation_poses(
            self._robot.min_inflation_radius)

        # 3 - Find paths to all accessible navigation cells and only keep these
        nav_pose_to_cost_and_real_path = multi_goal_a_star_real_path(
            binary_inflated_occupancy_grid.get_grid(), robot.pose, navigation_poses, dd.res, dd.grid_pose)

        # 4 - Only keep accessible cells in nav_cells
        for pose, (cost, _) in nav_pose_to_cost_and_real_path.items():
            if cost == float("inf"):
                del nav_pose_to_cost_and_real_path[pose]

        # 5 - Explore robot action space
        # 5.a - Create action roots starting at each valid navigation pose
        action_roots = []
        for nav_pose in nav_pose_to_cost_and_real_path.keys():
            translation_to_nav_pose = (nav_pose[0] - robot.pose[0], nav_pose[1] - robot.pose[1])
            rotation_to_nav_pose = nav_pose[2] - robot.pose[2]
            robot_state_in_root = copy.deepcopy(robot).translate(
                translation_to_nav_pose[0], translation_to_nav_pose[1], dd.res).rotate(rotation_to_nav_pose)
            robot_state_in_root.pose = nav_pose  # Fix floating-point error on robot pose caused by translation
            action_root_node = ActionTreeNode(0., 0., robot=robot_state_in_root, obstacle=obstacle)
            action_roots.append(action_root_node)

        # 5.b - Iterate over action trees and extract the best action leaf out of each
        best_action_leaf_for_tree = []
        for action_root in action_roots:
            # Action tree nodes to currently explore. Ordered by min phys cost.
            cur_action_leaves_to_explore = [ActionHeapNode(action_root.phys_cost, action_root)]
            # Action tree nodes to explore at next Breadth-FS iteration. Ordered by min phys cost.
            next_action_leaves_to_explore = []
            # Action tree nodes for which there is a successful plan. Ordered by total min cost.
            successful_action_tree_nodes = []

            best_in_cur_action_leaves_to_explore = cur_action_leaves_to_explore[0]
            best_in_successful_action_tree_nodes = None

            self._rp.publish_sim(action_root.robot.polygon, action_root.obstacle.polygon, "/init")

            evaluated_configurations = {(self.round_pose(action_root.robot.pose),
                                         self.round_pose(action_root.obstacle.pose))}

            # As long as the ascending physical cost of the currently explored action leaf is lower than the best
            # successful action leaf, and there are leaves to be explored,
            # we explore the action tree (this bounds the exploration).
            while ((best_in_successful_action_tree_nodes is None or
                    best_in_cur_action_leaves_to_explore is None or
                    (int(best_in_cur_action_leaves_to_explore.phys_cost * self.rounder)
                     < int(best_in_successful_action_tree_nodes.total_cost * self.rounder)))
                   and cur_action_leaves_to_explore):
                for action_heap_node in cur_action_leaves_to_explore:
                    leaf = action_heap_node.action_tree_node
                    for action in self.actions:
                        old_robot = leaf.robot
                        old_obstacle = leaf.obstacle

                        predicted_pose = action.predicted_pose(old_robot, old_obstacle)
                        predicted_configuration = (
                            self.round_pose(predicted_pose[0]),
                            self.round_pose(predicted_pose[1])
                        )

                        if predicted_configuration in evaluated_configurations:
                            continue
                        else:
                            evaluated_configurations.add(predicted_configuration)

                        new_robot = old_robot.light_copy()
                        new_obstacle = old_obstacle.light_copy()
                        try:
                            new_robot, new_obstacle = action.apply(new_robot, new_obstacle, other_entities)
                        except IntersectionError:
                            continue

                        self._rp.publish_sim(new_robot.polygon, new_obstacle.polygon, "/target")

                        if self._check_new_local_opening_activated:
                            other_entities_polygons = [entity.polygon for entity in self._world.entities.values()
                                                       if entity.uid != self._robot_uid
                                                       and entity.uid != obstacle.uid]
                            has_new_local_opening, _ = check_new_local_opening(
                                old_obstacle.polygon, new_obstacle.polygon, other_entities_polygons,
                                robot.min_inflation_radius)
                            has_new_local_opening = True if leaf.has_new_local_opening else has_new_local_opening
                            # Don't prevent full evaluation of plans when obstacle would pass over the goal
                            moved_polygons = [old_robot.polygon, new_robot.polygon, old_obstacle.polygon,
                                              new_obstacle.polygon]
                            move_passes_over_goal = is_move_passing_over_pose(moved_polygons, r_f)
                        else:
                            has_new_local_opening = True
                            move_passes_over_goal = True
                        is_it_worth_fully_evaluating = has_new_local_opening or move_passes_over_goal

                        if is_it_worth_fully_evaluating:
                            robot_cell = utils.real_to_grid(
                                new_robot.pose[0], new_robot.pose[1], dd.res, dd.grid_pose)
                            binary_inflated_occupancy_grid.update_buffered_entities(
                                {obstacle.uid: obstacle}, {new_obstacle.uid: new_obstacle})
                            # self._rp.publish_robot_sim_costmap(w_t_plus_2, self._robot_uid)
                            # cc_grid.update_freed_and_invaded_cells(
                            #     {obstacle.uid: obstacle}, {new_obstacle.uid: new_obstacle},
                            #     dd, binary_inflated_occupancy_grid.grid
                            # )
                            # cc_grid.re_init_grid(binary_inflated_occupancy_grid.get_grid())
                            # freed_cells, invaded_cells = binary_inflated_occupancy_grid.update_grid_and_return_freed_and_invaded_cells()
                            # cc_grid.update_freed_and_invaded_cells_alternative(freed_cells, invaded_cells)
                            # cc_grid.force_update_grid()

                            # self._rp.publish_connected_components_grid(cc_grid.get_grid(), dd)

                            is_there_opening_to_c_1 = self._is_there_opening_to_c_1(
                                binary_inflated_occupancy_grid.get_grid(),
                                dd.res, dd.grid_pose, robot_cell, c_1_cells_set)
                            binary_inflated_occupancy_grid.update_buffered_entities(
                                {new_obstacle.uid: new_obstacle}, {obstacle.uid: obstacle})
                        else:
                            is_there_opening_to_c_1 = False

                        phys_cost = leaf.phys_cost + self.__manip_e(old_robot.pose, new_robot.pose)
                        # TODO: Get relative social cost as is done in behavior report
                        social_cost = 0.  # compute_social_cost()
                        new_leaf = ActionTreeNode(phys_cost=phys_cost, social_cost=social_cost,
                                                  parent=leaf, action=action,
                                                  robot=new_robot, obstacle=new_obstacle,
                                                  has_new_local_opening=has_new_local_opening)
                        heapq.heappush(
                            next_action_leaves_to_explore, ActionHeapNode(new_leaf.phys_cost, new_leaf))

                        if is_there_opening_to_c_1:
                            heapq.heappush(
                                successful_action_tree_nodes, ActionHeapNode(new_leaf.total_cost, new_leaf))

                cur_action_leaves_to_explore = next_action_leaves_to_explore
                next_action_leaves_to_explore = []
                best_in_cur_action_leaves_to_explore = (None if not cur_action_leaves_to_explore
                                                        else cur_action_leaves_to_explore[0].action_tree_node)
                best_in_successful_action_tree_nodes = (None if not successful_action_tree_nodes
                                                        else successful_action_tree_nodes[0].action_tree_node)

            # TODO: Add combination function that applies social cost
            if best_in_successful_action_tree_nodes is not None:
                nav_plus_manip_total_cost = (best_in_successful_action_tree_nodes.total_cost
                                             + nav_pose_to_cost_and_real_path[action_root.robot.pose][0])
                heapq.heappush(
                    best_action_leaf_for_tree, ActionHeapNode(nav_plus_manip_total_cost,
                                                              best_in_successful_action_tree_nodes))

        # 6. Get best action branch from best action tree leaf and get its corresponding contact pose,
        # transit path to contact pose tho_n, transfer path from contact pose tho_m and overall cost
        best_action_tree_leaf = best_action_leaf_for_tree[0].action_tree_node
        best_action_branch = self.__get_actions_branch(best_action_tree_leaf)
        best_contact_pose = best_action_branch[0].robot.pose
        cost = best_action_branch[-1].total_cost
        nav_cost, nav_path = nav_pose_to_cost_and_real_path[best_contact_pose]
        tho_n = Path(nav_path, o_uid=obstacle.uid, phys_cost=nav_cost)
        tho_m = self.__actions_branch_to_path(best_action_branch)

        # 7. Update w_t_plus_2 to represent with final state that corresponds to best action branch result
        final_robot = best_action_branch[-1].robot
        final_obstacle = best_action_branch[-1].obstacle
        w_t_plus_2.set_entity_polygon(final_robot.uid, final_robot.polygon, final_robot.full_geometry_acquired)
        w_t_plus_2.set_entity_polygon(final_obstacle.uid, final_obstacle.polygon, final_obstacle.full_geometry_acquired)

        # Update displays
        self._rp.cleanup_robot_sim()
        self._rp.publish_sim(final_robot.polygon, final_obstacle.polygon, "/target")
        # cc_grid.re_init_grid(binary_inflated_occupancy_grid.get_grid())
        # self._rp.publish_connected_components_grid(cc_grid.get_grid(), dd)
        return w_t_plus_2, tho_n, tho_m, cost

    def round_pose(self, pose):
        return (int(round(pose[0] * self.rounder)),
                int(round(pose[1] * self.rounder)),
                int(round(pose[2] * self.rounder)))

    @staticmethod
    def _is_there_opening_to_c_1(inflated_grid, res, grid_pose, robot_cell, c_1_cells_set):
        """
        Checks if there is a path between robot_cell and a random cell in c_1_cells_set that is not covered by an
        obstacle (especially the one considered for manipulation).
        :param inflated_grid: 2D matrix of occupation data where free is 0 and > 0 is occupied
        :type inflated_grid: numpy.array([[int16]])
        :param res: grid resolution in [m] / cell
        :type res: float
        :param grid_pose: grid pose in real coords
        :type grid_pose: tuple(float, float, float)
        :param robot_cell: robot cell coordinates in grid
        :type robot_cell: tuple(int, int)
        :param c_1_cells_set: set of coordinates in c_1 free space component
        :type c_1_cells_set: set(tuple(int, int))
        :raises:
            ValueError: if the cells set given for c_1 is empty : either it means that c_1 is fully covered by an
            obstacle, or that not updating c_1 with freed cells was actually a bad idea.
        :return: True if a path is found, False otherwise
        """
        if c_1_cells_set:
            c_1_cells_set_iterator = iter(c_1_cells_set)
            cell_in_c_1 = next(c_1_cells_set_iterator)
            while inflated_grid[cell_in_c_1[0]][cell_in_c_1[1]] != 0:
                try:
                    cell_in_c_1 = next(c_1_cells_set_iterator)
                except StopIteration:
                    raise ValueError("_has_created_new_opening_to_c_1 could not find a new opening because"
                                     " c_1_cells_set is entirely inaccessible to the robot")
            return bool(astar(inflated_grid, robot_cell, cell_in_c_1, res, grid_pose))
        else:
            raise ValueError("c_1_cells_set should never be empty !")

    def _find_path(self, w_t, r_t, r_f):
        grid = w_t.get_binary_inflated_occupancy_grid((self._robot.uid,)).get_grid()
        return a_star_real_path(grid, r_t, r_f, w_t.dd.res, w_t.dd.grid_pose, restrict_4_neighbors=False)

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
        rotation_energy = 0. if self.rot_angles.size == 0 else self.basic_rot_moment * self._world.dd.res * (
                    abs(r_j[2] - r_i[2]) / self.rot_angles[0])
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
                if cell in w_t.entities[uid].get_discrete_inflated_cells_set(w_t.dd):
                    return uid

        raise RuntimeError("__robot_exc_contained_in_obs should never reach this point.")

    @staticmethod
    def __get_actions_branch(action_node):
        branch = [action_node]
        while branch[-1].parent is not None:
            branch.append(branch[-1].parent)
        branch.reverse()
        return branch

    @staticmethod
    def __actions_branch_to_path(actions_branch):
        real_path = [action.robot.pose for action in actions_branch]
        phys_cost = actions_branch[-1].phys_cost
        social_cost = actions_branch[-1].social_cost
        o_uid = actions_branch[-1].obstacle.uid
        return Path(real_path, is_transfer=True, o_uid=o_uid, phys_cost=phys_cost, social_cost=social_cost)


class Action:

    def __init__(self):
        pass

    def apply(self, robot, obstacle, other_entities):
        raise NotImplementedError

    def predicted_pose(self, robot, obstacle):
        raise NotImplementedError


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
    def __init__(self, translation_vector, res):
        self.translation_vector = translation_vector
        self.res = res

    def apply(self, robot, obstacle, other_entities):
        translation_linestring = LineString([(0., 0.), self.translation_vector])
        rotated_linestring = affinity.rotate(translation_linestring, robot.pose[2], origin=(0., 0.))
        rotated_translation_vector = rotated_linestring.coords[1]
        return (robot.translate(
            rotated_translation_vector[0], rotated_translation_vector[1], self.res, other_entities=other_entities),
                obstacle.translate(
            rotated_translation_vector[0], rotated_translation_vector[1], self.res, other_entities=other_entities))

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
    def __init__(self, phys_cost, social_cost, parent=None, action=None, robot=None, obstacle=None,
                 has_new_local_opening=False):
        self.parent = parent
        self.action = action
        self.robot = robot
        self.obstacle = obstacle
        self.has_new_local_opening = has_new_local_opening

        self.phys_cost = phys_cost
        self.social_cost = social_cost
        self.total_cost = self.phys_cost * (1. + self.social_cost)


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
