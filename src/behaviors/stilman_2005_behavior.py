import copy
import heapq
import math
import time

import numpy as np
from shapely.geometry import LineString
from shapely import affinity

from baseline_behavior import BaselineBehavior
from src.behaviors.algorithms.a_star import a_star_real_path
from src.behaviors.plan.path import Path
from src.behaviors.plan.plan import Plan
from src.behaviors.algorithms.multi_goal_a_star import multi_goal_a_star_real_path, multi_goal_astar
from src.utils import utils
from src.worldreps.entity_based.obstacle import Obstacle
from src.behaviors.algorithms.new_local_opening_check import check_new_local_opening, is_move_passing_over_pose


class Stilman2005Behavior(BaselineBehavior):
    """
    TODO (as documented on 2019-09-12):
      - Add visualization to all methods (about 1-days work) and try to have them all run
      - Debug and integrate the algorithm into the bigger frame of the simulator (1 to 2 days)
    """

    def __init__(self, ref_world, initial_world, robot_uid, navigation_goals, behavior_config):
        BaselineBehavior.__init__(self, ref_world, initial_world, robot_uid, navigation_goals, behavior_config)

        # Configuration parameters
        self.alpha = 0.5
        self.neighborhood = utils.TAXI_NEIGHBORHOOD
        self.cost_for_obstacle_occupied_cells = 2.
        self.trans_force = 2.
        self.rot_force = 2.
        self._check_new_local_opening_activated = True
        self.forbid_rotations = False

        self.robot_type = "diff"  # Possible types: ["omni", "diff"]
        trans_vectors = []
        rot_angles = []
        if self.robot_type == "omni":
            if self.neighborhood == utils.CHESSBOARD_NEIGHBORHOOD:
                trans_vectors = np.array(utils.OMNI_ROBOT_CHESSBOARD_TRANS_VECTORS) * self._world.dd.res
                rot_angles = np.array(utils.OMNI_ROBOT_CHESSBOARD_ROT_ANGLES)
            elif self.neighborhood == utils.TAXI_NEIGHBORHOOD:
                trans_vectors = np.array(utils.OMNI_ROBOT_TAXI_TRANS_VECTORS) * self._world.dd.res
                rot_angles = np.array(utils.OMNI_ROBOT_TAXI_ROT_ANGLES)
        elif self.robot_type == "diff":
            if self.neighborhood == utils.CHESSBOARD_NEIGHBORHOOD:
                trans_vectors = np.array(utils.DIFF_ROBOT_CHESSBOARD_TRANS_VECTORS) * self._world.dd.res
                rot_angles = np.array(utils.DIFF_ROBOT_CHESSBOARD_ROT_ANGLES)
            elif self.neighborhood == utils.TAXI_NEIGHBORHOOD:
                trans_vectors = np.array(utils.DIFF_ROBOT_TAXI_TRANS_VECTORS) * self._world.dd.res
                rot_angles = np.array(utils.DIFF_ROBOT_TAXI_ROT_ANGLES)

        if self.forbid_rotations:
            rot_angles = np.array([])

        self.actions = []
        for trans_vector in trans_vectors:
            self.actions.append(self.__make_translate(trans_vector, self._world.dd.res))
        for rot_angle in rot_angles:
            self.actions.append(self.__make_rotate(rot_angle))

    def think(self):
        pass

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
        x_t = utils.real_to_grid(r_t[0], r_t[1], w_t.dd.res, w_t.dd.grid_pose)

        avoid_list = set()

        simple_path_to_goal = self._find_path(w_t, x_t, r_f)
        if simple_path_to_goal:
            # If the goal is in the same free space component as the robot in simulated w_t
            # Orig. condition in pseudo-code is : x^f in C^acc_R(W)
            return Plan(simple_path_to_goal)

        o_1, c_1 = self._rch(w_t, avoid_list, prev_list, r_f)
        while (o_1, c_1) != (None, None):
            c_1_cells_set = w_t.get_connected_components_grid((self._robot_uid,)).components[c_1]
            w_t_plus_2, tho_n, tho_m, cost = self._manip_search(w_t, o_1, c_1_cells_set)

            if tho_m is not None:
                future_plan = self._select_connect(w_t_plus_2, prev_list.append(c_1), r_f)
                if future_plan is not None:
                    # Following line comes from original algorithm, but does not make sense ?
                    # tho_n = self._find_path(w_t, x_t, tho_m[0])
                    return Plan([tho_n, tho_m]).append(future_plan)

            avoid_list.add((o_1, c_1))

            o_1, c_1 = self._rch(w_t, avoid_list, prev_list, r_f)

        return None

    def _rch(self, w_t, avoid_list, prev_list, r_f):
        """
        Relaxed Constraint Heuristic (RCH)
        It is a navigation planner derived from A* that allows collisions with movable obstacles.
        It selects obstacles for _select_connect to displace.
        :param w_t: state of the world at time t
        :param avoid_list: list of obstacle+free space component (o_i , c_i) pairs to avoid
        :param prev_list: list of previously visited free space components c_j
        :param r_f: goal robot configuration [x, y, theta] in {m, m, degrees}
        :return: the pair (o_1 , c_1) of the first obstacle in the path, and the first component id of free space.
        Returns (None, None) if no path exists.

        TODO: - Add visualization of closed_set, open_queue, x_1 and x_2 as GridCells
              and self.connected_grid as OccupancyGrid.
        """
        r_t = w_t.entities[self._robot_uid].pose
        x_t = utils.real_to_grid(r_t[0], r_t[1], w_t.dd.res, w_t.dd.grid_pose)
        x_f = utils.real_to_grid(r_f[0], r_f[1], w_t.dd.res, w_t.dd.grid_pose)
        x_i_to_data = {x_t: CellData(g=0.0)}

        closed_set = set()
        connected_components_grid = w_t.get_connected_components_grid((self._robot_uid,)).grid
        c_0 = connected_components_grid[x_t[0]][x_t[1]]
        if c_0 <= 0:
            raise ValueError("Initial pose cell ({x}, {y}) is not in robot configuration space!".format(
                x=x_t[0], y=x_t[1]
            ))
        open_queue = self.__make_priority_queue(x_i=x_t, f=self.__h(x_t, x_f), o_f=0, c_f=c_0)

        while open_queue:
            popped = self.__remove_first(open_queue)
            x_1, o_f, c_f = popped.x_i, popped.o_f, popped.c_f

            if x_1 in closed_set:
                continue

            if x_1 == x_f and o_f != 0 and c_f != c_0:
                return o_f, c_f

            closed_set.add(x_1)

            for x_2 in self.__adjacent(x_1):
                g_x_1 = x_i_to_data[x_1].g
                g_x_2 = self.__g(x_1, x_2, g_x_1)
                x_i_to_data[x_2] = CellData(g_x_2)
                f_x_2 = self.__f(x_2, g_x_2, x_f)

                if c_f != c_0:
                    self.__enqueue(open_queue, x_2, f_x_2, o_f, c_f)
                    continue

                x_2_in_c_r_free = connected_components_grid[x_2[0]][x_2[1]] <= 0
                if o_f != 0 and x_2_in_c_r_free:
                    self.__enqueue(open_queue, x_2, f_x_2, o_f, c_0)

                # Note: second condition of this if statement is exclusive with the second condition of the previous
                # if statement: these two could be merged together with a or
                o_i = self.__robot_exc_contained_in_obs(x_2)
                r_exc_contained_in_o_f = o_i == o_f
                if o_f != 0 and r_exc_contained_in_o_f:
                    self.__enqueue(open_queue, x_2, f_x_2, o_f, c_0)

                c_i = connected_components_grid[x_2[0]][x_2[1]]
                c_i_is_valid_component = c_i <= -1
                if o_f != 0 and c_i_is_valid_component and c_i not in prev_list and (o_f, c_i) not in avoid_list:
                    self.__enqueue(open_queue, x_2, f_x_2, o_f, c_i)

                if o_f == 0 and x_2_in_c_r_free:
                    self.__enqueue(open_queue, x_2, f_x_2, 0, c_0)

                r_exc_contained_in_o_i = not r_exc_contained_in_o_f and o_i > 0
                if o_f == 0 and r_exc_contained_in_o_i:
                    self.__enqueue(open_queue, x_2, f_x_2, o_i, c_0)
        return None, None

    def _manip_search(self, w_t, o_1, c_1_cells_set):
        # 1 - Initialize manip search simulation world and some shortcut variables
        w_t_plus_2 = copy.deepcopy(w_t)
        obstacle = w_t_plus_2.entities[o_1]
        robot = w_t_plus_2.entities[self._robot.uid]
        binary_inflated_occupancy_grid = w_t_plus_2.get_binary_inflated_occupancy_grid((self._robot.uid,))
        dd = w_t_plus_2.dd
        other_entities = [entity for entity in w_t_plus_2.entities.values()
                          if entity.uid != robot.uid and entity.uid != obstacle.uid]

        # 2 - Get sampled navigation points around obstacle
        # TODO Implement generic method that can have three possibilities:
        #  - points from middle of sides (DONE)
        #  - points sampled along buffered polygon (to create from scratch)
        #  - points sampled along lines parallel to sides, s.t. we have at least a robot width from endpoints (scratch)
        navigation_poses = obstacle.get_middle_of_sides_manipulation_poses(
            self._robot.min_inflation_radius)

        # 3 - Find paths to all accessible navigation cells and only keep these
        nav_pose_to_cost_and_real_path = multi_goal_a_star_real_path(
            binary_inflated_occupancy_grid.grid, robot.pose, navigation_poses, dd.res, dd.grid_pose)

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
                    (best_in_cur_action_leaves_to_explore.phys_cost <= best_in_successful_action_tree_nodes.total_cost))
                   and cur_action_leaves_to_explore):
                for action_heap_node in cur_action_leaves_to_explore:
                    leaf = action_heap_node.action_tree_node
                    for action in self.actions:
                        # TODO : Compute poses only from action and check whether it's worth evaluating for collisions
                        #  (it is not if the two poses are in evaluated_configurations)
                        old_robot = leaf.robot
                        old_obstacle = leaf.obstacle

                        new_robot = old_robot.light_copy()
                        new_obstacle = old_obstacle.light_copy()
                        new_robot, new_obstacle = action(new_robot, new_obstacle)

                        evaluated_configuration = (self.round_pose(new_robot.pose), self.round_pose(new_obstacle.pose))
                        if evaluated_configuration in evaluated_configurations:
                            continue
                        else:
                            evaluated_configurations.add(evaluated_configuration)

                        self._rp.publish_sim(new_robot.polygon, new_obstacle.polygon, "/target")

                        # Is the robot intersecting with ANOTHER obstacle ?
                        # is_robot_colliding = utils.is_cells_set_colliding_in_grid(
                        #     new_robot.get_discrete_cells_set(dd), binary_occupancy_grid)
                        is_robot_colliding = new_robot.intersects(other_entities)

                        # Is the obstacle intersecting with ANOTHER obstacle ?
                        # is_obstacle_colliding = utils.is_cells_set_colliding_in_grid(
                        #     new_obstacle.get_discrete_cells_set(dd), binary_occupancy_grid)
                        is_obstacle_colliding = new_obstacle.intersects(other_entities)

                        is_manip_success = not (is_robot_colliding or is_obstacle_colliding)

                        if is_manip_success:
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
                                move_passes_over_goal = is_move_passing_over_pose(moved_polygons, self._q_goal)
                            else:
                                has_new_local_opening = True
                                move_passes_over_goal = True
                            is_it_worth_fully_evaluating = has_new_local_opening or move_passes_over_goal

                            if is_it_worth_fully_evaluating:
                                robot_cell = utils.real_to_grid(
                                    new_robot.pose[0], new_robot.pose[1], dd.res, dd.grid_pose)
                                binary_inflated_occupancy_grid.update_buffered_entities(
                                    {obstacle.uid: obstacle}, {new_obstacle.uid: new_obstacle})
                                self._rp.publish_robot_sim_costmap(w_t_plus_2, self._robot_uid)
                                is_there_opening_to_c_1 = self._is_there_opening_to_c_1(
                                    binary_inflated_occupancy_grid.grid,
                                    dd.res, dd.grid_pose, robot_cell, c_1_cells_set)
                                binary_inflated_occupancy_grid.update_buffered_entities(
                                    {new_obstacle.uid: new_obstacle}, {obstacle.uid: obstacle})
                            else:
                                is_there_opening_to_c_1 = False

                            phys_cost = leaf.phys_cost + self.__e(old_robot.pose, new_robot.pose)
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

        return w_t_plus_2, tho_n, tho_m, cost

    def round_pose(self, pose):
        return int(pose[0] * self.rounder), int(pose[1] * self.rounder), int(pose[2] * self.rounder)


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
            path_to_cell_in_c_1 = multi_goal_astar(
                inflated_grid, robot_cell, c_1_cells_set, res, grid_pose, break_at_first_goal_found=True)
            return path_to_cell_in_c_1
        else:
            raise ValueError("c_1_cells_set should never be empty !")

    def _find_path(self, w_t, r_t, r_f):
        return a_star_real_path(w_t.get_binary_inflated_occupancy_grid.get_grid(self._robot.uid),
                                r_t, r_f, w_t.dd.res,
                                w_t.dd.grid_pose, restrict_4_neighbors=False)

    @staticmethod
    def __make_priority_queue(x_i, f, o_f, c_f):
        return heapq.heappush([], HeapQueueElement(x_i, f, o_f, c_f))

    def __h(self, x_i, x_j):
        return math.sqrt((x_j[0] - x_i[0]) ** 2 + (x_j[1] - x_i[1]) ** 2)

    def __f(self, x_j, g_x_j, x_f):
        return g_x_j + self.__h(x_j, x_f)

    def __g(self, x_i, x_j, g_x_i):
        return g_x_i + (1 - self.alpha) + self.alpha * self.__e(x_i, x_j)

    def __e(self, x_i, x_j):
        # TODO Add proper computation of rotation energy and use it in return value
        translation_energy = self.trans_force * np.linalg.norm((x_j[0] - x_i[0], x_j[1] - x_i[1]))
        # rotation_energy = self.rot_force * (x_j[2] - x_i[2])
        return translation_energy  # + rotation_energy

    @staticmethod
    def __remove_first(queue):
        return heapq.heappop(queue)

    def __adjacent(self, x_1):
        return utils.get_neighbors(x_1, self._world.dd.d_width, self._world.dd.d_height, self.neighborhood)

    @staticmethod
    def __enqueue(queue, x_i, f, o_f, c_f):
        heapq.heappush(queue, HeapQueueElement(x_i, f, o_f, c_f))

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

    def __robot_exc_contained_in_obs(self, x_cur, grid):
        """
        If x_cur cell is contained only by one obstacle o_i, returns o_i.
        If contained by no obstacle or by more than one, returns None.
        :param x_cur: cell coordinates as integer tuple (x, y)
        :return: obstacle uid or None
        """
        if grid[x_cur[0]][x_cur[1]] == 0:
            return None
        o_i = None
        count = 0
        # current_robot_cells = self.__compute_current_robot_cells(x_init, x_cur, init_robot_cells)
        for obs, entity in self._world.entities.items():
            if isinstance(entity, Obstacle):
                if x_cur in self._world.get_discrete_inflated_cells_set_for_entity_uid(obs):
                    count += 1
                    if count == 1:
                        o_i = obs
                    else:
                        return None
        return o_i

    @staticmethod
    def __make_translate(translation_vector, res):
        def translate(robot, obstacle):
            translation_linestring = LineString([(0., 0.), translation_vector])
            rotated_linestring = affinity.rotate(translation_linestring, robot.pose[2], origin=(0., 0.))
            rotated_translation_vector = rotated_linestring.coords[1]
            return (robot.translate(rotated_translation_vector[0], rotated_translation_vector[1], res),
                    obstacle.translate(rotated_translation_vector[0], rotated_translation_vector[1], res))
        return translate

    @staticmethod
    def __make_rotate(angle):
        def rotate(robot, obstacle):
            return robot.rotate(angle), obstacle.rotate(angle, rot_center=(robot.pose[0], robot.pose[1]))
        return rotate

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


class HeapQueueElement:
    def __init__(self, x_i, f, o_f, c_f):
        self.x_i = x_i
        self.o_f = o_f
        self.c_f = c_f
        self.f = f

    def __cmp__(self, other):
        return cmp(self.f, other.f)

    def __lt__(self, other):
        return self.f < other.f


class CellData:
    def __init__(self, g):
        self.g = g


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
