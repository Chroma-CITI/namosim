import copy
import heapq
import math

import numpy as np

import src.behaviors.algorithms.a_star
from src.utils import utils
from src.worldreps.entity_based.obstacle import Obstacle
import shapely.affinity as affinity

from src.display.ros_publisher import RosPublisher


class Stilman2005Behavior:
    """
    TODO (as documented on 2019-09-12):
      - Finish up manip_search method by completing tasks written in-place (about 1-days work)
      - Add visualization to all methods (about 1-days work) and try to have them all run
      - Debug and integrate the algorithm into the bigger frame of the simulator (1 to 2 days)
    """

    def __init__(self, simulator, sim_world, robot_uid):
        self.simulator = simulator
        self.world = sim_world
        self.robot = self.world.entities[robot_uid]
        self.grid = self.world.get_grid()
        self.connected_grid, self.nb_connected_components = self.world.compute_discrete_connected_components()

        # Configuration parameters
        self.alpha = 0.5
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD
        self.cost_for_obstacle_occupied_cells = 2.0
        self.trans_force = 1.0
        self.rot_force = 1.0

        self.robot_type = "omni" # Possible types: ["omni", "diff"]
        trans_vectors = []
        rot_angles = []
        if self.robot_type == "omni":
            if self.neighborhood == utils.CHESSBOARD_NEIGHBORHOOD:
                trans_vectors = np.array(utils.OMNI_ROBOT_CHESSBOARD_TRANS_VECTORS) * self.world.dd.res
                rot_angles = np.array(utils.OMNI_ROBOT_CHESSBOARD_ROT_ANGLES)
            elif self.neighborhood == utils.TAXI_NEIGHBORHOOD:
                trans_vectors = np.array(utils.OMNI_ROBOT_TAXI_TRANS_VECTORS) * self.world.dd.res
                rot_angles = np.array(utils.OMNI_ROBOT_TAXI_ROT_ANGLES)
        elif self.robot_type == "diff":
            if self.neighborhood == utils.CHESSBOARD_NEIGHBORHOOD:
                trans_vectors = np.array(utils.DIFF_ROBOT_CHESSBOARD_TRANS_VECTORS) * self.world.dd.res
                rot_angles = np.array(utils.DIFF_ROBOT_CHESSBOARD_ROT_ANGLES)
            elif self.neighborhood == utils.TAXI_NEIGHBORHOOD:
                trans_vectors = np.array(utils.DIFF_ROBOT_TAXI_TRANS_VECTORS) * self.world.dd.res
                rot_angles = np.array(utils.DIFF_ROBOT_TAXI_ROT_ANGLES)

        self.actions = []
        for trans_vector in trans_vectors:
            self.actions.append(self.__make_translate(trans_vector))
        for rot_angle in rot_angles:
            self.actions.append(self.__make_rotate(rot_angle))

        self.rp = RosPublisher()

    def execute(self, q_init, q_goal):
        self.rp.publish_goal(q_init, q_goal, self.robot.polygon)
        q_r = q_init

        x_f = utils.real_to_grid(q_goal[0], q_goal[1], self.world.dd)

    def _select_connect(self, w_t, prev_list, x_f):
        """

        :param w_t:
        :param prev_list:
        :param x_f:
        :return:
        """
        r_t = w_t.entities[self.robot.uid].pose
        x_t = utils.real_to_grid(r_t[0], r_t[1], w_t.dd)

        avoid_list = set()

        simple_path_to_goal = self._find_path(w_t, x_t, x_f)
        if simple_path_to_goal:
            # If the goal is in the same free space component as the robot in simulated w_t
            # Orig. condition in pseudo-code is : x^f in C^acc_R(W)
            return simple_path_to_goal

        o_1, c_1 = self._rch(avoid_list, prev_list, x_f)
        while (o_1, c_1) != (None, None):
            w_t_plus_2, tho_n, tho_m, cost = self._manip_search(w_t, o_1, c_1)

            if tho_m is not None:
                future_plan = self._select_connect(w_t_plus_2, prev_list.append(c_1), x_f)
                if future_plan is not None:
                    # tho_n = self._find_path(w_t, x_t, tho_m[0])  # Line comes from original algorithm, but does not make sense ?
                    return [tho_n, tho_m].append(future_plan)

            avoid_list.add((o_1, c_1))

            o_1, c_1 = self._rch(avoid_list, prev_list, x_f)

        return None

    def _rch(self, avoid_list, prev_list, x_f):
        """
        Relaxed Constraint Heuristic (RCH)
        It is a navigation planner derived from A* that allows collisions with movable obstacles.
        It selects obstacles for _select_connect to displace.
        :param avoid_list: list of obstacle+free space component (o_i , c_i) pairs to avoid
        :param prev_list: list of previously visited free space components c_j
        :param r_f: goal robot configuration [x, y, theta] in {m, m, degrees}
        :return: the pair (o_1 , c_1) of the first obstacle in the path, and the first component of free space.
        Returns (None, None) if no path exists.

        TODO: - Add visualization of closed_set, open_queue, x_1 and x_2 as GridCells
              and self.connected_grid as OccupancyGrid.
        """
        r_t = self.robot.pose
        x_t = utils.real_to_grid(r_t[0], r_t[1], self.world.dd)
        x_i_to_data = {x_t: CellData(g=0.0)}

        closed_set = set()
        c_0 = self.connected_grid[x_t[0]][x_t[1]]
        if c_0 >= 0:
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
                g_x_2 = self.__g(x_2, g_x_1)
                x_i_to_data[x_2] = CellData(g_x_2)
                f_x_2 = self.__f(x_2, g_x_2)

                if c_f != c_0:
                    self.__enqueue(open_queue, x_2, f_x_2, o_f, c_f)
                    continue

                x_2_in_c_r_free = self.grid[x_2[0]][x_2[1]] == 0
                if o_f != 0 and x_2_in_c_r_free:
                    self.__enqueue(open_queue, x_2, f_x_2, o_f, c_0)

                # Note: second condition of this if statement is exclusive with the second condition of the previous
                # if statement: these two could be merged together with a or
                o_i = self.__robot_exc_contained_in_obs(x_2)
                r_exc_contained_in_o_f = o_i == o_f
                if o_f != 0 and r_exc_contained_in_o_f:
                    self.__enqueue(open_queue, x_2, f_x_2, o_f, c_0)

                c_i = self.connected_grid[x_2[0]][x_2[1]]
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
        w_t_plus_2 = copy.deepcopy(w_t)
        obstacle = w_t_plus_2.entities[o_1]
        robot = w_t_plus_2.entities[self.robot.uid]
        # grid = w_t_plus_2.get_obstacle_counter_grid()
        dd = w_t_plus_2.dd
        start_cell = utils.real_to_grid(robot.pose[0], robot.pose[1], dd)

        # 1 - Get sampled navigation points around obstacle
        # TODO Implement generic method that can have three possibilities:
        #  - points from middle of sides
        #  - points sampled along buffered polygon (to create from scratch)
        #  - points sampled along lines parallel to sides, s.t. we have at least a robot width from endpoints (scratch)
        navigation_poses = obstacle.get_middle_of_sides_manipulation_poses(
            self.robot.dist_between_robot_front_and_center)

        # 2 - Convert navigation points to navigation cells
        nav_cells = set()
        nav_cell_to_nav_pose = dict()
        for nav_pose in navigation_poses:
            nav_cell = utils.real_to_grid(nav_pose[0], nav_pose[1], dd)
            nav_cells.add(nav_cell)
            nav_cell_to_nav_pose[nav_cell] = nav_pose

        # 3 - Find paths to all accessible navigation cells and only keep these
        paths_to_nav_cells = dict()  # src.behaviors.algorithms.a_star.multi_goal_astar(grid, start_cell, nav_cells, dd)  # Is a dict TODO FIXME

        # 4 - Only keep accessible cells in nav_cells
        for cell, cost_and_path in paths_to_nav_cells:
            if cost_and_path[0] == float("inf"):
                nav_cells.remove(cell)

        # 5 - Compute obstacle counter grid without robot and obstacle
        grid_without = w_t_plus_2.get_obstacle_counter_grid(excluded_entites=[self.robot.uid, o_1])

        # 6 - Explore robot action space
        # Action tree nodes to currently explore. Ordered by min phys cost.
        best_action_tree_node_for_cell = []
        for cell in nav_cells:
            # Action tree nodes to currently explore. Ordered by min phys cost.
            cur_action_leaves_to_explore = []
            # Action tree nodes to explore at next Breadth-FS iteration. Ordered by min phys cost.
            next_action_leaves_to_explore = []
            # Action tree nodes for which there is a successful plan. Ordered by total min cost.
            successful_action_tree_nodes = []

            best_in_cur_action_leaves_to_explore = cur_action_leaves_to_explore[0].action_tree_node
            best_in_successful_action_tree_nodes = successful_action_tree_nodes[0].action_tree_node

            while best_in_cur_action_leaves_to_explore.phys_cost <= best_in_successful_action_tree_nodes.total_cost:
                for leaf in cur_action_leaves_to_explore:
                    for action in self.actions:
                        old_robot_poly = leaf.robot_poly
                        old_obstacle_poly = leaf.obstacle_poly

                        new_robot_poly = action(old_robot_poly)
                        new_obstacle_poly = action(old_obstacle_poly)

                        # TODO: Implement collision detection between obstacles/robot:
                        #  - first, the same as it was done previously, by iterating over polygonal entities and using
                        #  shapely
                        #  - second, by applying the fully discretized method described by Stilman,
                        #  get_obstacle_counter_grid following Appendix A
                        # Is the robot intersecting with ANOTHER obstacle ?
                        is_robot_colliding = True # TODO FIXME

                        # Is the obstacle intersecting with ANOTHER obstacle ?
                        is_obstacle_colliding = True # TODO FIXME

                        is_manip_success = not (is_robot_colliding or is_obstacle_colliding)

                        if is_manip_success:
                            # Add new leaf to next_action_leaves_to_explore
                            # TODO: properly write def __e(r_i, r_j, o) to compute the energy cost of moving between
                            #  configuration r_i and r_j, while (or not) moving an obstacle
                            phys_cost = leaf.phys_cost # + self.__e(r_i, r_j, obstacle) FIXME
                            # TODO: get social grid and extract social cost value, as is done in s_namo_behavior class
                            social_cost = 1. # compute_social_cost() FIXME
                            new_leaf = ActionTreeNode(phys_cost=phys_cost, social_cost=social_cost,
                                                      parent=leaf, action=action,
                                                      robot_poly=new_robot_poly, obstacle_poly=new_obstacle_poly)
                            heapq.heappush(
                                next_action_leaves_to_explore, ActionHeapNode(new_leaf.phys_cost, new_leaf))

                            # TODO: Check if there is a navigation plan from current robot cell to any cell of c_1 that
                            #   is not covered by the new_obstacle_poly. Extra: apply new local opening detection
                            #   algorithm to gain performance ?
                            has_created_new_opening_to_c_1 = True # find_path() FIXME

                            if has_created_new_opening_to_c_1:
                                heapq.heappush(
                                    successful_action_tree_nodes, ActionHeapNode(new_leaf.total_cost, new_leaf))
                best_in_cur_action_leaves_to_explore = cur_action_leaves_to_explore[0].action_tree_node
                best_in_successful_action_tree_nodes = successful_action_tree_nodes[0].action_tree_node
                cur_action_leaves_to_explore = next_action_leaves_to_explore

            # TODO:
            nav_plus_manip_total_cost = best_in_successful_action_tree_nodes.total_cost # + FIXME
            heapq.heappush(
                best_action_tree_node_for_cell, CellActionHeapNode(nav_plus_manip_total_cost,
                                                                   cell,
                                                                   best_in_successful_action_tree_nodes))

        best_action_tree_node = best_action_tree_node_for_cell[0].action_tree_node
        best_contact_cell = best_action_tree_node_for_cell[0].cell
        cost = best_action_tree_node_for_cell[0].cost
        tho_n = utils.grid_path_to_real_path(
            paths_to_nav_cells[best_contact_cell], robot.pose, nav_cell_to_nav_pose[best_contact_cell], dd)
        best_action_branch = self.__get_actions_branch(best_action_tree_node)
        tho_m = self.__actions_branch_to_path(best_action_branch)
        return w_t_plus_2, tho_n, tho_m, cost

    def _find_path(self, w_t, x_t, x_f):
        return True # FIXME
        # return src.behaviors.algorithms.a_star.astar(grid=w_t.get_grid(),
        #                                              start=x_t, goal_s=x_f,
        #                                              dd=w_t.dd, restrict_4_neighbors=False)

    def __make_priority_queue(self, x_i, f, o_f, c_f):
        return heapq.heappush([], HeapQueueElement(x_i, f, o_f, c_f))

    def __h(self, x_i, x_j):
        return math.sqrt((x_j[0] - x_i[0]) ** 2 + (x_j[1] - x_i[1]) ** 2)

    def __f(self, x_j, g_x_j, x_f):
        return g_x_j + self.__h(x_j, x_f)

    def __g(self, x_j, g_x_i):
        return g_x_i + (1 - self.alpha) + self.alpha * self.__e(x_j)

    def __e(self, x_j):
        return 1.0 if self.grid[x_j[0]][x_j[1]] == 0.0 else self.cost_for_obstacle_occupied_cells

    def __remove_first(self, queue):
        return heapq.heappop(queue)

    def __adjacent(self, x_1):
        return utils.get_neighbors(x_1, self.world.dd.d_width, self.world.dd.d_height, self.neighborhood)

    def __enqueue(self, queue, x_i, f, o_f, c_f):
        heapq.heappush(queue, HeapQueueElement(x_i, f, o_f, c_f))

    def __compute_current_robot_cells(self, x_init, x_cur, init_robot_cells):
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

    def __robot_exc_contained_in_obs(self, x_cur):
        """
        If x_cur cell is contained only by one obstacle o_i, returns o_i.
        If contained by no obstacle or by more than one, returns None.
        :param x_cur: cell coordinates as integer tuple (x, y)
        :return: obstacle uid or None
        """
        if self.grid[x_cur[0]][x_cur[1]] == 0:
            return None
        o_i = None
        count = 0
        # current_robot_cells = self.__compute_current_robot_cells(x_init, x_cur, init_robot_cells)
        for obs, entity in self.world.entities.items():
            if isinstance(entity, Obstacle):
                if x_cur in self.world.get_discrete_inflated_cells_set_for_entity_uid(obs):
                    count += 1
                    if count == 1:
                        o_i = obs
                    else:
                        return None
        return o_i

    def __make_translate(self, translation_vector):
        def translate(geom):
            return affinity.translate(geom, translation_vector[0], translation_vector[1])
        return translate

    def __make_rotate(self, angle):
        def rotate(geom):
            return affinity.rotate(geom, angle)
        return rotate

    def __get_actions_branch(self, action_node):
        branch = [action_node]
        while branch[-1].parent is not None:
            branch.append(branch[-1].parent)
        return branch

    def __actions_branch_to_path(self, actions_branch):
        pass


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
    def __init__(self, phys_cost, social_cost, parent=None, action=None, robot_poly=None, obstacle_poly=None):
        self.parent = parent
        self.action = action
        self.robot_poly = robot_poly
        self.obstacle_poly = obstacle_poly

        self.phys_cost = phys_cost
        self.social_cost = social_cost
        self.total_cost = self.phys_cost * (1 + self.social_cost)


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
