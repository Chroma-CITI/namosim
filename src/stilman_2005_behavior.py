import a_star
from path import Path
from plan import Plan
import numpy as np
import copy
from obstacle import Obstacle
from shapely import affinity
from shapely.ops import cascaded_union
from shapely.geometry import Polygon, Point
from shapely.errors import TopologicalError
import heapq
import utils
import math


class Stilman2005Behavior:

    def __init__(self, simulator, sim_world, robot_uid):
        self.simulator = simulator
        self.world = sim_world
        self.robot = self.world.entities[robot_uid]
        self.grid = self.world.get_inflated_grid()
        self.connected_grid, self.nb_connected_components = self.world.compute_discrete_connected_components()
        self.rp = None

        # Configuration parameters
        self.alpha = 0.5
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD
        self.cost_for_obstacle_occupied_cells = 2.0
        self.trans_force = 1.0
        self.rot_force = 1.0

    def execute(self, q_init, q_goal, rp):
        self.rp = rp
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

        c_r = self.connected_grid[x_t[0]][x_t[1]]  # TODO FIX THIS TO USE DATA FROM W_T ?
        c_f = self.connected_grid[x_f[0]][x_f[1]]  # TODO FIX THIS TO USE DATA FROM W_T ?

        avoid_list = set()
        # TODO IN FACT CHANGE THIS CONDITION TO USE THE RESULT OF A SIMPLE A* ? Orig. condition is : x^f ∈ C^acc_R(W)
        if c_r == c_f:
            # If the goal is in the same connected free space as the projected robot pose at time t during computation
            return self._find_path(w_t, x_t, x_f)

        o_1, c_1 = self._rch(avoid_list, prev_list, x_f)
        while (o_1, c_1) != (None, None):
            w_t_plus_2, tho_m, c = self._manip_search(w_t, o_1, c_1)

            if tho_m is not None:
                future_plan = self._select_connect(w_t_plus_2, prev_list.append(c_1), x_f)
                if future_plan is not None:
                    tho_n = self._find_path(w_t, x_t, tho_m[0])
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

        TODO: Add visualization of closed_set, open_queue, x_1 and x_2 as GridCells
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

    def _manip_search(self, w_t, o_1, c_1):
        raise NotImplementedError

    def _find_path(self, w_t, x_t, x_f):
        return a_star.astar(grid=w_t.get_inflated_grid(),
                            start=x_t, goal_s=x_f,
                            dd=w_t.dd, rp=self.rp, restrict_4_neighbors=False)

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
