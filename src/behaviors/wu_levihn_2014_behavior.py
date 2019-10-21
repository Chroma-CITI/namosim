import src.behaviors.algorithms.a_star
from src.behaviors.plan.path import Path
from src.behaviors.plan.plan import Plan
from src.behaviors.algorithms.multi_goal_a_star import two_way_multi_goal_a_star
import numpy as np
import copy
from src.worldreps.entity_based.obstacle import Obstacle
from shapely import affinity
from shapely.ops import cascaded_union
from shapely.geometry import Polygon, Point
from shapely.errors import TopologicalError
from src.display.ros_publisher import RosPublisher


class WuLevihn2014Behavior:
    def __init__(self, simulator, initial_world, robot_uid, navigation_goals, behavior_config):
        self._simulator = simulator
        self._initial_world = initial_world
        self._robot_uid = robot_uid
        self._navigation_goals = navigation_goals
        self._behavior_config = behavior_config

        self._check_new_opening_activated = behavior_config["parameters"]["check_new_opening_activated"]
        self._social_placement_choice_activated = behavior_config["parameters"]["social_placement_choice_activated"]
        self._social_movability_evaluation_activated = behavior_config["parameters"]["social_movability_evaluation_activated"]
        self._reset_knowledge_activated = behavior_config["parameters"]["reset_knowledge_activated"]
        self._use_social_layer = behavior_config["parameters"]["use_social_layer"]
        self._manip_weight = behavior_config["parameters"]["manip_weight"]

        self._world = copy.deepcopy(self._initial_world)
        self._robot = self._world.entities[self._robot_uid]
        self._blocked_obstacles = set()

        self._rp = RosPublisher()

    def execute(self, q_init, q_goal):
        self._world = copy.deepcopy(self._initial_world) if self._reset_knowledge_activated else self._world
        self._robot = self._world.entities[self._robot_uid]
        self._rp.publish_goal(q_init, q_goal, self._world.entities[self._robot_uid].polygon)

        q_r = q_init
        e_l, m_l = [], []
        exec_success = True
        p_opt = [Plan([Path(
            src.behaviors.algorithms.a_star.a_star_real_path(self._world.get_grid(), q_r, q_goal, self._world.dd))])]
        self._rp.publish_p_opt(p_opt[0])

        while not all(np.isclose(q_r, q_goal, rtol=0.00001)):
            self._simulator.update_robot_knowledge(self._world, self._robot_uid)
            q_r = self._world.entities[self._robot_uid].pose
            self._rp.publish_robot_world(self._world, self._robot_uid)

            is_p_opt_valid = p_opt[0].is_valid(self._world, self._blocked_obstacles)
            if not is_p_opt_valid or not exec_success:
                self._rp.cleanup_p_opt()
                p_opt = [Plan([Path(
                    src.behaviors.algorithms.a_star.a_star_real_path(self._world.get_grid(), q_r, q_goal, self._world.dd))])]
                self._rp.publish_p_opt(p_opt[0])
                self.make_plan(q_r, q_goal, p_opt, e_l, m_l)

            if not p_opt[0].is_empty() and not p_opt[0].has_infinite_cost():
                step = p_opt[0].pop_next_step()
                exec_success = self._simulator.act(self._robot_uid, step)
                # If execution of a manipulation step failed, then obstacle is set as unmovable and remembered
                if not exec_success and step.is_transfer:
                    blocked_obstacle = self._world.entities[step.obstacle_uid]
                    self._blocked_obstacles.add(blocked_obstacle.uid)
                    self._initial_world.add_entity(blocked_obstacle)
                # If an object is moved, free space is created, thus we invalidate m_l
                if exec_success and step is not None and step.is_transfer:
                    m_l = []
            else:
                return False
        return True

    def make_plan(self, q_r, q_goal, p_opt, e_l, m_l):
        # Update e_l
        for entity in self._world.entities.values():
            if isinstance(entity, Obstacle):
                entity_movability = self._robot.deduce_movability(entity.type)
                if (entity.uid not in self._blocked_obstacles
                        and entity_movability == "movable" or entity_movability == "unknown"):
                    c3_est = float("inf")
                    for q_manip in entity.get_actions(self._world.dd, self._robot.deduce_push_only(entity.type)).values():
                        c3_est = min(c3_est, np.linalg.norm([q_goal[0] - q_manip[0], q_goal[1] - q_manip[1]]))
                        self.update_list(e_l, entity.uid, c3_est)
                elif entity_movability == "unmovable" or entity.uid in self._blocked_obstacles:
                    self.remove_from_list(e_l, entity.uid)
                    self.remove_from_list(m_l, entity.uid)

        index_e_l, index_m_l = 0, 0
        evaluated_obstacles_uids = set()

        # Update m_l
        while (min(self._get_cost_at_index(m_l, index_m_l), self._get_cost_at_index(e_l, index_e_l))
               < p_opt[0].total_cost):
            if self._get_cost_at_index(m_l, index_m_l) < self._get_cost_at_index(e_l, index_e_l):
                o_best_uid = self._get_obs_uid_at_index(m_l, index_m_l)
                if o_best_uid not in evaluated_obstacles_uids:
                    p_o_best = self.make_plan_for_obs(q_r, q_goal, o_best_uid, p_opt)
                    if not p_o_best.has_infinite_cost():
                        self.update_list(
                            m_l, o_best_uid, p_o_best.path_components[1].phys_cost + p_o_best.path_components[2].phys_cost)
                    evaluated_obstacles_uids.add(o_best_uid)
                index_m_l = index_m_l + 1
            else:
                o_best_uid = self._get_obs_uid_at_index(e_l, index_e_l)
                if o_best_uid not in evaluated_obstacles_uids:
                    # If the min_cost_L doesn't contain the obstacle, use best obstacle found in e_l
                    if self.find_in_list(m_l, o_best_uid) is None:
                        p_o_best = self.make_plan_for_obs(q_r, q_goal, o_best_uid, p_opt)
                        if not p_o_best.has_infinite_cost():
                            self.update_list(
                                m_l, o_best_uid, p_o_best.path_components[1].phys_cost + p_o_best.path_components[2].phys_cost)
                        evaluated_obstacles_uids.add(o_best_uid)
                index_e_l = index_e_l + 1

    def make_plan_for_obs(self, q_r, q_goal, o_uid, p_opt):
        p_best = Plan([Path([])])
        obs = self._world.entities[o_uid]
        robot = self._world.entities[self._robot_uid]

        obs_is_push_only = self._robot.deduce_push_only(obs.type)
        self._rp.publish_q_manips_for_obs(obs.get_actions(self._world.dd, obs_is_push_only).values())

        for unit_translation, q_manip in obs.get_actions(self._world.dd, obs_is_push_only).items():
            c_1 = Path(src.behaviors.algorithms.a_star.a_star_real_path(self._world.get_grid(), q_r, q_manip, self._world.dd), o_uid=o_uid)
            self._rp.publish_c_1(c_1)
            if not c_1.has_infinite_cost():
                c_0_is_valid, c_1_is_valid = True, True

                if self._social_movability_evaluation_activated:
                    if self._robot.deduce_movability(obs.type) == "unknown":
                        q_look_index = self._get_last_look_q(robot, obs, c_1)
                        if q_look_index is not None:
                            c_0, c_1 = self._split_at_pose(c_1, q_look_index, o_uid)
                        else:
                            c_0, c_1 = self.compute_c_0_c_1(self._world, robot, obs, q_r, q_manip)
                        c_0_is_valid, c_1_is_valid = not c_0.has_infinite_cost(), not c_1.has_infinite_cost()

                if c_0_is_valid and c_1_is_valid:
                    init_robot_polygon = affinity.translate(robot.polygon, q_manip[0] - q_r[0], q_manip[1] - q_r[1])
                    init_robot_polygon = affinity.rotate(init_robot_polygon, q_manip[2] - q_r[2] % 360.0)

                    self._rp.publish_sim(init_robot_polygon, obs.polygon, "/init")

                    total_translation, is_step_success, q_sim, c_est, target_obs_polygon = self._sim_one_step(
                        self._world, obs, [0.0, 0.0], unit_translation, q_manip, q_goal, c_1, init_robot_polygon)

                    while c_est <= p_opt[0].total_cost and is_step_success:
                        if (self._check_new_opening(self._world, obs, target_obs_polygon, q_goal)
                                and self._not_in_taboo(self._world.taboos, target_obs_polygon)):
                            world_copy = copy.deepcopy(self._world)
                            world_copy.translate_entity(o_uid, total_translation)
                            if self._use_social_layer:
                                social_cost = self.get_social_cost_for_entity(o_uid, self._world, world_copy)
                                c_2 = Path.line_path(q_manip, q_sim, weigth=self._manip_weight,
                                                     unit_translation=unit_translation, is_transfer=True, o_uid=o_uid,
                                                     social_cost=social_cost)
                            else:
                                c_2 = Path.line_path(q_manip, q_sim, weigth=self._manip_weight,
                                                     unit_translation=unit_translation, is_transfer=True, o_uid=o_uid)
                            self._rp.publish_c_2(c_2)
                            c_3 = Path(
                                src.behaviors.algorithms.a_star.a_star_real_path(world_copy.get_grid(), q_sim, q_goal, world_copy.dd),
                                o_uid=o_uid)
                            self._rp.publish_c_3(c_3)
                            if not c_3.has_infinite_cost():
                                p = Plan([c_1, c_2, c_3])
                                if p.total_cost < p_best.total_cost:
                                    p_best = p
                                    if p.total_cost < p_opt[0].total_cost:
                                        p_opt[0] = p
                                        self._rp.publish_robot_sim_costmap(world_copy, self._robot_uid)
                                        self._rp.publish_p_opt(p_opt[0])
                        # Increment one step
                        total_translation, is_step_success, q_sim, c_est, target_obs_polygon = self._sim_one_step(
                            self._world, obs, total_translation, unit_translation, q_manip, q_goal,
                            c_1, init_robot_polygon)

            self._rp.cleanup_eval_c1_c2_c3_sim_init_target()
        self._rp.cleanup_q_manips_for_obs()
        return p_best

    # --- From original algorithm: simulate the move of the object for one step

    def _sim_one_step(self, world, obs, p_total_translation, unit_translation, q_manip, q_goal,
                      c_1, init_robot_polygon):
        total_translation = p_total_translation + np.array(unit_translation)
        target_robot_polygon = affinity.translate(
            init_robot_polygon, total_translation[0], total_translation[1])
        target_obs_polygon = affinity.translate(obs.polygon, total_translation[0], total_translation[1])
        self._rp.publish_sim(target_robot_polygon, target_obs_polygon, "/target")

        is_step_success = self._is_step_success(world, obs.uid, init_robot_polygon, target_robot_polygon,
                                                obs.polygon, target_obs_polygon)
        q_sim = (target_robot_polygon.centroid.coords[0][0],
                 target_robot_polygon.centroid.coords[0][1],
                 q_manip[2])
        c_est = c_1.phys_cost + np.linalg.norm(total_translation) * self._manip_weight + np.linalg.norm(
            [q_goal[0] - q_sim[0], q_goal[1] - q_sim[1]])

        return total_translation, is_step_success, q_sim, c_est, target_obs_polygon

    def _is_step_success(self, world, o_uid, init_robot_polygon, target_robot_polygon,
                         init_obs_polygon, target_obs_polygon):
        robot_swept_area = cascaded_union([init_robot_polygon, target_robot_polygon]).convex_hull
        obs_swept_area = cascaded_union([init_obs_polygon, target_obs_polygon]).convex_hull

        for entity_uid, entity in world.entities.items():
            if entity_uid != self._robot_uid and entity_uid != o_uid:
                if entity.polygon.intersects(robot_swept_area) or entity.polygon.intersects(obs_swept_area):
                    return False
        return True

    # --- From original algorithm: check new local openings ---

    def _check_new_opening(self, world, obs, target_obs_polygon, q_goal):
        # TODO: DEBUG THIS METHOD, SEEMS NOT TO WORK PROPERLY !
        #  - For this, add visualization of the blocking areas and other various polygons
        if not self._check_new_opening_activated:
            return True

        init_inflated_obs_polygon = obs.polygon.buffer(2 * world.dd.inflation_radius)
        target_inflated_robot_polygon = target_obs_polygon.buffer(2 * world.dd.inflation_radius)

        # Our improvement over the original method: _check_new_opening does not prevent evaluation of plans where the
        # obstacle would pass over the goal
        if Point([q_goal[0], q_goal[1]]).intersects(
                cascaded_union([init_inflated_obs_polygon, target_inflated_robot_polygon]).convex_hull):
            return True

        # Under the assumption that obstacles are convex polygons, there can only be one blocking area per entity
        init_blocking_areas = dict()
        target_blocking_areas = dict()

        for entity_uid, entity in world.entities.items():
            if entity_uid != self._robot_uid and entity_uid != obs.uid:
                try:
                    init_blocking_area = init_inflated_obs_polygon.intersection(entity.polygon)
                    if isinstance(init_blocking_area, Polygon):
                        init_blocking_areas[entity_uid] = init_blocking_area
                except TopologicalError:
                    pass
                try:
                    target_blocking_area = target_inflated_robot_polygon.intersection(entity.polygon)
                    if isinstance(target_blocking_area, Polygon):
                        target_blocking_areas[entity_uid] = target_blocking_area
                except TopologicalError:
                    pass  # There is no intersection...

        for init_blocking_area in init_blocking_areas.values():
            if self._check_still_blocked(init_blocking_area, target_blocking_areas):
                continue
            return True  # If even one blocking area is no longer blocked, return True
        return False  # If no blocking areas has been unblocked, return False

    def _check_still_blocked(self, init_blocking_area, target_blocking_areas):
        for target_blocking_area in target_blocking_areas.values():
            if init_blocking_area.intersects(target_blocking_area):
                return True  # If area is still blocked, there is no local opening here, get next init
        # If initial blocking area does not intersect with any of the target ones, then it is no longer blocked
        return False

    # --- Method for ensuring that the object is not left in taboo zones ---

    def _not_in_taboo(self, taboos, target_obs_polygon):
        if self._social_placement_choice_activated:
            for taboo in taboos.values():
                if target_obs_polygon.intersects(taboo.polygon):
                    return False
        return True

    # --- Methods for ensuring that the path allows observation of target ---

    def _get_last_look_q(self, robot, obs, path):
        index = len(path.path) - 1
        while index != -1:
            look_pose = path.path[index]
            trans = [look_pose[0] - robot.pose[0], look_pose[1] - robot.pose[1]]
            rot = look_pose[2] - robot.pose[2] % 360.0
            displaced_fov_polygon = affinity.rotate(affinity.translate(robot.s_fov_polygon, trans[0], trans[1]), rot)
            if obs.polygon.within(displaced_fov_polygon):
                return index
        return None

    def _split_at_pose(self, c_1_in, q_look_index, o_uid, c_0_in=None):
        c_1_in_is_c_0 = (q_look_index == (len(c_1_in.path) - 1))
        c_0_out = Path((c_0_in.path if c_0_in is not None else []) + (
            c_1_in.path[0:q_look_index + 1] if not c_1_in_is_c_0 else c_1_in.path), is_observation=True, o_uid=o_uid)
        c_1_out = Path((c_1_in.path[q_look_index + 1:len(c_1_in.path)] if not c_1_in_is_c_0
                        else [c_1_in.path[len(c_1_in.path) - 1]]), o_uid=o_uid)
        return c_0_out, c_1_out

    def compute_c_0_c_1(self, world, robot, obs, q_r, q_manip):
        if self._social_movability_evaluation_activated:
            q_l = obs.get_q_l(world)
            c_0_path, c_1_path = two_way_multi_goal_a_star(world.get_grid(), q_r, q_l, q_manip, world.dd)
            q_look_index = self._get_last_look_q(robot, obs, c_1_path)
            if q_look_index is not None:
                return self._split_at_pose(c_1_path, q_look_index, obs.uid, c_0_path)
            else:
                return Path([]), Path([])

    # --- BIG PROPOSITION 1: COMPUTE A FREE SPACE AFFORDANCE FOR LEAVING OBJECTS ---

    def get_social_cost_for_entity(self, entity_uid, world, world_copy, aggregation_type='avg'):

        social_costmap = world.get_social_costmap((entity_uid,))
        entity_cells = world_copy.get_discrete_cells_set_for_entity_uid(entity_uid)
        self._rp.publish_social_cells(entity_cells, world.dd)

        if aggregation_type == 'avg':
            _avg = 0.0
            counter = 0
            for cell in entity_cells:
                _avg += social_costmap[cell[0]][cell[1]]
                counter += 1
            return _avg / float(counter)
        elif aggregation_type == 'sum':
            _sum = 0.0
            for cell in entity_cells:
                _sum += social_costmap[cell[0]][cell[1]]
            return _sum
        elif aggregation_type == 'max':
            _max = 0.0
            for cell in entity_cells:
                cell_value = social_costmap[cell[0]][cell[1]]
                if cell_value > _max:
                    _max = cell_value
            return _max
        else:
            print("ALERT: Should not pass through here in get_social_cost_for_entity function...")

    # --- Helper methods ---

    def find_in_list(self, array, uid):
        items = [item for item in array if item[0] == uid]
        return items[0] if len(items) == 1 else None

    def update_list(self, array, uid, cost):
        item = self.find_in_list(array, uid)
        if item is None:
            array.append([uid, cost])
            array.sort(key=lambda x: x[1])
        else:
            item[1] = cost

    def remove_from_list(self, array, uid):
        item = self.find_in_list(array, uid)
        if item is not None:
            array.remove(item)

    def _get_cost_at_index(self, array, index):
        try:
            return array[index][1]
        except IndexError:
            # If cost not found at index, it means
            return float('inf')

    def _get_obs_uid_at_index(self, array, index):
        try:
            return array[index][0]
        except IndexError as e:
            raise e
