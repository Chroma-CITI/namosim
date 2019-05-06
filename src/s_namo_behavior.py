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


class SNAMOBehavior:
    def __init__(self, simulator, initial_world, robot_uid):
        self.simulator = simulator
        self.robot_uid = robot_uid
        self.initial_world = initial_world
        self.world = copy.deepcopy(self.initial_world)

        self.check_new_opening_activated = True
        self.social_placement_choice_activated = True
        self.social_movability_evaluation_activated = False
        self.reset_knowledge_activated = False
        self.manip_weight = 1.0

    def execute(self, q_init, q_goal, rp):
        world = copy.deepcopy(self.initial_world) if self.reset_knowledge_activated else self.world
        rp.publish_goal(q_init, q_goal, world.entities[self.robot_uid].polygon)

        q_r = q_init
        e_l, m_l = [], []
        exec_success = True
        p_opt = [Plan([Path(a_star.a_star_real_path(world.get_grid(), q_r, q_goal, world.dd, rp))])]
        rp.publish_p_opt(p_opt[0])

        while not all(np.isclose(q_r, q_goal, rtol=0.00001)):
            self.simulator.update_robot_knowledge(world)
            q_r = world.entities[self.robot_uid].pose
            rp.publish_world(world, "/robot")

            is_p_opt_valid = p_opt[0].is_valid(world)
            if not is_p_opt_valid or not exec_success:
                rp.cleanup_p_opt()
                p_opt = [Plan([Path(a_star.a_star_real_path(world.get_grid(), q_r, q_goal, world.dd, rp))])]
                rp.publish_p_opt(p_opt[0])
                self.make_plan(world, q_r, q_goal, p_opt, e_l, m_l, rp)

            if p_opt[0].is_not_empty() and not p_opt[0].has_infinite_cost():
                step = p_opt[0].pop_next_step()
                exec_success = self.simulator.try_exe_next_step(self.robot_uid, step)
                # If execution of a manipulation step failed, then obstacle is set as unmovable and remembered
                if not exec_success and step.is_transfer:
                    blocked_obstacle = world.entities[step.obstacle_uid]
                    blocked_obstacle.movability = "unmovable"
                    self.initial_world.add_entity(blocked_obstacle)
                # If an object is moved, free space is created, thus we invalidate m_l
                if exec_success and step is not None and step.is_transfer:
                    m_l = []
            else:
                return False
        return True

    def make_plan(self, world, q_r, q_goal, p_opt, e_l, m_l, rp):
        for entity in world.entities.values():
            if isinstance(entity, Obstacle):
                if entity.movability == "movable" or entity.movability == "unknown":
                    # c1_est = float("inf")
                    c3_est = float("inf")
                    for q_manip in entity.get_actions(world.dd).values():
                        # c1_est = min(c1_est, np.linalg.norm([q_manip[0] - q_r[0], q_manip[1] - q_r[1]]))
                        c3_est = min(c3_est, np.linalg.norm([q_goal[0] - q_manip[0], q_goal[1] - q_manip[1]]))
                        self.update_list(e_l, entity.uid, c3_est)
                    # self.update_list(e_l, entity.uid, c1_est + c3_est)
                if entity.movability == "unmovable":
                    self.remove_from_list(e_l, entity.uid)
                    self.remove_from_list(m_l, entity.uid)

        index_e_l, index_m_l = 0, 0
        evaluated_obstacles_uids = set()

        while (min(self._get_cost_at_index(m_l, index_m_l), self._get_cost_at_index(e_l, index_e_l))
               < p_opt[0].cost):
            if self._get_cost_at_index(m_l, index_m_l) < self._get_cost_at_index(e_l, index_e_l):
                o_best_uid = self._get_obs_uid_at_index(m_l, index_m_l)
                if o_best_uid not in evaluated_obstacles_uids:
                    p_o_best = self.make_plan_for_obs(world, q_r, q_goal, o_best_uid, p_opt, rp)
                    if not p_o_best.has_infinite_cost():
                        self.update_list(
                            m_l, o_best_uid, p_o_best.path_components[1].cost + p_o_best.path_components[2].cost)
                    evaluated_obstacles_uids.add(o_best_uid)
                index_m_l = index_m_l + 1
            else:
                o_best_uid = self._get_obs_uid_at_index(e_l, index_e_l)
                if o_best_uid not in evaluated_obstacles_uids:
                    # If the min_cost_L doesn't contain the obstacle, use best obstacle found in e_l
                    if self.find_in_list(m_l, o_best_uid) is None:
                        p_o_best = self.make_plan_for_obs(world, q_r, q_goal, o_best_uid, p_opt, rp)
                        if not p_o_best.has_infinite_cost():
                            self.update_list(
                                m_l, o_best_uid, p_o_best.path_components[1].cost + p_o_best.path_components[2].cost)
                        evaluated_obstacles_uids.add(o_best_uid)
                index_e_l = index_e_l + 1

    def make_plan_for_obs(self, world, q_r, q_goal, o_uid, p_opt, rp):
        p_best = Plan([Path([])])
        obs = world.entities[o_uid]
        robot = world.entities[world.robot_uid]

        rp.publish_q_manips_for_obs(obs.get_actions(world.dd).values())

        for unit_translation, q_manip in obs.get_actions(world.dd).items():
            c_1 = Path(a_star.a_star_real_path(world.get_grid(), q_r, q_manip, world.dd, rp), o_uid=o_uid)
            rp.publish_c_1(c_1)
            if not c_1.has_infinite_cost():
                c_0_is_valid, c_1_is_valid = True, True

                if self.social_movability_evaluation_activated:
                    if obs.movability == "unknown":
                        q_look_index = self._get_last_look_q(robot, obs, c_1)
                        if q_look_index is not None:
                            c_0, c_1 = self._split_at_pose(c_1, q_look_index, o_uid)
                        else:
                            c_0, c_1 = self.compute_c_0_c_1(world, robot, obs, q_r, q_manip, rp)
                        c_0_is_valid, c_1_is_valid = not c_0.has_infinite_cost(), not c_1.has_infinite_cost()

                if c_0_is_valid and c_1_is_valid:
                    init_robot_polygon = affinity.translate(robot.polygon, q_manip[0] - q_r[0], q_manip[1] - q_r[1])
                    init_robot_polygon = affinity.rotate(init_robot_polygon, q_manip[2] - q_r[2] % 360.0)

                    rp.publish_sim(init_robot_polygon, obs.polygon, "/init")

                    total_translation, is_step_success, q_sim, c_est, target_obs_polygon = self._sim_one_step(
                        world, obs, [0.0, 0.0], unit_translation, q_manip, q_goal, c_1, init_robot_polygon, rp)
                    
                    while c_est <= p_opt[0].cost and is_step_success:
                        if (self._check_new_opening(world, obs, target_obs_polygon, q_goal)
                                and self._not_in_taboo(world.taboos, target_obs_polygon)):
                            c_2 = Path.line_path(q_manip, q_sim, weigth=self.manip_weight,
                                                 unit_translation=unit_translation, is_transfer=True, o_uid=o_uid)
                            rp.publish_c_2(c_2)
                            world_copy = copy.deepcopy(world)
                            world_copy.translate_entity(o_uid, total_translation)
                            c_3 = Path(a_star.a_star_real_path(world_copy.get_grid(), q_sim, q_goal, world_copy.dd, rp),
                                       o_uid=o_uid)
                            rp.publish_c_3(c_3)
                            if not c_3.has_infinite_cost():
                                p = Plan([c_1, c_2, c_3])
                                if p.cost < p_best.cost:
                                    p_best = p
                                    if p.cost < p_opt[0].cost:
                                        p_opt[0] = p
                                        rp.publish_costmap(world_copy, "/robot_sim")
                                        rp.publish_p_opt(p_opt[0])
                        # Increment one step
                        total_translation, is_step_success, q_sim, c_est, target_obs_polygon = self._sim_one_step(
                            world, obs, total_translation, unit_translation, q_manip, q_goal,
                            c_1, init_robot_polygon, rp)

            rp.cleanup_eval_c1_c2_c3_sim_init_target()
        rp.cleanup_q_manips_for_obs()
        return p_best

    def _sim_one_step(self, world, obs, p_total_translation, unit_translation, q_manip, q_goal,
                      c_1, init_robot_polygon, rp):
        total_translation = p_total_translation + np.array(unit_translation)
        target_robot_polygon = affinity.translate(
            init_robot_polygon, total_translation[0], total_translation[1])
        target_obs_polygon = affinity.translate(obs.polygon, total_translation[0], total_translation[1])
        rp.publish_sim(target_robot_polygon, target_obs_polygon, "/target")

        is_step_success = self._is_step_success(world, obs.uid, init_robot_polygon, target_robot_polygon,
                                                obs.polygon, target_obs_polygon)
        q_sim = (target_robot_polygon.centroid.coords[0][0],
                 target_robot_polygon.centroid.coords[0][1],
                 q_manip[2])
        c_est = c_1.cost + np.linalg.norm(total_translation) * self.manip_weight + np.linalg.norm(
            [q_goal[0] - q_sim[0], q_goal[1] - q_sim[1]])

        return total_translation, is_step_success, q_sim, c_est, target_obs_polygon

    def _is_step_success(self, world, o_uid, init_robot_polygon, target_robot_polygon,
                         init_obs_polygon, target_obs_polygon):
        robot_swept_area = cascaded_union([init_robot_polygon, target_robot_polygon]).convex_hull
        obs_swept_area = cascaded_union([init_obs_polygon, target_obs_polygon]).convex_hull

        for entity_uid, entity in world.entities.items():
            if entity_uid != world.robot_uid and entity_uid != o_uid:
                if entity.polygon.intersects(robot_swept_area) or entity.polygon.intersects(obs_swept_area):
                    return False
        return True

    def _check_new_opening(self, world, obs, target_obs_polygon, q_goal):
        if not self.check_new_opening_activated:
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
            if entity_uid != world.robot_uid and entity_uid != obs.uid:
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

    def _not_in_taboo(self, taboos, target_obs_polygon):
        if self.social_placement_choice_activated:
            for taboo in taboos.values():
                if target_obs_polygon.intersects(taboo.polygon):
                    return False
        return True

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

    def compute_c_0_c_1(self, world, robot, obs, q_r, q_manip, rp):
        if self.social_movability_evaluation_activated:
            q_l = obs.get_q_l(world, rp)
            c_0_path, c_1_path = a_star.two_way_multi_goal_a_star(world.get_grid(), q_r, q_l, q_manip, world.dd, rp)
            q_look_index = self._get_last_look_q(robot, obs, c_1_path)
            if q_look_index is not None:
                return self._split_at_pose(c_1_path, q_look_index, obs.uid, c_0_path)
            else:
                return Path([]), Path([])

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
