import numpy as np
import copy
import time
from shapely import affinity
from shapely.ops import cascaded_union

from src.behaviors.algorithms.graph_search import a_star_real_path
from src.behaviors.plan.path import Path
from src.behaviors.plan.plan import Plan
from src.behaviors.algorithms.multi_goal_a_star import two_way_multi_goal_a_star
from src.behaviors.algorithms.new_local_opening_check import check_new_local_opening, is_move_passing_over_pose

from plan.basic_actions import GoalFailed, GoalsFinished, GoalSuccess
from src.worldreps.entity_based.obstacle import Obstacle
from src.behaviors.plan.action_result import ActionSuccess
from baseline_behavior import BaselineBehavior


class WuLevihn2014Behavior(BaselineBehavior):
    """
    Implementation of Wu and Levihn's NAMO algorithm in unknown environments
    """

    def __init__(self, initial_world, robot_uid, navigation_goals, behavior_config, abs_path_to_logs_dir):
        BaselineBehavior.__init__(self, initial_world, robot_uid, navigation_goals, behavior_config, abs_path_to_logs_dir)

        self._check_new_opening_activated = behavior_config["parameters"]["check_new_opening_activated"]
        self._social_placement_choice_activated = behavior_config["parameters"]["social_placement_choice_activated"]
        self._social_movability_evaluation_activated = behavior_config["parameters"]["social_movability_evaluation_activated"]
        self._reset_knowledge_activated = behavior_config["parameters"]["reset_knowledge_activated"]
        self._use_social_layer = behavior_config["parameters"]["use_social_layer"]
        self._manip_weight = behavior_config["parameters"]["manip_weight"]

        self._blocked_obstacles = set()
        self._e_l, self._m_l = [], []

    def think(self):
        if self._navigation_goals or self._q_goal is not None:
            # If we need to get to the next goal, reset plan, heuristic obstacle lists e_l and m_l (+ world if needed)
            if self._q_goal is None:
                self._q_goal = self._navigation_goals.pop(0)
                self._world = copy.deepcopy(self._initial_world) if self._reset_knowledge_activated else self._world
                q_r = self._robot.pose
                self._rp.publish_goal(q_r, self._q_goal, self._robot.polygon, ns=self._robot_name)

                self._e_l, self._m_l = [], []
                self._last_action_result = None
                grid = self._world.get_binary_inflated_occupancy_grid((self._robot_uid,)).get_grid()
                self._p_opt = Plan(
                    [Path(a_star_real_path(grid, q_r, self._q_goal, self._world.dd.res, self._world.dd.grid_pose, ns=self._robot_name))],
                    self._q_goal)
                self._rp.publish_p_opt(self._p_opt, ns=self._robot_name)

            q_r = self._robot.pose

            # TODO Extract abs_tol constant and make it a parameter for each goal
            is_close_enough_to_goal = all(np.isclose(q_r, self._q_goal, rtol=1e-5))
            if is_close_enough_to_goal:
                print("SUCCESS: Agent '{name}' has successfully reached pose {nav_goal}.".format(
                    name=self._robot.name, nav_goal=str(self._q_goal)))
                action = GoalSuccess(self._q_goal)
                self._q_goal = None
                return action

            # If execution of a manipulation step failed, then obstacle is set as unmovable and remembered
            if self._last_action_result is not None:
                last_action = self._last_action_result.action
                # If an object is moved, free space is created, thus we invalidate m_l
                if last_action.is_transfer:
                    self._m_l = []
                    # If the manipulation action failed, then the obstacle is marked as blocked
                    is_last_action_success = isinstance(self._last_action_result, ActionSuccess)
                    if not is_last_action_success:
                        blocked_obstacle = self._world.entities[last_action.obstacle_uid]
                        self._blocked_obstacles.add(blocked_obstacle.uid)
                        self._initial_world.add_entity(blocked_obstacle)

            if not self._p_opt.is_valid(self._world, self._robot_uid) or self._last_action_result is None:
                grid = self._world.get_binary_inflated_occupancy_grid((self._robot_uid,)).get_grid()

                self._rp.cleanup_p_opt(ns=self._robot_name)
                self._p_opt = Plan(
                    [Path(a_star_real_path(grid, q_r, self._q_goal, self._world.dd.res, self._world.dd.grid_pose, ns=self._robot_name))],
                    self._q_goal
                )
                self._rp.publish_p_opt(self._p_opt, ns=self._robot_name)
                self.make_plan(q_r, self._q_goal)

            if not self._p_opt.is_empty():
                next_step = self._p_opt.pop_next_step()
                return next_step
            elif self._p_opt.has_infinite_cost():
                print("FAILURE: Agent '{name}' has failed to reach pose {nav_goal}.".format(
                    name=self._robot.name, nav_goal=str(self._q_goal)))
                action = GoalFailed(self._q_goal)
                self._q_goal = None
                return action

        else:
            print("FINISH: Agent '{name}' has finished trying to reach its goals !".format(name=self._robot.name))
            return GoalsFinished()

    def make_plan(self, q_r, q_goal):
        # Update e_l
        for entity in self._world.entities.values():
            if isinstance(entity, Obstacle):
                entity_movability = self._robot.deduce_movability(entity.type)
                if (entity.uid not in self._blocked_obstacles
                        and entity_movability == "movable" or entity_movability == "unknown"):
                    c3_est = float("inf")
                    for q_manip in entity.get_actions(self._world.dd.inflation_radius, self._world.dd.res,
                                                      self._robot.deduce_push_only(entity.type)).values():
                        c3_est = min(c3_est, np.linalg.norm([q_goal[0] - q_manip[0], q_goal[1] - q_manip[1]]))
                        self.update_list(self._e_l, entity.uid, c3_est)
                elif entity_movability == "unmovable" or entity.uid in self._blocked_obstacles:
                    self.remove_from_list(self._e_l, entity.uid)
                    self.remove_from_list(self._m_l, entity.uid)

        index_e_l, index_m_l = 0, 0
        evaluated_obstacles_uids = set()

        # Update m_l
        while (min(self._get_cost_at_index(self._m_l, index_m_l), self._get_cost_at_index(self._e_l, index_e_l))
               < self._p_opt.total_cost):
            if self._get_cost_at_index(self._m_l, index_m_l) < self._get_cost_at_index(self._e_l, index_e_l):
                o_best_uid = self._get_obs_uid_at_index(self._m_l, index_m_l)
                if o_best_uid not in evaluated_obstacles_uids:
                    p_o_best = self.make_plan_for_obs(q_r, q_goal, o_best_uid)
                    if not p_o_best.has_infinite_cost():
                        self.update_list(
                            self._m_l, o_best_uid, p_o_best.path_components[1].phys_cost + p_o_best.path_components[2].phys_cost)
                    evaluated_obstacles_uids.add(o_best_uid)
                index_m_l = index_m_l + 1
            else:
                o_best_uid = self._get_obs_uid_at_index(self._e_l, index_e_l)
                if o_best_uid not in evaluated_obstacles_uids:
                    # If the min_cost_L doesn't contain the obstacle, use best obstacle found in e_l
                    if self.find_in_list(self._m_l, o_best_uid) is None:
                        p_o_best = self.make_plan_for_obs(q_r, q_goal, o_best_uid)
                        if not p_o_best.has_infinite_cost():
                            self.update_list(
                                self._m_l, o_best_uid, p_o_best.path_components[1].phys_cost + p_o_best.path_components[2].phys_cost)
                        evaluated_obstacles_uids.add(o_best_uid)
                index_e_l = index_e_l + 1

    def make_plan_for_obs(self, q_r, q_goal, o_uid):
        p_best = Plan([], self._q_goal)
        obs = self._world.entities[o_uid]
        robot = self._world.entities[self._robot_uid]

        obs_is_push_only = self._robot.deduce_push_only(obs.type)
        self._rp.publish_q_manips_for_obs(obs.get_actions(self._world.dd.inflation_radius, self._world.dd.res,
                                                          obs_is_push_only).values(), ns=self._robot_name)

        world_copy = copy.deepcopy(self._world)
        self._rp.publish_robot_sim_costmap(world_copy, self._robot_uid, ns=self._robot_name)

        for unit_translation, q_manip in obs.get_actions(self._world.dd.inflation_radius, self._world.dd.res,
                                                         obs_is_push_only).items():
            grid = self._world.get_binary_inflated_occupancy_grid((self._robot_uid,)).get_grid()
            c_1 = Path(a_star_real_path(grid, q_r, q_manip, self._world.dd.res, self._world.dd.grid_pose, ns=self._robot_name), o_uid=o_uid)
            self._rp.publish_c_1(c_1, ns=self._robot_name)
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

                    self._rp.publish_sim(init_robot_polygon, obs.polygon, "/init", ns=self._robot_name)

                    total_translation, is_step_success, q_sim, c_est, target_robot_polygon, target_obs_polygon =\
                        self._sim_one_step(
                            self._world, obs, [0.0, 0.0], unit_translation, q_manip, q_goal, c_1, init_robot_polygon)

                    init_blocking_areas = None

                    while c_est <= self._p_opt.total_cost and is_step_success:
                        if self._check_new_opening_activated:
                            other_entities_polygons = [entity.polygon for entity in self._world.entities.values()
                                                       if entity.uid != self._robot_uid and entity.uid != obs.uid]
                            has_new_local_opening, init_blocking_areas = check_new_local_opening(
                                obs.polygon, target_obs_polygon, other_entities_polygons,
                                robot.min_inflation_radius, init_blocking_areas, ns=self._robot_name)
                            # Don't prevent full evaluation of plans when obstacle would pass over the goal
                            moved_polygons = [init_robot_polygon, target_robot_polygon, obs.polygon, target_obs_polygon]
                            move_passes_over_goal = is_move_passing_over_pose(moved_polygons, q_goal)
                            is_it_worth_fully_evaluating = has_new_local_opening or move_passes_over_goal
                        else:
                            is_it_worth_fully_evaluating = True
                        if is_it_worth_fully_evaluating:

                            world_copy.translate_entity(o_uid, total_translation)
                            # self._rp.publish_robot_sim_costmap(world_copy, self._robot_uid)
                            if self._use_social_layer:
                                social_grid = world_copy.get_social_topological_occupation_cost_grid((self._robot_uid, o_uid))
                                social_cost = world_copy.agg_grid_cost_for_entities((o_uid,), social_grid)
                                c_2 = Path.line_path(q_manip, q_sim, weigth=self._manip_weight,
                                                     unit_translation=unit_translation, is_transfer=True, o_uid=o_uid,
                                                     social_cost=social_cost)
                            else:
                                c_2 = Path.line_path(q_manip, q_sim, weigth=self._manip_weight,
                                                     unit_translation=unit_translation, is_transfer=True, o_uid=o_uid)
                            self._rp.publish_c_2(c_2, ns=self._robot_name)
                            world_copy_grid = world_copy.get_binary_inflated_occupancy_grid((self._robot_uid,)).get_grid()
                            c_3 = Path(a_star_real_path(world_copy_grid, q_sim, q_goal,
                                                        world_copy.dd.res, world_copy.dd.grid_pose, ns=self._robot_name),
                                       o_uid=o_uid)
                            self._rp.publish_c_3(c_3, ns=self._robot_name)
                            if not c_3.has_infinite_cost():
                                p = Plan([c_1, c_2, c_3], q_goal)
                                if p.total_cost < p_best.total_cost:
                                    p_best = p
                                    if p.total_cost < self._p_opt.total_cost:
                                        self._p_opt = p
                                        self._rp.publish_robot_sim_costmap(world_copy, self._robot_uid, ns=self._robot_name)
                                        self._rp.publish_p_opt(self._p_opt, ns=self._robot_name)

                            world_copy.translate_entity(o_uid, -total_translation)
                            # self._rp.publish_robot_sim_costmap(world_copy, self._robot_uid)

                        # Increment one step
                        total_translation, is_step_success, q_sim, c_est, target_robot_polygon, target_obs_polygon =\
                            self._sim_one_step(
                                self._world, obs, total_translation, unit_translation, q_manip, q_goal,
                                c_1, init_robot_polygon)

            self._rp.cleanup_eval_c1_c2_c3_sim_init_target(ns=self._robot_name)
        self._rp.cleanup_q_manips_for_obs(ns=self._robot_name)
        return p_best

    # --- From original algorithm: simulate the move of the object for one step

    def _sim_one_step(self, world, obs, p_total_translation, unit_translation, q_manip, q_goal,
                      c_1, init_robot_polygon):
        total_translation = p_total_translation + np.array(unit_translation)
        target_robot_polygon = affinity.translate(
            init_robot_polygon, total_translation[0], total_translation[1])
        target_obs_polygon = affinity.translate(obs.polygon, total_translation[0], total_translation[1])
        self._rp.publish_sim(target_robot_polygon, target_obs_polygon, "/target", ns=self._robot_name)

        is_step_success = self._is_step_success(world, obs.uid, init_robot_polygon, target_robot_polygon,
                                                obs.polygon, target_obs_polygon)
        q_sim = (target_robot_polygon.centroid.coords[0][0],
                 target_robot_polygon.centroid.coords[0][1],
                 q_manip[2])
        c_est = c_1.phys_cost + np.linalg.norm(total_translation) * self._manip_weight + np.linalg.norm(
            [q_goal[0] - q_sim[0], q_goal[1] - q_sim[1]])

        return total_translation, is_step_success, q_sim, c_est, target_robot_polygon, target_obs_polygon

    def _is_step_success(self, world, o_uid, init_robot_polygon, target_robot_polygon,
                         init_obs_polygon, target_obs_polygon):
        robot_swept_area = cascaded_union([init_robot_polygon, target_robot_polygon]).convex_hull
        obs_swept_area = cascaded_union([init_obs_polygon, target_obs_polygon]).convex_hull

        for entity_uid, entity in world.entities.items():
            if entity_uid != self._robot_uid and entity_uid != o_uid:
                if entity.polygon.intersects(robot_swept_area) or entity.polygon.intersects(obs_swept_area):
                    return False
        return True

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
            q_l = obs.get_q_l(world, ns=self._robot_name)
            grid = world.get_binary_inflated_occupancy_grid(robot.uid)
            c_0_path, c_1_path = two_way_multi_goal_a_star(grid, q_r, q_l, q_manip, world.dd.res, world.dd.grid_pose, ns=self._robot_name)
            q_look_index = self._get_last_look_q(robot, obs, c_1_path)
            if q_look_index is not None:
                return self._split_at_pose(c_1_path, q_look_index, obs.uid, c_0_path)
            else:
                return Path([]), Path([])

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
