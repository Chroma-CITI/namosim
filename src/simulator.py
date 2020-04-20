import time
import copy
import yaml
import json
import os
import random
from datetime import datetime

from src.behaviors.navigation_only_behavior import NavigationOnlyBehavior
from src.behaviors.wu_levihn_2014_behavior import WuLevihn2014Behavior
from src.behaviors.stilman_2005_behavior import Stilman2005Behavior

from src.behaviors.plan.basic_actions import ActionGoalsFinished, ActionGoalResult
from src.behaviors.plan.action_result import IntersectionFailure, UnmanipulableFailure, ActionSuccess
from src.worldreps.entity_based.custom_exceptions import IntersectionError

from src.display.ros_publisher import RosPublisher

from src.worldreps.entity_based.world import World
from src.worldreps.entity_based.robot import Robot
from src.worldreps.entity_based.obstacle import Obstacle

from src.utils import stats_utils


class Simulator:
    def __init__(self, simulation_file_path):
        # Import YAML world configuration file
        self.sim_start_timestring = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_%f")

        behavior_yaml_abs_path = os.path.abspath(simulation_file_path)
        config = yaml.load(open(behavior_yaml_abs_path))

        # Save general simulation parameters
        self.provide_walls = config["provide_walls"]
        self.display_sim_knowledge_only_once = config["display_sim_knowledge_only_once"]
        self.human_inflation_radius = 0.55/2.  # [m]
        simulation_file_parent_dirname = os.path.basename(
            os.path.normpath(os.path.abspath(os.path.join(behavior_yaml_abs_path, '..'))))
        simulation_filename = os.path.splitext(os.path.basename(behavior_yaml_abs_path))[0]

        rel_path_to_main_sim_logs_dir = os.path.join('../logs/', simulation_file_parent_dirname, simulation_filename)
        abs_path_to_main_sim_logs_dir = os.path.join(os.path.dirname(__file__), rel_path_to_main_sim_logs_dir)
        self.abs_path_to_logs_dir = os.path.join(abs_path_to_main_sim_logs_dir, self.sim_start_timestring + "/")
        os.makedirs(self.abs_path_to_logs_dir)

        # Reinitialize rviz display
        self.rp = RosPublisher()
        self.rp.cleanup_all()

        # Create world from world description yaml file
        world_file_path = config["files"]["world_file"]
        world_yaml_abs_path = os.path.join(os.path.dirname(behavior_yaml_abs_path), world_file_path)
        self.ref_world = World()
        goals_geometries = self.ref_world.load_from_yaml(world_yaml_abs_path)
        self.init_ref_world = copy.deepcopy(self.ref_world)

        # Associate autonomous agents with behaviors
        self.agent_uid_to_behavior = dict()
        for agent_to_behavior_config in config["agents_behaviors"]:
            agent_name = agent_to_behavior_config["agent_name"]
            agent_uid = self.ref_world.get_entity_uid_from_name(agent_name)
            if agent_name in self.agent_uid_to_behavior:
                raise RuntimeError("You can only associate a single behavior with entity: {entity_name}.".format(
                    entity_name=agent_name
                ))
            else:
                behavior_config = agent_to_behavior_config["behavior"]
                agent_behavior_name = behavior_config["name"]

                agent_navigation_goals = []

                for config_goal in behavior_config["navigation_goals"]:
                    if config_goal["name"] in goals_geometries:
                        agent_navigation_goals.append(goals_geometries[config_goal["name"]])

                if "randomization" in behavior_config:
                    if "activated" in behavior_config["randomization"]:
                        if "goal_multiplier" in behavior_config["randomization"]:
                            agent_navigation_goals *= behavior_config["randomization"]["goal_multiplier"]
                        random.shuffle(agent_navigation_goals)

                if agent_behavior_name == "navigation_only_behavior":
                    agent_world = self._create_robot_world_from_sim_world()
                    self.rp.cleanup_robot_world()
                    self.agent_uid_to_behavior[agent_uid] = NavigationOnlyBehavior(
                        agent_world, agent_uid, agent_navigation_goals, behavior_config, self.abs_path_to_logs_dir)
                elif agent_behavior_name == "wu_levihn_2014_behavior":
                    agent_world = self._create_robot_world_from_sim_world()
                    self.rp.cleanup_robot_world()
                    self.agent_uid_to_behavior[agent_uid] = WuLevihn2014Behavior(
                        agent_world, agent_uid, agent_navigation_goals, behavior_config, self.abs_path_to_logs_dir)
                elif agent_behavior_name == "stilman_2005_behavior":
                    agent_world = copy.deepcopy(self.ref_world)
                    self.rp.cleanup_robot_world()
                    self.agent_uid_to_behavior[agent_uid] = Stilman2005Behavior(
                        agent_world, agent_uid, agent_navigation_goals, behavior_config, self.abs_path_to_logs_dir)
                else:
                    raise NotImplementedError("You tried to associate entity '{agent_name}' with a behavior named"
                                              "'{b_name}' that is not implemented yet."
                                              "Maybe you mispelled something ?".format(agent_name=agent_name,
                                                                                       b_name=agent_behavior_name))
        self.rp.cleanup_sim_world()

        if self.display_sim_knowledge_only_once:
            time.sleep(2.0)
            self.rp.cleanup_sim_world()

        # Time stats
        self.agent_uid_to_think_time = {agent_uid: 0. for agent_uid in self.agent_uid_to_behavior.keys()}
        self.agent_uid_to_action_results = {agent_uid: [] for agent_uid in self.agent_uid_to_behavior.keys()}
        self.agent_uid_and_goal_to_action_results = {agent_uid: {} for agent_uid in self.agent_uid_to_behavior.keys()}
        self.run_duration = 0.
        self.agent_uid_and_goal_to_world_snapshot = {agent_uid: [] for agent_uid in self.agent_uid_to_behavior.keys()}


        self.init_nb_cc, self.init_biggest_cc_size, self.init_all_cc_sum_size, self.init_frag_percentage = stats_utils.get_connectivity_stats(
            self.init_ref_world, self.human_inflation_radius, tuple()
        )

    def run(self):
        print("Run started")
        run_start_time = time.time()

        # TODO Test this execution loop to see if it works with multiple (agent_uid, behavior) tuples at once
        #  (Especially check if properly deterministic)

        active_agents = set(self.agent_uid_to_behavior.keys())

        while active_agents:
            for agent_uid, behavior in self.agent_uid_to_behavior.items():
                last_action_result = (self.agent_uid_to_action_results[agent_uid][-1]
                                      if self.agent_uid_to_action_results[agent_uid]
                                      else ActionSuccess)
                behavior.sense(self.ref_world, last_action_result)

                planning_start_time = time.time()
                action = behavior.think()
                self.agent_uid_to_think_time[agent_uid] += time.time() - planning_start_time

                # If there are no more goals to execute for the agent behavior, then remove it
                if isinstance(action, ActionGoalsFinished):
                    active_agents.remove(agent_uid)
                elif not isinstance(action, ActionGoalResult):
                    action_result = self.act(agent_uid, action)
                    self.agent_uid_to_action_results[agent_uid].append(action_result)
                    if action.goal in self.agent_uid_and_goal_to_action_results[agent_uid]:
                        self.agent_uid_and_goal_to_action_results[agent_uid][action.goal].append(action_result)
                    else:
                        self.agent_uid_and_goal_to_action_results[agent_uid][action.goal] = [action_result]

                elif isinstance(action, ActionGoalResult):
                    self.agent_uid_and_goal_to_world_snapshot[agent_uid].append({
                        "goal": action.goal,
                        "goal_status": str(action),
                        "world_snapshot": copy.deepcopy(self.ref_world)
                    })
                    if action.goal not in self.agent_uid_and_goal_to_action_results[agent_uid]:
                        self.agent_uid_and_goal_to_action_results[agent_uid][action.goal] = []

                if not self.display_sim_knowledge_only_once:
                    self.rp.publish_sim_world(self.ref_world, agent_uid)

        # Print simulation results
        self.run_duration = time.time() - run_start_time

        simulation_report = self.create_simulation_report()
        simulation_report_json = json.dumps(simulation_report, indent=4, sort_keys=True)

        print(simulation_report_json)

        log_filepath = os.path.join(
                os.path.dirname(self.abs_path_to_logs_dir), "sim_results.json")
        with open(log_filepath, 'w') as f:
            f.write(simulation_report_json)

        return simulation_report

    def act(self, robot_uid, next_step):
        if next_step is None:
            return True

        robot = self.ref_world.entities[robot_uid]

        target_trans = [next_step.target_pose[0] - robot.pose[0], next_step.target_pose[1] - robot.pose[1]]
        target_rot = (next_step.target_pose[2] - robot.pose[2])
        if -180. <= target_rot <= 180.:
            fixed_target_rot = target_rot
        elif 180. < target_rot:
            fixed_target_rot = -(360. - target_rot)
        else:  # i.e. if target_rot < -180.
            fixed_target_rot = 360 + target_rot

        try:
            other_entities = [entity for entity_uid, entity in self.ref_world.entities.items()
                if entity_uid != robot_uid and entity_uid != next_step.obstacle_uid]
            if next_step.is_transfer:
                obstacle = self.ref_world.entities[next_step.obstacle_uid]
                if robot.deduce_movability(obstacle.type) == "unmovable":
                    return UnmanipulableFailure(next_step, next_step.obstacle_uid)
                obstacle.rotate(fixed_target_rot, robot.pose, other_entities, 5.)
                obstacle.translate(target_trans[0], target_trans[1], self.ref_world.dd.res, other_entities)
            robot.translate(target_trans[0], target_trans[1], self.ref_world.dd.res, other_entities)
            robot.rotate(fixed_target_rot, 'centroid', other_entities, 5.)
        except IntersectionError as e:
            return IntersectionFailure(next_step, robot_uid, 0)  # TODO Raise proper entity id

        return ActionSuccess(next_step)

    def _create_robot_world_from_sim_world(self):
        entities = dict()
        for entity_uid, entity in self.ref_world.entities.items():
            if (isinstance(entity, Robot)
                    or ((isinstance(entity, Obstacle) and entity.type == "wall") if self.provide_walls else True)):
                entities[entity_uid] = copy.deepcopy(entity)

        return World(entities=entities,
                     taboo_zones=copy.deepcopy(self.ref_world.taboo_zones),
                     dd=copy.deepcopy(self.ref_world.dd))

    def create_simulation_report(self):
        all_movable_types = set()
        for entity in self.init_ref_world.entities.values():
            if isinstance(entity, Robot):
                all_movable_types.update(set(entity.movable_whitelist))

        all_movables_uids = tuple({
            entity_uid for entity_uid, entity in self.init_ref_world.entities.items()
            if isinstance(entity, Robot) or (isinstance(entity, Obstacle) and entity.type in all_movable_types)})

        init_abs_social_cost = stats_utils.get_social_costs_stats(self.init_ref_world, tuple(all_movables_uids))

        report = {
            "total_run_time": self.run_duration,
            "number_of_connected_components_initial": self.init_nb_cc,
            "biggest_free_component_size_initial": self.init_biggest_cc_size,
            "free_space_size_initial": self.init_all_cc_sum_size,
            "space_fragmentation_percentage_initial": self.init_frag_percentage,
            "absolute_social_cost_initial": init_abs_social_cost,
            "agents": []
        }

        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            goals_reports = []

            goal_world_snapshots = self.agent_uid_and_goal_to_world_snapshot[agent_uid]

            for goal_world_snapshot in goal_world_snapshots:
                goal = goal_world_snapshot["goal"]
                goal_status = goal_world_snapshot["goal_status"]
                world_snapshot = goal_world_snapshot["world_snapshot"]
                actions_results_to_goal = self.agent_uid_and_goal_to_action_results[agent_uid][goal]
                transit_path_length, transfer_path_length = stats_utils.get_total_path_lengths(actions_results_to_goal)

                end_nb_cc, end_biggest_cc_size, end_all_cc_sum_size, end_frag_percentage = stats_utils.get_connectivity_stats(
                    world_snapshot, self.human_inflation_radius, tuple()
                )

                end_abs_social_cost = stats_utils.get_social_costs_stats(world_snapshot, all_movables_uids)

                total_path_length = transit_path_length + transfer_path_length

                nb_reallocated_obstacles = stats_utils.get_nb_reallocated_obstacles(self.init_ref_world, world_snapshot)

                goal_report = {
                    "goal": goal,
                    "goal_status": goal_status,
                    "number_of_transferred_obstacles": stats_utils.get_nb_transferred_obstacles(actions_results_to_goal),
                    "number_of_reallocated_obstacles": nb_reallocated_obstacles,
                    "total_path_length": total_path_length,
                    "transit_path_length": transit_path_length,
                    "transfer_path_length": transfer_path_length,
                    "transit_transfer_ratio": (
                        1. if transfer_path_length == 0. else transit_path_length / transfer_path_length),
                    "transfer_percentage": (
                        0. if total_path_length == 0. else transfer_path_length / total_path_length * 100),
                    "number_of_connected_components_after_goal": end_nb_cc,
                    "biggest_free_component_size_after_goal": end_biggest_cc_size,
                    "free_space_size_after_goal": end_all_cc_sum_size,
                    "space_fragmentation_percentage_after_goal": end_frag_percentage,
                    "absolute_social_cost_after_goal": end_abs_social_cost,
                    "number_of_connected_components_relative_change": stats_utils.relative_change(
                        self.init_nb_cc, end_nb_cc),
                    "biggest_free_component_size_relative_change": stats_utils.relative_change(
                        self.init_biggest_cc_size, end_biggest_cc_size),
                    "free_space_size_relative_change": stats_utils.relative_change(
                        self.init_all_cc_sum_size, end_all_cc_sum_size),
                    "space_fragmentation_percentage_relative_change": stats_utils.relative_change(
                        self.init_frag_percentage, end_frag_percentage, False) * 100.,
                    "absolute_social_cost_relative_change": stats_utils.relative_change(
                        init_abs_social_cost, end_abs_social_cost)
                }
                goals_reports.append(goal_report)

            agent_report = {
                "agent_uid": agent_uid,
                "agent_name": self.ref_world.entities[agent_uid].name,
                "agent_behavior_name": behavior.name,
                "total_planning_time": self.agent_uid_to_think_time[agent_uid],
                "goals_reports": goals_reports
            }

            report["agents"].append(agent_report)

        return report

    def create_simulation_light_report(self):
        report = {
            "total_run_time": self.run_duration,
            "agents": []
        }

        for agent_uid, behavior in self.agent_uid_to_behavior.items():
            agent_report = {
                "agent_uid": agent_uid,
                "agent_name": self.ref_world.entities[agent_uid].name,
                "agent_behavior_name": behavior.name,
                "total_planning_time": self.agent_uid_to_think_time[agent_uid]
            }
            report["agents"].append(agent_report)

        return report
