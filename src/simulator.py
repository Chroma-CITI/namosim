from src.display.ros_publisher import RosPublisher
from src.worldreps.entity_based.world import World
from src.worldreps.entity_based.robot import Robot
from src.worldreps.entity_based.obstacle import Obstacle

import time
import copy
import yaml
import os

from shapely import affinity
from shapely.ops import cascaded_union

from src.behaviors.navigation_only_behavior import NavigationOnlyBehavior
from src.behaviors.wu_levihn_2014_behavior import WuLevihn2014Behavior
from src.behaviors.stilman_2005_behavior import Stilman2005Behavior

from src.behaviors.plan.basic_actions import ActionGoalsFinished, ActionGoalResult
from src.behaviors.plan.action_result import IntersectionFailure, UnmanipulableFailure, ActionSuccess


class Simulator:
    def __init__(self, simulation_file_path):
        # Import YAML world configuration file
        behavior_yaml_abs_path = os.path.abspath(simulation_file_path)
        config = yaml.load(open(behavior_yaml_abs_path))

        # Save general simulation parameters
        self.provide_walls = config["provide_walls"]
        self.display_sim_knowledge_only_once = config["display_sim_knowledge_only_once"]

        # Reinitialize rviz display
        self.rp = RosPublisher()
        self.rp.cleanup_all()

        # Create world from world description yaml file
        world_file_path = config["files"]["world_file"]
        world_yaml_abs_path = os.path.join(os.path.dirname(behavior_yaml_abs_path), world_file_path)
        self.ref_world = World()
        goals = self.ref_world.load_from_yaml(world_yaml_abs_path)

        # Associate autonomous agents with behaviors
        self.temp_agent_uid = 0  # FIXME Remove this variable after refactoring publish world so that
        self.agent_uid_to_behavior = dict()
        for agent_to_behavior_config in config["agents_behaviors"]:
            agent_name = agent_to_behavior_config["agent_name"]
            agent_uid = self.ref_world.get_entity_uid_from_name(agent_name)
            self.temp_agent_uid = agent_uid
            if agent_name in self.agent_uid_to_behavior:
                raise RuntimeError("You can only associate a single behavior with entity: {entity_name}.".format(
                    entity_name=agent_name
                ))
            else:
                agent_world = self._create_robot_world_from_sim_world()
                self.rp.cleanup_robot_world()
                self.rp.publish_robot_world(agent_world, self.temp_agent_uid)

                behavior_config = agent_to_behavior_config["behavior"]
                agent_behavior_name = behavior_config["name"]

                agent_navigation_goals = []
                for config_goal in behavior_config["navigation_goals"]:
                    if config_goal["name"] in goals:
                        agent_navigation_goals.append(goals[config_goal["name"]])

                if agent_behavior_name == "navigation_only_behavior":
                    self.agent_uid_to_behavior[agent_uid] = NavigationOnlyBehavior(
                        self.ref_world, agent_world, agent_uid, agent_navigation_goals, behavior_config)
                elif agent_behavior_name == "wu_levihn_2014_behavior":
                    self.agent_uid_to_behavior[agent_uid] = WuLevihn2014Behavior(
                        self.ref_world, agent_world, agent_uid, agent_navigation_goals, behavior_config)
                elif agent_behavior_name == "stilman_2005_behavior":
                    self.agent_uid_to_behavior[agent_uid] = Stilman2005Behavior(
                        self.ref_world, agent_world, agent_uid, agent_navigation_goals, behavior_config)
                else:
                    raise NotImplementedError("You tried to associate entity '{agent_name}' with a behavior named"
                                              "'{b_name}' that is not implemented yet."
                                              "Maybe you mispelled something ?".format(agent_name=agent_name,
                                                                                       b_name=agent_behavior_name))
        self.rp.cleanup_sim_world()
        self.rp.publish_sim_world(self.ref_world, self.temp_agent_uid)

        if self.display_sim_knowledge_only_once:
            time.sleep(2.0)
            self.rp.cleanup_sim_world()

        # self.user_preempted = False
        # self.run_thread = Thread(target=self.run)
        # self.run_thread.start()

    def run(self):
        print("Run started")

        # TODO Test this execution loop to see if it works with multiple (agent_uid, behavior) tuples at once
        #  (Especially check if properly deterministic)
        agent_uid_to_last_action_result = dict()
        while self.agent_uid_to_behavior:
            for agent_uid, behavior in self.agent_uid_to_behavior.items():
                last_action_result = (ActionSuccess if agent_uid not in agent_uid_to_last_action_result
                                      else agent_uid_to_last_action_result[agent_uid])
                behavior.sense(self.ref_world, last_action_result)

                planning_start_time = time.time()
                action = behavior.think()
                behavior.add_planning_duration_to_report(time.time() - planning_start_time)

                # If there are no more goals to execute for the agent behavior, then remove it
                if isinstance(action, ActionGoalsFinished):
                    del self.agent_uid_to_behavior[agent_uid]
                elif not isinstance(action, ActionGoalResult):
                    action_result = self.act(agent_uid, action)
                    agent_uid_to_last_action_result[agent_uid] = action_result

    def act(self, robot_uid, next_step):
        if next_step is None:
            return True

        robot = self.ref_world.entities[robot_uid]

        current_pose = robot.pose

        target_trans = [next_step.target_pose[0] - current_pose[0], next_step.target_pose[1] - current_pose[1]]
        target_rot = (next_step.target_pose[2] - current_pose[2]) % 360.

        target_robot_polygon_rot = affinity.rotate(copy.deepcopy(robot.polygon), target_rot, 'centroid')
        target_robot_polygon_rot_trans = affinity.translate(target_robot_polygon_rot, target_trans[0], target_trans[1])
        robot_rot_convex_hull = cascaded_union([robot.polygon, target_robot_polygon_rot])
        robot_trans_convex_hull = cascaded_union([target_robot_polygon_rot, target_robot_polygon_rot_trans])

        obs_rot_convex_hull, obs_trans_convex_hull, obstacle = None, None, None
        if next_step.is_transfer:
            obstacle = self.ref_world.entities[next_step.obstacle_uid]
            if robot.deduce_movability(obstacle.type) == "unmovable":
                return UnmanipulableFailure(next_step, next_step.obstacle_uid)
            target_obs_polygon_rot = affinity.rotate(obstacle.polygon, target_rot, 'centroid')
            target_obs_polygon_rot_trans = affinity.translate(target_obs_polygon_rot, target_trans[0],
                                                              target_trans[1])
            obs_rot_convex_hull = cascaded_union([obstacle.polygon, target_obs_polygon_rot]).convex_hull
            obs_trans_convex_hull = cascaded_union([target_obs_polygon_rot, target_obs_polygon_rot_trans]).convex_hull

        for entity_uid, entity in self.ref_world.entities.items():
            if (entity_uid != robot_uid
                    and (True if next_step.obstacle_uid is None else entity_uid != next_step.obstacle_uid)):
                if (robot_rot_convex_hull.intersects(entity.polygon)
                        or robot_trans_convex_hull.intersects(entity.polygon)):
                    return IntersectionFailure(next_step, robot_uid, entity.uid)
                if next_step.is_transfer:
                    if (obs_rot_convex_hull.intersects(entity.polygon)
                            or obs_trans_convex_hull.intersects(entity.polygon)):
                        return IntersectionFailure(next_step, obstacle.uid, entity.uid)

        # If all collision checks have passed, apply step and return success
        self.ref_world.translate_entity(robot.uid, target_trans)
        self.ref_world.rotate_entity(robot.uid, target_rot)
        if next_step.is_transfer:
            self.ref_world.translate_entity(obstacle.uid, target_trans)
            # No rotation for now

        if not self.display_sim_knowledge_only_once:
            self.rp.publish_sim_world(self.ref_world, self.temp_agent_uid)

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
