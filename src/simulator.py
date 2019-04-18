from ros_publisher import RosPublisher
from world import World
from robot import Robot
from obstacle import Obstacle

import time
import copy
from threading import Thread
from ptpython.repl import embed

from shapely import affinity
from shapely.ops import cascaded_union

from standard_behavior import StandardBehavior
from s_namo_behavior import SNAMOBehavior


class Simulator:
    def __init__(self, config_file_path, world_file_path):
        # Create ros_publisher
        self.rp = RosPublisher(config_file_path)

        # Create world from world description yaml file
        self.world = World()
        self.world.load_from_yaml(world_file_path)
        self.rp.cleanup_world("/sim")
        self.rp.publish_world(self.world, "/sim")

        self.goals = []

        robot_world = self._create_robot_world_from_sim_world()
        self.rp.cleanup_world("/robot")
        self.rp.publish_world(robot_world, "/robot")

        # self.robot_behavior = StandardBehavior(self, robot_world, robot_world.robot_uid)
        self.robot_behavior = SNAMOBehavior(self, robot_world, robot_world.robot_uid)

        self.user_preempted = False

        self.run_thread = Thread(target=self.run)
        self.run_thread.start()

    def run(self):
        print("Run thread started")
        while True:
            # print("Run thread runnning")
            if not self.goals:
                time.sleep(1)

            else:
                while self.goals:
                    self.robot_behavior.execute(self.world.entities[self.world.robot_uid].pose, self.goals[0], self.rp)
                    self.goals.pop(0)

    def add_goal(self, x, y, yaw):
        self.goals.append([x, y, yaw])

    def override_goal(self, x, y, yaw):
        self.goals = [[x, y, yaw]]
        # May need to stop the run thread and relaunch it from here: BAD IDEA !

    def update_robot_knowledge(self, robot_world):
        robot_pose = robot_world.entities[robot_world.robot_uid].pose

        # Update robot pose in sim_world and sim_robot_pose subsequently
        sim_robot = self.world.entities[robot_world.robot_uid]
        trans = [sim_robot.pose[0] - robot_pose[0], sim_robot.pose[1] - robot_pose[1]]
        rot = (sim_robot.pose[2] - robot_pose[2]) % 360
        robot_world.entities[robot_world.robot_uid].translate(trans[0], trans[1], robot_world.dd)
        robot_world.entities[robot_world.robot_uid].rotate(rot, robot_world.dd)

        # Get entities in real-world fovs, merge the result and update in robot_world
        entities_in_g_fov = self.world.get_entities_in_g_fov_seethrough(robot_world.robot_uid)
        robot_world.update_from_g_fov(entities_in_g_fov)
        entities_in_s_fov = self.world.get_entities_in_s_fov_seethrough(robot_world.robot_uid)
        robot_world.update_from_s_fov(entities_in_s_fov)

        # if self.user_preempted:
        #     pass  # TODO PASS IT TO THE EXECUTION LOOP OF BEHAVIOR

    def try_exe_next_step(self, robot_uid, next_step):
        if next_step is None:
            return True

        robot = self.world.entities[robot_uid]

        current_pose = robot.pose

        target_trans = [next_step.target_pose[0] - current_pose[0], next_step.target_pose[1] - current_pose[1]]
        target_rot = (next_step.target_pose[2] - current_pose[2]) % 360

        target_robot_polygon_rot = affinity.rotate(copy.deepcopy(robot.polygon), target_rot, 'centroid')
        target_robot_polygon_rot_trans = affinity.translate(target_robot_polygon_rot, target_trans[0], target_trans[1])
        robot_rot_convex_hull = cascaded_union([robot.polygon, target_robot_polygon_rot])
        robot_trans_convex_hull = cascaded_union([target_robot_polygon_rot, target_robot_polygon_rot_trans])

        obs_rot_convex_hull, obs_trans_convex_hull, obstacle = None, None, None
        if next_step.is_transfer:
            obstacle = self.world.entities[next_step.obstacle_uid]
            if obstacle.movability == "unmovable":
                return False
            target_obs_polygon_rot = affinity.rotate(obstacle.polygon, target_rot, 'centroid')
            target_obs_polygon_rot_trans = affinity.translate(target_obs_polygon_rot, target_trans[0],
                                                              target_trans[1])
            obs_rot_convex_hull = cascaded_union([obstacle.polygon, target_obs_polygon_rot]).convex_hull
            obs_trans_convex_hull = cascaded_union([target_obs_polygon_rot, target_obs_polygon_rot_trans]).convex_hull

        for entity_uid, entity in self.world.entities.items():
            if (entity_uid != robot_uid
                    and (True if next_step.obstacle_uid is None else entity_uid != next_step.obstacle_uid)):
                if (robot_rot_convex_hull.intersects(entity.polygon)
                        or robot_trans_convex_hull.intersects(entity.polygon)):
                    return False
                if next_step.is_transfer:
                    if (obs_rot_convex_hull.intersects(entity.polygon)
                            or obs_trans_convex_hull.intersects(entity.polygon)):
                        return False

        # If all collision checks have passed, apply step and return True
        robot.translate(target_trans[0], target_trans[1], self.world.dd)
        robot.rotate(target_rot, self.world.dd)
        if next_step.is_transfer:
            obstacle.translate(target_trans[0], target_trans[1], self.world.dd)
            # No rotation for now

        self.world._update_grid() # TODO: Again, make this automatic on world change...

        self.rp.publish_world(self.world, "/sim")

        return True

    def _create_robot_world_from_sim_world(self):
        entities = dict()
        for entity_uid, entity in self.world.entities.items():
            if isinstance(entity, Robot) or (isinstance(entity, Obstacle) and entity.type == "wall"):
                entities[entity_uid] = copy.deepcopy(entity)

        return World(entities, copy.deepcopy(self.world.taboos), self.world.robot_uid,
                     copy.deepcopy(self.world.whitelist), copy.deepcopy(self.world.push_only_list),
                     self.world.force_pushes_only, copy.deepcopy(self.world.dd))


if __name__ == '__main__':
    import rospy
    rospy.init_node('world_gui_test_node', log_level=rospy.INFO)

    sim = Simulator(config_file_path="/home/xia0ben/catkin_ws/src/namo_navigation/config/config.yaml",
                    world_file_path="/home/xia0ben/catkin_ws/src/namo_navigation/worlds/03.yaml")

    # sim.add_goal(3.0, 0.0, 0.0)
    sim.add_goal(3.0, 3.0, 0.0)

    # sim.add_goal(-3.5, -2.0, 270)
    # sim.add_goal(4.0, -2.0, 180)

    banner = """
    Welcome in the S-NAMO simulator !
    
    You are within the program now: you can access the simulator's
    functions simply by calling it's python functions, like so:
    
    sim.help()
    
    To Exit, press Ctrl+z.
    """

    print(banner)

    embed(globals(), locals())
