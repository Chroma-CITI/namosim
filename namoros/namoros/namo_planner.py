import typing as t

from namosim import agents
from namosim.display.ros2_publisher import RosPublisher
from namosim.navigation import basic_actions as ba
from namosim.navigation.action_result import ActionSuccess
from namosim.navigation.navigation_path import (
    TransferPath,
    TransitPath,
    EvasionTransitPath,
)
from namosim.navigation.navigation_plan import Plan
from namosim.utils.utils import NamosimLogger
from namosim.world.entity import Movability, Style
from namosim.world.goal import Goal
from namosim.world.obstacle import Obstacle
from namosim.world.world import World
from nav_msgs.msg import Path
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from shapely.geometry import Point, Polygon
from std_msgs.msg import Header
import namosim.svg_styles as svg_styles
import namoros.utils as utils
from namoros.utils import Pose2D
from shapely.geometry import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from namoros_msgs.msg import NamoPlan, NamoPath
from namoros_msgs.msg import NamoPath, NamoPlan, NamoAction
from namosim.navigation.basic_actions import (
    Wait,
    Translation,
    Rotation,
    Advance,
    Grab,
    Release,
    Action,
)
from namosim.navigation.conflict import Conflict
from namoros_msgs.msg import NamoEntity


class StepResult(t.NamedTuple):
    action: ba.Action | None
    expected_pose_after_action: t.Tuple[float, float, float]


class NamoRosPath:
    def __init__(
        self,
        path: TransferPath | TransitPath,
        header: Header,
    ):
        self.namo_path = path
        self.is_transfer = path.is_transfer
        self.obstacle_id: str | None = None
        self.header = header
        if isinstance(path, TransferPath):
            if path.is_transfer:
                self.obstacle_id = path.obstacle_uid
        self.nav2_path = self.to_nav2_path(header=header)

    def to_nav2_path(
        self,
        header: Header,
    ) -> Path:
        path_msg = Path()
        path_msg.header = header

        for pose in self.namo_path.robot_path.poses:
            x, y, theta = pose
            pose_stamped = utils.construct_ros_pose(x, y, 0.0, theta, header=header)
            path_msg.poses.append(pose_stamped)  # type: ignore
        return path_msg


def namo_path_to_nav2_path(header: Header, path: TransitPath | TransferPath) -> Path:
    path_msg = Path()
    path_msg.header = header
    for pose in path.robot_path.poses:
        x, y, theta = pose
        pose_stamped = utils.construct_ros_pose(x, y, 0.0, theta, header=header)
        path_msg.poses.append(pose_stamped)  # type: ignore
    return path_msg


def plan_to_msg(plan: Plan, header: Header):
    plan_msg = NamoPlan()
    for path in plan.paths:
        if path.is_fully_executed():
            continue
        path_msg = NamoPath()
        path_msg.path = namo_path_to_nav2_path(header, path)
        for action in path.actions[path.action_index :]:
            action_msg = action_to_message(action)
            path_msg.actions.append(action_msg)
        if isinstance(path, TransferPath):
            path_msg.is_transfer = True
            path_msg.obstacle_id = path.obstacle_uid
        if isinstance(path, EvasionTransitPath):
            path_msg.is_evasion = True
        plan_msg.paths.append(path_msg)

    postpone_steps = 0
    if plan.postpone.is_running():
        postpone_steps = plan.postpone.duration
    plan_msg.postpone_steps = postpone_steps

    return plan_msg


def action_to_message(action: Action) -> NamoAction:
    msg = NamoAction()
    if isinstance(action, Wait):
        msg.action_type = NamoAction.WAIT
    elif isinstance(action, Translation):
        msg.action_type = NamoAction.TRANSLATION
        msg.translation_x = action.v[0]
        msg.translation_y = action.v[1]
    elif isinstance(action, Rotation):
        msg.action_type = NamoAction.ROTATION
        msg.rotation_angle_degrees = action.angle
    elif isinstance(action, Grab):
        msg.action_type = NamoAction.GRAB
        msg.distance = action.distance
        msg.obstacle_id = action.entity_uid
    elif isinstance(action, Release):
        msg.action_type = NamoAction.RELEASE
        msg.distance = action.distance
        msg.obstacle_id = action.entity_uid
    elif isinstance(action, Advance):
        msg.action_type = NamoAction.ADVANCE
        msg.distance = action.distance
    else:
        raise Exception("Unsupported action type")
    return msg


class NamoPlanner:
    def __init__(
        self,
        ros_node: Node,
        scenario_file: str,
        agent_id: str,
        omniscient_obstacle_perception: bool,
        logger: RcutilsLogger,
    ):
        nammosim_logger = NamosimLogger(printout=True, ros2_logger=logger)
        if scenario_file.strip().lower().endswith(".svg"):
            self.world = World.load_from_svg(scenario_file, logger=nammosim_logger)
        else:
            self.world = World.load_from_yaml(scenario_file, logger=nammosim_logger)

        if not omniscient_obstacle_perception:
            for obstacle in self.world.get_movable_obstacles():
                self.world.remove_entity(obstacle.uid)

        agent = self.world.agents[agent_id]
        if not isinstance(agent, agents.Stilman2005Agent):
            raise Exception("Agent must have a stilman_2005_behavior")

        self.agent: agents.Stilman2005Agent = agent
        self.ros_publisher = RosPublisher(
            ros_node=ros_node,
            agent_ids=[self.agent.uid],
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.step_count = 0
        self.logger = logger

    def add_movable_obstacle(self, entity_id: str, pose: Pose2D, polygon: Polygon):
        self.logger.info(f"Adding movable obstacle. id={entity_id}, pose={pose}")
        movable = Obstacle(
            type_="movable",
            uid=entity_id,
            polygon=polygon,
            pose=pose,
            style=Style.from_string(svg_styles.DEFAULT_MOVABLE_ENTITY_STYLE),
            movability=Movability.MOVABLE,
            full_geometry_acquired=True,
        )
        self.world.add_entity(movable)
        self.world.resolve_collisions(movable.uid)
        for agent in self.world.agents.values():
            agent.init(self.world)

    def get_movable_obstacle_polygon(self, entity_id: str) -> Polygon | None:
        if entity_id not in self.world.dynamic_entities:
            return None
        return self.world.dynamic_entities[entity_id].polygon

    def get_agent_pose(self, agent_id="robot_0") -> utils.Pose2D:
        pose = self.world.agents[agent_id].pose
        return utils.Pose2D(x=pose[0], y=pose[1], degrees=pose[2])

    def reset_robot_pose(self, agent: agents.Agent, pose: Pose2D):
        agent.move_to_pose(pose)
        self.world.resolve_collisions(agent.uid)
        self.publish_world()

    def reset_goal_pose(self, pose: Pose2D):
        goal = Goal(
            uid="goal_0",
            polygon=Point(pose[0], pose[1]).buffer(self.agent.circumscribed_radius),
            pose=pose,
        )
        self.agent.set_navigation_goals([goal])

    def set_initial_and_goal_pose(self, initial_pose: Pose2D, goal_pose: Pose2D):
        self.agent.move_to_pose(initial_pose)
        goal = Goal(
            uid="goal_0",
            polygon=Point(goal_pose[0], goal_pose[1]).buffer(
                self.agent.circumscribed_radius
            ),
            pose=goal_pose,
        )
        self.agent.set_navigation_goals([goal])

    def compute_plan(self, header: Header) -> t.Tuple[Plan, NamoPlan] | None:
        self.ros_publisher.clear_robot_plan(self.agent.uid)

        # sense
        self.agent.sense(self.world, ActionSuccess(), self.step_count)

        # think
        think_result = self.agent.think(ros_publisher=self.ros_publisher)
        plan = think_result.plan

        if plan is None:
            plan.reset()
            self.ros_publisher.publish_robot_plan(plan, self.agent, map=self.world.map)
            self.logger.info("Computed namo plan")
            return None

        plan_msg = plan_to_msg(plan, header)

        return plan, plan_msg

    def think(self, header: Header) -> t.Tuple[Plan, NamoPlan] | None:
        self.step_count += 1
        self.ros_publisher.clear_robot_plan(self.agent.uid)
        self.world.resolve_collisions(self.agent.uid)

        # sense
        self.agent.sense(self.world, ActionSuccess(), self.step_count)

        # think/act
        plan = self.agent.get_plan()
        if plan:
            if plan.postpone.is_running():
                plan.postpone.go_to_end()
                self.agent.think(ros_publisher=self.ros_publisher)

        think_result = self.agent.think(ros_publisher=self.ros_publisher)
        plan = think_result.plan

        if plan:
            self.ros_publisher.publish_robot_plan(plan, self.agent, map=self.world.map)
            self.logger.info(f"Udated namo plan: {plan.paths}")
            plan_msg = plan_to_msg(plan, header)
            return plan, plan_msg

    def detect_conflicts(self) -> t.Set[Conflict]:
        self.agent.sense(self.world, ActionSuccess(), self.step_count)
        plan = self.agent.get_plan()
        if plan:
            return plan.get_conflicts(
                world=self.world,
                robot_inflated_grid=self.agent.robot_inflated_grid,
                grab_start_distance=self.agent.grab_start_distance,
                rp=None,
                check_horizon=self.agent.conflict_horizon,
            )
        return set()

    def step_simulation(self, actions: t.Dict[str, ba.Action]):
        self.agent.sense(self.world, ActionSuccess(), self.step_count)
        self.world.step(actions, self.step_count)
        self.ros_publisher.publish_world(self.world)

    def publish_world(self):
        self.ros_publisher.clear_world()
        self.ros_publisher.publish_world(self.world)

    def synchronize_state(self, other_robots: t.List[NamoEntity]):
        self.step_count += 1

        for robot in other_robots:
            if robot.entity_id in self.world.agents:
                agent = self.world.agents[robot.entity_id]
                pose = Pose2D(robot.pose.x, robot.pose.y, robot.pose.angle_degrees)
                self.reset_robot_pose(agent, pose)
            else:
                # TODO
                pass

        self.agent.sense(self.world, ActionSuccess(), self.step_count)
        self.publish_world()

    def end_postpone(self):
        # sense
        self.agent.sense(self.world, ActionSuccess(), self.step_count)

        # think/act
        plan = self.agent.get_plan()
        if plan and plan.postpone.is_running():
            plan.postpone.go_to_end()
