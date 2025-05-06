import functools
import math
import random
import typing as t

from geometry_msgs.msg import Polygon, Point, Pose, PoseArray, PoseWithCovarianceStamped
from kobuki_ros_interfaces.msg import BumperEvent
from kobuki_ros_interfaces.msg._sound import Sound
from namoros_msgs.msg._namo_plan import NamoPlan
from namoros.data_models import load_namoros_config
from namoros.movable_obstacle_tracker import MovableObstacleTracker
from namoros.navigator import BasicNavigator, GoalStatus
from namoros.utils import Pose2D
from nav2_msgs.action import (
    FollowPath,
    BackUp,
    DriveOnHeading,
    Wait,
    ComputePathToPose,
    SmoothPath,
    Spin,
)
from nav_msgs.msg import Path
from rclpy import Future
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter as RosParam
from ros_gz_interfaces.msg import ParamVec
from builtin_interfaces.msg import Duration
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,  # type: ignore
    ConnectivityException,  # type: ignore
    ExtrapolationException,  # type: ignore
)
import rclpy.time
from tf2_geometry_msgs import PoseStamped
import namoros.utils as utils
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav2_msgs.srv import ClearEntireCostmap
from numpy.linalg import LinAlgError
from rcl_interfaces.msg import Parameter, ParameterValue
import subprocess
import shapely.geometry as geom
from shapely import affinity
from namoros_msgs.srv import (
    AddOrUpdateMovableObstacle,
    ComputePlan,
    UpdatePlan,
    SimulatePath,
    GetEntityPolygon,
    SynchronizeState,
    DetectConflicts,
)
from namoros_msgs.msg import NamoPath, NamoEntity, NamoConflict
from namosim.world.world import World
from namoros.robot_tracker import RobotTracker
from std_msgs.msg import Header


class NamoBehaviorNode(Node):
    def __init__(self):
        super().__init__("namo_behavior", parameter_overrides=[])
        self.declare_parameters(
            namespace="",
            parameters=[
                ("scenario_file", RosParam.Type.STRING),
                ("config_file", RosParam.Type.STRING),
                ("agent_id", RosParam.Type.STRING),
                ("omniscient_obstacle_perception", RosParam.Type.BOOL),
            ],
        )
        self.declare_parameter("is_sim", False)
        self.scenario_file = t.cast(str, self.get_parameter("scenario_file").value)
        self.agent_id = t.cast(str, self.get_parameter("agent_id").value)
        self.namoros_config = load_namoros_config(
            t.cast(str, self.get_parameter("config_file").value)
        )
        self.omniscient_obstacle_perception = t.cast(
            bool, self.get_parameter("omniscient_obstacle_perception").value
        )
        self.is_sim = t.cast(bool, self.get_parameter("is_sim").value)

        if self.scenario_file.strip().lower().endswith(".svg"):
            world = World.load_from_svg(self.scenario_file)
        else:
            world = World.load_from_yaml(self.scenario_file)
        self.agent = world.agents[self.agent_id]
        self.robot_radius = world.agents[self.agent_id].circumscribed_radius

        self.main_cb_group = MutuallyExclusiveCallbackGroup()
        self.namo_cb_group = ReentrantCallbackGroup()

        # actions
        self.follow_path_client = ActionClient(self, FollowPath, "follow_path")
        self.backup_client = ActionClient(self, BackUp, "backup")
        self.drive_on_heading_client = ActionClient(
            self, DriveOnHeading, "drive_on_heading"
        )
        self.spin_client = ActionClient(self, Spin, "spin")
        self.compute_path_client = ActionClient(
            self, ComputePathToPose, "compute_path_to_pose"
        )
        self.smooth_path_client = ActionClient(self, SmoothPath, "smooth_path")
        self.wait_client = ActionClient(self, Wait, "wait")

        # subscribers
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, f"goal_pose_namo", self.goal_pose_callback, 10
        )
        self.sub_bumper = self.create_subscription(
            BumperEvent, "events/bumper", self.bumper_callback, 10
        )
        self.sub_laser_scan = self.create_subscription(
            LaserScan, "scan", self.laser_scan_callback, 10
        )
        self.sub_robot_info = self.create_subscription(
            NamoEntity, "/namo/robots", self._robot_info_callback, 10
        )

        if self.is_sim:
            # subscriptions
            self.obstacle_pose_subscriptions = {}
            for obstacle in self.namoros_config.obstacles:
                self.obstacle_pose_subscriptions[obstacle.name] = (
                    self.create_subscription(
                        PoseArray,
                        f"/model/{obstacle.name}/pose",
                        self.create_obstacle_pose_callback(obstacle.name),
                        10,
                        callback_group=self.main_cb_group,
                    )
                )

        # publishers
        self.pub_init_pose = self.create_publisher(
            PoseWithCovarianceStamped, "initialpose", 10
        )
        self.grab_publisher = self.create_publisher(ParamVec, "/namo_grab", 10)
        self.release_publisher = self.create_publisher(ParamVec, "/namo_release", 10)
        self.pub_sound = self.create_publisher(Sound, "/commands/sound", 10)
        self.local_footprint_publisher = self.create_publisher(
            Polygon, "local_costmap/footprint", 10
        )
        self.global_footprint_publisher = self.create_publisher(
            Polygon, "global_costmap/footprint", 10
        )
        self.path_publisher = self.create_publisher(Path, "current_namo_path", 10)
        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub_robot_info = self.create_publisher(NamoEntity, "/namo/robots", 10)
        self.robot_info_timer = self.create_timer(
            1 / 2.0, self.publish_robot_info, MutuallyExclusiveCallbackGroup()
        )

        self.nav = BasicNavigator(namespace=self.get_namespace())
        self.add_or_update_movable_obstacle_cb_group = MutuallyExclusiveCallbackGroup()

        # services
        self.srv_clear_local_costmap = self.create_client(
            ClearEntireCostmap, "local_costmap/clear_entirely_local_costmap"
        )
        self.srv_clear_global_costmap = self.create_client(
            ClearEntireCostmap, "global_costmap/clear_entirely_global_costmap"
        )
        self.srv_add_movable_obstacle = self.create_client(
            AddOrUpdateMovableObstacle,
            "namo_planner/add_or_update_movable_obstacle",
            callback_group=self.add_or_update_movable_obstacle_cb_group,
        )
        self.srv_compute_plan = self.create_client(
            ComputePlan,
            "namo_planner/compute_plan",
            callback_group=ReentrantCallbackGroup(),
        )
        self.srv_update_plan = self.create_client(
            UpdatePlan,
            "namo_planner/update_plan",
            callback_group=ReentrantCallbackGroup(),
        )
        self.srv_simulate_path = self.create_client(
            SimulatePath,
            "namo_planner/simulate_path",
            callback_group=self.namo_cb_group,
        )

        self.sync_cb_group = ReentrantCallbackGroup()
        self.srv_synchronize_planner = self.create_client(
            SynchronizeState,
            "namo_planner/synchronize_state",
            callback_group=self.sync_cb_group,
        )
        self.srv_detect_conflicts = self.create_client(
            DetectConflicts,
            "namo_planner/detect_conflicts",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.srv_get_entity_polygon = self.create_client(
            GetEntityPolygon,
            "namo_planner/get_entity_polygon",
            callback_group=self.namo_cb_group,
        )

        # transform listener
        self.tf_buffer = Buffer(cache_time=rclpy.time.Duration(seconds=20))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # state
        self.replan_count: int = 0
        self.reset()
        self.get_logger().info(f"Namespace: {self.get_namespace()}")
        self.publish_initial_pose_timer = self.create_timer(
            5.0, self.publish_initial_pose
        )
        self.publish_init_pose_count = 0
        self.init_goals()

    def publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()

        # Set the header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # Adjust frame_id as needed

        # Set the pose (example values: x=0.0, y=0.0, z=0.0, yaw=0.0)
        msg.pose.pose.position.x = self.agent.pose[0]
        msg.pose.pose.position.y = self.agent.pose[1]
        msg.pose.pose.position.z = 0.0

        orientation = utils.euler_to_quat(0, 0, math.radians(self.agent.pose[2]))
        msg.pose.pose.orientation.x = orientation[0]
        msg.pose.pose.orientation.y = orientation[1]
        msg.pose.pose.orientation.z = orientation[2]
        msg.pose.pose.orientation.w = orientation[3]

        # Set covariance (example: low uncertainty)
        msg.pose.covariance = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        self.pub_init_pose.publish(msg)
        self.get_logger().info(
            "Published initial pose: x={}, y={}".format(
                msg.pose.pose.position.x, msg.pose.pose.position.y
            )
        )

        self.publish_init_pose_count += 1
        if self.publish_init_pose_count > 1:
            self.publish_initial_pose_timer.cancel()

    def reset(self):
        self.replan_flag: bool = False
        self.update_plan_flag: bool = False
        self.goal_pose: PoseStamped | None = None
        self.grabbed = False
        self.bumper_pressed: bool = False
        self.goal_handle: ClientGoalHandle | None = None
        self.movable_obstacle_tracker = MovableObstacleTracker(self)  # type: ignore
        self.forward_dist_to_obstacle: float = float("inf")
        self.obstacle_poses: t.Dict[str, Pose] = {}
        self.plan: NamoPlan | None = None
        self.robot_tracker = RobotTracker()
        self.conflicts: t.List[NamoConflict] = []

    def init_goals(self):
        goal = self.agent.get_current_or_next_goal()
        if goal:
            header = Header()
            header.frame_id = "map"
            header.stamp = self.get_clock().now().to_msg()
            self.goal_pose = utils.construct_ros_pose(
                x=goal.pose[0], y=goal.pose[1], z=0.0, theta=goal.pose[2], header=header
            )

    def create_obstacle_pose_callback(self, obstacle_name: str):
        def cb(msg: PoseArray):
            self.obstacle_poses[obstacle_name] = msg.poses[0]  # type: ignore

        return cb

    def trigger_a_replan(self):
        self.get_logger().info("Triggering a replan.")
        self.replan_flag = True
        self.update_plan_flag = False
        self.replan_count += 1

    def trigger_update_plan(self):
        self.get_logger().info("Triggering a plan update.")
        self.replan_flag = False
        self.update_plan_flag = True
        self.replan_count += 1

    def laser_scan_callback(self, data: LaserScan):
        total_len = len(data.ranges)
        center_index = int(round((total_len / 2.0) - 1))
        min_dist = float("inf")
        min_index = -1
        values_around = 5
        for i in range(0, values_around + 1):
            if min_dist > data.ranges[center_index + i]:
                min_index = center_index + i
                min_dist = data.ranges[center_index + i]
            if min_dist > data.ranges[center_index - i]:
                min_index = center_index + i
                min_dist = data.ranges[center_index - i]
        self.forward_dist_to_obstacle = min_dist

    def _robot_info_callback(self, entity: NamoEntity):
        if entity.entity_id == self.agent_id:
            return
        self.robot_tracker.update(entity)

    def publish_robot_info(self):
        entity = NamoEntity()
        entity.entity_id = self.agent_id

        robot_pose = self.lookup_robot_pose()
        if not robot_pose:
            return

        robot_pose = utils.entity_pose_to_pose2d(robot_pose.pose)
        robot_polygon = self.lookup_robot_polygon()
        robot_polygon = utils.shapely_to_ros_polygon(robot_polygon)

        entity.pose.x = robot_pose.x
        entity.pose.y = robot_pose.y
        entity.pose.angle_degrees = robot_pose.degrees
        entity.polygon = robot_polygon

        self.pub_robot_info.publish(entity)

    def publish_cmd_vel(self, cmd_vel: Twist):
        self.pub_cmd_vel.publish(cmd_vel)

    def publish_namo_path(self, path: Path):
        self.clear_namo_path()
        self.path_publisher.publish(path)

    def clear_namo_path(self):
        empty_path = Path()
        empty_path.header.frame_id = "map"
        empty_path.header.stamp = self.get_clock().now().to_msg()
        self.path_publisher.publish(empty_path)

    def transform_pose(
        self, input_pose: PoseStamped, target_frame: str
    ) -> PoseStamped | None:
        try:
            # Transform the pose to the target frame
            transformed_pose = self.tf_buffer.transform(
                input_pose, target_frame, timeout=rclpy.time.Duration(seconds=1)
            )
            return t.cast(PoseStamped, transformed_pose)
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
            LinAlgError,
        ) as e:
            self.get_logger().error(f"Failed to transform pose: {e}")
            return None

    def _goal_response_callback(self, future: Future, final_result_future: Future):
        goal_handle: ClientGoalHandle = future.result()  # type: ignore
        if not goal_handle.accepted:
            self.get_logger().error(f"Goal rejected {goal_handle}")
            final_result_future.set_exception(Exception("goal rejected"))
            return

        self.goal_handle = goal_handle
        self.get_result_future: Future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(
            functools.partial(
                self._get_result_callback, final_result_future=final_result_future
            )
        )

    def _get_result_callback(self, future: Future, final_result_future: Future):
        try:
            result = future.result()
            final_result_future.set_result(result)
        except Exception as ex:
            self.get_logger().warn(f"action failed {ex}")
            final_result_future.set_exception(ex)

    def _wrap_send_goal_future(self, send_goal_future: Future):
        final_result_future = Future()
        send_goal_future.add_done_callback(
            functools.partial(
                self._goal_response_callback,
                final_result_future=final_result_future,
            )
        )
        return final_result_future

    def get_plan(self):
        if not self.goal_pose:
            raise Exception("No goal pose")
        try:
            robot_tf = self.tf_buffer.lookup_transform(
                source_frame=f"base_link",
                target_frame="map",
                time=rclpy.time.Time(seconds=0),
            )
        except Exception as ex:
            self.get_logger().warn(f"Failed to lookup robot tf: {ex}")
            raise ex
        robot_pose = utils.transform_to_pose(robot_tf)

        req = ComputePlan.Request()
        req.start_pose = robot_pose
        req.goal_pose = self.goal_pose
        self.get_logger().info(
            f"{self.agent_id} computing plan to goal {self.goal_pose}"
        )
        res: ComputePlan.Response = self.srv_compute_plan.call(req)
        return res.plan

    def update_plan(self):
        req = UpdatePlan.Request()
        res: UpdatePlan.Response = self.srv_update_plan.call(req)
        return res.plan

    def get_entity_polygon(self, uid: str):
        req = GetEntityPolygon.Request()
        req.entity_id = uid
        res: GetEntityPolygon.Response = self.srv_compute_plan.call(req)
        return utils.ros_polygon_to_shapely_polygon(res.polygon)

    def add_or_update_movable_ostable(
        self, uid: str, pose: Pose2D, polygon: geom.Polygon
    ):
        self.get_logger().info("Adding obstacle")
        req = AddOrUpdateMovableObstacle.Request()
        req.polygon = utils.shapely_to_ros_polygon(polygon)
        req.pose.x = float(pose.x)
        req.pose.y = float(pose.y)
        req.pose.angle_degrees = float(pose.degrees)
        req.obstacle_id = uid
        res: AddOrUpdateMovableObstacle.Response = self.srv_add_movable_obstacle.call(
            req
        )
        self.get_logger().info("Done adding obstacle")

    def simulate_path(self, path: NamoPath):
        req = SimulatePath.Request()
        req.path = path
        res: SimulatePath.Response = self.srv_simulate_path.call(req)

    def synchronize_planner(self):
        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            self.get_logger().warn("Failed to lookup robot pose")
            return
        robot_pose = utils.entity_pose_to_pose2d(robot_pose.pose)
        req = SynchronizeState.Request()
        req.observed_robot_pose.x = robot_pose.x
        req.observed_robot_pose.y = robot_pose.y
        req.observed_robot_pose.angle_degrees = robot_pose.degrees

        for other_robot in self.robot_tracker.robots.values():
            req.other_observed_robots.append(other_robot)
        self.srv_synchronize_planner.call(req)

    def detect_conflicts(self):
        req = DetectConflicts.Request()
        res: DetectConflicts.Response = self.srv_detect_conflicts.call(req)
        self.conflicts = res.conflicts
        return res.conflicts

    def lookup_robot_pose(self) -> PoseStamped | None:
        try:
            robot_tf = self.tf_buffer.lookup_transform(
                source_frame=f"base_link",
                target_frame="map",
                time=rclpy.time.Time(seconds=0),
            )
        except Exception as ex:
            self.get_logger().warn(f"Failed to lookup robot tf: {ex}")
            return None
        robot_pose = utils.transform_to_pose(robot_tf)
        return robot_pose

    def lookup_robot_polygon(self) -> geom.Polygon:
        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            raise Exception("Failed to get robot pose")
        return geom.Point(
            robot_pose.pose.position.x, robot_pose.pose.position.y
        ).buffer(self.robot_radius)

    def lookup_pose(self, frame_id: str) -> PoseStamped | None:
        try:
            robot_tf = self.tf_buffer.lookup_transform(
                source_frame=frame_id,
                target_frame="map",
                time=rclpy.time.Time(seconds=0),
            )
        except Exception as ex:
            self.get_logger().warn(f"Failed to lookup pose for frame {frame_id}: {ex}")
            return None
        pose = utils.transform_to_pose(robot_tf)
        return pose

    def goal_pose_callback(self, msg: PoseStamped):
        self.goal_pose = msg  # entity_pose_to_pose2d(msg.pose)
        self.get_logger().info(
            f"Received goal pose: {self.goal_pose}, frame: {msg.header.frame_id}"
        )

    def wait_for_successful_task_completion(self):
        while not self.nav.isTaskComplete():
            pass

        if not self.nav.status == GoalStatus.STATUS_SUCCEEDED:
            raise Exception("Failed to run nav task")

    def obstacle_marker_id_to_name(self, marker_id: str):
        for obs in self.namoros_config.obstacles:
            if obs.marker_id == marker_id:
                return obs.name
        return None

    def grab(self, obs_marker_id: str):
        self.grabbed = True
        self.get_logger().info(f"Grabbing obstacle {obs_marker_id}.")
        obstacle_name = self.obstacle_marker_id_to_name(obs_marker_id)
        if obstacle_name is None:
            obstacle_name = obs_marker_id
        if self.is_sim:
            self.set_obstacle_pose(obstacle_name=obstacle_name)

        params = ParamVec()
        robot_id_param = Parameter(name="robot_name")
        robot_id_param.value = ParameterValue(string_value=self.agent_id, type=4)
        robot_link_param = Parameter(name="robot_link")
        robot_link_param.value = ParameterValue(string_value="base_link", type=4)
        obs_id_param = Parameter(name="obstacle_name")
        obs_id_param.value = ParameterValue(string_value=obstacle_name, type=4)
        obs_link_param = Parameter(name="obstacle_link")
        obs_link_param.value = ParameterValue(string_value="box", type=4)
        params.params = [
            robot_id_param,
            robot_link_param,
            obs_id_param,
            obs_link_param,
        ]
        self.get_logger().info(f"Publishing params {str(params.__slots__)}.")
        self.grab_publisher.publish(params)
        self.update_robot_footprint_for_grab(obs_marker_id=obs_marker_id)

    def release(self):
        self.grabbed = False
        self.get_logger().info(f"Release.")
        if self.is_sim:
            params = ParamVec()
            robot_id = Parameter(name="robot_name")
            robot_id.value = ParameterValue(string_value=self.agent_id, type=4)
            robot_link = Parameter(name="robot_link")
            robot_link.value = ParameterValue(string_value="base_link", type=4)
            params.params = [robot_id, robot_link]
            self.release_publisher.publish(params)
        self.update_robot_footprint_for_release()

    def pose_info_callback(self, msg: PoseArray):
        for pose in msg.poses:
            self.get_logger().info(str(pose))

    def set_obstacle_pose(self, obstacle_name: str) -> Pose | None:
        if obstacle_name not in self.obstacle_poses:
            return
        new_box_pose = self.obstacle_poses[obstacle_name]
        new_box_pose.position.z += 0.4

        # Define the shell command
        pose_str = f"{{x: {new_box_pose.position.x}, y: {new_box_pose.position.y}, z: {new_box_pose.position.z}}}"
        orient_str = f"{{w: {new_box_pose.orientation.w}, x: {new_box_pose.orientation.x}, y: {new_box_pose.orientation.y}, z: {new_box_pose.orientation.z}}}"

        shell_command = f"ign service -s /world/namo_world/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 300 --req 'name: \"{obstacle_name}\", position: {pose_str}, orientation: {orient_str}'"
        self.get_logger().info(f"set obstacle pose command: {shell_command}")
        # Execute the shell command synchronously
        subprocess.run(shell_command, shell=True)
        return new_box_pose

    def follow_path(self, path: Path, controller_id: str):
        """Returns a future that should resolve to a FollowPath.Result object"""
        self.publish_namo_path(path)
        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = controller_id
        goal_msg.goal_checker_id = ""
        send_goal_future = self.follow_path_client.send_goal_async(goal_msg)
        final_result_future = self._wrap_send_goal_future(send_goal_future)
        return final_result_future

    def cancel_nav_task(self) -> Future:
        if self.goal_handle:
            self.get_logger().info("Canceling current task.")
            future = self.goal_handle.cancel_goal_async()
        else:
            future = Future()
            future.set_result(None)
        return future

    def advance(self, distance: float, speed: float = 0.05):
        """Returns a future that should resolve to a DriveOnHeading.Result object"""
        goal_msg = DriveOnHeading.Goal()
        goal_msg.target = Point(x=float(distance))
        goal_msg.speed = speed
        goal_msg.time_allowance = Duration(sec=20)
        send_goal_future = self.drive_on_heading_client.send_goal_async(goal_msg)
        final_result_future = self._wrap_send_goal_future(send_goal_future)
        return final_result_future

    def backup(self, distance: float, speed: float = 0.05):
        """Returns a future that should resolve to a BackUp.Result object"""
        goal_msg = BackUp.Goal()
        goal_msg.target = Point(x=float(distance))
        goal_msg.speed = speed
        goal_msg.time_allowance = Duration(sec=20)
        send_goal_future = self.backup_client.send_goal_async(goal_msg)
        final_result_future = self._wrap_send_goal_future(send_goal_future)
        return final_result_future

    def spin(self, target_yaw: float):
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = target_yaw
        goal_msg.time_allowance = Duration(sec=10)
        send_goal_future = self.spin_client.send_goal_async(goal_msg)
        final_result_future = self._wrap_send_goal_future(send_goal_future)
        return final_result_future

    def clear_bumper(self):
        self.bumper_pressed = False

    def request_bumper(self):
        self.play_random_sound()

    def play_sound(self, sound: int):
        self.pub_sound.publish(Sound(value=sound))

    def play_random_sound(self):
        random_sound = random.choice(
            [
                Sound.BUTTON,
                Sound.CLEANINGEND,
                Sound.CLEANINGSTART,
                Sound.ERROR,
                Sound.OFF,
                Sound.ON,
                Sound.RECHARGE,
            ]
        )
        self.pub_sound.publish(Sound(value=random_sound))

    def bumper_callback(self, event: BumperEvent):
        if event.state == BumperEvent.PRESSED:
            self.bumper_pressed = True
        else:
            self.bumper_pressed = False

    def compute_path(self, start: PoseStamped, goal: PoseStamped):
        msg = ComputePathToPose.Goal()
        msg.start = start
        msg.goal = goal
        msg.use_start = True
        send_goal_future = self.compute_path_client.send_goal_async(msg)
        final_result_future = self._wrap_send_goal_future(send_goal_future)
        return final_result_future

    def smooth_path(self, path: Path):
        msg = SmoothPath.Goal()
        msg.path = path
        msg.check_for_collisions = False
        send_goal_future = self.smooth_path_client.send_goal_async(msg)
        final_result_future = self._wrap_send_goal_future(send_goal_future)
        return final_result_future

    def clear_local_costmap(self) -> Future:
        req = ClearEntireCostmap.Request()
        future = self.srv_clear_local_costmap.call_async(req)
        return future

    def clear_global_costmap(self) -> Future:
        req = ClearEntireCostmap.Request()
        future = self.srv_clear_global_costmap.call_async(req)
        return future

    def update_robot_footprint_for_grab(self, obs_marker_id: str):
        if self.omniscient_obstacle_perception:
            # TODO
            return

        obstacle_polygon = self.movable_obstacle_tracker.get_obstacle_polygon(
            obs_marker_id
        )
        if obstacle_polygon is None:
            raise Exception("Obstacle not found in namosim world")
        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            raise Exception("Failed to lookup robot pose")
        robot_pose = utils.entity_pose_to_pose2d(robot_pose.pose)
        robot_polygon: geom.Polygon = geom.Point(robot_pose.x, robot_pose.y).buffer(self.robot_radius)  # type: ignore
        footprint: t.Any = geom.MultiPolygon(
            [robot_polygon, obstacle_polygon]
        ).convex_hull

        footprint = affinity.translate(
            footprint, xoff=-robot_pose.x, yoff=-robot_pose.y
        )
        footprint = affinity.rotate(
            footprint,
            angle=robot_pose.degrees,
            origin=(0, 0),  # type: ignore
            use_radians=False,
        )
        footprint = utils.shapely_to_ros_polygon(footprint)

        self.local_footprint_publisher.publish(footprint)
        self.global_footprint_publisher.publish(footprint)

    def update_robot_footprint_for_release(self):
        robot_polygon: geom.Polygon = geom.Point(0, 0).buffer(self.robot_radius)  # type: ignore
        footprint = utils.shapely_to_ros_polygon(robot_polygon)
        self.local_footprint_publisher.publish(footprint)
        self.global_footprint_publisher.publish(footprint)
