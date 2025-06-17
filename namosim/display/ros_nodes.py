# pyright: reportUnusedImport=false

from collections import OrderedDict
import copy
import subprocess
import time
import typing as t

import numpy as np
from namosim.algorithms.rrt_node import RRTNode
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseArray,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Vector3,
)
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.qos_event import PublisherEventCallbacks
from rclpy.qos_overriding_options import QoSOverridingOptions
from rclpy.utilities import ok  # noqa: F401 forwarding to this module
from shapely import affinity
from std_msgs.msg import ColorRGBA, Header  # type: ignore  # type: ignore
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import namosim.display.colors as colors
import namosim.display.ros_publisher_config as cfg
import namosim.navigation.navigation_plan as nav_plan
import namosim.world.world as world
from namosim.agents import agent
from namosim.config import DEACTIVATE_RVIZ
from namosim.data_models import GridCellModel, Pose2D
from namosim.display.conversions import (
    costmap_to_grid_map,
    make_delete_all_marker,
    plan_to_markerarray,
    polygon_to_line_strip,
    polygon_to_triangle_list,
    pose_to_ros_pose,
    string_to_text,
)
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import (
    BinaryOccupancyGrid,
)
from namosim.world.entity import Entity, Style
from namosim.world.obstacle import Obstacle
import numpy.typing as npt


def init_header(stamp: Time = Time()):
    return Header(stamp=stamp, frame_id=cfg.main_frame_id)


def poses_to_poses_array(poses: t.Iterable[Pose2D], stamp: Time = Time()):
    pose_array = PoseArray(header=init_header(stamp), poses=[])
    for pose in poses:
        pose_array.poses.append(pose_to_ros_pose(pose))  # type: ignore
    return pose_array


class NamespaceCache:
    def __init__(self):
        self.current_cell_to_marker = dict()
        self.current_cell_marker_current_id = 1
        self.cells_to_path_marker = dict()
        self.cells_path_marker_current_id = 1
        self.manip_search_neighbors_markers_p_ids = []
        self.current_fixed_robot_pose_to_marker = dict()
        self.current_fixed_robot_pose_marker_current_id = 1


def create_publisher(
    node: Node,
    msg_type: type,
    topic: str,
    *,
    callback_group: t.Optional[CallbackGroup] = None,
) -> Publisher:
    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,  # Ensures messages get delivered
        depth=10,
    )
    return node.create_publisher(
        msg_type=msg_type,
        topic=topic,
        qos_profile=10,
        callback_group=callback_group,
    )


class DefaultRosPublisherNode(Node):
    def __init__(self, node_name: str):
        # Shutdown the ROS Context if it is already running.
        # This is necessary when running multiple unit tests since each may create their own context.
        if ok():
            rclpy.shutdown()
        rclpy.init(args=None)
        super().__init__(node_name=node_name, parameter_overrides=[])


class BasePublisher:
    def __init__(
        self,
        msg_type: type,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        self.node = node
        self.topic = topic
        self._publisher = create_publisher(
            node, msg_type, topic, callback_group=callback_group
        )
        self.is_active = is_active and not DEACTIVATE_RVIZ
        self._rate = rate

        self._duration = 1.0 / self.rate
        self._last_time = time.time()
        self.get_subscription_count = self._publisher.get_subscription_count

    def get_timestamp(self):
        return self.node.get_clock().now().to_msg()

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, r: float):
        self._rate = r
        self._duration = 1.0 / self.rate

    def publish(self, msg: t.Any):
        if not DEACTIVATE_RVIZ and self.is_active:
            connections = self.get_subscription_count()
            if connections > 0:
                elapsed_time = time.time() - self._last_time
                time_to_wait = self._duration - elapsed_time
                if time_to_wait > 0.0:
                    time.sleep(time_to_wait)
                self._publisher.publish(msg)
                self._last_time = time.time()

    def reset(
        self,
        reset_msg: (
            Marker | MarkerArray | PoseArray | GridMap | OccupancyGrid | None
        ) = None,
    ):
        if not DEACTIVATE_RVIZ and reset_msg is not None:
            self._publisher.publish(reset_msg)


class ObstaclePublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        super().__init__(
            msg_type=MarkerArray,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            callback_group=callback_group,
        )

    def publish_obstacles(self, world: "world.World"):
        if DEACTIVATE_RVIZ:
            return
        movables = world.get_all_obstacles()
        marker_array = MarkerArray()
        markers = []
        for entity in movables:
            if not isinstance(entity, Obstacle):
                continue
            markers += self.obstacle_to_markers(
                entity=entity,
                p_id=utils.hash_to_32_bit_int(entity.uid),
                z_index=cfg.entities_z_index,
            )
        marker_array.markers = markers
        self.publish(marker_array)

    def obstacle_to_markers(
        self,
        *,
        entity: Obstacle,
        p_id: int,
        z_index: float,
    ) -> t.List[Marker]:
        polygon = entity.polygon
        markers = []
        markers.append(
            polygon_to_triangle_list(
                polygon=polygon,
                p_id=p_id,
                frame_id=cfg.main_frame_id,
                color=ColorRGBA(**colors.hex_to_rgba(entity.style.fill)),
                z_index=z_index,
                stamp=self.get_timestamp(),
            )
        )
        return markers

    def reset(self):
        if not DEACTIVATE_RVIZ:
            self._publisher.publish(make_delete_all_marker(cfg.main_frame_id))


class DynamicEntitiesPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        callback_group: t.Optional["CallbackGroup"] = None,
        rate: int = cfg.rate,
    ):
        super().__init__(
            msg_type=MarkerArray,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            callback_group=callback_group,
        )

    def update(self, world: "world.World"):
        msg = self.get_all_markers(world)
        self.publish(msg)

    def agent_to_markers(self, agent: "agent.Agent"):
        body_color = ColorRGBA(
            **colors.hex_to_rgba(Style.from_string(agent.agent_style.shape).fill)
        )
        orientation_fill = Style.from_string(agent.agent_style.orientation).fill
        orientation_fill = orientation_fill if orientation_fill != "none" else "#FFFFFF"
        orientation_color = ColorRGBA(**colors.hex_to_rgba(orientation_fill))
        shape_marker = polygon_to_triangle_list(
            polygon=agent.polygon,
            p_id=utils.hash_to_32_bit_int(agent.uid),
            frame_id=cfg.main_frame_id,
            color=body_color,
            z_index=cfg.entities_z_index,
            stamp=self.get_timestamp(),
            namespace="agents",
        )
        orientation_marker = polygon_to_triangle_list(
            polygon=agent.get_orientation_polygon(),
            p_id=utils.hash_to_32_bit_int(agent.uid + "orientation"),
            frame_id=cfg.main_frame_id,
            color=orientation_color,
            z_index=cfg.entities_z_index + 1,
            stamp=self.get_timestamp(),
            namespace="agents",
        )
        return [shape_marker, orientation_marker]

    def get_all_markers(
        self,
        world: "world.World",
        entities_to_ignore: t.Set[str] | None = None,
    ):
        if entities_to_ignore is None:
            entities_to_ignore = set()
        marker_array = MarkerArray()
        markers = []
        for entity in world.dynamic_entities.values():
            if entity.uid not in entities_to_ignore:
                if isinstance(entity, agent.Agent):
                    markers += self.agent_to_markers(entity)
                elif isinstance(entity, Obstacle):
                    markers += self.obstacle_to_markers(
                        entity=entity,
                        z_index=cfg.entities_z_index,
                    )
        marker_array.markers = markers
        return marker_array

    def obstacle_to_markers(
        self,
        *,
        entity: Obstacle,
        z_index: float,
    ) -> t.List[Marker]:
        polygon = entity.polygon
        markers = []
        markers.append(
            polygon_to_triangle_list(
                polygon=polygon,
                p_id=utils.hash_to_32_bit_int(entity.uid),
                frame_id=cfg.main_frame_id,
                color=ColorRGBA(**colors.hex_to_rgba(entity.style.fill)),
                z_index=z_index,
                stamp=self.get_timestamp(),
                namespace="obstacles",
            )
        )
        return markers

    def reset(self):
        if not DEACTIVATE_RVIZ:
            self._publisher.publish(make_delete_all_marker(cfg.main_frame_id))


class ManipSearchPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        super().__init__(
            msg_type=MarkerArray,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            callback_group=callback_group,
        )

    def reset(self):
        if not DEACTIVATE_RVIZ:
            self._publisher.publish(make_delete_all_marker(cfg.main_frame_id))


class WorldMapPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        super().__init__(
            msg_type=OccupancyGrid,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            callback_group=callback_group,
        )

    def publish(self, world: "world.World", agent_id: str | None = None):
        if not self.is_active:
            return
        msg = self.world_to_costmap(world, agent_id)
        super().publish(msg)

    def world_to_costmap(self, world: "world.World", agent_id: str | None = None):
        grid = world.map
        costmap = OccupancyGrid(header=init_header(self.get_timestamp()))
        costmap.info.map_load_time = costmap.header.stamp
        costmap.info.resolution = grid.cell_size
        costmap.info.width = grid.d_width
        costmap.info.height = grid.d_height
        costmap.info.origin.position.x = float(grid.grid_pose[0])
        costmap.info.origin.position.y = float(grid.grid_pose[1])
        costmap_grid = np.transpose(grid.grid == 1) * 255
        costmap.data = costmap_grid.flatten().astype(np.int8).tolist()

        return costmap

    def reset(self):
        super().reset(OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0]))


class GridMapPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        super().__init__(
            msg_type=GridMap,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            callback_group=callback_group,
        )

    def publish(self, costmap: npt.NDArray[t.Any], cell_size: float):
        grid_map = costmap_to_grid_map(costmap, cell_size, stamp=self.get_timestamp())
        self._publisher.publish(grid_map)

    def reset(self):
        super().reset(costmap_to_grid_map(np.full((1000, 1000), np.nan), 1.0))


class CombinedCostmapPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        super().__init__(
            msg_type=GridMap,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            callback_group=callback_group,
        )

    def publish(
        self,
        sorted_cell_to_combined_cost: OrderedDict[GridCellModel, float],
        inflated_grid_by_obstacle: BinaryOccupancyGrid,
    ):
        combined_costmap = np.zeros(
            (inflated_grid_by_obstacle.d_width, inflated_grid_by_obstacle.d_height)
        )
        for cell, combined_cost in sorted_cell_to_combined_cost.items():
            combined_costmap[cell[0]][cell[1]] = combined_cost

        # re-scale and shift the costmap so it displays nicely below the 2D environment in RVIZ
        H = combined_costmap.shape[0] * inflated_grid_by_obstacle.cell_size
        M = np.ptp(combined_costmap)
        m = np.min(combined_costmap)
        cc = 0.5 * (combined_costmap - m) / M  # make the costmap 0.5 meters tall
        cc = cc - 2  # display 2.0 meters below 0

        grid_map = costmap_to_grid_map(
            cc,
            inflated_grid_by_obstacle.cell_size,
            frame_id=cfg.combined_gridmap_frame_id,
            stamp=self.get_timestamp(),
        )
        super().publish(grid_map)

    def reset(self):
        super().reset(costmap_to_grid_map(np.full((1000, 1000), np.nan), 1.0))


class GoalPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        super().__init__(
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=MarkerArray,
            callback_group=callback_group,
        )

    def publish(self, robot: "agent.Agent", q_init: t.Any, q_goal: t.Any):
        if q_goal is None:
            msg = MarkerArray()
        else:
            polygon_at_goal_pose = affinity.translate(
                robot.polygon, q_goal[0] - q_init[0], q_goal[1] - q_init[1]
            )

            color = ColorRGBA(
                **colors.hex_to_rgba(
                    colors.darken(Style.from_string(robot.agent_style.shape).fill)
                )
            )
            msg = MarkerArray(
                markers=[
                    polygon_to_line_strip(
                        polygon=polygon_at_goal_pose,
                        p_id=0,
                        frame_id=cfg.main_frame_id,
                        color=color,
                        z_index=cfg.goal_z_index,
                        line_width=robot.min_inflation_radius / 4,
                    )
                ]
            )
        super().publish(msg)

    def reset(self):
        super().reset(make_delete_all_marker(cfg.main_frame_id))


class PosesPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        super().__init__(
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=PoseArray,
            callback_group=callback_group,
        )

    def publish(self, poses: t.Iterable[Pose2D]):
        super().publish(poses_to_poses_array(poses, self.get_timestamp()))

    def reset(self):
        super().reset(PoseArray(header=init_header(self.get_timestamp()), poses=[]))


class PlanPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        super().__init__(
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=MarkerArray,
            callback_group=callback_group,
        )

    def publish(self, map: t.Any, plan: t.Any, robot: t.Any):
        super().publish(
            plan_to_markerarray(
                map=map,
                plan=plan,
                robot=robot,
                frame_id=cfg.main_frame_id,
                stamp=self.get_timestamp(),
            )
        )

    def reset(self):
        super().reset(make_delete_all_marker(cfg.main_frame_id))


class RRTPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        super().__init__(
            msg_type=MarkerArray,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            callback_group=callback_group,
        )

    def _create_marker(
        self,
        marker_type: int,
        marker_id: int,
        scale: float,
        r: float,
        g: float,
        b: float,
        a: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = cfg.main_frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        return marker

    def publish(self, rrt_nodes: t.List[RRTNode]):
        # Create MarkerArray
        marker_array = MarkerArray()

        # # Create points marker for nodes
        # points_marker = self._create_marker(
        #     marker_type=Marker.POINTS,
        #     marker_id=0,
        #     scale=0.02,  # Size of points
        #     r=1.0,
        #     g=1.0,
        #     b=1.0,
        #     a=1.0,  # Green color
        # )
        # for node in rrt_nodes:
        #     point = Point()
        #     point.x = float(node.pose[0])
        #     point.y = float(node.pose[1])
        #     point.z = float(0)
        #     points_marker.points.append(point)

        # Create lines marker for edges
        lines_marker = self._create_marker(
            marker_type=Marker.LINE_LIST,
            marker_id=1,
            scale=0.003,  # Line thickness
            r=0.0,
            g=0.0,
            b=0.0,
            a=1.0,  # Blue color
        )
        for node in rrt_nodes:
            if node.parent is not None:
                # Add start point (parent)
                start = Point()
                start.x = float(node.parent.pose[0])
                start.y = float(node.parent.pose[1])
                start.z = float(0)
                lines_marker.points.append(start)  # type: ignore
                # Add end point (current node)
                end = Point()
                end.x = float(node.pose[0])
                end.y = float(node.pose[1])
                end.z = float(0)
                lines_marker.points.append(end)  # type: ignore

        # Add both markers to the MarkerArray
        marker_array.markers.append(lines_marker)  # type: ignore

        super().publish(marker_array)

    def reset(self):
        super().reset(make_delete_all_marker(cfg.main_frame_id))
