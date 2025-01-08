# pyright: reportUnusedImport=false

import copy
import subprocess
import time
import typing as t

import numpy as np
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
from rclpy.qos import QoSProfile
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
from namosim.data_models import PoseModel
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


def init_header(stamp: Time = Time()):
    return Header(stamp=stamp, frame_id=cfg.main_frame_id)


def poses_to_poses_array(poses: t.List[PoseModel], stamp: Time = Time()):
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
    qos_profile: t.Union[QoSProfile, int] = cfg.default_queue_size,
    *,
    callback_group: t.Optional[CallbackGroup] = None,
    event_callbacks: t.Optional[PublisherEventCallbacks] = None,
    qos_overriding_options: t.Optional[QoSOverridingOptions] = None,
    publisher_class: t.Type[Publisher] = Publisher,
) -> Publisher:
    return node.create_publisher(
        msg_type=msg_type,
        topic=topic,
        qos_profile=qos_profile,
        callback_group=callback_group,
        event_callbacks=event_callbacks,
        qos_overriding_options=qos_overriding_options,
        publisher_class=publisher_class,
    )


class DefaultRosPublisherNode(Node):
    def __init__(self, node_name: str):
        # Shutdown the ROS Context if it is already running.
        # This is necessary when running multiple unit tests since each may create their own context.
        if ok():
            rclpy.shutdown()
        rclpy.init(args=None)
        super().__init__(node_name=node_name, parameter_overrides=[])


class RosObserver:
    def __init__(
        self,
        msg_type: type,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
    ):
        self.node = node
        self.topic = topic
        self._publisher = create_publisher(node, msg_type, topic)
        self.is_active = is_active
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

    def update(self, **kwargs: t.Any):
        if not DEACTIVATE_RVIZ and self.is_active:
            connections = self.get_subscription_count()
            if connections > 0:
                elapsed_time = time.time() - self._last_time
                time_to_wait = self._duration - elapsed_time
                if time_to_wait > 0.0:
                    time.sleep(time_to_wait)
                self._publisher.publish(self._convert(**kwargs))
                self._last_time = time.time()

    def _convert(self, **kwargs: t.Any) -> t.Any:
        """
        Receives arguments related to the world and simulation and converts them in to RViz messages
        for visualization.
        """
        raise NotImplementedError

    def reset(
        self,
        reset_msg: (
            Marker | MarkerArray | PoseArray | GridMap | OccupancyGrid | None
        ) = None,
    ):
        if not DEACTIVATE_RVIZ and reset_msg is not None:
            self._publisher.publish(reset_msg)


class BasePublisher:
    def __init__(
        self,
        msg_type: type,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
    ):
        self.node = node
        self.topic = topic
        self._publisher = create_publisher(node, msg_type, topic)
        self.is_active = is_active
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
        self, node: Node, topic: str, is_active: bool = True, rate: int = cfg.rate
    ):
        super().__init__(
            msg_type=MarkerArray, node=node, topic=topic, is_active=is_active, rate=rate
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
                namespace="obstacle/polygon",
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


class WorldPublisher(BasePublisher):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
    ):
        super().__init__(
            msg_type=MarkerArray, node=node, topic=topic, is_active=is_active, rate=rate
        )
        self.prev_sim_world_draw_data = None

    def update(self, world: "world.World"):
        current_world_draw_data = {
            entity.uid: {
                "polygon": entity.polygon,
                "type": "robot" if isinstance(entity, agent.Agent) else entity.type_,
                "pose": entity.pose,
            }
            for entity in world.dynamic_entities.values()
        }
        entities_to_ignore = {
            entity_uid
            for entity_uid, drawable_data in current_world_draw_data.items()
            if (
                self.prev_sim_world_draw_data is not None
                and entity_uid in self.prev_sim_world_draw_data
                and drawable_data["polygon"]
                == self.prev_sim_world_draw_data[entity_uid]["polygon"]
                and drawable_data["type"]
                == self.prev_sim_world_draw_data[entity_uid]["type"]
                and drawable_data["pose"]
                == self.prev_sim_world_draw_data[entity_uid]["pose"]
            )
        }
        self.prev_sim_world_draw_data = current_world_draw_data
        msg = self.world_to_marker_array(world, entities_to_ignore)
        self.publish(msg)
        self.publish_obstacles(world)

    def agent_to_marker_array(self, agent: "agent.Agent"):
        body_color = ColorRGBA(
            **colors.hex_to_rgba(Style.from_string(agent.agent_style.shape).fill)
        )
        orientation_fill = Style.from_string(agent.agent_style.orientation).fill
        orientation_fill = orientation_fill if orientation_fill != "none" else "#FFFFFF"
        orientation_color = ColorRGBA(**colors.hex_to_rgba(orientation_fill))
        shape_marker = polygon_to_triangle_list(
            polygon=agent.polygon,
            namespace=f"/robot/polygon",
            p_id=utils.hash_to_32_bit_int(agent.uid),
            frame_id=cfg.main_frame_id,
            color=body_color,
            z_index=cfg.entities_z_index,
            stamp=self.get_timestamp(),
        )
        orientation_marker = polygon_to_triangle_list(
            polygon=agent.get_orientation_polygon(),
            namespace=f"/robot/orientation",
            p_id=utils.hash_to_32_bit_int(agent.uid + "orientation"),
            frame_id=cfg.main_frame_id,
            color=orientation_color,
            z_index=cfg.entities_z_index + 1,
            stamp=self.get_timestamp(),
        )
        return [shape_marker, orientation_marker]

    def world_to_marker_array(
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
                    markers += self.agent_to_marker_array(entity)
                elif isinstance(entity, Obstacle):
                    continue
                else:
                    raise ValueError(
                        "Only Robot and Obstacle can be displayed in Rviz, current entity is: {}".format(
                            entity
                        )
                    )
        marker_array.markers = markers
        return marker_array

    def publish_obstacles(self, world: "world.World"):
        markers = self.get_obstacle_markers(world)
        msg = MarkerArray()
        msg.markers = markers
        self.publish(msg)

    def get_obstacle_markers(self, world: "world.World"):
        movables = world.get_all_obstacles()
        markers = []
        for entity in movables:
            if not isinstance(entity, Obstacle):
                continue
            markers += self.obstacle_to_markers(
                entity=entity,
                p_id=utils.hash_to_32_bit_int(entity.uid),
                z_index=cfg.entities_z_index,
            )
        return markers

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
                namespace="obstacle/polygon",
                p_id=p_id,
                frame_id=cfg.main_frame_id,
                color=ColorRGBA(**colors.hex_to_rgba(entity.style.fill)),
                z_index=z_index,
                stamp=self.get_timestamp(),
            )
        )
        return markers

    def entity_to_markers(
        self,
        *,
        entity: Entity,
        namespace: str,
        p_id: int,
        frame_id: str,
        color: ColorRGBA,
        border_color: ColorRGBA,
        text_color_filling: ColorRGBA,
        text_color_empty: ColorRGBA,
        z_index: float,
        text_height: float,
        add_filling: bool = True,
        add_border: bool = True,
        add_text: bool = True,
    ) -> t.List[Marker]:
        polygon = entity.polygon
        markers = []
        if add_filling:
            markers.append(
                polygon_to_triangle_list(
                    polygon,
                    namespace + "/polygon",
                    p_id,
                    frame_id,
                    color,
                    z_index,
                    self.get_timestamp(),
                )
            )
        if add_border:
            markers.append(
                polygon_to_line_strip(
                    polygon,
                    namespace + "/border",
                    p_id,
                    frame_id,
                    border_color,
                    z_index,
                    entity.circumscribed_radius / 4,
                    self.get_timestamp(),
                )
            )
        if add_text:
            string = "Name: " + entity.uid + "\n"
            text_coordinates = polygon.centroid.coords[0]
            markers.append(
                string_to_text(
                    string,
                    text_coordinates,
                    namespace + "/text",
                    p_id,
                    frame_id,
                    text_color_filling if add_filling else text_color_empty,
                    z_index,
                    text_height,
                    self.get_timestamp(),
                )
            )
        return markers

    def reset(self):
        if not DEACTIVATE_RVIZ:
            self._publisher.publish(make_delete_all_marker(cfg.main_frame_id))


class ManipSearchPublisher(BasePublisher):
    def __init__(
        self, node: Node, topic: str, is_active: bool = True, rate: int = cfg.rate
    ):
        super().__init__(
            msg_type=MarkerArray, node=node, topic=topic, is_active=is_active, rate=rate
        )

    def reset(self):
        if not DEACTIVATE_RVIZ:
            self._publisher.publish(make_delete_all_marker(cfg.main_frame_id))


class CostmapObserver(RosObserver):
    def __init__(
        self, node: Node, topic: str, is_active: bool = True, rate: int = cfg.rate
    ):
        RosObserver.__init__(
            self,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=OccupancyGrid,
        )

    def _convert(self, **kwargs: t.Any):
        world, robot_uid = kwargs["world"], kwargs["robot_uid"]
        return self.world_to_costmap(world, robot_uid)

    def world_to_costmap(self, world: "world.World", robot_uid: str | None = None):
        if robot_uid:
            robot_max_inflation_radius = utils.get_circumscribed_radius(
                world.dynamic_entities[robot_uid].polygon
            )
            grid = copy.deepcopy(world.map).inflate_map(robot_max_inflation_radius)
        else:
            grid = copy.deepcopy(world.map)

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

    def reset(self, reset_msg: OccupancyGrid | None = None):
        RosObserver.reset(
            self, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0])
        )


class GridMapObserver(RosObserver):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        msg_type: type = GridMap,
    ):
        RosObserver.__init__(
            self,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=msg_type,
        )

    def _convert(self, **kwargs: t.Any):
        costmap, res = kwargs["costmap"], kwargs["res"]
        grid_map = costmap_to_grid_map(costmap, res, stamp=self.get_timestamp())
        return grid_map

    def reset(self):
        RosObserver.reset(self, costmap_to_grid_map(np.full((1000, 1000), np.nan), 1.0))


class CombinedCostGridMapObserver(GridMapObserver):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        msg_type: type = GridMap,
    ):
        RosObserver.__init__(
            self,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=msg_type,
        )

    def _convert(self, **kwargs: t.Any):
        sorted_cell_to_combined_cost, inflated_grid_by_obstacle = (
            kwargs["sorted_cell_to_combined_cost"],
            kwargs["inflated_grid_by_obstacle"],
        )
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
        return grid_map


class GoalObserver(RosObserver):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        msg_type: type = MarkerArray,
    ):
        RosObserver.__init__(
            self,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=msg_type,
        )

    def _convert(self, **kwargs: t.Any):
        robot: "agent.Agent" = kwargs["entity"]
        q_init, q_goal = kwargs["q_init"], kwargs["q_goal"]

        if q_goal is None:
            return MarkerArray()

        polygon_at_goal_pose = affinity.translate(
            robot.polygon, q_goal[0] - q_init[0], q_goal[1] - q_init[1]
        )

        color = ColorRGBA(
            **colors.hex_to_rgba(
                colors.darken(Style.from_string(robot.agent_style.shape).fill)
            )
        )
        marker_array = MarkerArray(
            markers=[
                polygon_to_line_strip(
                    polygon_at_goal_pose,
                    "/polygon",
                    0,
                    cfg.main_frame_id,
                    color,
                    cfg.goal_z_index,
                    line_width=robot.min_inflation_radius / 4,
                )
            ]
        )
        return marker_array

    def reset(self):
        super().reset(make_delete_all_marker(cfg.main_frame_id))


class PosesObserver(RosObserver):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        msg_type: type = PoseArray,
    ):
        RosObserver.__init__(
            self,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=msg_type,
        )

    def _convert(self, **kwargs: t.Any):
        poses = kwargs["poses"]
        return poses_to_poses_array(poses, self.get_timestamp())

    def reset(self):
        RosObserver.reset(
            self, PoseArray(header=init_header(self.get_timestamp()), poses=[])
        )


class PlanObserver(RosObserver):
    def __init__(
        self,
        node: Node,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
        msg_type: type = MarkerArray,
    ):
        RosObserver.__init__(
            self,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=msg_type,
        )

    def _convert(self, **kwargs: t.Any):
        map, plan, robot = kwargs["map"], kwargs["plan"], kwargs["robot"]
        return plan_to_markerarray(
            map=map,
            plan=plan,
            robot=robot,
            frame_id=cfg.main_frame_id,
            stamp=self.get_timestamp(),
        )

    def reset(self, reset_msg: t.Optional[t.Any] = None):
        RosObserver.reset(self, make_delete_all_marker(cfg.main_frame_id))


class PlanPublisher:
    """Publishes robot navigation plans as `MarkerArray`s to /agent_id/plan"""

    def __init__(self, node: Node, agent_id: str):
        if DEACTIVATE_RVIZ:
            return
        self.node = node
        self._publisher = create_publisher(node, MarkerArray, f"/{agent_id}/plan")

    def publish(
        self,
        plan: "nav_plan.Plan",
        robot: "agent.Agent",
        map: BinaryOccupancyGrid,
        scale: float = 1.0,
    ):
        if DEACTIVATE_RVIZ:
            return

        if self._publisher.get_subscription_count() > 0:
            markers = plan_to_markerarray(
                plan=plan,
                robot=robot,
                map=map,
                frame_id=cfg.main_frame_id,
                stamp=self.node.get_clock().now().to_msg(),
            )

            if scale != 1.0:
                for marker in markers.markers:
                    marker.scale.x *= scale
                    marker.scale.y *= scale
                    marker.scale.z *= scale

            self._publisher.publish(markers)

    def reset(self):
        if DEACTIVATE_RVIZ:
            return
        self._publisher.publish(make_delete_all_marker(cfg.main_frame_id))
