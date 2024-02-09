# pyright: reportUnusedImport=false

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
import namosim.world.world as world
from namosim.agents import agent
from namosim.config import DEACTIVATE_RVIZ
from namosim.data_models import UID, PoseModel
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
    BinaryInflatedOccupancyGrid,
    BinaryOccupancyGrid,
)
from namosim.world.entity import Entity
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


class MyNode(Node):
    def __init__(self, node_name: str):
        # Shutdown the ROS Context if it is already running.
        # This is necessary when running multiple unit tests since each may create their own context.
        if ok():
            rclpy.shutdown()
        rclpy.init(args=None)
        super().__init__(node_name=node_name)

    def get_timestamp(self) -> Time:
        return self.get_clock().now().to_msg()

    def log_warn(self, text: str):
        self.get_logger().warn(text)

    def get_transform_broadcaster(self):
        return StaticTransformBroadcaster(self)

    @staticmethod
    def get_nodes_names():
        cmd_str = "ros2 node list"
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        nodes_names = [name for name in result.stdout.split("\n") if name]
        return nodes_names

    def create_publisher(
        self,
        msg_type: type,
        topic: str,
        qos_profile: t.Union[QoSProfile, int] = cfg.default_queue_size,
        *,
        callback_group: t.Optional[CallbackGroup] = None,
        event_callbacks: t.Optional[PublisherEventCallbacks] = None,
        qos_overriding_options: t.Optional[QoSOverridingOptions] = None,
        publisher_class: t.Type[Publisher] = Publisher,
    ) -> Publisher:
        return super(MyNode, self).create_publisher(
            msg_type=msg_type,
            topic=topic,
            qos_profile=qos_profile,
            callback_group=callback_group,
            event_callbacks=event_callbacks,
            qos_overriding_options=qos_overriding_options,
            publisher_class=publisher_class,
        )


class RosObserver:
    def __init__(
        self,
        msg_type: type,
        node: MyNode,
        topic: str,
        is_active: bool = True,
        rate: int = cfg.rate,
    ):
        self.node = node
        self.topic = topic
        self._publisher = node.create_publisher(msg_type, topic)
        self.is_active = is_active
        self._rate = rate

        self._duration = 1.0 / self.rate
        self._last_time = time.time()
        self.get_subscription_count = self._publisher.get_subscription_count

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
        reset_msg: Marker
        | MarkerArray
        | PoseArray
        | GridMap
        | OccupancyGrid
        | None = None,
    ):
        if not DEACTIVATE_RVIZ and reset_msg is not None:
            self._publisher.publish(reset_msg)


class WorldObserver(RosObserver):
    def __init__(
        self, node: MyNode, topic: str, is_active: bool = True, rate: int = cfg.rate
    ):
        RosObserver.__init__(
            self,
            node=node,
            topic=topic,
            is_active=is_active,
            rate=rate,
            msg_type=MarkerArray,
        )
        self.prev_sim_world_draw_data = None

    def _convert(self, **kwargs: t.Any):
        world, robot_uid = kwargs["world"], kwargs["robot_uid"]

        current_world_draw_data = {
            entity.uid: {
                "polygon": entity.polygon,
                "type": "robot" if isinstance(entity, agent.Agent) else entity.type_,
                "pose": entity.pose,
            }
            for entity in world.entities.values()
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
        return self.world_to_marker_array(world, robot_uid, entities_to_ignore)

    def world_to_marker_array(
        self,
        world: "world.World",
        robot_uid: UID | None = None,
        entities_to_ignore: t.Set[UID] | None = None,
    ):
        if entities_to_ignore is None:
            entities_to_ignore = set()
        marker_array = MarkerArray()
        markers = []
        for entity in world.entities.values():
            if entity.uid not in entities_to_ignore:
                entity_color = ColorRGBA(**colors.hex_to_rgba(entity.style.fill))
                if isinstance(entity, agent.Agent):
                    namespace = "/robot"
                elif isinstance(entity, Obstacle):
                    namespace = "obstacles"
                else:
                    raise ValueError(
                        "Only Robot and Obstacle can be displayed in Rviz, current entity is: {}".format(
                            entity
                        )
                    )

                markers = markers + self.entity_to_markers(
                    entity=entity,
                    namespace=namespace,
                    p_id=entity.uid,
                    frame_id=cfg.main_frame_id,
                    color=entity_color,
                    border_color=entity_color,
                    text_color_filling=colors.text_color_on_filling,
                    text_color_empty=colors.text_color_on_empty,
                    z_index=cfg.entities_z_index,
                    text_height=entity.circumscribed_radius / 5,
                    add_border=False,
                    add_text=False,
                )
        marker_array.markers = markers
        return marker_array

    def entity_to_markers(
        self,
        *,
        entity: Entity,
        namespace: str,
        p_id: UID,
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
        add_uid: bool = True,
        add_name: bool = True,
    ) -> t.List[Marker]:
        markers = []
        if add_filling:
            markers.append(
                polygon_to_triangle_list(
                    entity.polygon,
                    namespace + "/polygon",
                    p_id,
                    frame_id,
                    color,
                    z_index,
                    self.node.get_timestamp(),
                )
            )
        if add_border:
            markers.append(
                polygon_to_line_strip(
                    entity.polygon,
                    namespace + "/border",
                    p_id,
                    frame_id,
                    border_color,
                    z_index,
                    entity.circumscribed_radius / 4,
                    self.node.get_timestamp(),
                )
            )
        if add_text:
            string = (("UID: " + str(entity.uid) + "\n") if add_uid else "") + (
                ("Name: " + entity.name + "\n") if add_name else ""
            )
            text_coordinates = entity.polygon.centroid.coords[0]
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
                    self.node.get_timestamp(),
                )
            )
        return markers

    def reset(self, reset_msg: Marker | None = None):
        RosObserver.reset(self, make_delete_all_marker(cfg.main_frame_id))


class CostmapObserver(RosObserver):
    def __init__(
        self, node: MyNode, topic: str, is_active: bool = True, rate: int = cfg.rate
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

    def world_to_costmap(self, world: "world.World", robot_uid: UID | None = None):
        polygons = {
            uid: entity.polygon
            for uid, entity in world.entities.items()
            if not isinstance(entity, agent.Agent)
        }
        if robot_uid:
            robot_max_inflation_radius = utils.get_circumscribed_radius(
                world.entities[robot_uid].polygon
            )
            grid = BinaryInflatedOccupancyGrid(
                polygons,
                world.discretization_data.res,
                robot_max_inflation_radius,
                neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
            )
        else:
            grid = BinaryOccupancyGrid(
                polygons,
                world.discretization_data.res,
                neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
            )

        costmap = OccupancyGrid(header=init_header(self.node.get_timestamp()))
        costmap.info.map_load_time = costmap.header.stamp
        costmap.info.resolution = grid.res
        costmap.info.width = grid.d_width
        costmap.info.height = grid.d_height
        costmap.info.origin.position.x = grid.grid_pose[0]
        costmap.info.origin.position.y = grid.grid_pose[1]
        costmap.info.origin.position.z = -0.1
        costmap.data = (
            np.fliplr(np.rot90(grid.grid, 3)).flatten().astype(np.int8).tolist()
        )

        return costmap

    def reset(self, reset_msg: OccupancyGrid | None = None):
        RosObserver.reset(
            self, OccupancyGrid(info=MapMetaData(width=1, height=1), data=[0])
        )


class GridMapObserver(RosObserver):
    def __init__(
        self,
        node: MyNode,
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

        grid_map = costmap_to_grid_map(costmap, res, stamp=self.node.get_timestamp())
        return grid_map

    def reset(self):
        RosObserver.reset(self, costmap_to_grid_map(np.full((1000, 1000), np.nan), 1.0))


class CombinedCostGridMapObserver(GridMapObserver):
    def __init__(
        self,
        node: MyNode,
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
        H = combined_costmap.shape[0]
        M = np.ptp(combined_costmap)
        m = np.min(combined_costmap)
        cc = (combined_costmap - m) / M
        cc = cc * H - (4 * H)

        grid_map = costmap_to_grid_map(
            cc,
            inflated_grid_by_obstacle.res,
            frame_id=cfg.combined_gridmap_frame_id,
            stamp=self.node.get_timestamp(),
        )
        return grid_map


class GoalObserver(RosObserver):
    def __init__(
        self,
        node: MyNode,
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
        color = ColorRGBA(**colors.hex_to_rgba(colors.darken(robot.style.fill)))
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
        node: MyNode,
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
        return poses_to_poses_array(poses, self.node.get_timestamp())

    def reset(self):
        RosObserver.reset(
            self, PoseArray(header=init_header(self.node.get_timestamp()), poses=[])
        )


class PlanObserver(RosObserver):
    def __init__(
        self,
        node: MyNode,
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
        plan, robot = kwargs["plan"], kwargs["robot"]
        return plan_to_markerarray(
            plan, robot, cfg.main_frame_id, stamp=self.node.get_timestamp()
        )

    def reset(self, reset_msg: t.Optional[t.Any] = None):
        RosObserver.reset(self, make_delete_all_marker(cfg.main_frame_id))
