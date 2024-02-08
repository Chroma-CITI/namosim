# pyright: reportUnboundVariable=false

import copy
import math
import typing as t
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from shapely import GeometryCollection, Polygon

import namosim.display.ros_publisher_config as cfg
import namosim.navigation.navigation_plan as navigation_plan
import namosim.world.world as world
from namosim.agents import agent
from namosim.config import DEACTIVATE_RVIZ
from namosim.data_models import UID, GridCellModel, PoseModel
from namosim.display import colors
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryInflatedOccupancyGrid

if not DEACTIVATE_RVIZ:
    import namosim.display.conversions as conversions
    import namosim.display.ros_nodes as ros_nodes


class RosPublisher:  # noqa: F821
    def __init__(
        self,
        node_name: str,
        agent_names: t.List[str],
        prefix_topics_with_node_name: bool = False,
    ):
        if DEACTIVATE_RVIZ:
            return

        # HACK: Must necessarily be invoked in the init method of this singleton and not at module-level (rclpy bug...)
        self.ros_node = ros_nodes.MyNode(
            node_name=self.create_valid_node_name(node_name)
        )
        self.prefix = (
            "" if not prefix_topics_with_node_name else self.ros_node.get_name()
        )

        self.my_publishers: t.Dict[str, ros_nodes.Publisher] = {}  # DEPRECATED
        self.observers: t.Dict[str, ros_nodes.RosObserver] = {}

        # Add simulation-specific publishers
        self.sim_knowledge_topic = self.prefix + "/simulation" + cfg.sim_knowledge_topic
        self.observers[self.sim_knowledge_topic] = ros_nodes.WorldObserver(
            self.ros_node, self.sim_knowledge_topic
        )
        self.sim_costmap_topic = self.prefix + "/simulation" + cfg.sim_costmap_topic
        self.observers[self.sim_costmap_topic] = ros_nodes.CostmapObserver(
            self.ros_node, self.sim_costmap_topic
        )
        self.sim_gridmap_topic = (
            self.prefix + "/simulation" + cfg.test_social_gridmap_topic
        )
        self.observers[self.sim_gridmap_topic] = ros_nodes.GridMapObserver(
            self.ros_node, self.sim_gridmap_topic
        )
        self.sim_cc_topic = (
            self.prefix + "/simulation" + cfg.test_connected_components_topic
        )
        self.observers[self.sim_cc_topic] = ros_nodes.GridMapObserver(
            self.ros_node, self.sim_cc_topic
        )

        self.my_publishers[
            "/simulation" + cfg.sim_latest_message_topic
        ] = self.ros_node.create_publisher(
            ros_nodes.MarkerArray, "/simulation" + cfg.sim_latest_message_topic
        )

        self.agents_names = agent_names

        # Add robot-specific publishers for each robot namespace
        for agent_name in self.agents_names:
            ns = self.prefix + "/" + agent_name
            self.observers[ns + cfg.robot_knowledge_topic] = ros_nodes.WorldObserver(
                self.ros_node, ns + cfg.robot_knowledge_topic
            )
            self.observers[ns + cfg.robot_costmap_topic] = ros_nodes.CostmapObserver(
                self.ros_node, ns + cfg.robot_costmap_topic
            )
            self.observers[ns + cfg.robot_sim_world_topic] = ros_nodes.WorldObserver(
                self.ros_node, ns + cfg.robot_sim_world_topic
            )
            self.observers[
                ns + cfg.robot_sim_costmap_topic
            ] = ros_nodes.CostmapObserver(
                self.ros_node, ns + cfg.robot_sim_costmap_topic
            )
            self.observers[
                ns + cfg.test_connected_components_topic
            ] = ros_nodes.GridMapObserver(
                self.ros_node, ns + cfg.test_connected_components_topic
            )
            self.observers[
                ns + cfg.test_combined_gridmap_topic
            ] = ros_nodes.CombinedCostGridMapObserver(
                self.ros_node, ns + cfg.test_combined_gridmap_topic
            )
            self.observers[
                ns + cfg.test_social_gridmap_topic
            ] = ros_nodes.GridMapObserver(
                self.ros_node, ns + cfg.test_social_gridmap_topic
            )
            self.observers[ns + cfg.robot_goal_topic] = ros_nodes.GoalObserver(
                self.ros_node, ns + cfg.robot_goal_topic
            )
            self.observers[ns + cfg.obs_manip_poses_topic] = ros_nodes.PosesObserver(
                self.ros_node, ns + cfg.obs_manip_poses_topic
            )
            self.observers[ns + cfg.plan_topic] = ros_nodes.PlanObserver(
                self.ros_node, ns + cfg.plan_topic
            )
            # TODO: Refactor the following publisher with the Observer pattern
            self.my_publishers[
                ns + cfg.conflicts_check_topic
            ] = self.ros_node.create_publisher(
                ros_nodes.MarkerArray, ns + cfg.conflicts_check_topic
            )
            self.my_publishers[
                ns + cfg.conflict_horizon_topic
            ] = self.ros_node.create_publisher(
                ros_nodes.MarkerArray, ns + cfg.conflict_horizon_topic
            )
            self.my_publishers[
                ns + cfg.swept_area_topic
            ] = self.ros_node.create_publisher(
                ros_nodes.MarkerArray, ns + cfg.swept_area_topic
            )
            # TODO: Last publisher to refactor, as it requires separating it into smaller meaningful units
            self.my_publishers[
                ns + cfg.robot_sim_topic
            ] = self.ros_node.create_publisher(
                ros_nodes.MarkerArray, ns + cfg.robot_sim_topic, cfg.default_queue_size
            )

        # HACK: Necessary because ROS1 pub/sub system is not really reliable : wait a second for subscribers to listen
        ros_nodes.time.sleep(cfg.hack_duration_wait)

        # Setup Static Transform for grid map (Hack so that it is properly placed in view)
        broadcaster = self.ros_node.get_transform_broadcaster()

        for frame_id, z_index in cfg.gridmap_frame_ids_to_z_indexes.items():
            transform = ros_nodes.TransformStamped(
                header=ros_nodes.Header(
                    stamp=self.ros_node.get_timestamp(), frame_id=cfg.main_frame_id
                ),
                child_frame_id=frame_id,
                transform=ros_nodes.Transform(
                    translation=ros_nodes.Vector3(z=z_index),
                    rotation=ros_nodes.Quaternion(x=0.0, y=0.0, z=1.0, w=0.0),
                ),
            )
            broadcaster.sendTransform(transform)
            ros_nodes.time.sleep(0.5)  # Hack so that transform is properly sent...

        # Initialize caches for each top level namespace
        self.prev_sim_world_draw_data = {}

        self.namespaces_caches = {}
        for ns in self.agents_names:
            self.namespaces_caches[ns] = ros_nodes.NamespaceCache()

    @staticmethod
    def create_valid_node_name(root_name: str):
        nodes_names = ros_nodes.MyNode.get_nodes_names()
        node_name = (
            root_name
            if (root_name and not root_name[0].isdigit())
            else ("node_" + root_name)
        )
        i = 0
        while node_name in nodes_names:
            node_name = root_name + "_" + str(i)
            i += 1
        return node_name

    def publish(self, topic: str, msg: t.Any):
        if DEACTIVATE_RVIZ:
            return
        publisher = self.my_publishers[topic]
        connections = publisher.get_subscription_count()
        if connections > 0:
            ros_nodes.time.sleep(1.0 / cfg.rate)
            publisher.publish(msg)

    def is_activated(self, topic: str = ""):
        if DEACTIVATE_RVIZ or (topic and topic not in self.my_publishers):
            return False
        if not DEACTIVATE_RVIZ and not topic:
            return True
        return topic in self.my_publishers

    # region SIM WORLD
    def publish_sim_world(self, world: "world.World", robot_uid: UID | None = None):
        if DEACTIVATE_RVIZ:
            return
        self.observers[self.sim_knowledge_topic].update(
            world=world, robot_uid=robot_uid
        )
        self.observers[self.sim_costmap_topic].update(world=world, robot_uid=robot_uid)
        world.discretization_data.d_height

    def cleanup_sim_world(self):
        if not DEACTIVATE_RVIZ:
            self.observers[self.sim_knowledge_topic].reset()
            self.observers[self.sim_costmap_topic].reset()

    # endregion

    # region ROBOT WORLD
    def publish_robot_world(self, world: "world.World", robot_uid: UID):
        if DEACTIVATE_RVIZ:
            return
        world_topic = (
            self.prefix
            + "/"
            + world.entities[robot_uid].name
            + cfg.robot_knowledge_topic
        )
        self.observers[world_topic].update(world=world, robot_uid=robot_uid)
        costmap_topic = (
            self.prefix + "/" + world.entities[robot_uid].name + cfg.robot_costmap_topic
        )
        self.observers[costmap_topic].update(world=world, robot_uid=robot_uid)

    def cleanup_robot_world(self, ns: str = ""):
        if DEACTIVATE_RVIZ:
            return

        world_topic = self.prefix + "/" + ns + cfg.robot_knowledge_topic
        if world_topic in self.observers:
            self.observers[world_topic].reset()
        costmap_topic = self.prefix + "/" + ns + cfg.robot_costmap_topic
        if costmap_topic in self.observers:
            self.observers[costmap_topic].reset()

    # endregion

    # region ROBOT SIM
    def publish_robot_sim_world(self, world: "world.World", robot_uid: UID):
        if DEACTIVATE_RVIZ:
            return
        topic = (
            self.prefix
            + "/"
            + world.entities[robot_uid].name
            + cfg.robot_sim_world_topic
        )
        self.observers[topic].update(world=world, robot_uid=robot_uid)

    def cleanup_robot_sim_world(self, ns: str = ""):
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + "/" + ns + cfg.robot_sim_world_topic
        self.observers[topic].reset()

    def publish_robot_sim_costmap(self, world: "world.World", robot_uid: UID):
        if DEACTIVATE_RVIZ:
            return
        topic = (
            self.prefix
            + "/"
            + world.entities[robot_uid].name
            + cfg.robot_sim_costmap_topic
        )
        self.observers[topic].update(world=world, robot_uid=robot_uid)

    def cleanup_robot_sim_costmap(self, world: "world.World", robot_uid: UID):
        if DEACTIVATE_RVIZ:
            return
        topic = (
            self.prefix
            + "/"
            + world.entities[robot_uid].name
            + cfg.robot_sim_costmap_topic
        )
        self.observers[topic].reset()

    # endregion

    # region GRID MAP
    def publish_social_grid_map(
        self, costmap: npt.NDArray[np.float_], res: float, ns: str = ""
    ):
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (
            cfg.test_social_gridmap_topic
            if not ns
            else "/" + ns + cfg.test_social_gridmap_topic
        )

        # re-scale and shift the costmap so it displays nicely below the 2D environment in RVIZ
        costmap = np.copy(costmap)
        costmap[costmap == -1.0] = 0.0
        H = costmap.shape[0]
        M = np.ptp(costmap)
        m = np.min(costmap)
        costmap = (costmap - m) / M
        costmap = costmap * H - (2 * H)

        self.observers[topic].update(costmap=costmap, res=res)

    def cleanup_social_grid_map(self, ns: str = ""):
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (
            cfg.test_social_gridmap_topic
            if not ns
            else "/" + ns + cfg.test_social_gridmap_topic
        )
        self.observers[topic].reset()

    def publish_combined_costmap(
        self,
        sorted_cell_to_combined_cost: OrderedDict[GridCellModel, float],
        inflated_grid_by_obstacle: BinaryInflatedOccupancyGrid,
        ns: str = "",
    ):
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (
            cfg.test_combined_gridmap_topic
            if not ns
            else "/" + ns + cfg.test_combined_gridmap_topic
        )

        self.observers[topic].update(
            sorted_cell_to_combined_cost=sorted_cell_to_combined_cost,
            inflated_grid_by_obstacle=inflated_grid_by_obstacle,
        )

    def cleanup_combined_costmap(self, ns: str = ""):
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (
            cfg.test_combined_gridmap_topic
            if not ns
            else "/" + ns + cfg.test_combined_gridmap_topic
        )
        self.observers[topic].reset()

    # endregion

    # region CONNECTED COMPONENTS GRID
    def publish_connected_components_grid(
        self, costmap: npt.NDArray[np.float_], res: float, ns: str = ""
    ):
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (
            cfg.test_connected_components_topic
            if not ns
            else "/" + ns + cfg.test_connected_components_topic
        )

        # re-scale and shift the costmap so it displays nicely below the 2D environment in RVIZ
        costmap = np.copy(costmap)
        costmap[costmap == -1.0] = 0.0
        H = costmap.shape[0]
        M = np.ptp(costmap)
        m = np.min(costmap)
        costmap = (costmap - m) / M
        costmap = costmap * H - (6 * H)

        self.observers[topic].update(costmap=costmap, res=res)

    def cleanup_connected_components_grid(self, ns: str = ""):
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (
            cfg.test_connected_components_topic
            if not ns
            else "/" + ns + cfg.test_connected_components_topic
        )
        self.observers[topic].reset()

    # endregion

    # region STILMAN 2005 RCH DATA
    def publish_rch_data(
        self,
        current: t.Any,
        came_from: t.Any,
        neighbors: t.Any,
        traversed_obstacles_ids: t.Any,
        res: float,
        grid_pose: PoseModel,
        ns: str = "",
    ):
        full_topic = cfg.robot_sim_topic if not ns else "/" + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = ros_nodes.MarkerArray(markers=[])

            # Publish current cell
            current_marker = self._grid_cells_to_cube_list_markers(
                [current.cell],
                res,
                grid_pose,
                z_index=0.9,
                color=colors.flashy_purple,
                ns="/rch_current_cell",
            )
            marker_array.markers.append(current_marker)  # type: ignore

            # Publish neighbors
            neighbors_marker = self._grid_cells_to_cube_list_markers(
                [neighbor.cell for neighbor in neighbors],
                res,
                grid_pose,
                z_index=0.9,
                color=colors.flashy_red,
                ns="/rch_current_cell_neighbors",
            )
            marker_array.markers.append(neighbors_marker)  # type: ignore

            # Publish close_set
            if traversed_obstacles_ids:
                obstacle_id_to_color = dict(
                    zip(
                        traversed_obstacles_ids,
                        colors.generate_equally_spread_ros_colors(
                            len(traversed_obstacles_ids)
                        ),
                    )
                )
                color = obstacle_id_to_color[current.first_obstacle_uid]
            else:
                color = colors.generate_equally_spread_ros_colors(1)[0]

            if current.cell in self.namespaces_caches[ns].current_cell_to_marker:
                original_marker = self.namespaces_caches[ns].current_cell_to_marker[
                    current.cell
                ]
                blended_color = colors.blend_colors(original_marker.color, color)
                original_marker.color = blended_color
                close_set_marker = original_marker
            else:
                _id = self.namespaces_caches[ns].current_cell_marker_current_id
                close_set_marker = self._grid_cell_to_cube_marker(
                    current.cell,
                    res,
                    grid_pose,
                    color,
                    _id,
                    z_index=0.8,
                    ns="/rch_close_set",
                )
                self.namespaces_caches[ns].current_cell_to_marker[
                    current.cell
                ] = close_set_marker
                self.namespaces_caches[ns].current_cell_marker_current_id += 1

            marker_array.markers.append(close_set_marker)  # type: ignore

            # Publish open_heap
            # TODO

            # Publish came_from as paths between cells poses
            if current in came_from:
                path_color = ros_nodes.ColorRGBA(r=color.r, g=color.g, b=color.b, a=1.0)
                cells = (current.cell, came_from[current].cell)
                if cells in self.namespaces_caches[ns].cells_to_path_marker:
                    original_marker = self.namespaces_caches[ns].cells_to_path_marker[
                        cells
                    ]
                    blended_color = colors.blend_colors(
                        original_marker.color, path_color
                    )
                    original_marker.color = blended_color
                    came_from_marker = original_marker
                else:
                    _id = self.namespaces_caches[ns].cells_path_marker_current_id
                    cur_pose = utils.grid_to_real(
                        current.cell[0], current.cell[1], res, grid_pose
                    )
                    from_pose = utils.grid_to_real(
                        came_from[current].cell[0],
                        came_from[current].cell[1],
                        res,
                        grid_pose,
                    )
                    came_from_marker = conversions.real_path_to_triangle_list(
                        [cur_pose, from_pose],
                        "/rch_came_from",
                        _id,
                        cfg.main_frame_id,
                        ros_nodes.ColorRGBA(r=color.r, g=color.g, b=color.b, a=1.0),
                        res / 10.0,
                        cfg.path_line_z_index,
                    )
                    self.namespaces_caches[ns].cells_to_path_marker[
                        cells
                    ] = came_from_marker
                    self.namespaces_caches[ns].cells_path_marker_current_id += 1

                marker_array.markers.append(came_from_marker)  # type: ignore

            self.publish(full_topic, marker_array)

    # endregion

    # region MANIP SEARCH
    def publish_manip_search_data(
        self,
        current_manip_pose_id: int,
        robot_pose: PoseModel,
        obstacle_pose: PoseModel,
        robot_fixed_precision_pos: PoseModel,
        robot_polygon: Polygon,
        obstacle_polygon: Polygon,
        manip_poses_ids: t.List[int],
        neighbor_poses: t.List[PoseModel],
        line_width: float,
        res: float,
        ns: str = "",
    ):
        full_topic = cfg.robot_sim_topic if not ns else "/" + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = ros_nodes.MarkerArray(markers=[])

            arrow_length, shaft_diameter, head_diameter, head_length = (
                res / 1.5,
                res / 10.0,
                res / 5.0,
                res / 5.0,
            )
            manip_pose_id_to_color = dict(
                zip(
                    manip_poses_ids,
                    colors.generate_equally_spread_ros_colors(len(manip_poses_ids)),
                )
            )

            # Publish current configuration
            current_robot_pose_marker = self._pose_to_arrow(
                pose=robot_pose,
                namespace="/manip_search/current/robot/pose",
                p_id=0,
                frame_id=cfg.main_frame_id,
                color=colors.flashy_cyan,
                z_index=1.1,
                arrow_length=arrow_length,
                shaft_diameter=shaft_diameter,
                head_diameter=head_diameter,
                head_length=head_length,
            )
            current_obstacle_pose_marker = self._pose_to_arrow(
                pose=obstacle_pose,
                namespace="/manip_search/current/obstacle/pose",
                p_id=0,
                frame_id=cfg.main_frame_id,
                color=colors.flashy_dark_cyan,
                z_index=1.1,
                arrow_length=arrow_length,
                shaft_diameter=shaft_diameter,
                head_diameter=head_diameter,
                head_length=head_length,
            )
            marker_array.markers.append(current_robot_pose_marker)  # type: ignore
            marker_array.markers.append(current_obstacle_pose_marker)  # type: ignore

            current_robot_polygon_marker = self._polygon_to_line_strip(
                robot_polygon,
                "/manip_search/current/robot/polygon",
                0,
                cfg.main_frame_id,
                colors.flashy_cyan,
                cfg.entities_z_index,
                line_width=line_width,
            )
            current_obstacle_polygon_marker = self._polygon_to_line_strip(
                obstacle_polygon,
                "/manip_search/current/obstacle/polygon",
                0,
                cfg.main_frame_id,
                colors.flashy_dark_cyan,
                cfg.entities_z_index,
                line_width=line_width,
            )
            marker_array.markers.append(current_robot_polygon_marker)  # type: ignore
            marker_array.markers.append(current_obstacle_polygon_marker)  # type: ignore

            # Publish neighbors
            neighbors_markers = [
                self._pose_to_arrow(
                    pose=neighbor,
                    namespace="/manip_search_neighbors",
                    p_id=p_id,
                    frame_id=cfg.main_frame_id,
                    color=colors.flashy_green,
                    z_index=1.1,
                    arrow_length=arrow_length,
                    shaft_diameter=shaft_diameter,
                    head_diameter=head_diameter,
                    head_length=head_length,
                )
                for p_id, neighbor in enumerate(neighbor_poses)
            ]
            marker_array.markers += neighbors_markers  # type: ignore
            neighbor_markers_ids = {n.id for n in neighbors_markers}
            for p_id in self.namespaces_caches[ns].manip_search_neighbors_markers_p_ids:
                if p_id not in neighbor_markers_ids:
                    marker_array.markers.append(
                        self._make_delete_marker(
                            frame_id=cfg.main_frame_id,
                            namespace="/manip_search_neighbors",
                            p_id=p_id,
                        )
                    )
            self.namespaces_caches[
                ns
            ].manip_search_neighbors_markers_p_ids = neighbor_markers_ids

            # Publish close_set
            color = manip_pose_id_to_color[current_manip_pose_id]
            if (
                robot_fixed_precision_pos
                in self.namespaces_caches[ns].current_fixed_robot_pose_to_marker
            ):
                original_marker = self.namespaces_caches[
                    ns
                ].current_fixed_robot_pose_to_marker[robot_fixed_precision_pos]
                blended_color = colors.blend_colors(original_marker.color, color)
                original_marker.color = blended_color
                close_set_marker = original_marker
            else:
                _id = self.namespaces_caches[
                    ns
                ].current_fixed_robot_pose_marker_current_id
                close_set_marker = copy.deepcopy(current_robot_pose_marker)
                close_set_marker.ns = "/manip_search/close_set"
                close_set_marker.id = _id
                close_set_marker.color = color
                self.namespaces_caches[ns].current_fixed_robot_pose_to_marker[
                    robot_fixed_precision_pos
                ] = close_set_marker
                self.namespaces_caches[
                    ns
                ].current_fixed_robot_pose_marker_current_id += 1

            marker_array.markers.append(close_set_marker)

            # # Publish open_heap
            # # TODO
            #
            # # Publish came_from as paths between cells poses
            # if current in came_from:
            #     path_color = ColorRGBA(color.r, color.g, color.b, 1.)
            #     cells = (current.cell, came_from[current].cell)
            #     if cells in self.namespaces_caches[ns].cells_to_path_marker:
            #         original_marker = self.namespaces_caches[ns].cells_to_path_marker[cells]
            #         blended_color = colors.blend_colors(original_marker.color, path_color)
            #         original_marker.color = blended_color
            #         came_from_marker = original_marker
            #     else:
            #         _id = self.namespaces_caches[ns].cells_path_marker_current_id
            #         cur_pose = utils.grid_to_real(current.cell[0], current.cell[1], res, grid_pose)
            #         from_pose = utils.grid_to_real(came_from[current].cell[0], came_from[current].cell[1], res, grid_pose)
            #         came_from_marker = real_path_to_linestrip(
            #             [cur_pose, from_pose],
            #             '/manip_search_came_from', _id, cfg.main_frame_id, ColorRGBA(color.r, color.g, color.b, 1.),
            #             res / 10., cfg.path_line_z_index
            #         )
            #         self.namespaces_caches[ns].cells_to_path_marker[cells] = came_from_marker
            #         self.namespaces_caches[ns].cells_path_marker_current_id += 1
            #
            #     marker_array.markers.append(came_from_marker)

            self.publish(full_topic, marker_array)

    # endregion

    # region Q MANIPS FOR OBS
    def publish_q_manips_for_obs(self, poses: t.List[PoseModel], ns: str = ""):
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (
            cfg.obs_manip_poses_topic
            if not ns
            else "/" + ns + cfg.obs_manip_poses_topic
        )
        self.observers[topic].update(poses=poses)

    def cleanup_q_manips_for_obs(self, ns: str = ""):
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (
            cfg.obs_manip_poses_topic
            if not ns
            else "/" + ns + cfg.obs_manip_poses_topic
        )
        self.observers[topic].reset()

    # endregion

    # region P_OPT
    def publish_p_opt(
        self,
        plan: "navigation_plan.Plan",
        robot: "agent.Agent",
        ns: str = "",
    ):
        """Publishes the optimal path to observers"""
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (cfg.plan_topic if not ns else "/" + ns + cfg.plan_topic)
        self.observers[topic].update(plan=plan, robot=robot)

    def cleanup_p_opt(self, ns: str = ""):
        """Clears the optimal path from observers"""
        if DEACTIVATE_RVIZ:
            return
        topic = self.prefix + (cfg.plan_topic if not ns else "/" + ns + cfg.plan_topic)
        self.observers[topic].reset()

    # endregion

    # region ROBOT SIM
    def publish_sim(
        self,
        *,
        robot_polygon: Polygon,
        obs_polygon: Polygon,
        line_width: float,
        namespace: str = "/init",
        robot_name: str = "",
    ):
        full_topic = (
            cfg.robot_sim_topic
            if not robot_name
            else "/" + robot_name + cfg.robot_sim_topic
        )

        if self.is_activated(full_topic):
            robot_color = (
                colors.robot_border_color
                if namespace == "/target"
                else colors.robot_color
            )
            obs_color = (
                colors.movable_obstacle_border_color
                if namespace == "/target"
                else colors.movable_obstacle_color
            )
            marker_array = ros_nodes.MarkerArray(
                markers=[
                    self._polygon_to_line_strip(
                        robot_polygon,
                        namespace + "/robot/polygon",
                        0,
                        cfg.main_frame_id,
                        robot_color,
                        cfg.entities_z_index,
                        line_width=line_width,
                    ),
                    self._polygon_to_line_strip(
                        obs_polygon,
                        namespace + "/obstacle/polygon",
                        0,
                        cfg.main_frame_id,
                        obs_color,
                        cfg.entities_z_index,
                        line_width=line_width,
                    ),
                ]
            )
            self.publish(full_topic, marker_array)

    def publish_blocking_areas(
        self,
        init_blocking_areas: t.List[Polygon],
        target_blocking_areas: t.List[Polygon],
        ns: str = "",
    ):
        if DEACTIVATE_RVIZ:
            return
        full_topic = cfg.robot_sim_topic if not ns else "/" + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            init_blocking_areas_markers = []
            for i in range(len(init_blocking_areas)):
                init_blocking_areas_markers.append(
                    conversions.polygon_to_triangle_list(
                        polygon=init_blocking_areas[i],
                        namespace="/blocking_areas/init",
                        p_id=i,
                        frame_id=cfg.main_frame_id,
                        color=colors.init_blocking_areas_color,
                        z_index=cfg.entities_z_index,
                    )
                )

            target_blocking_areas_markers = []
            for i in range(len(target_blocking_areas)):
                target_blocking_areas_markers.append(
                    conversions.polygon_to_triangle_list(
                        polygon=target_blocking_areas[i],
                        namespace="/blocking_areas/target",
                        p_id=i,
                        frame_id=cfg.main_frame_id,
                        color=colors.target_blocking_areas_color,
                        z_index=cfg.entities_z_index,
                    )
                )

            marker_array = ros_nodes.MarkerArray(
                markers=init_blocking_areas_markers + target_blocking_areas_markers
            )
            self.publish(full_topic, marker_array)

    def cleanup_blocking_areas(self, ns: str = ""):
        # FIXME Not implemented correctly in ROS...
        #  https://answers.ros.org/question/263031/delete-all-rviz-markers-in-a-specific-namespace/
        full_topic = cfg.robot_sim_topic if not ns else "/" + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(
                full_topic,
                self._make_delete_all_marker(cfg.main_frame_id, "/blocking_areas"),
            )

    def publish_diameter_inflated_polygons(
        self,
        init_entity_inflated_polygon: Polygon,
        target_entity_inflated_polygon: Polygon,
        line_width: float,
        ns: str = "",
    ):
        if DEACTIVATE_RVIZ:
            return
        full_topic = cfg.robot_sim_topic if not ns else "/" + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = ros_nodes.MarkerArray(
                markers=[
                    self._polygon_to_line_strip(
                        init_entity_inflated_polygon,
                        "/diameter_inflated_polygon/init",
                        0,
                        cfg.main_frame_id,
                        colors.init_diameter_inflated_polygon_color,
                        cfg.entities_z_index,
                        line_width=line_width,
                    ),
                    self._polygon_to_line_strip(
                        target_entity_inflated_polygon,
                        "/diameter_inflated_polygon/target",
                        0,
                        cfg.main_frame_id,
                        colors.target_diameter_inflated_polygon_color,
                        cfg.entities_z_index,
                        line_width=line_width,
                    ),
                ]
            )
            self.publish(full_topic, marker_array)

    def cleanup_diameter_inflated_polygons(self, ns: str = ""):
        # FIXME Not implemented correctly in ROS...
        #  https://answers.ros.org/question/263031/delete-all-rviz-markers-in-a-specific-namespace/
        full_topic = cfg.robot_sim_topic if not ns else "/" + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(
                full_topic,
                self._make_delete_all_marker(
                    cfg.main_frame_id, "/diameter_inflated_polygon"
                ),
            )

    def publish_debug_polygons(
        self, polygons: t.List[Polygon], line_width: float, ns: str = ""
    ):
        # FIXME Not implemented correctly in ROS...
        #  https://answers.ros.org/question/263031/delete-all-rviz-markers-in-a-specific-namespace/
        full_topic = cfg.robot_sim_topic if not ns else "/" + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            marker_array = self._polygons_to_line_strips_marker_array(
                polygons,
                "/debug/polygons",
                cfg.main_frame_id,
                colors.robot_color,
                cfg.entities_z_index,
                line_width=line_width,
            )
            self.publish(full_topic, marker_array)

    def cleanup_debug_polygons(self, ns: str = ""):
        full_topic = cfg.robot_sim_topic if not ns else "/" + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.publish(
                full_topic,
                self._make_delete_all_marker(cfg.main_frame_id, "/debug/polygons"),
            )

    def cleanup_robot_sim(self, ns: str = ""):
        if DEACTIVATE_RVIZ:
            return
        full_topic = cfg.robot_sim_topic if not ns else "/" + ns + cfg.robot_sim_topic
        if self.is_activated(full_topic):
            self.namespaces_caches[ns] = ros_nodes.NamespaceCache()
            self.publish(full_topic, self._make_delete_all_marker(cfg.main_frame_id))

    # endregion

    # region GOAL
    def publish_goal(
        self,
        q_init: PoseModel,
        q_goal: PoseModel,
        entity: "agent.Agent",
        ns: str = "",
    ):
        if DEACTIVATE_RVIZ:
            return

        topic = self.prefix + (
            cfg.robot_goal_topic if not ns else "/" + ns + cfg.robot_goal_topic
        )
        self.observers[topic].update(q_init=q_init, q_goal=q_goal, entity=entity)

    def cleanup_goal(self, ns: str = ""):
        topic = self.prefix + (
            cfg.robot_goal_topic if not ns else "/" + ns + cfg.robot_goal_topic
        )
        self.observers[topic].reset()

    # endregion

    # region MESSAGE TEXT
    def publish_message(
        self, message: str, pose: PoseModel = (0.0, 0.0, 0.0), font_size: float = 1.0
    ):
        if self.is_activated("/simulation" + cfg.sim_latest_message_topic):
            marker_array = ros_nodes.MarkerArray(
                markers=[
                    self._string_to_text_marker(
                        message=message,
                        pose=pose,
                        ns="",
                        p_id=0,
                        z_index=cfg.goal_z_index,
                        font_size=font_size,
                        frame_id=cfg.main_frame_id,
                        color=colors.black,
                    )
                ]
            )
            self.publish("/simulation" + cfg.sim_latest_message_topic, marker_array)

    def cleanup_message(self):
        if self.is_activated("/simulation" + cfg.sim_latest_message_topic):
            marker_array = ros_nodes.MarkerArray(
                markers=[
                    self._string_to_text_marker(
                        message="_",
                        pose=(0.0, 0.0, 0.0),
                        ns="",
                        p_id=0,
                        z_index=cfg.goal_z_index,
                        font_size=0.01,
                        frame_id=cfg.main_frame_id,
                        color=colors.black,
                    )
                ]
            )
            self.publish("/simulation" + cfg.sim_latest_message_topic, marker_array)

    # endregion

    # region CONFLICTS CHECK
    def publish_transit_horizon_cells(
        self,
        poses: t.List[PoseModel],
        start_index: int,
        check_horizon: int,
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        robot_name: str,
    ):
        if check_horizon <= 0:
            return

        full_topic = "/" + robot_name + cfg.conflict_horizon_topic

        if self.is_activated(full_topic):
            horizon_cells: t.Set[GridCellModel] = set()
            for pose in poses[start_index : start_index + check_horizon]:
                cell = utils.real_to_grid(
                    pose[0],
                    pose[1],
                    inflated_grid_by_robot.res,
                    inflated_grid_by_robot.grid_pose,
                )
                horizon_cells.add(cell)
            cube_list_marker = self._grid_cells_to_cube_list_markers(
                horizon_cells,
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
                colors.flashy_green,
                z_index=cfg.horizon_markers_z_index,
                ns="/transit_horizon_cells",
            )
            marker_array = ros_nodes.MarkerArray(markers=[cube_list_marker])
            self.publish(full_topic, marker_array)

    def publish_transit_conflicting_cells(
        self,
        conflicting_cells: t.List[GridCellModel],
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        ns: str,
    ):
        full_topic = (
            cfg.conflicts_check_topic
            if not ns
            else "/" + ns + cfg.conflicts_check_topic
        )
        if self.is_activated(full_topic):
            cube_list_marker = self._grid_cells_to_cube_list_markers(
                conflicting_cells,
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
                colors.flashy_red,
                z_index=cfg.conflicting_cells_z_index,
                ns="/transit_conflicting_cells",
            )
            marker_array = ros_nodes.MarkerArray(markers=[cube_list_marker])
            self.publish(full_topic, marker_array)

    def publish_transit_conflicting_polygons_cells(
        self,
        conflicting_entities_cells: t.List[GridCellModel],
        inflated_grid_by_robot: BinaryInflatedOccupancyGrid,
        ns: str,
    ):
        full_topic = (
            cfg.conflicts_check_topic
            if not ns
            else "/" + ns + cfg.conflicts_check_topic
        )
        if self.is_activated(full_topic):
            cube_list_marker = self._grid_cells_to_cube_list_markers(
                conflicting_entities_cells,
                inflated_grid_by_robot.res,
                inflated_grid_by_robot.grid_pose,
                colors.flashy_cyan,
                z_index=cfg.conflict_markers_z_index,
                ns="/transit_conflicting_entities_cells",
            )
            marker_array = ros_nodes.MarkerArray(markers=[cube_list_marker])
            self.publish(full_topic, marker_array)

    def publish_transfer_horizon_convex_polygons(
        self,
        robot_csv_polygons: t.Dict[t.Tuple[int], GeometryCollection],
        obstacle_csv_polygons: t.Dict[t.Tuple[int], GeometryCollection],
        start_index: int,
        check_horizon: int,
        robot_name: str,
    ):
        full_topic = "/" + robot_name + cfg.swept_area_topic

        if not self.is_activated(full_topic):
            return

        horizon_csv_polygons = []
        for i in range(
            start_index, min(start_index + check_horizon, len(robot_csv_polygons))
        ):
            key = (i,)
            if key in robot_csv_polygons:
                horizon_csv_polygons.append(robot_csv_polygons[key])
            if key in obstacle_csv_polygons:
                horizon_csv_polygons.append(obstacle_csv_polygons[key])

        markers = []
        for p_id, polygon in enumerate(horizon_csv_polygons):
            marker = conversions.polygon_to_triangle_list(
                polygon=polygon,
                namespace="/transfer_horizon_csv_polygons",
                p_id=p_id,
                frame_id=cfg.main_frame_id,
                color=colors.flashy_green,
                z_index=cfg.swept_area_z_index,
            )
            markers.append(marker)
        marker_array = ros_nodes.MarkerArray(markers=markers)
        self.publish(full_topic, marker_array)

    def publish_transfer_conflicting_intersections(self):
        pass

    def publish_transfer_conflicting_convex_polygons(self):
        pass

    def cleanup_swept_area(self, ns: str):
        full_topic = "/" + ns + cfg.swept_area_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, self._make_delete_all_marker(cfg.main_frame_id))

    def cleanup_conflict_horizon(self, ns: str):
        full_topic = "/" + ns + cfg.conflict_horizon_topic
        if self.is_activated(full_topic):
            self.publish(full_topic, self._make_delete_all_marker(cfg.main_frame_id))

    def cleanup_conflicts_checks(self, ns: str):
        full_topic = (
            cfg.conflicts_check_topic
            if not ns
            else "/" + ns + cfg.conflicts_check_topic
        )
        if self.is_activated(full_topic):
            self.publish(full_topic, self._make_delete_all_marker(cfg.main_frame_id))

    # endregion

    # region EXTRA COMBINED CLEANUP METHODS
    def cleanup_all(self):
        if DEACTIVATE_RVIZ:
            return
        self.cleanup_sim_world()
        self.cleanup_message()
        for ns in self.agents_names:
            self.cleanup_robot_world(ns=ns)
            self.cleanup_robot_sim_world(ns=ns)
            self.cleanup_p_opt(ns=ns)
            self.cleanup_q_manips_for_obs(ns=ns)
            self.cleanup_goal(ns=ns)
            self.cleanup_social_grid_map(ns=ns)
            self.cleanup_combined_costmap(ns=ns)
            self.cleanup_conflicts_checks(ns=ns)
            self.cleanup_swept_area(ns=ns)
            self.cleanup_conflict_horizon(ns=ns)

    def init_header(self):
        if DEACTIVATE_RVIZ:
            return
        return ros_nodes.Header(stamp=self.ros_node.get_timestamp(), frame_id="map")

    def _grid_cells_to_cube_list_markers(
        self,
        grid_cells: t.Iterable[GridCellModel],
        res: float,
        grid_pose: PoseModel,
        color: t.Any,
        z_index: float = -0.5,
        cube_list: t.Any | None = None,
        ns: str = "",
    ):
        if cube_list is None:
            cube_list = ros_nodes.Marker(
                type=ros_nodes.Marker.CUBE_LIST,
                ns=ns,
                id=0,
                header=ros_nodes.Header(
                    frame_id=cfg.main_frame_id, stamp=self.ros_node.get_timestamp()
                ),
                color=color,
                scale=ros_nodes.Vector3(x=res, y=res, z=1e-6),
                points=[],
            )
        for cell in grid_cells:
            point = ros_nodes.Point()
            point.x, point.y = utils.grid_to_real(cell[0], cell[1], res, grid_pose)
            point.z = z_index
            cube_list.points.append(point)  # type: ignore
        return cube_list

    def _grid_cell_to_cube_marker(
        self,
        cell: GridCellModel,
        res: float,
        grid_pose: PoseModel,
        color: t.Any,
        _id: int,
        z_index: float,
        ns: str = "",
    ):
        x, y = utils.grid_to_real(cell[0], cell[1], res, grid_pose)
        z = z_index

        cube = ros_nodes.Marker(
            type=ros_nodes.Marker.CUBE,
            ns=ns,
            id=_id,
            header=ros_nodes.Header(
                frame_id=cfg.main_frame_id, stamp=self.ros_node.get_timestamp()
            ),
            color=color,
            scale=ros_nodes.Vector3(x=res, y=res, z=res),
            pose=ros_nodes.Pose(position=(ros_nodes.Point(x=x, y=y, z=z))),
        )
        return cube

    def _polygon_to_line_strip(
        self,
        polygon: Polygon | None,
        namespace: str,
        p_id: int,
        frame_id: str,
        color: t.Any,
        z_index: float,
        line_width: float,
    ):
        marker = ros_nodes.Marker(
            type=ros_nodes.Marker.LINE_STRIP,
            ns=namespace,
            id=p_id,
            header=ros_nodes.Header(
                frame_id=frame_id, stamp=self.ros_node.get_timestamp()
            ),
            color=color,
            scale=ros_nodes.Vector3(x=line_width, y=0.0, z=0.0),
            points=[],
        )
        if polygon is not None:
            for i in range(len(polygon.exterior.coords) - 1):
                point = polygon.exterior.coords[i]
                next_point = polygon.exterior.coords[i + 1]
                marker.points.append(ros_nodes.Point(x=point[0], y=point[1], z=z_index))  # type: ignore
                marker.points.append(  # type: ignore
                    ros_nodes.Point(x=next_point[0], y=next_point[1], z=z_index)
                )
            marker.points.append(  # type: ignore
                ros_nodes.Point(
                    x=polygon.exterior.coords[0][0],
                    y=polygon.exterior.coords[0][1],
                    z=z_index,
                )
            )
            marker.points.append(  # type: ignore
                ros_nodes.Point(
                    x=polygon.exterior.coords[1][0],
                    y=polygon.exterior.coords[1][1],
                    z=z_index,
                )
            )
        return marker

    def _polygons_to_line_strips_marker_array(
        self,
        polygons: t.List[Polygon],
        namespace: str,
        frame_id: str,
        color: t.Any,
        z_index: float,
        line_width: float,
    ):
        marker_array = ros_nodes.MarkerArray()
        markers = []
        p_id = 0
        for polygon in polygons:
            markers.append(
                self._polygon_to_line_strip(
                    polygon, namespace, p_id, frame_id, color, z_index, line_width
                )
            )
            p_id += 1
        marker_array.markers = markers
        return marker_array

    def _pose_to_arrow(
        self,
        pose: PoseModel,
        namespace: str,
        p_id: int,
        frame_id: str,
        color: t.Any,
        z_index: float,
        arrow_length: float,
        shaft_diameter: float,
        head_diameter: float,
        head_length: float,
    ):
        marker = ros_nodes.Marker(
            type=ros_nodes.Marker.ARROW,
            ns=namespace,
            id=p_id,
            # pose=Pose(Point(pose[0], pose[1], z_index), geom_quat_from_yaw(pose[2])),
            points=[
                ros_nodes.Point(x=pose[0], y=pose[1], z=z_index),
                ros_nodes.Point(
                    x=pose[0] + arrow_length * math.cos(math.radians(pose[2])),
                    y=pose[1] + arrow_length * math.sin(math.radians(pose[2])),
                    z=z_index,
                ),
            ],
            scale=ros_nodes.Vector3(x=shaft_diameter, y=head_diameter, z=head_length),
            header=ros_nodes.Header(
                frame_id=frame_id, stamp=self.ros_node.get_timestamp()
            ),
            color=color,
        )
        return marker

    def _make_delete_marker(self, namespace: str, p_id: int, frame_id: str):
        return ros_nodes.Marker(
            ns=namespace,
            id=p_id,
            header=ros_nodes.Header(
                frame_id=frame_id, stamp=self.ros_node.get_timestamp()
            ),
            action=ros_nodes.Marker.DELETE,
        )

    def _make_delete_all_marker(self, frame_id: str, ns: str = ""):
        return ros_nodes.MarkerArray(
            markers=[
                ros_nodes.Marker(
                    ns=ns,
                    header=ros_nodes.Header(
                        frame_id=frame_id, stamp=self.ros_node.get_timestamp()
                    ),
                    action=ros_nodes.Marker.DELETEALL,
                )
            ]
        )

    def _string_to_text_marker(
        self,
        message: str = "",
        pose: PoseModel = (0.0, 0.0, 0.0),
        ns: str = "",
        p_id: int = 0,
        z_index: float = 0.0,
        font_size: float = 1.0,
        frame_id: str = "/map",
        color: t.Any | None = None,
    ):
        if color is None:
            color = colors.black
        x, y, z = pose[0], pose[1], z_index
        marker = ros_nodes.Marker(
            type=ros_nodes.Marker.TEXT_VIEW_FACING,
            ns=ns,
            id=p_id,
            pose=ros_nodes.Pose(
                position=(ros_nodes.Point(x=x, y=y, z=z)),
                orientation=conversions.geom_quat_from_yaw(pose[2]),
            ),
            points=[ros_nodes.Point(x=pose[0], y=pose[1], z=z_index)],
            scale=ros_nodes.Vector3(x=0.0, y=0.0, z=font_size),
            header=ros_nodes.Header(
                frame_id=frame_id, stamp=self.ros_node.get_timestamp()
            ),
            color=color,
            text=message,
        )
        return marker

    # endregion
