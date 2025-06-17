# pyright: reportUnboundVariable=false

import math
import subprocess
import typing as t
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon

from namosim.agents.stilman_configurations import RCHConfiguration
from namosim.algorithms.rrt_node import RRTNode
import namosim.display.ros_publisher_config as cfg
import namosim.navigation.navigation_plan as nav_plan
import namosim.world.world as world
from namosim.agents import agent
from namosim.config import DEACTIVATE_RVIZ
from namosim.data_models import GridCellModel, Pose2D
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid

if not DEACTIVATE_RVIZ:
    from rclpy.node import Node
    from rclpy.callback_groups import CallbackGroup
    import namosim.display.conversions as conversions
    import namosim.display.ros_nodes as ros_nodes
    from namosim.display import colors


class RosPublisher:  # noqa: F821
    def __init__(
        self,
        agent_ids: t.List[str],
        ros_node: t.Optional["Node"] = None,
        callback_group: t.Optional["CallbackGroup"] = None,
    ):
        if DEACTIVATE_RVIZ or not ros_node:
            return

        self.ros_node = ros_node
        # Add simulation-specific publishers
        self.dynamic_entities_publisher = ros_nodes.DynamicEntitiesPublisher(
            node=self.ros_node,
            topic="namosim/dynamic_entities",
            callback_group=callback_group,
        )
        self.map_publisher = ros_nodes.WorldMapPublisher(
            self.ros_node, "namosim/map", callback_group=callback_group
        )
        self.text_publisher = ros_nodes.create_publisher(
            self.ros_node,
            ros_nodes.Marker,
            f"namosim/text",
            callback_group=callback_group,
        )
        self.agent_ids = agent_ids

        # Add robot-specific publishers for each robot namespace
        self.robot_world_publishers: t.Dict[str, ros_nodes.DynamicEntitiesPublisher] = (
            {}
        )
        self.robot_map_publishers: t.Dict[str, ros_nodes.WorldMapPublisher] = {}
        self.robot_combined_costmap_publishers: t.Dict[
            str, ros_nodes.CombinedCostmapPublisher
        ] = {}
        self.robot_social_costmap_publishers: t.Dict[
            str, ros_nodes.GridMapPublisher
        ] = {}
        self.robot_goal_publishers: t.Dict[str, ros_nodes.GoalPublisher] = {}
        self.robot_manip_search_publishers: t.Dict[
            str, ros_nodes.ManipSearchPublisher
        ] = {}
        self.robot_manip_pose_publishers: t.Dict[str, ros_nodes.PosesPublisher] = {}
        self.robot_plan_publishers: t.Dict[str, ros_nodes.PlanPublisher] = {}
        self.robot_conflict_check_publishers: t.Dict[str, ros_nodes.Publisher] = {}
        self.robot_conflict_horizon_publishers: t.Dict[str, ros_nodes.Publisher] = {}
        self.robot_swept_area_publishers: t.Dict[str, ros_nodes.Publisher] = {}
        self.robot_rch_publishers: t.Dict[str, ros_nodes.Publisher] = {}
        self.robot_rrt_publishers: t.Dict[str, ros_nodes.RRTPublisher] = {}
        self.robot_connected_components_publishers: t.Dict[
            str, ros_nodes.GridMapPublisher
        ] = {}

        for agent_id in self.agent_ids:
            ns = f"namosim/{agent_id}"
            self.robot_world_publishers[agent_id] = ros_nodes.DynamicEntitiesPublisher(
                self.ros_node, f"{ns}/world", callback_group=callback_group
            )
            self.robot_map_publishers[agent_id] = ros_nodes.WorldMapPublisher(
                self.ros_node, f"{ns}/map", callback_group=callback_group
            )
            self.robot_combined_costmap_publishers[agent_id] = (
                ros_nodes.CombinedCostmapPublisher(
                    self.ros_node,
                    f"{ns}/combined_costmap",
                    callback_group=callback_group,
                )
            )
            self.robot_social_costmap_publishers[agent_id] = ros_nodes.GridMapPublisher(
                self.ros_node, f"{ns}/social_costmap", callback_group=callback_group
            )
            self.robot_goal_publishers[agent_id] = ros_nodes.GoalPublisher(
                self.ros_node, f"{ns}/goals", callback_group=callback_group
            )
            self.robot_manip_search_publishers[agent_id] = (
                ros_nodes.ManipSearchPublisher(self.ros_node, f"{ns}/manip_search")
            )
            self.robot_manip_pose_publishers[agent_id] = ros_nodes.PosesPublisher(
                self.ros_node, f"{ns}/manip_search/poses", callback_group=callback_group
            )
            self.robot_plan_publishers[agent_id] = ros_nodes.PlanPublisher(
                self.ros_node, f"{ns}/plan", callback_group=callback_group
            )
            self.robot_conflict_check_publishers[agent_id] = ros_nodes.create_publisher(
                self.ros_node,
                ros_nodes.MarkerArray,
                f"{ns}/conflict_check",
                callback_group=callback_group,
            )
            self.robot_conflict_horizon_publishers[agent_id] = (
                ros_nodes.create_publisher(
                    self.ros_node,
                    ros_nodes.MarkerArray,
                    f"{ns}/conflict_horizon",
                    callback_group=callback_group,
                )
            )
            self.robot_swept_area_publishers[agent_id] = ros_nodes.create_publisher(
                self.ros_node,
                ros_nodes.MarkerArray,
                f"{ns}/swept_area",
                callback_group=callback_group,
            )
            self.robot_rch_publishers[agent_id] = ros_nodes.create_publisher(
                self.ros_node,
                ros_nodes.MarkerArray,
                f"{ns}/rch",
                callback_group=callback_group,
            )
            self.robot_connected_components_publishers[agent_id] = (
                ros_nodes.GridMapPublisher(
                    self.ros_node,
                    f"{ns}/connected_components",
                    callback_group=callback_group,
                )
            )
            self.robot_rrt_publishers[agent_id] = ros_nodes.RRTPublisher(
                node=self.ros_node,
                topic=f"{ns}/rrt",
                callback_group=callback_group,
            )

        # Setup Static Transform for grid map (Hack so that it is properly placed in view)
        broadcaster = self.get_transform_broadcaster()

        for frame_id, z_index in cfg.gridmap_frame_ids_to_z_indexes.items():
            transform = ros_nodes.TransformStamped(
                header=ros_nodes.Header(stamp=self.get_timestamp(), frame_id="map"),
                child_frame_id=frame_id,
                transform=ros_nodes.Transform(
                    translation=ros_nodes.Vector3(z=z_index),
                    rotation=ros_nodes.Quaternion(x=0.0, y=0.0, z=1.0, w=0.0),
                ),
            )
            broadcaster.sendTransform(transform)
            ros_nodes.time.sleep(0.5)  # Hack so that transform is properly sent...

    def get_transform_broadcaster(self):
        return ros_nodes.StaticTransformBroadcaster(self.ros_node)

    def get_timestamp(self) -> "ros_nodes.Time":
        return self.ros_node.get_clock().now().to_msg()

    @staticmethod
    def get_nodes_names():
        cmd_str = "ros2 node list"
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        nodes_names = [name for name in result.stdout.split("\n") if name]
        return nodes_names

    @staticmethod
    def create_valid_node_name(root_name: str):
        nodes_names = RosPublisher.get_nodes_names()
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

    def publish_namespace(self):
        marker = ros_nodes.Marker()
        marker.header.frame_id = cfg.main_frame_id
        marker.type = ros_nodes.Marker.TEXT_VIEW_FACING
        marker.text = f"Namespace: {self.ros_node.get_namespace()}"
        marker.pose.position.x = 0.0
        marker.pose.position.y = -1.0
        marker.pose.position.z = 1.0
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        self.text_publisher.publish(marker)

    def clear_namespace(self):
        self.text_publisher.publish(
            ros_nodes.Marker(
                header=ros_nodes.Header(
                    frame_id=cfg.main_frame_id, stamp=self.get_timestamp()
                ),
                action=ros_nodes.Marker.DELETEALL,
            )
        )

    # region TRUE WORLD
    def publish_world(self, world: "world.World"):
        self.publish_namespace()
        self.map_publisher.publish(world)
        self.dynamic_entities_publisher.update(world=world)

    def clear_world(self):
        self.clear_namespace()
        self.map_publisher.reset()
        self.dynamic_entities_publisher.reset()

    # endregion

    # region ROBOT OBSERVED WORLD
    def publish_robot_observed_world(self, world: "world.World", agent_id: str):

        self.robot_world_publishers[agent_id].update(world=world)
        self.robot_map_publishers[agent_id].publish(world, agent_id)

    def cleanup_robot_observed_world(self, agent_id: str):

        self.robot_world_publishers[agent_id].reset()
        self.robot_map_publishers[agent_id].reset()

    # endregion

    # region SOCIAL AND COMBINED COSTMAPS
    def publish_social_costmap(
        self,
        costmap: npt.NDArray[np.float32],
        cell_size: float,
        agent_id: str,
    ):

        # re-scale and shift the costmap so it displays nicely below the 2D environment in RVIZ
        costmap = np.copy(costmap)
        costmap[costmap == -1.0] = 0.0
        M = np.ptp(costmap)
        m = np.min(costmap)
        costmap = 0.5 * (costmap - m) / M  # make the costmap 0.5 meters tall
        costmap = costmap - 1  # display 1 meters below 0

        self.robot_social_costmap_publishers[agent_id].publish(
            costmap=costmap, cell_size=cell_size
        )

    def clear_social_costmap(self, agent_id: str):

        self.robot_social_costmap_publishers[agent_id].reset()

    def publish_combined_costmap(
        self,
        sorted_cell_to_combined_cost: OrderedDict[GridCellModel, float],
        inflated_grid_by_obstacle: BinaryOccupancyGrid,
        agent_id: str,
    ):

        self.robot_combined_costmap_publishers[agent_id].publish(
            sorted_cell_to_combined_cost=sorted_cell_to_combined_cost,
            inflated_grid_by_obstacle=inflated_grid_by_obstacle,
        )

    def clear_combined_costmap(self, agent_id: str):

        self.robot_combined_costmap_publishers[agent_id].reset()

    # endregion

    # region CONNECTED COMPONENTS
    def publish_connected_components(
        self, cc_grid: npt.NDArray[np.float32], cell_size: float, agent_id: str
    ):

        # re-scale and shift the costmap so it displays nicely below the 2D environment in RVIZ
        cc_grid = np.copy(cc_grid)
        cc_grid[cc_grid == -1.0] = 0.0
        H = cc_grid.shape[0]
        M = np.ptp(cc_grid)
        m = np.min(cc_grid)
        cc_grid = (cc_grid - m) / M
        cc_grid = cc_grid * H - (6 * H)

        self.robot_connected_components_publishers[agent_id].publish(
            costmap=cc_grid, cell_size=cell_size
        )

    def clear_connected_components(self, agent_id: str):

        self.robot_connected_components_publishers[agent_id].reset()

    # endregion

    # region RCH
    def publish_rch_search(
        self,
        map: BinaryOccupancyGrid,
        current: RCHConfiguration,
        came_from: t.Dict[RCHConfiguration, RCHConfiguration],
        neighbors: t.Iterable[RCHConfiguration],
        traversed_obstacles_ids: t.List[str],
        res: float,
        grid_pose: Pose2D,
        agent_id: str,
    ):
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

        _id = 1
        close_set_marker = self._grid_cell_to_cube_marker(
            current.cell,
            res,
            grid_pose,
            color,
            _id,
            z_index=0.8,
            ns="/rch_close_set",
        )

        marker_array.markers.append(close_set_marker)  # type: ignore

        # Publish open_heap
        # TODO

        # Publish came_from as paths between cells poses
        if current in came_from:
            _id += 1
            path_color = ros_nodes.ColorRGBA(r=color.r, g=color.g, b=color.b, a=1.0)
            cells = (current.cell, came_from[current].cell)

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
                map,
                _id,
                cfg.main_frame_id,
                ros_nodes.ColorRGBA(r=color.r, g=color.g, b=color.b, a=1.0),
                res / 10.0,
                cfg.path_line_z_index,
            )

            marker_array.markers.append(came_from_marker)  # type: ignore

        self.robot_rch_publishers[agent_id].publish(marker_array)

    def publish_manip_search(
        self,
        *,
        agent_id: str,
        current: GridCellModel,
        visited_cells: t.Set[GridCellModel],
        cell_size: float,
        grid_pose: Pose2D,
        robot_pose: Pose2D,
        obstacle_pose: Pose2D,
        robot_polygon: Polygon,
        obstacle_polygon: Polygon,
    ):
        marker_array = ros_nodes.MarkerArray(markers=[])

        # Publish visited cells
        current_marker = self._grid_cells_to_cube_list_markers(
            visited_cells,
            cell_size,
            grid_pose,
            z_index=0.9,
            color=colors.dark_blue,
            ns="/visited_cells",
        )
        marker_array.markers.append(current_marker)  # type: ignore
        # Publish current cell
        current_marker = self._grid_cells_to_cube_list_markers(
            [current],
            cell_size,
            grid_pose,
            z_index=0.9,
            color=colors.flashy_purple,
            ns="/current_cell",
        )
        marker_array.markers.append(current_marker)  # type: ignore

        arrow_length, shaft_diameter, head_diameter, head_length = (
            cell_size / 1.5,
            cell_size / 10.0,
            cell_size / 5.0,
            cell_size / 5.0,
        )
        # Publish current configuration
        current_robot_pose_marker = self._pose_to_arrow(
            pose=robot_pose,
            namespace="/manip_search/robot/pose",
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
            namespace="/manip_search/obstacle/pose",
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
            "/manip_search/robot/polygon",
            0,
            cfg.main_frame_id,
            colors.flashy_cyan,
            cfg.entities_z_index,
            line_width=0.01,
        )
        marker_array.markers.append(current_robot_polygon_marker)  # type: ignore

        current_obstacle_polygon_marker = self._polygon_to_line_strip(
            obstacle_polygon,
            "/manip_search/obstacle/polygon",
            0,
            cfg.main_frame_id,
            colors.flashy_dark_cyan,
            cfg.entities_z_index,
            line_width=0.01,
        )
        marker_array.markers.append(current_obstacle_polygon_marker)  # type: ignore

        self.robot_manip_search_publishers[agent_id].publish(marker_array)

    def publish_blocking_areas(
        self,
        init_blocking_areas: t.List[Polygon],
        target_blocking_areas: t.List[Polygon],
        agent_id: str,
    ):

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
        self.robot_manip_search_publishers[agent_id].publish(marker_array)

    def cleanup_blocking_areas(self, agent_id: str):

        self.robot_manip_search_publishers[agent_id]._publisher.publish(
            self._make_delete_all_marker(cfg.main_frame_id, "/blocking_areas")
        )

    def publish_diameter_inflated_polygons(
        self,
        init_entity_inflated_polygon: Polygon,
        target_entity_inflated_polygon: Polygon,
        line_width: float,
        agent_id: str,
    ):

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
        self.robot_manip_search_publishers[agent_id].publish(marker_array)

    def cleanup_diameter_inflated_polygons(self, agent_id: str):

        self.robot_manip_search_publishers[agent_id]._publisher.publish(
            self._make_delete_all_marker(
                cfg.main_frame_id, "/diameter_inflated_polygon"
            )
        )

    def cleanup_manip_search(self, agent_id: str):

        self.robot_manip_search_publishers[agent_id].reset()

    def publish_obstacle_grab_poses(self, poses: t.List[Pose2D], agent_id: str):

        self.robot_manip_pose_publishers[agent_id].publish(poses)

    def clear_obstacle_grab_poses(self, agent_id: str):

        self.robot_manip_pose_publishers[agent_id].reset()

    # region GOAL
    def publish_goal(
        self,
        q_init: Pose2D,
        q_goal: Pose2D,
        entity: "agent.Agent",
    ):

        self.robot_goal_publishers[entity.uid].publish(
            robot=entity,
            q_init=q_init,
            q_goal=q_goal,
        )

    def clear_goal(self, agent_id: str):

        self.robot_goal_publishers[agent_id].reset()

    # endregion

    # region PLAN

    def publish_robot_plan(
        self,
        plan: "nav_plan.Plan",
        robot: "agent.Agent",
        map: BinaryOccupancyGrid,
    ):

        self.robot_plan_publishers[robot.uid].publish(map=map, robot=robot, plan=plan)

    def clear_robot_plan(self, agent_id: str):

        self.robot_plan_publishers[agent_id].reset()

    # endregion

    # region RRT

    def publish_robot_rrt(self, agent_id: str, rrt_nodes: t.List[RRTNode]):
        self.robot_rrt_publishers[agent_id].publish(rrt_nodes=rrt_nodes)

    def clear_robot_rrt(self, agent_id: str):
        self.robot_rrt_publishers[agent_id].reset()

    # endregion

    # region CONFLICTS CHECK
    def publish_transit_horizon_cells(
        self,
        poses: t.List[Pose2D],
        start_index: int,
        horizon: int,
        robot_inflated_grid: BinaryOccupancyGrid,
        agent_id: str,
    ):

        if horizon <= 0:
            return

        horizon_cells: t.Set[GridCellModel] = set()
        for pose in poses[start_index : start_index + horizon]:
            cell = utils.real_to_grid(
                pose[0],
                pose[1],
                robot_inflated_grid.cell_size,
                robot_inflated_grid.grid_pose,
            )
            horizon_cells.add(cell)
        cube_list_marker = self._grid_cells_to_cube_list_markers(
            horizon_cells,
            robot_inflated_grid.cell_size,
            robot_inflated_grid.grid_pose,
            colors.flashy_green,
            z_index=cfg.horizon_markers_z_index,
            ns="/transit_horizon_cells",
        )
        marker_array = ros_nodes.MarkerArray(markers=[cube_list_marker])
        self.robot_conflict_check_publishers[agent_id].publish(marker_array)

    def publish_transit_conflicting_cells(
        self,
        conflicting_cells: t.Iterable[GridCellModel],
        robot_inflated_grid: BinaryOccupancyGrid,
        agent_id: str,
    ):

        cube_list_marker = self._grid_cells_to_cube_list_markers(
            conflicting_cells,
            robot_inflated_grid.cell_size,
            robot_inflated_grid.grid_pose,
            colors.flashy_red,
            z_index=cfg.conflicting_cells_z_index,
            ns="/transit_conflicting_cells",
        )
        marker_array = ros_nodes.MarkerArray(markers=[cube_list_marker])
        self.robot_conflict_check_publishers[agent_id].publish(marker_array)

    def publish_transit_conflicting_polygons_cells(
        self,
        conflicting_entities_cells: t.Iterable[GridCellModel],
        robot_inflated_grid: BinaryOccupancyGrid,
        agent_id: str,
    ):

        cube_list_marker = self._grid_cells_to_cube_list_markers(
            conflicting_entities_cells,
            robot_inflated_grid.cell_size,
            robot_inflated_grid.grid_pose,
            colors.flashy_cyan,
            z_index=cfg.conflict_markers_z_index,
            ns="/transit_conflicting_entities_cells",
        )
        marker_array = ros_nodes.MarkerArray(markers=[cube_list_marker])
        self.robot_conflict_check_publishers[agent_id].publish(marker_array)

    def publish_transfer_horizon_polygons(
        self,
        robot_polygons: t.List[Polygon],
        obstacle_polygons: t.List[Polygon],
        start_index: int,
        horizon: int,
        agent_id: str,
    ):

        horizon_csv_polygons = []
        for i in range(start_index, min(start_index + horizon, len(obstacle_polygons))):
            horizon_csv_polygons.append(robot_polygons[i])
            horizon_csv_polygons.append(obstacle_polygons[i])

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
        self.robot_conflict_horizon_publishers[agent_id].publish(marker_array)

    def publish_transfer_conflicting_intersections(self):
        pass

    def publish_transfer_conflicting_convex_polygons(self):
        pass

    def cleanup_swept_area(self, agent_id: str):

        self.robot_swept_area_publishers[agent_id].publish(
            self._make_delete_all_marker(cfg.main_frame_id)
        )

    def cleanup_conflict_horizon(self, agent_id: str):

        self.robot_conflict_horizon_publishers[agent_id].publish(
            self._make_delete_all_marker(cfg.main_frame_id)
        )

    def cleanup_conflicts_checks(self, agent_id: str):

        self.robot_conflict_check_publishers[agent_id].publish(
            self._make_delete_all_marker(cfg.main_frame_id)
        )

    # endregion

    # region CLEANUP ALL
    def cleanup_all(self):

        self.clear_world()
        for agent_id in self.agent_ids:
            self.cleanup_robot_observed_world(agent_id=agent_id)
            self.clear_robot_plan(agent_id=agent_id)
            self.clear_obstacle_grab_poses(agent_id=agent_id)
            self.clear_goal(agent_id=agent_id)
            self.clear_social_costmap(agent_id=agent_id)
            self.clear_combined_costmap(agent_id=agent_id)
            self.cleanup_conflicts_checks(agent_id=agent_id)
            self.cleanup_swept_area(agent_id=agent_id)
            self.cleanup_conflict_horizon(agent_id=agent_id)

    def init_header(self):

        return ros_nodes.Header(stamp=self.get_timestamp(), frame_id=cfg.main_frame_id)

    def _grid_cells_to_cube_list_markers(
        self,
        grid_cells: t.Iterable[GridCellModel],
        res: float,
        grid_pose: Pose2D,
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
                    frame_id=cfg.main_frame_id, stamp=self.get_timestamp()
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
        grid_pose: Pose2D,
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
                frame_id=cfg.main_frame_id, stamp=self.get_timestamp()
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
            header=ros_nodes.Header(frame_id=frame_id, stamp=self.get_timestamp()),
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
        pose: Pose2D,
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
            header=ros_nodes.Header(frame_id=frame_id, stamp=self.get_timestamp()),
            color=color,
        )
        return marker

    def _make_delete_marker(self, namespace: str, p_id: int, frame_id: str):
        return ros_nodes.Marker(
            ns=namespace,
            id=p_id,
            header=ros_nodes.Header(frame_id=frame_id, stamp=self.get_timestamp()),
            action=ros_nodes.Marker.DELETE,
        )

    def _make_delete_all_marker(self, frame_id: str, ns: str = ""):
        return ros_nodes.MarkerArray(
            markers=[
                ros_nodes.Marker(
                    ns=ns,
                    header=ros_nodes.Header(
                        frame_id=frame_id, stamp=self.get_timestamp()
                    ),
                    action=ros_nodes.Marker.DELETEALL,
                )
            ]
        )

    def _string_to_text_marker(
        self,
        message: str = "",
        pose: Pose2D = Pose2D(0.0, 0.0, 0.0),
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
            header=ros_nodes.Header(frame_id=frame_id, stamp=self.get_timestamp()),
            color=color,
            text=message,
        )
        return marker

    # endregion


def create_default_ros_publisher(
    node_name: str, agent_ids: t.List[str]
) -> RosPublisher:
    if DEACTIVATE_RVIZ:
        return RosPublisher(agent_ids=agent_ids, ros_node=None)
    node = ros_nodes.DefaultRosPublisherNode(
        RosPublisher.create_valid_node_name(node_name)
    )
    return RosPublisher(agent_ids=agent_ids, ros_node=node)
