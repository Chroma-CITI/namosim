"""
This module contains classes describing robot and obstacle configurations used during the Stilman obstacle selection sub-routines.
"""

import typing as t
from abc import ABC

from shapely.geometry import Polygon

import namosim.navigation.basic_actions as ba
from namosim.algorithms import graph_search
from namosim.data_models import FixedPrecisionPose2D, GridCellModel, Pose2D


class BaseConfiguration(ABC):
    pass


class RCHConfiguration(BaseConfiguration):
    def __init__(
        self,
        cell: t.Tuple[int, int],
        first_obstacle_uid: str,
        first_component_uid: str,
    ):
        self.cell = cell
        self.first_obstacle_uid = first_obstacle_uid
        self.first_component_uid = first_component_uid

    def __eq__(self, other: object):
        if isinstance(other, tuple):
            return self.cell == other
        if isinstance(other, RCHConfiguration):
            return (
                self.cell == other.cell
                and self.first_obstacle_uid == other.first_obstacle_uid
                and self.first_component_uid == other.first_component_uid
            )
        raise Exception("Invalid comparison")

    def __hash__(self):
        return hash((self.cell, self.first_obstacle_uid, self.first_component_uid))


class RobotConfiguration(BaseConfiguration):
    def __init__(
        self,
        *,
        floating_point_pose: Pose2D,
        polygon: Polygon,
        cell_in_grid: t.Tuple[int, int],
        fixed_precision_pose: FixedPrecisionPose2D,
        action: ba.Action | None = None,
        csv_polygon: Polygon | None = None,
    ):
        self.floating_point_pose = floating_point_pose
        self.polygon = polygon
        self.cell_in_grid = cell_in_grid
        self.fixed_precision_pose = fixed_precision_pose
        self.action = action
        self.csv_polygon = csv_polygon
        self.bb_vertices = self.polygon.exterior.coords

    def __eq__(self, other: object) -> bool:
        if isinstance(other, graph_search.HeapNode):
            return self.fixed_precision_pose == other.element.fixed_precision_pose
        if isinstance(other, tuple):
            return self.fixed_precision_pose == other
        if isinstance(other, RobotConfiguration):
            return self.fixed_precision_pose == other.fixed_precision_pose
        raise Exception("Invalid comparison")

    def __hash__(self):
        return hash(self.fixed_precision_pose)


class RobotObstacleConfiguration(BaseConfiguration):
    def __init__(
        self,
        *,
        robot_pose: Pose2D,
        robot_polygon: Polygon,
        robot_cell_in_grid: GridCellModel,
        robot_fixed_precision_pose: FixedPrecisionPose2D,
        obstacle_floating_point_pose: Pose2D,
        obstacle_polygon: Polygon,
        obstacle_cell_in_grid: GridCellModel,
        obstacle_fixed_precision_pose: FixedPrecisionPose2D,
        manip_pose_id: int,
        action: ba.Action,
        prev_robot_pose: Pose2D,
        prev_robot_polygon: Polygon,
        robot_csv_polygon: Polygon | None = None,
        obstacle_csv_polygon: Polygon | None = None,
    ):
        self.robot = RobotConfiguration(
            floating_point_pose=robot_pose,
            polygon=robot_polygon,
            cell_in_grid=robot_cell_in_grid,
            fixed_precision_pose=robot_fixed_precision_pose,
            action=action,
            csv_polygon=robot_csv_polygon,
        )
        self.obstacle = RobotConfiguration(
            floating_point_pose=obstacle_floating_point_pose,
            polygon=obstacle_polygon,
            cell_in_grid=obstacle_cell_in_grid,
            fixed_precision_pose=obstacle_fixed_precision_pose,
            action=action,
            csv_polygon=obstacle_csv_polygon,
        )
        self.action = action
        self.manip_pose_id = manip_pose_id
        self.prev_robot_pose = prev_robot_pose
        self.prev_robot_polygon = prev_robot_polygon

    def __eq__(self, other: object) -> bool:
        if isinstance(other, graph_search.HeapNode):
            return (
                self.robot.fixed_precision_pose
                == other.element.robot.fixed_precision_pose
                and self.obstacle.fixed_precision_pose
                == other.element.obstacle.fixed_precision_pose
            )
        if isinstance(other, tuple):
            return (
                self.robot.fixed_precision_pose == other[0]
                and self.obstacle.fixed_precision_pose == other[1]
            )
        if isinstance(other, RobotObstacleConfiguration):
            return (
                self.robot.fixed_precision_pose == other.robot.fixed_precision_pose
                and self.obstacle.fixed_precision_pose
                == other.obstacle.fixed_precision_pose
            )

        raise Exception("Invalid comparison")

    def __hash__(self):
        return hash(
            (self.robot.fixed_precision_pose, self.obstacle.fixed_precision_pose)
        )
