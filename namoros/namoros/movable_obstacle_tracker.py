import typing as t
from collections import deque

import numpy as np
import shapely.geometry as geom
from aruco_markers_msgs.msg import Marker, MarkerArray
from tf2_geometry_msgs import (
    PoseStamped,
    TransformStamped,
    PointStamped,
    tf2_geometry_msgs,
)
import namoros.behavior_node as behavior_node
from namoros import utils
from namoros.config import Config
from namoros.utils import Pose2D
from namoros.data_models import ShapeEnum


class MovableObstacleTracker:
    def __init__(self, node: "behavior_node.NamoBehaviorNode"):
        self.node = node
        self.aruco_pose = self.node.create_subscription(
            MarkerArray, f"aruco/markers", self.markers_callback, 10
        )
        self.detected_movables: t.Dict[str, t.Deque[PoseStamped]] = {}
        self.newly_detected_obstacle_ids: t.Deque[str] = deque(
            maxlen=Config.MOVABLE_OBSTACLE_DETECTION_AVG_N
        )
        self.possible_marker_ids = set()
        for obs in self.node.namoros_config.obstacles:
            self.possible_marker_ids.add(obs.marker_id)

    def reset_obstacle(self, obstacle_id: str):
        self.detected_movables[obstacle_id] = deque(
            maxlen=Config.MOVABLE_OBSTACLE_DETECTION_AVG_N
        )

    def markers_callback(self, msg: MarkerArray):
        for marker in msg.markers:
            self.process_marker(marker)

    def process_marker(self, marker: Marker):
        if self.node.omniscient_obstacle_perception:
            return

        marker_id = str(marker.id)
        if marker_id not in self.possible_marker_ids:
            return

        # marker_pose = self._translate_aruco_pose(
        #     marker.pose, distance=Config.DEFAULT_OBSTACLE_RADIUS
        # )
        marker_pose = self.node.transform_pose(marker.pose, target_frame="map")
        if marker_pose is None:
            self.node.get_logger().warn("Failed to transform marker pose to map frame")
            return

        robot_pose = self.node.lookup_robot_pose()
        if robot_pose is None:
            self.node.get_logger().warn("Failed to lookup robot pose")
            return

        distance = utils.get_distance(marker_pose, robot_pose)
        if distance >= Config.MAX_MOVABLE_OBSTACLE_DETECTION_DISTANCE:
            return

        if marker_id not in self.detected_movables:
            self.detected_movables[marker_id] = deque(
                maxlen=Config.MOVABLE_OBSTACLE_DETECTION_AVG_N
            )
            self.newly_detected_obstacle_ids.append(marker_id)

        self.detected_movables[marker_id].append(marker_pose)

    def is_obstacle_fully_detected(self, marker_id: str) -> bool:
        if marker_id not in self.detected_movables:
            return False
        return (
            len(self.detected_movables[marker_id])
            == Config.MOVABLE_OBSTACLE_DETECTION_AVG_N
        )

    def get_averaged_marker_pose(self, marker_id: str) -> PoseStamped | None:
        if marker_id not in self.detected_movables:
            return None

        # Initialize accumulators for position and orientation
        pos_x, pos_y, pos_z = 0.0, 0.0, 0.0
        quat_x, quat_y, quat_z, quat_w = 0.0, 0.0, 0.0, 0.0

        # Iterate over the poses and sum their components
        pose_list = self.detected_movables[marker_id]
        num_poses = len(pose_list)
        if num_poses == 0:
            return

        for pose in pose_list:
            pos_x += pose.pose.position.x
            pos_y += pose.pose.position.y
            pos_z += pose.pose.position.z
            quat_x += pose.pose.orientation.x
            quat_y += pose.pose.orientation.y
            quat_z += pose.pose.orientation.z
            quat_w += pose.pose.orientation.w

        # Compute the averages

        avg_position = [pos_x / num_poses, pos_y / num_poses, pos_z / num_poses]
        avg_orientation = [
            quat_x / num_poses,
            quat_y / num_poses,
            quat_z / num_poses,
            quat_w / num_poses,
        ]

        # Normalize the averaged quaternion to make it valid
        norm = np.sqrt(sum(x**2 for x in avg_orientation))
        avg_orientation = [x / norm for x in avg_orientation]

        # Create the averaged PoseStamped
        avg_pose = PoseStamped()
        avg_pose.header.stamp = self.node.get_clock().now().to_msg()
        avg_pose.header.frame_id = pose_list[
            0
        ].header.frame_id  # Use frame_id of first pose
        assert avg_pose.header.frame_id == "map"

        (
            avg_pose.pose.position.x,
            avg_pose.pose.position.y,
            avg_pose.pose.position.z,
        ) = avg_position
        (
            avg_pose.pose.orientation.x,
            avg_pose.pose.orientation.y,
            avg_pose.pose.orientation.z,
            avg_pose.pose.orientation.w,
        ) = avg_orientation

        return avg_pose

    def _get_estimated_obstacle_polygon(self, marker_id: str) -> geom.Polygon:
        """
        Gets the obstacle footprint polygon in the frame of the marker
        """
        for obs in self.node.namoros_config.obstacles:
            if obs.marker_id == marker_id:
                if obs.shape == ShapeEnum.circle:
                    polygon = geom.Point(0, 0).buffer(obs.radius)
                    return polygon
                elif obs.shape == ShapeEnum.rectangle:
                    polygon = geom.Polygon(
                        [
                            [-obs.width / 2, 0],
                            [-obs.width / 2, -obs.length],
                            [obs.width / 2, -obs.length],
                            [obs.width / 2, 0],
                            [-obs.width / 2, 0],
                        ]
                    )
                    return polygon
        raise Exception(f"Failed to lookup obstacle polygon for marker id {marker_id}")

    def transform_polygon_to_map_frame(
        self, marker_id: str, marker_pose: PoseStamped, polygon: geom.Polygon
    ) -> geom.Polygon:
        transform = self.pose_to_transform(
            marker_pose,
            "map",
            f"aruco_marker_{marker_id}",  # Source frame
        )
        transformed_points = []
        for point in polygon.exterior.coords:
            transformed_point = self.transform_point(point, transform)
            transformed_points.append(transformed_point)
        if len(transformed_points) > 0:
            return geom.Polygon(transformed_points)
        raise Exception("Failed to transform polygon into the map frame")

    def transform_point(
        self, point: t.Tuple[float, float], transform: TransformStamped
    ) -> t.Tuple[float, float]:
        # Create a PointStamped object for the input point
        point_stamped = PointStamped()
        point_stamped.header.frame_id = transform.child_frame_id
        point_stamped.point.x = point[0]
        point_stamped.point.y = 0.0
        point_stamped.point.z = point[1]

        # Transform the point
        transformed_point_stamped = tf2_geometry_msgs.do_transform_point(
            point_stamped, transform
        )

        # Return the transformed point as a tuple
        return (
            transformed_point_stamped.point.x,
            transformed_point_stamped.point.y,
        )

    def pose_to_transform(
        self, pose: PoseStamped, parent_frame_id: str, child_frame_id: str
    ):
        transform = TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame_id
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = pose.pose.position.x
        transform.transform.translation.y = pose.pose.position.y
        transform.transform.translation.z = pose.pose.position.z
        transform.transform.rotation.x = pose.pose.orientation.x
        transform.transform.rotation.y = pose.pose.orientation.y
        transform.transform.rotation.z = pose.pose.orientation.z
        transform.transform.rotation.w = pose.pose.orientation.w
        return transform

    def update_obstacle_polygons(self):
        for marker_id in self.detected_movables.keys():
            marker_pose = self.get_averaged_marker_pose(marker_id)
            if marker_pose is None:
                continue
            polygon_in_marker_frame = self._get_estimated_obstacle_polygon(marker_id)
            polygon_in_map = self.transform_polygon_to_map_frame(
                marker_id, marker_pose, polygon_in_marker_frame
            )
            pose2d = Pose2D(polygon_in_map.centroid.x, polygon_in_map.centroid.y, 0.0)
            self.node.add_or_update_movable_ostable(
                uid=marker_id, pose=pose2d, polygon=polygon_in_map
            )

    def get_obstacle_polygon(self, marker_id: str) -> geom.Polygon | None:
        marker_pose = self.get_averaged_marker_pose(marker_id)
        if marker_pose is None:
            return
        polygon_in_marker_frame = self._get_estimated_obstacle_polygon(marker_id)
        polygon_in_map = self.transform_polygon_to_map_frame(
            marker_id, marker_pose, polygon_in_marker_frame
        )
        return polygon_in_map
