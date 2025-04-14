import math
import typing as t

import numpy as np
import transforms3d
from geometry_msgs.msg import Point32, Polygon, Pose, PoseStamped
from scipy.spatial.transform import Rotation
from shapely import affinity
from shapely.geometry import Polygon as ShapelyPolygon
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TransformStamped


def euler_to_quat(
    roll: float, pitch: float, yaw: float
) -> t.Tuple[float, float, float, float]:
    """Converts euler angles in degrees to a quaternion tuple (x,y,z,w)"""
    rot = Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=True)
    quat = rot.as_quat()  # type: ignore
    return quat  # type: ignore


def quat_to_euler(
    quat: t.Tuple[float, float, float, float],
) -> t.Tuple[float, float, float]:
    """Converts quaternion to euler angles in degrees"""
    rot = Rotation.from_quat(quat)
    return rot.as_euler("xyz", degrees=True)  # type: ignore


def shapely_to_ros_polygon(shapely_polygon: ShapelyPolygon):
    ros_polygon = Polygon()
    ros_polygon.points = [
        Point32(x=float(p[0]), y=float(p[1]), z=0.0)  # type: ignore
        for p in shapely_polygon.exterior.coords  # type: ignore
    ]
    return ros_polygon


def ros_polygon_to_shapely_polygon(ros_polygon: Polygon):
    shapely_polygon = ShapelyPolygon([(p.x, p.y) for p in ros_polygon.points])
    return shapely_polygon


def scale_polygon(polygon: ShapelyPolygon, scale: float) -> ShapelyPolygon:
    scaled_polygon = affinity.scale(polygon, xfact=scale, yfact=scale, origin=(0, 0, 0))  # type: ignore
    return scaled_polygon  # type: ignore


class Pose2D(t.NamedTuple):
    x: float
    y: float
    degrees: float


def entity_pose_to_pose2d(p: Pose) -> Pose2D:
    """returns (x, y, z, yaw)"""
    orientation = quat_to_euler(
        (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)  # type: ignore
    )

    return Pose2D(p.position.x, p.position.y, orientation[2])  # type: ignore


def transform_pose(source: Pose2D, target: Pose2D) -> Pose2D:
    """
    Transforms a pose from source frame to target frame.
    """
    # Step 1: Create rotation matrix for source (rotation around z-axis)
    R_src = transforms3d.euler.euler2mat(
        0, 0, math.radians(source.degrees), axes="sxyz"
    )

    # Step 2: Create 4x4 transformation matrix for source (world to source)
    wTs = transforms3d.affines.compose(
        T=[source.x, source.y, 0],  # Translation
        R=R_src,  # Rotation
        Z=[1, 1, 1],  # No scaling
    )

    # Step 3: Create rotation matrix for target (rotation around z-axis)
    R_tgt = transforms3d.euler.euler2mat(
        0, 0, math.radians(target.degrees), axes="sxyz"
    )

    # Step 4: Create 4x4 transformation matrix for target (world to target)
    wTt = transforms3d.affines.compose(
        T=[target.x, target.y, 0],  # Translation
        R=R_tgt,  # Rotation
        Z=[1, 1, 1],  # No scaling
    )

    # Step 5: Compute transformation from source to target (sTt = inv(wTs) * wTt)
    sTt = np.matmul(np.linalg.inv(wTs), wTt)

    # Step 6: Extract position and quaternion
    # Position is the translation part (last column, first three rows)
    position = sTt[:3, 3]
    # Rotation matrix is the top-left 3x3
    R = sTt[:3, :3]
    # Convert rotation matrix to quaternion
    quaternion = transforms3d.quaternions.mat2quat(R)

    # Step 7: Convert quaternion to Euler angles
    angles = transforms3d.euler.quat2euler(quaternion, axes="sxyz")

    # Step 8: Return Pose2D with x, y, and angle in degrees
    return Pose2D(position[0], position[1], math.degrees(angles[2]))


def construct_ros_pose(
    x: float,
    y: float,
    z: float,
    theta: float,
    header: Header,
) -> PoseStamped:
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = float(z)

    (x, y, z, w) = euler_to_quat(0, 0, theta)  #
    p.orientation.x = x
    p.orientation.y = y
    p.orientation.z = z
    p.orientation.w = w

    stamped = PoseStamped()
    stamped.header = header
    stamped.pose = p
    return stamped


def normalize_angle(angle: float):
    return np.arctan2(np.sin(angle), np.cos(angle))


def transform_to_pose(transform: TransformStamped) -> PoseStamped:
    # Create a Pose message
    pose = PoseStamped()

    pose.header = transform.header

    # Copy translation to position
    pose.pose.position.x = transform.transform.translation.x
    pose.pose.position.y = transform.transform.translation.y
    pose.pose.position.z = transform.transform.translation.z

    # Copy rotation (quaternion) to orientation
    pose.pose.orientation.x = transform.transform.rotation.x
    pose.pose.orientation.y = transform.transform.rotation.y
    pose.pose.orientation.z = transform.transform.rotation.z
    pose.pose.orientation.w = transform.transform.rotation.w

    return pose


def get_distance2d(pose1: Pose2D, pose2: Pose2D) -> float:
    return math.sqrt((pose1.x - pose2.x) ** 2 + (pose1.y - pose2.y) ** 2)


def get_distance(pose1: PoseStamped, pose2: PoseStamped) -> float:
    return math.sqrt(
        (pose1.pose.position.x - pose2.pose.position.x) ** 2
        + (pose1.pose.position.y - pose2.pose.position.y) ** 2
        + (pose1.pose.position.z - pose2.pose.position.z) ** 2
    )
