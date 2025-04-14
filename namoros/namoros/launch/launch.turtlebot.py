import os

from ament_index_python.packages import get_package_share_directory  # type: ignore
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution


def generate_launch_description() -> LaunchDescription:
    namo_config_arg = DeclareLaunchArgument(
        "namo_config", description="Path to a namo_planner yamo config file"
    )
    map_arg = DeclareLaunchArgument("map", description="Path to a map yaml config file")
    pkg_share = get_package_share_directory("namoros")

    # rviz
    default_rviz_config_path = os.path.join(pkg_share, "nav2_default_view.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config_path],
        parameters=[{"use_sim_time": False}],
    )

    # nav2
    nav2_bringup_share = get_package_share_directory("nav2_bringup")
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_bringup_share + "/launch/bringup_launch.py"),
        launch_arguments={
            "params_file": os.path.join(pkg_share, "params/nav2_params.yaml"),
            "map": LaunchConfiguration("map"),
            "use_sim_time": "False",
        }.items(),
    )

    # aruco_ros
    aruco_ros_share = get_package_share_directory("aruco_ros")
    aruco_ros = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            aruco_ros_share + "/launch/marker_publisher.launch.py"
        ),
        launch_arguments={
            "camera_frame": TextSubstitution(text="obot_0_camera_color_optical_frame"),
            "marker_size": TextSubstitution(text="0.1"),
            "reference_frame": TextSubstitution(text="base_link"),
        }.items(),
    )

    namoros = Node(
        package="namoros",
        namespace="namoros",
        executable="bt",
        name="namoros",
        parameters=[
            {
                "namo_config": LaunchConfiguration("namo_config"),
            }
        ],
    )
    return LaunchDescription(
        [
            map_arg,
            namo_config_arg,
            rviz_node,
            TimerAction(period=5.0, actions=[nav2_bringup, aruco_ros]),
            TimerAction(period=10.0, actions=[namoros]),
        ]
    )
