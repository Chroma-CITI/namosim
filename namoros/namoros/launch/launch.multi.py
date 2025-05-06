import logging
import os
import typing as t

from ament_index_python import get_package_prefix
import xacro
from ament_index_python.packages import get_package_share_directory  # type: ignore
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import Node
from namosim.world import world as _world
from launch.logging import get_logger


def spawn_robots(context: t.Any, *args, **kwargs):  # type: ignore
    logger = get_logger()
    pkg_share = get_package_share_directory("namoros")
    urdf_path = os.path.join(
        pkg_share,
        "models/turtlebot_description/robots/kobuki_hexagons_astra.urdf.xacro",
    )
    scenario_file = LaunchConfiguration("scenario_file").perform(context)
    w = _world.World.load_from_svg(scenario_file)
    robots = []
    for agent in w.agents.values():
        try:
            robot_description = xacro.process_file(
                urdf_path, mappings={"robot_name": agent.uid}
            )
        except Exception as e:
            logging.error("An error occurred: %s", e, exc_info=True)
            raise e

        robot_description_topic = f"/{agent.uid}/robot_description"
        joint_states_topic = f"/{agent.uid}/joint_states"
        logging.info(
            f"Spawning robot {agent.uid} at pose ({agent.pose[0], agent.pose[1], agent.pose[2]})"
        )
        robot_spawn = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-world",
                "namo_world",
                # "-file",
                # urdf_path,
                "-topic",
                robot_description_topic,
                "-name",
                agent.uid,
                "-x",
                str(agent.pose[0]),
                "-y",
                str(agent.pose[1]),
                "-Y",
                str(agent.pose[2]),
            ],
            output="screen",
        )
        robots.append(robot_spawn)

        node_robot_state_publisher = Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            namespace=f"/{agent.uid}",
            output="screen",
            parameters=[
                {
                    "robot_description": robot_description.toxml(),  # type: ignore
                    "use_sim_time": True,
                }
            ],
            arguments=[],
            remappings=[
                ("/robot_description", "robot_description"),
                ("/joint_states", "joint_states"),
                ("/tf", "tf"),
                ("/tf_static", "tf_static"),
            ],
        )
        robots.append(node_robot_state_publisher)

        namo_planner = Node(
            package="namoros",
            executable="namo_planner",
            name="namo_planner",
            namespace=f"/{agent.uid}",
            output="screen",
            parameters=[
                {
                    "scenario_file": LaunchConfiguration("scenario_file"),
                    "omniscient_obstacle_perception": LaunchConfiguration(
                        "omniscient_obstacle_perception"
                    ),
                    "agent_id": agent.uid,
                }
            ],
            remappings=[
                ("/tf", "tf"),
                ("/tf_static", "tf_static"),
            ],
        )

        namoros = Node(
            package="namoros",
            executable="namoros_bt",
            name=f"namoros_bt",
            namespace=f"/{agent.uid}",
            output="screen",
            parameters=[
                {
                    "scenario_file": LaunchConfiguration("scenario_file"),
                    "config_file": LaunchConfiguration("config_file"),
                    "omniscient_obstacle_perception": LaunchConfiguration(
                        "omniscient_obstacle_perception"
                    ),
                    "agent_id": agent.uid,
                    "is_sim": True,
                    "use_sim_time": True,
                }
            ],
            remappings=[
                ("/tf", "tf"),
                ("/tf_static", "tf_static"),
            ],
        )

        # rviz
        default_rviz_config_path = os.path.join(pkg_share, "nav2_default_view.rviz")
        rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            namespace=f"/{agent.uid}",
            name=f"rviz2",
            output="screen",
            arguments=["-d", default_rviz_config_path],
            parameters=[{"use_sim_time": True}],
            remappings=[
                ("/tf", "tf"),
                ("/tf_static", "tf_static"),
                ("/map", "map"),
                ("/initialpose", "initialpose"),
                ("/trajectories", "trajectories"),
                ("/plan", "plan"),
                ("/goal_pose", "goal_pose"),
                ("/scan", "scan"),
                ("/global_costmap/costmap", "global_costmap/costmap"),
                ("/local_costmap/costmap", "local_costmap/costmap"),
                (
                    "/global_costmap/published_footprint",
                    "global_costmap/published_footprint",
                ),
                (
                    "/local_costmap/published_footprint",
                    "local_costmap/published_footprint",
                ),
                ("/current_namo_path", "current_namo_path"),
                ("/namosim/dynamic_entities", "namosim/dynamic_entities"),
                ("/namosim/text", "namosim/text"),
                (
                    f"/namosim/combined_costmap",
                    f"namosim/{agent.uid}/combined_costmap",
                ),
                (
                    f"/namosim/social_costmap",
                    f"namosim/{agent.uid}/social_costmap",
                ),
                ("/namosim/robot_0/manip_search", "namosim/robot_0/manip_search"),
                ("/namosim/robot_1/manip_search", "namosim/robot_1/manip_search"),
            ],
        )

        # nav2
        nav2_bringup_share = get_package_share_directory("nav2_bringup")
        nav2_bringup = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                nav2_bringup_share + "/launch/bringup_launch.py"
            ),
            launch_arguments={
                "namespace": f"/{agent.uid}",
                "use_namespace": "True",
                "params_file": os.path.join(pkg_share, "config/nav2_params.yaml"),
                "map": LaunchConfiguration("map_yaml"),
                "use_sim_time": "True",
            }.items(),
        )

        # aruco_markers
        aruco_markers = Node(
            package="aruco_markers",
            executable="aruco_markers",
            name="aruco_markers",
            namespace=f"/{agent.uid}",
            output="screen",
            parameters=[
                {
                    "marker_size": 0.1,
                    "camera_frame": f"camera_rgb_optical_frame",
                    "image_topic": f"/{agent.uid}/camera/color/image_raw",
                    "camera_info_topic": f"/{agent.uid}/camera/color/camera_info",
                    "dictionary": "DICT_ARUCO_ORIGINAL",
                    "use_sim_time": True,
                }
            ],
            remappings=[
                ("/tf", "tf"),
                ("/tf_static", "tf_static"),
            ],
        )

        scan_relay = Node(
            package="topic_tools",
            executable="relay",
            name="scan_relay",
            namespace=f"/{agent.uid}",
            arguments=[f"scan", f"local_costmap/scan"],
            output="screen",
        )

        robots += [
            rviz_node,
            scan_relay,
            TimerAction(period=0.0, actions=[nav2_bringup]),
            TimerAction(period=0.0, actions=[aruco_markers]),
            TimerAction(period=5.0, actions=[namo_planner, namoros]),
        ]
    return robots


def spawn_obstacles(context: t.Any, *args, **kwargs):  # type: ignore
    scenario_file = LaunchConfiguration("scenario_file").perform(context)
    w = _world.World.load_from_svg(scenario_file)
    agent_ids = list(w.agents.values())
    robot_start_pose = agent_ids[0].pose
    pkg_share = get_package_share_directory("namoros")

    urdfs = [
        os.path.join(pkg_share, "models/movable_box/model_300.urdf"),
        os.path.join(pkg_share, "models/movable_box/model_301.urdf"),
    ]

    actions = []
    for i, obstacle in enumerate(w.get_movable_obstacles()):
        obs_node = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-world",
                "namo_world",
                "-file",
                urdfs[i],
                "-name",
                obstacle.uid,
                "-x",
                str(obstacle.pose[0]),
                "-y",
                str(obstacle.pose[1]),
                "-z",
                "0.5",
            ],
            output="screen",
        )
        actions.append(obs_node)

    return actions


def namo_planner_bringup(context: t.Any, *args, **kwargs):  # type: ignore
    pkg_share = get_package_share_directory("namoros")
    plugin_prefix = get_package_prefix("namoros_gz")
    plugin_path = os.path.join(plugin_prefix, "lib")
    default_rviz_config_path = os.path.join(pkg_share, "nav2_default_view.rviz")
    bridge_config_path = os.path.join(pkg_share, "ros_gz_bridge.yaml")
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{"use_sim_time": True, "config_file": bridge_config_path}],
    )

    gazebo = ExecuteProcess(
        cmd=[
            "ign",
            "gazebo",
            "-r",
            "--render-engine",
            "ogre",
            "-v",
            "4",
            LaunchConfiguration("sdf_file"),
        ],
        output="screen",
        additional_env={
            "GZ_SIM_SYSTEM_PLUGIN_PATH": plugin_path,
            "IGN_GAZEBO_RESOURCE_PATH": pkg_share,
        },
    )
    gz_cmds = [ros_gz_bridge, gazebo]

    return [
        *gz_cmds,
    ]


def generate_launch_description() -> LaunchDescription:
    declare_sdf_arg = DeclareLaunchArgument(
        "sdf_file", description="Path to gazebo sdf world file"
    )
    declare_scenario_arg = DeclareLaunchArgument(
        "scenario_file", description="Path to a namo_planner scenario file"
    )
    declare_config_arg = DeclareLaunchArgument(
        "config_file", description="Path to a namoros config yaml file"
    )
    declare_map_arg = DeclareLaunchArgument(
        "map_yaml", description="Path to map yaml file"
    )
    declare_perception_arg = DeclareLaunchArgument(
        "omniscient_obstacle_perception",
        default_value="false",
        choices=["true", "false"],
    )
    namo_planner_bringup_launch = OpaqueFunction(function=namo_planner_bringup)

    ld = LaunchDescription()
    ld.add_action(declare_sdf_arg)
    ld.add_action(declare_scenario_arg)
    ld.add_action(declare_config_arg)
    ld.add_action(declare_map_arg)
    ld.add_action(declare_perception_arg)
    ld.add_action(namo_planner_bringup_launch)

    ld.add_action(OpaqueFunction(function=spawn_robots))
    ld.add_action(OpaqueFunction(function=spawn_obstacles))

    return ld
