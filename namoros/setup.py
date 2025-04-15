import glob
import os

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

package_name = "namoros"


data_files: list[tuple[str, list[str]]] = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    (
        "share/" + package_name,
        [
            "package.xml",
            "ros_gz_bridge.yaml",
            "rviz/urdf_config.rviz",
            "rviz/nav2_default_view.rviz",
            "namoros/launch/launch.multi.py",
            "namo_world_template.sdf",
        ],
    ),
    (
        "share/" + package_name + "/config",
        [
            "rviz/nav2_default_view.rviz",
        ],
    ),
]

for filepath in glob.glob("config/**/*", recursive=True):
    if os.path.isfile(filepath):
        data_files.append(
            (os.path.join("share", package_name, os.path.dirname(filepath)), [filepath])
        )


for filepath in glob.glob("models/**/*", recursive=True):
    if os.path.isfile(filepath):
        data_files.append(
            (os.path.join("share", package_name, os.path.dirname(filepath)), [filepath])
        )

for filepath in glob.glob("maps/**/*", recursive=True):
    if os.path.isfile(filepath):
        data_files.append(
            (os.path.join("share", package_name, os.path.dirname(filepath)), [filepath])
        )

for filepath in glob.glob("params/**/*", recursive=True):
    if os.path.isfile(filepath):
        data_files.append(
            (os.path.join("share", package_name, os.path.dirname(filepath)), [filepath])
        )

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,  # type: ignore
    install_requires=requirements,
    zip_safe=True,
    maintainer="chroma",
    maintainer_email="david.brown@inria.fr",
    description="A ROS package for Navigation Among Movable Obstacles (NAMO)",
    license="AGPLv3",
    entry_points={
        "console_scripts": [
            "namo_planner = namoros.planner_node:main",
            "namoros_bt = namoros.bt:main",
            "namoros = namoros.main_real:main",
            "scenario2sdf = namoros.scripts.scenario2sdf:app",
        ],
    },
)
