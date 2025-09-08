from setuptools import find_packages, setup
from glob import glob

package_name = "namosim"


def read_requirements():
    with open("requirements.txt", "r") as file:
        return [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]


setup(
    name=package_name,
    version="0.0.3",
    packages=find_packages(
        exclude=["tests", "tests.*"]
    ),  # Exclude tests and subpackages
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include non-Python files like launch files if needed
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=read_requirements(),
    zip_safe=True,
    maintainer="David Brown",
    maintainer_email="davewbrwn@gmail.com",
    description="A mobile-robot motion planner for Navigation Among Obstacles (NAMO)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
