from setuptools import find_packages, setup

package_name = "namosim"


def read_requirements():
    with open("requirements.txt", "r") as file:
        # Strip whitespace and ignore empty lines or comments
        return [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=read_requirements(),
    maintainer="chroma",
    maintainer_email="david.brown@inria.fr",
    description="TODO: Package description",
    license="MIT",
)
