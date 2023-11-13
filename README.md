# NAMOSIM

A simulator for NAMO problems. NAMO is an acronym for Navigation Among Movable Obstacles.

## System Requirements

* Python >=3.9,<3.13
* ROS2 (we have tested ros-iron but others may work too)
* RVIZ2

## Quickstart

This project uses [poetry](https://python-poetry.org/) for packaging and dependency management. If you
don't already have it, please install it before proceeding.

Install dependencies:

```bash
poetry install
```

Activate the poetry environment:
```bash
poetry shell
```

You should be all set to start.

## Examples

### IROS 2021 Experiments

The following command should launch all scenarios presented in our ICRA2022 paper submission.

First fetch the git submodule containing the IROS 2021 data:

```bash
git submodule update --init
git pull --recurse-submodules
```

```bash
python -m namosim.tests.integration_tests.namo-socials.iros_2021 IROS2021Tests.test_for_10_hours 0 199
```

Results should be saved in the 'logs' folder that is automatically created the first time in the repository folder. 
To get the full visual feedback, please install ROS2 and RVIZ. You will also need to install the grid-map ROS package if you don't already have it (e.g. `sudo apt install ros-iron-grid-map`).

### Run a Basic Scenario and Visualize in RVIZ

The following example runs the most basic scenario with the (Stillman,2005) algorithm and assumes you have `ros2` and `rviz2` installed.

Start rviz2:

```
rviz2 -d rviz/ROS2/basic_view.rviz
```

Then, in a new terminal, run:
```
python -m namosim.tests.integration_tests.namo-socials.basic_with_opening_test BasicWithOpeningTest.test_stilman_2005_behavior
```

## Run Unit Tests

```bash
poetry run poe test
```

## Credits

If you reuse (even partially) of the provided data/code, please do cite the associated paper:

```bibtex
@inproceedings{renault:hal-02912925,
  TITLE = {{Modeling a Social Placement Cost to Extend Navigation Among Movable Obstacles (NAMO) Algorithms}},
  AUTHOR = {Renault, Benoit and Saraydaryan, Jacques and Simonin, Olivier},
  URL = {https://hal.archives-ouvertes.fr/hal-02912925},
  BOOKTITLE = {{IROS 2020 - IEEE/RSJ International Conference on Intelligent Robots and Systems}},
  ADDRESS = {Las Vegas, United States},
  SERIES = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) Conference Proceedings},
  PAGES = {11345-11351},
  YEAR = {2020},
  MONTH = Oct,
  DOI = {10.1109/IROS45743.2020.9340892},
  KEYWORDS = {Navigation Among Movable Obstacles (NAMO) ; Socially- Aware Navigation (SAN) ; Path planning ; Simulation},
  PDF = {https://hal.archives-ouvertes.fr/hal-02912925/file/IROS_2020_Camera_Ready.pdf},
  HAL_ID = {hal-02912925},
  HAL_VERSION = {v1},
}
```

## Contributing

To contribute to this project, please make your changes in a new branch and open a merge request when ready. Don't forget to format the code with:

```
ruff format .
```