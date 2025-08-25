[![CI](https://github.com/Chroma-CITI/namosim/actions/workflows/ci.yml/badge.svg?branch=humble)](https://github.com/Chroma-CITI/namosim/actions/workflows/ci.yml)

# NAMOSIM

![NAMO Simulator](docs/source/_static/namo.gif)

NAMOSIM is a robot motion planner designed for the problem of navigation among movable obstacles (NAMO). It simulates mobile robots navigating in 2D polygonal environments in which certain obstacles can be grabbed and relocated. It currently supports **holonomic** and **differential drive** motion models. A variety of agent types are implemented, including primarily our **Stilman2005** baseline agent. New agents utilizing alternative algorithmic approaches can be created and plugged into the planner in a straightforward manner by implementing the **Agent** base class.

## System Requirements

- Ubuntu 22.04
- ROS2 Humble

## Setup

First, clone the repo and `cd` into it.

Next, use `rosdep` to install the dependencies listed in the `package.xml` file:

```bash
rosdep install --from-paths . -r -y
```

Some dependencies are only available as pip packages. Please install them with:

```bash
pip install -r requirements.txt
```

## Usage

You can launch a scenario as follows

```bash
python3 -m namosim.main run tests/scenarios/minimal_stilman_2005.svg
```

The best way to experiment with other scenarios is to open the repo in VSCode and use the python test explorer to run the `e2e` tests.

Alternatively you can launch a test from the command line like so:

```bash
python3 -m pytest tests/e2e/e2e_test.py::TestE2E::test_social_dr_success_d
```

### Run a Basic Scenario and Visualize in RVIZ

The following example runs the most basic scenario with the (Stillman,2005) algorithm and assumes you have `ROS2` and `RViz2` installed.

Start rviz2:

```bash
rviz2 -d rviz/basic_view.rviz
```

Then, in a new terminal, run:

```bash
python3 -m pytest tests/e2e/e2e_test.py::TestE2E::test_social_dr_success_d
```

## Run Unit Tests

```bash
./scripts/test_unit.sh
```

## Documentation

You can find the docs site [here](https://chroma-citi.github.io/namosim/).

To build the docs site locally, run:

```bash
./scripts/make_docs.sh
```

## Authors

- Benoit Renault
- Jacques Saraydaryan
- David Brown
- Olivier Simonin

## Affiliated Teams and Organisations

|                                                 | Org/Team                                      |
| ----------------------------------------------- | --------------------------------------------- |
| ![Inria Logo](docs/source/_static/inria.png)    | [Inria](https://inria.fr/fr)                  |
| ![INSA Lyon Logo](docs/source/_static/insa.png) | [INSA Lyon](https://www.insa-lyon.fr/)        |
| ![CITI Logo](docs/source/_static/citi.png)      | [CITI Laboratory](https://www.citi-lab.fr/)   |
| CHROMA                                          | [CHROMA Team](https://www.inria.fr/en/chroma) |

## Cite Us

If you reuse any of the provided data/code, please cite the associated papers:

```bibtex
@inproceedings{renault:hal-04705395,
  TITLE = {{Multi-Robot Navigation among Movable Obstacles: Implicit Coordination to Deal with Conflicts and Deadlocks}},
  AUTHOR = {Renault, Benoit and Saraydaryan, Jacques and Brown, David and Simonin, Olivier},
  URL = {https://hal.science/hal-04705395},
  BOOKTITLE = {{IROS 2024 - IEEE/RSJ International Conference on Intelligent Robots and Systems}},
  ADDRESS = {Abu DHABI, United Arab Emirates},
  PUBLISHER = {{IEEE}},
  PAGES = {1-7},
  YEAR = {2024},
  MONTH = Oct,
  KEYWORDS = {Planning ; Scheduling and Coordination ; Path Planning for Multiple Mobile Robots or Agents ; Multi-Robot Systems},
  PDF = {https://hal.science/hal-04705395v1/file/IROS24_1134_FI.pdf},
  HAL_ID = {hal-04705395},
  HAL_VERSION = {v1},
}
```

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
