# NAMOSIM

NAMOSIM is a robot motion-planning simulator designed for the problem of navigation among movable obstacles (NAMO).

![NAMO Simulator](docs/source/_static/namo.gif)

## System Requirements

- Python 3.10
- ROS2 (we have tested ros-humble but others may work too)
- RVIZ2

You might also need the following apt packages

```bash
sudo apt install python3-tk
sudo apt install libcairo2-dev
sudo apt install libopencv-dev
sudo apt install ros-humble-grid-map
```

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

The best way is to open the repo in VSCode and use the pythong test explorer to run the `e2e` tests.

Alternativley you can launch a test from the command line like so:
```bash
pytest tests/e2e/e2e_test.py::TestE2E::test_social_dr_success_d
```

### Run a Basic Scenario and Visualize in RVIZ

The following example runs the most basic scenario with the (Stillman,2005) algorithm and assumes you have `ros2` and `rviz2` installed.

Start rviz2:

```
rviz2 -d rviz/ROS2/basic_view.rviz
```

Then, in a new terminal, run:

```
pytest tests/e2e/e2e_test.py::TestE2E::test_social_dr_success_d
```

## Run Unit Tests

```bash
poetry run poe test
```

## Documentation

You can find the docs site [here](https://chroma.gitlabpages.inria.fr/namo/namosim/).

To build the docs site locally, run:

```bash
./scripts/make_docs.sh
```

The poetry shell will need to be activated.

## Authors

* Benoit Renault
* Jacques Saraydaryan
* David Brown
* Olivier Simonin

## Affiliated Teams and Organisations

|          | Org/Team |
|----------|----------|
|  ![Inria Logo](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAM1BMVEXiJCfrVmb97e/+9vfoQ1P5zdLlMD/1qLDxi5btaXf////84+bzmqT4wsj72dzveof2tbwOBJyHAAAAXUlEQVR4AY1MNwKAIBALXFFC/f9r9SbLRKZ07CNl0bdWMf/o4/z2RcpX868/A69sceNIjoCxAxg04QhdmB3o5Bg8wqj3oQ5KO9k9HHJ2oSQIlyxEOfdaE5DFCjZwAf9lAphlCmBfAAAAAElFTkSuQmCC)    | [Inria](https://inria.fr/fr)   |
|  ![INSA Lyon Logo](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAh1BMVEX////yZ2LyYFv+8/L1g3/0d3L++Pj97Ovza2b3o6HxVU/uMy3vOTP6wL795uXyXlns3NyBf3+HhITuKSLxUUvuLij84N/uLCX5u7n2mJX/zsxWU1P6xcT5s7H0fXnwQzz70dDwSELEyMj2jYry+PjxWVTz0dBnZWX4q6l3eHiZm5vGuLf4qKWVXt2FAAAA80lEQVR4Ae2PR2LCQBAEe5WzbErLkJNM5v/fcxb8gBN17qgXfzgXhJFzcZJmziVJXpRVUEtqWufc2/tIQNqBb8YGivllIk0BZvP5IGAx9lgDtlxBJxVg5KOHIE49VsM62UCpcM12i9s9BPwL6EN903hWDv9xF+xZeiwEsEjSApYRHI6D4ETmsWTjAaqNImhy6M+DYFZWhoVKMgNKVfhwAtvLIBivAUt+2jPwM7BuC+V1EDQxYD1stQGL+MMOd8EMsBuUiwh8C+uo97C8V+j/6i/ZHnopgFj7fRfG+y5Vv9/vk2a19lVbJ25f1NKp2Gd6Fi++AMY8G/1/xJtEAAAAAElFTkSuQmCC)    | [INSA Lyon](https://www.insa-lyon.fr/)  |
| ![CITI Logo](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAS1BMVEVHcEw9sMg8rsU8rMI8rcQ8rsQ9scg8rsY9sMhDsXo9r8ZTullUvlo8rcQ9sMdBtUhLuFJwxHXk8uWW0pnB48L///88rcRUvVpUvVqXMmfmAAAAGXRSTlMATbv/96YqzmURkP/46Hv/////////3b2xKLx8LgAAASRJREFUeAF8kAe2qCAMRAMjgeAXscD+l/oT6+tji+emDfTKeWAITD8pDkmIM7LGMv4zPWzyJCUkkiEbPHQzjlQQcgQL+As0DBuXQHxWztXgKzeJoQMu61Y/wQlJQUoK573tX6EjKsg/Qlf0FZCOtvszU+T1e247z+MJywAMRe1EnjrQexjHdT0rI3rJmCghJFgecl3aOhsMCGfjjGJ2o/7Ure0GHTqd8nBk5QZtWYUd7nHCdJbLvLZNIWOgS8BdbnBRmOCfw53ucqmtVbWSkU9EDv4qd7K0Zj7DCTO4WMSnpa2tLwzw+iQtMq9R1Im1LZrKXRl1m1YwaCx72wwKTOVeNh+x6LIGib0vYkeULSNFtnjfRxX9KKn1D/h/QBlaSIpLgAAAMWYQImmdxu4AAAAASUVORK5CYII=)    | [CITI Laboratory](https://www.citi-lab.fr/)   |
|  CHROMA   | [CHROMA Team](https://www.inria.fr/en/chroma)   |

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

## Contributing

To contribute to this project, please make your changes in a new branch and open a merge request when ready. Don't forget to run the lint checks, type checks, and unit tests:

```bash
poetry run poe all_checks
```
