# NAMOSIM

A simulator for NAMO problems. NAMO is an acronym for Navigation Among Movable Obstacles.

![NAMO Simulator](docs/source/_static/namo-sim.jpg)

## System Requirements

- Python >=3.10,<3.13
- ROS2 (we have tested ros-iron but others may work too)
- RVIZ2

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

### Install ROS packages

```
sudo apt update
sudo apt install ros-iron-grid-map
```

## Examples

### IROS 2024 Experiments

Generate the scenarios:

```bash
./scripts/generate_scenarios_intersections.sh
```

Launch the experiments:

```bash
./scripts/launch_experiments_intersections.sh
```

Results should be saved in the `namo_logs` folder. You can transform the results into a single csv file with:

```bash
python -m namosim.scripts.generate_results_table --results-dir namo_logs/intersections --out report_intersections.csv
```

### Run a Basic Scenario and Visualize in RVIZ

The following example runs the most basic scenario with the (Stillman,2005) algorithm and assumes you have `ros2` and `rviz2` installed.

Start rviz2:

```
rviz2 -d rviz/ROS2/basic_view.rviz
```

Then, in a new terminal, run:

```
python -m tests.unit.basic_senarios_test BasicTest.test_social_dr_success_d
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

## Credits

If you reuse any of the provided data/code, please cite the associated paper:

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
