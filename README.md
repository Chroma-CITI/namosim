# S-NAMO Sim

S-NAMO simulator, scenarios data and algorithms.

**Please have a look at the [wiki](https://gitlab.inria.fr/brenault/s-namo-sim/-/wikis/home) for more results ([such as interactive graphs](https://gitlab.inria.fr/brenault/s-namo-sim/-/wikis/uploads/interactive_stats.zip)).** Raw logs from experiments are accessible on demand from the author following the instructions below (because of a size of several gigabytes).

*THIS IS AN ACTIVE WORK IN PROGRESS, DO NO HESITATE TO OPEN AN ISSUE / CONTACT THE AUTHOR (firstname.lastname@insa-lyon.fr, replace with Benoit Renault) IF YOU ENCOUNTER ANY TROUBLE.*

## System Requirements

* Python >=3.9,<3.13
* ROS2 (we have tested ros-iron but others may work too)
* RVIZ2

## Quickstart

Download the repo (there is a submodule with the ICRA2022 paper submission scenarios data, if you don't want it for now, don't use the --recurse-submodules option, it will save you about 250Mb) :

```bash
git clone --recurse-submodules https://gitlab.inria.fr/brenault/s-namo-sim-private.git
```

Install the python dependencies with:

```bash
pip install .
```

If using [poetry](https://python-poetry.org/) (recommended), install with:
```bash
poetry install
```
And remember to activate the poetry environment:
```bash
poetry shell
```

You should be all set to start experimenting ! Individual experiments can be easily launched through python tests, like in the command below that should launch all scenarios presented in our ICRA2022 paper submission:

```bash
python3 ~/s-namo-sim-private/snamosim/tests/integration_tests/s-namo_cases/iros_2021.py IROS2021Tests.test_for_10_hours 0 199
```

Results should be saved in the 'logs' folder that is automatically created the first time in the repository folder. 
To get the full visual feedback, please install ROS1 and RVIZ (+ the grid-map ROS package). ROS2 support and independent visualization capabilities are a work ni progress.

> If you want to edit the ROS compatibility layer using Pycharm, you may want to follow [this very good tutorial](https://www.youtube.com/watch?v=lTew9mbXrAs) as to how to properly setup the IDE so that it finds the ROS python files correctly.

## Example

The following example runs the most basic scenario with the (Stillman,2005) algorithm and assumes you have `ros2` and `rviz2` installed.

Start rviz2:

```
rviz2 -d rviz/ROS2/basic_view.rviz
```

Then, in a new terminal, run:
```
python -m snamosim.tests.integration_tests.namo-socials.basic_with_opening_test BasicWithOpeningTest.test_stilman_2005_behavior
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
