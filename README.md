[![CI](https://github.com/Chroma-CITI/namosim/actions/workflows/ci.yml/badge.svg?branch=humble)](https://github.com/Chroma-CITI/namosim/actions/workflows/ci.yml)

# NAMOSIM

![NAMO Simulator](docs/source/_static/namo.gif)

NAMOSIM is a robot motion planner designed for the problem of navigation among movable obstacles (NAMO). It simulates mobile robots navigating in 2D polygonal environments in which certain obstacles can be grabbed and relocated. It currently supports **holonomic** and **differential drive** motion models. A variety of agent types are implemented, including primarily our **Stilman2005** baseline agent. New agents utilizing alternative algorithmic approaches can be created and plugged into the planner in a straightforward manner by implementing the **Agent** base class.

## Documentation

Please check out the [docs site](https://chroma-citi.github.io/namosim/) for installation and usage instructions.

To build the docs site locally, run:

```bash
./scripts/make_docs.sh
```

# Demos

Here are a couple demo videos applying namosim on real and simulated robots.

#### NAMOSIM on a Turtlebo

[![NAMOSIM on Turtlebot](docs/source/_static/namo_demo_thumbnail.png)](https://youtu.be/076ecBfaBTw)


#### NAMOSIM on Multiple Robots in Gazebo

[![NAMOSIM on Multiple Robots in Gazebo](docs/source/_static/multi_robot_demo_thumbnail.png)](https://youtu.be/qgPz69Dk9bc)

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

If you reuse any part of this project in your research, please cite the associated papers:

```bibtex
@inproceedings{renault_2024_iros,
  author    = {Renault, Benoit and Saraydaryan, Jacques and Brown, David and Simonin, Olivier},
  booktitle = {2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title     = {Multi-Robot Navigation Among Movable Obstacles: Implicit Coordination to Deal with Conflicts and Deadlocks},
  year      = {2024},
  volume    = {},
  number    = {},
  pages     = {3505-3511},
  keywords  = {Machine learning algorithms;Costs;Navigation;Robot kinematics;Machine learning;System recovery;Benchmark testing;Multi-robot systems;Intelligent robots},
  doi       = {10.1109/IROS58592.2024.10802092}
}
```

```bibtex
@inproceedings{renault_2020_iros,
  title     = {Modeling a Social Placement Cost to Extend Navigation Among Movable Obstacles (NAMO) Algorithms},
  author    = {Renault, Benoit and Saraydaryan, Jacques and Simonin, Olivier},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  address   = {Las Vegas, United States},
  year      = {2020},
  month     = {October},
  pages     = {11345--11351},
  doi       = {10.1109/IROS45743.2020.9340892},
  url       = {https://hal.archives-ouvertes.fr/hal-02912925},
  pdf       = {https://hal.archives-ouvertes.fr/hal-02912925/file/IROS_2020_Camera_Ready.pdf}
}
```
