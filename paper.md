---
title: 'NAMOSIM: A Robot Motion Planner for Navigation Among Movable Obstacles'
tags:
  - robotics
  - motion planning
  - NAMO
  - ROS2
  - simulation
  - path planning
authors:
  - name: Benoit Renault
    orcid: 0000-000X-XXXX-XXXX  # Replace with actual ORCID
    affiliation: "2 4"
  - name: Jacques Saraydaryan
    orcid: 0000-000X-XXXX-XXXX  # Replace with actual ORCID
    affiliation: "1 3 4"
  - name: David Brown
    orcid: 0000-000X-XXXX-XXXX  # Replace with actual ORCID
    affiliation: "1 4"
  - name: Olivier Simonin
    orcid: 0000-000X-XXXX-XXXX  # Replace with actual ORCID
    affiliation: "1 2 4"
affiliations:
  - name: Inria, CHROMA Team
    index: 1
  - name: INSA Lyon
    index: 2
  - name: CPE Lyon
    index: 3
  - name: CITI Laboratory
    index: 4
date: 2025-07-07
repository: https://gitlab.inria.fr/chroma/namo/namosim
archive: 10.5281/zenodo.XXXXXXX  # Replace with actual Zenodo DOI after archiving
license: MIT
---

![NAMOSIM](docs/source/_static/namosim_example.png)

# Summary

*NAMOSIM* is a robot motion planning simulator designed for the Navigation Among Movable Obstacles (NAMO) problem. It enables simulation and evaluation of motion planning strategies in 2D environments where certain obstacles can be manipulated by robots to reach their goals. The simulator includes support for holonomic and differential drive models, and integrates with ROS2 for seamless use in both simulated and real robotic platforms.

NAMOSIM provides a modular agent-based architecture, including a baseline implementation of the well-known Stilman2005 NAMO planner. New planning strategies can be implemented via a clean `Agent` interface, facilitating experimentation and benchmarking. The system includes full ROS2 compatibility, allowing visualization of plans using RViz2, as well as utilities for testing and documentation generation.

This software is intended for researchers, educators, and developers working on robot navigation in dynamic environments, particularly where physical interaction with the environment is necessary.

# Statement of need

Most motion planning frameworks assume static environments, limiting their usefulness in cluttered or semi-structured domains. NAMO problems introduce the need for reasoning about which obstacles to move, where to move them, and how to coordinate motion and manipulation. NAMOSIM addresses this by offering a simulation platform explicitly designed to study and prototype NAMO-capable robots.

This package fills a gap in current robotics tooling by supporting both the simulation and real-time deployment of NAMO algorithms in ROS2, with full extensibility for research and teaching use cases.

# Major Features

NAMOSIM provides a robust set of features to support research and development in Navigation Among Movable Obstacles (NAMO):

- **Modular Agent-Based Architecture**: The simulator is built around a flexible `Agent` interface, allowing users to implement and test custom NAMO planning algorithms. A baseline implementation of the Stilman2005 planner is included for immediate use and benchmarking.
- **Support for Multiple Robot Models**: NAMOSIM supports both holonomic and differential drive robot models, enabling realistic simulation of various robotic platforms.
- **ROS2 Integration**: Full compatibility with ROS2 allows seamless deployment on both simulated and physical robots, with built-in support for visualization in RViz2 and integration with ROS2 navigation stacks.
- **2D Environment Simulation**: The simulator provides a customizable 2D environment where users can define static and movable obstacles, supporting complex scenarios for testing navigation and manipulation strategies.
- **Extensive Testing and Documentation Utilities**: NAMOSIM includes tools for automated testing of planning algorithms and generating comprehensive documentation, facilitating reproducible research and educational use.
- **Multi-Robot Coordination**: The simulator supports multi-robot scenarios, enabling the study of implicit coordination and conflict resolution in NAMO tasks, as explored in related research (Renault et al., 2024).

These features make NAMOSIM a versatile tool for prototyping, evaluating, and deploying NAMO algorithms in diverse robotic applications.

# Acknowledgements

This project was developed by the CHROMA team at the CITI Laboratory, INSA Lyon, in collaboration with Inria. It is part of ongoing research into autonomous navigation, multi-robot systems, and human-aware robotics.

# References

Renault, B., Saraydaryan, J., Brown, D., & Simonin, O. (2024). Multi-Robot Navigation among Movable Obstacles: Implicit Coordination to Deal with Conflicts and Deadlocks. *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. https://hal.science/hal-04705395

Renault, B., Saraydaryan, J., & Simonin, O. (2020). Modeling a Social Placement Cost to Extend Navigation Among Movable Obstacles (NAMO) Algorithms. *IEEE/RSJ IROS 2020*. https://doi.org/10.1109/IROS45743.2020.9340892