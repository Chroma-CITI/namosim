.. namosim documentation master file, created by
   sphinx-quickstart on Fri Nov 17 08:34:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NAMOSIM
===================================

NAMOSIM is a mobile robot motion planner for the problem of navigation of movable obstacles (NAMO).

.. image:: _static/namo.gif
  :width: 600
  :alt: NAMO Simulator

NAMOSIM is a mobile robot motion planner for the problem of navigation of movable obstacles (NAMO).
It computes plans for robots navigating in a 2D polygonal environment in which certain
obstacles may be grasped and moved. This problem is pertinent for real-world robotics applications such as indoor, social environments
where robots may need to move or manipulate objects in order to navigate and complete their tasks.

Statement of Need
-----------------------------------
Many interesting applications in autonomous mobile robotics involve some kind of physical interaction with the environment as well as social coordination with other agents. 
However, global navigation planners typically assume static, non-interactive environments, leaving higher-level behaviors to other parts of the software stack and thus complicating their implementation to some degree. 
Ideally, motion planners should be able to reason about physical and social interactions, continuously update their internal state in response to incoming data, and adapt to changing conditions. 
NAMOSIM takes a first step towards addressing this challenging problem by offering a simulation environment explicitly designed to study NAMO problems, which involve not only path planning but also reasoning about which obstacles to move, where to move them, and how to combine standard navigation with obstacle manipulation. Additionally, NAMOSIM supports multi-robot environments, facilitating reproducible research in social navigation.

DEMOS
-----------------------------------

Here are a couple demo videos applying namosim on real and simulated robots.

NAMOSIM on a Turtlebot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/076ecBfaBTw" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

NAMOSIM on Multiple Robots in Gazebo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/qgPz69Dk9bc" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


Cite Us
-----------------------------------
If you reuse any part of this project in your research, please cite the associated papers:

.. code-block:: bibtex

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

.. code-block:: bibtex

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.rst
   usage.rst
   testing.rst
   guides.rst
   contributing.rst
   namosim.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
