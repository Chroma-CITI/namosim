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

.. code-block:: bibtex

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
