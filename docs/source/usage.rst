Usage
==========

You can launch a scenario represented by an SVG or YAML file as follows:

.. code-block:: bash

   python3 -m namosim.main run tests/scenarios/minimal_stilman_2005.svg

.. code-block:: bash

   python3 -m namosim.main run tests/scenarios/citi_ing/namo.yaml

Instructions on how to create custom scenarios can be found on the :ref:`guides` page.

Examples
--------

A set of executable examples demonstrating key features are available in the ``examples/`` folder. They can be run like so:

.. code-block:: bash

   ./examples/3_robots.sh

.. code-block:: bash

   ./examples/with_slam_generated_map.sh

Examples Code Usage
----------------------------

Here is a quick example of using ``namosim`` in Python to load a scenario and compute a plan:

.. literalinclude:: ../../examples/compute_plan.py
   :language: python

This example can be executed with

.. code-block:: bash

   python3 -m examples.compute_plan

Visualization
-------------

By default, ``namosim`` renders the robots and environment in a Tkinter window. More detailed aspects of the operation of ``namosim`` can be visualized in RViz. To do this, first launch RViz with:

.. code-block:: bash

   rviz2 -d rviz/basic_view.rviz

Then, in a new terminal, launch a scenario:

.. code-block:: bash

   python3 -m namosim.main run tests/scenarios/minimal_stilman_2005.svg

Both Tkinter and RViz visualization can be disabled by setting the following environment variables. Doing this slightly improves the speed of the planner.

.. code-block:: bash

   export NAMO_DEACTIVATE_TKINTER="TRUE"
   export NAMO_DEACTIVATE_RVIZ="TRUE"