.. _creating_svg_scenario:

Creating an SVG Scenario
=============================

Each Namo scenario is fully contained in a single SVG file.

The Geometry File
-----------------

Here are the contents of a minimal svg geometry file. All geometries in the world must be svg **path**
elements and each must have an **id** attribute which is used by the **<namo_config>** to configure the geometry
as an entity in the simulation.

.. literalinclude:: ../../tests/scenarios/minimal_stilman_2005.svg
  :language: xml

Here is the same file rendered as an image:

.. image:: ../../tests/scenarios/minimal_stilman_2005.svg
  :width: 400
  :alt: NAMO Simulator

You can see the robot starting position in the top left. To the right of the robot is a movable box. The walls
are in black. The robot goal pose is visible in the bottom right.

We recommend using `Inkscape <https://inkscape.org/>`_ to edit your svg geometry file.

Units
-----

It is essential that all units in your SVG geometry file be in CENTIMETERs. The reasons for this is because, Inkscape
only supports centimeters and not meters. The NAMO planner will convert the units to meters during execution.

The Namo Config
---------------

The scenario file must contain a `<namo_config>` element that is a direct child of the root `<svg>` element.
This object configures the simulator and agents and it the place where all agent behavioral parameters are set.

The full specification for the `<namo_config>` is defined by the `NamoConfigModel` class which can be found in 
`namosim/data_models.py <https://github.com/Chroma-CITI/namosim/blob/humble/namosim/data_models.py?ref_type=heads>`_.

Robots and Navigation Goals
---------------------------

Each robot and goal listed in the `<namo_config>` must have a corresponding `<svg:path id="[your_agent_id]">` path element somewhere in the svg file. These elements 
provides the shape, position, and orientation of the robots and goals.

.. code-block:: xml

  <svg:path
      d="m 121.17572,125.39975 0,0 c 0,-3.87867 3.14428,-7.02295 7.02295,-7.02295 l 0,0 c 1.8626,0 3.64891,0.73992 4.96597,2.05698 1.31706,1.31705 2.05697,3.10337 2.05697,4.96597 l 0,0 c 0,3.87866 -3.14428,7.02294 -7.02294,7.02294 l 0,0 c -3.87867,0 -7.02295,-3.14428 -7.02295,-7.02294 z"
      id="goal_0"
      angle="90"
      style="fill:none;fill-rule:evenodd;stroke:#1155cc;stroke-linecap:square;stroke-miterlimit:10;stroke-opacity:1"
      inkscape:connector-curvature="0" inkscape:label="#path29-7" />
  </svg:g>

You can optionally add a `angle="90"` attribute to specify the orientation in degrees.