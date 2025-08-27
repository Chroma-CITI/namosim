Installation
============

System Requirements
-------------------------------

- Ubuntu 22.04
- ROS2 Humble

Setup
-------------------------------

First, clone the repository and navigate to its directory.

If using ROS2, use ``rosdep`` to install the dependencies listed in the ``package.xml`` file:

.. code-block:: bash

   rosdep install --from-paths . -r -y

Some dependencies are only available as pip packages. Install them with:

.. code-block:: bash

   pip install -r requirements.txt

It is often best practice to install pip packages in a virtual environment to avoid conflicts with system packages. You can do this as follows:

.. code-block:: bash

   python3 -m venv env  # Optional
   source env/bin/activate  # Optional
   pip install -r requirements.txt

.. note::

   You may need to install the ``python3-venv`` APT package.

Standalone (Without ROS2)
------------------------------

This project can also be used as a standalone Python module independently of ROS. To use it in this manner, skip the ``rosdep install`` step and install only the pip packages.

Dev Container
-------------------------------

If you are working on another system such as a different version of Ubuntu, or Windows or MacOS, you can use the provided `dev container <https://code.visualstudio.com/docs/devcontainers/containers>`_ to run and edit this project.

In order to access the X server of the dev container to view a scenario's Tkinter window, you may need to run the following command prior to launching.

.. code-block:: bash

   xhost +local:root