Contributing
===============================

Thank you for your interest in contributing to NAMOSIM! We welcome contributions from the community to improve the project, whether through code, documentation, bug reports, or feature suggestions. This page outlines the guidelines for contributing, with a specific focus on testing requirements and methods to ensure the quality and reliability of the NAMOSIM codebase.

How to Contribute
-------------------------------

1. **Fork the Repository**: Start by forking the NAMOSIM repository on GitHub to your own account.
2. **Clone the Fork**: Clone your forked repository to your local machine.
3. **Create a Branch**: Create a new branch for your changes (e.g., `feature/new-planner` or `bugfix/issue-123`).
4. **Make Changes**: Implement your changes, ensuring they align with the project's coding standards (see below).
5. **Test Your Changes**: Run the test suite and add new tests as needed (see Testing Requirements below).
6. **Submit a Pull Request**: Push your changes to your fork and submit a pull request to the main NAMOSIM repository. Include a clear description of your changes and reference any related issues.

Please ensure your contributions adhere to the following guidelines:

- Follow the existing code style (PEP 8 for Python code).
- Write clear, concise commit messages.
- Update documentation if your changes affect usage or functionality.
- Ensure all tests pass before submitting a pull request.

Testing Requirements
-------------------------------

To maintain the reliability and correctness of NAMOSIM, all code contributions must include appropriate tests. Testing ensures that new features or bug fixes do not introduce regressions and that the planner behaves as expected in various scenarios.

### Testing Environment
NAMOSIM uses the `osrf/ros:humble-desktop-full` Docker container for its CI pipeline, which includes ROS Humble and dependencies. To replicate this environment locally:

- **Use ROS Humble**: Ensure you have ROS Humble installed. Refer to `installation.rst` for setup instructions.
- **Install Dependencies**: Install required system packages and Python dependencies:
  .. code-block:: bash

      sudo apt-get update
      sudo apt-get install -y swig curl python3-pip ros-humble-grid-map-msgs
      rosdep install --from-paths . -r -y
      pip install -r requirements.txt

- **Set Up ROS Environment**: Source the ROS Humble setup script before building or testing:
  .. code-block:: bash

      source /opt/ros/humble/setup.bash

### Types of Tests
NAMOSIM uses a combination of the following tests to validate functionality:

- **Linting/Formatting**: Ensures code adheres to style guidelines using `black`.
- **Type Checking**: Verifies type correctness in Python code using the static type checkers `pyright`.
- **Unit Tests**: Test individual components (e.g., path planning algorithms, obstacle manipulation logic) in isolation. These should run fast and cover edge cases and common use cases.
- **End-to-End Tests**: Verify the entire system, from environment setup to path planning and obstacle manipulation, in realistic scenarios.

### Testing Framework
NAMOSIM tests are executed via custom scripts (`scripts/test_unit.sh`, `scripts/test_e2e.sh`, `scripts/test_types.sh`). Tests are organized in the `tests/` directory.

### Writing Tests
When adding new functionality or fixing bugs, include corresponding tests in the `tests/` directory. Follow these guidelines:

- **Test Structure**: Place tests in the `tests/` directory, organized by module (e.g., `tests/planner_test.py` for unit tests, `tests/e2e/e2e_test.py` for end-to-end tests).
- **Test Naming**: Use descriptive names for test files, post-fixed with `_test.py` (e.g., `collision_test.py`).
- **Coverage**: Aim for high test coverage, especially for critical components. Use test scripts to verify functionality.
- **Edge Cases**: Include tests for edge cases, such as empty environments, fully blocked paths, etc.
- **ROS Integration**: Ensure tests account for ROS-specific features, such as message passing (e.g., `grid_map_msgs`) and node communication.

### Example Test
Below is an example of a unit test for loading a scenario from an SVG file:

.. code-block:: python

    import os
    from namosim.world.world import World

    dirname = os.path.dirname(os.path.abspath(__file__))


    def test_load_from_svg():
        world = World.load_from_svg(f"{dirname}/../scenarios/minimal_stilman_2005.svg")
        assert len(world.agents) == 1
        assert "robot_0" in world.agents
        assert world.agents["robot_0"].is_initialized


### Running Tests
To build and test the project locally, follow these steps:


1. **Run Unit Tests**:
   .. code-block:: bash

       ./scripts/test_unit.sh

2. **Run End-to-End Tests**:
   .. code-block:: bash

       ./scripts/test_e2e.sh

3. **Run Type Checking**:
   .. code-block:: bash

       ./scripts/test_types.sh

4. **Apply Code Formatting**:
   Ensure code adheres to the project's formatting standards:
   .. code-block:: bash

       ./scripts/format.sh

5. **Run All Tests**:
   To execute all tests (unit, end-to-end, and type checking), use:
   .. code-block:: bash

       ./scripts/test_all.sh

6. **Colcon Build and Test**:
   Since namosim is also distributed as a ros2 package, please verify that it builds correctly with colcon:
   .. code-block:: bash

       source /opt/ros/humble/setup.bash
       colcon build
       colcon test

### Continuous Integration
NAMOSIM uses GitHub Actions for continuous integration (CI), running on the `humble` branch and all pull requests. The CI pipeline:

- Uses the `osrf/ros:humble-desktop-full` container.
- Installs dependencies (`swig`, `curl`, `python3-pip`, `ros-humble-grid-map-msgs`, etc.).
- Builds the project with `colcon build`.
- Runs tests with `colcon test`, `test_unit.sh`, `test_e2e.sh`, and `test_types.sh`.
- Ensures all tests pass and reports results.

Before submitting a pull request, verify locally that your changes pass the CI pipeline's testing steps. The CI pipeline will automatically run on your pull request, and maintainers will review the results.

Code Review Process
-------------------------------

Once you submit a pull request, it will be reviewed by the NAMOSIM maintainers. The review will focus on:

- Code quality and adherence to style guidelines.
- Test coverage and correctness.
- Compatibility with existing functionality and ROS Humble.
- Clarity of documentation and commit messages.

You may be asked to make revisions before your pull request is merged. Please respond promptly to review comments.

Reporting Issues
-------------------------------

If you encounter bugs or have feature suggestions, please open an issue on the GitHub repository. Include:

- A clear description of the issue or feature.
- Steps to reproduce (for bugs).
- Expected and actual behavior.
- Any relevant logs or screenshots.

Community Guidelines
-------------------------------

We strive to maintain a welcoming and inclusive community. Please adhere to the following:

- Be respectful and constructive in all interactions.
- Follow the project's code of conduct (available in the repository).
- Provide clear and actionable feedback in issues and pull requests.

Thank you for contributing to NAMOSIM! Your efforts help advance research and development in navigation among movable obstacles.