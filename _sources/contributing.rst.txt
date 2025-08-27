Contributing to NAMOSIM
======================

Thank you for your interest in contributing to NAMOSIM! We welcome contributions from the community to improve the project, whether through code, documentation, bug reports, or feature suggestions. This page outlines the guidelines for contributing, with a specific focus on testing requirements and methods to ensure the quality and reliability of the NAMOSIM codebase.

How to Contribute
-----------------

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
-------------------

To maintain the reliability and correctness of NAMOSIM, all code contributions must include appropriate tests. Testing ensures that new features or bug fixes do not introduce regressions and that the planner behaves as expected in various scenarios. NAMOSIM is built and tested in a ROS Humble environment using `colcon` and custom test scripts.

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

- **Linting/Formatting**: Ensures code adheres to style guidelines (e.g., PEP 8) using tools like `flake8` or `black`.
- **Type Checking**: Verifies type correctness in Python code using static type checkers like `mypy`.
- **Unit Tests**: Test individual components (e.g., path planning algorithms, obstacle manipulation logic) in isolation. These should cover edge cases and common use cases.
- **End-to-End Tests**: Verify the entire system, from environment setup to path planning and obstacle manipulation, in realistic scenarios.

### Testing Framework
NAMOSIM uses `colcon` for building and testing, with tests executed via custom scripts (`scripts/test_unit.sh`, `scripts/test_e2e.sh`, `scripts/test_types.sh`). Tests are organized in the `tests/` directory and should be compatible with the ROS Humble environment.

### Writing Tests
When adding new functionality or fixing bugs, include corresponding tests in the `tests/` directory. Follow these guidelines:

- **Test Structure**: Place tests in the `tests/` directory, organized by module (e.g., `tests/test_planner.py` for unit tests, `tests/e2e/e2e_test.py` for end-to-end tests).
- **Test Naming**: Use descriptive names for test functions, prefixed with `test_` (e.g., `test_path_planner_handles_collisions`).
- **Coverage**: Aim for high test coverage, especially for critical components like path planning and obstacle manipulation. Use `colcon test` to verify functionality.
- **Edge Cases**: Include tests for edge cases, such as empty environments, fully blocked paths, or maximum obstacle weight limits.
- **ROS Integration**: Ensure tests account for ROS-specific features, such as message passing (e.g., `grid_map_msgs`) and node communication.
- **Mocking**: Use `unittest.mock` to mock ROS nodes or external dependencies when testing specific components.

### Example Test
Below is an example of a unit test for a path planner function in a ROS context:

.. code-block:: python

    import pytest
    from namosim.planner import PathPlanner
    from geometry_msgs.msg import Point

    def test_path_planner_empty_environment():
        planner = PathPlanner(environment=[])
        start = Point(x=0.0, y=0.0)
        goal = Point(x=10.0, y=10.0)
        path = planner.compute_path(start, goal)
        assert path is not None, "Path should exist in an empty environment"
        assert path[0].x == start.x and path[0].y == start.y, "Path should start at the given start point"
        assert path[-1].x == goal.x and path[-1].y == goal.y, "Path should end at the given goal point"

### Running Tests
To build and test the project locally, follow these steps:

1. **Build the Project**:
   .. code-block:: bash

       source /opt/ros/humble/setup.bash
       colcon build

2. **Run Unit Tests**:
   .. code-block:: bash

       source /opt/ros/humble/setup.bash
       source install/setup.bash
       ./scripts/test_unit.sh

3. **Run End-to-End Tests**:
   .. code-block:: bash

       source /opt/ros/humble/setup.bash
       source install/setup.bash
       ./scripts/test_e2e.sh

4. **Run Type Checking**:
   .. code-block:: bash

       source /opt/ros/humble/setup.bash
       source install/setup.bash
       ./scripts/test_types.sh

5. **Apply Code Formatting**:
   Ensure code adheres to the project's formatting standards:
   .. code-block:: bash

       ./scripts/format.sh

6. **Run All Tests**:
   To execute all tests (unit, end-to-end, and type checking), use:
   .. code-block:: bash

       source /opt/ros/humble/setup.bash
       source install/setup.bash
       ./scripts/test_all.sh

### End-to-End Scenario Testing
To experiment with different scenarios and parameters, use the Python Test Explorer in VSCode to run end-to-end tests located in `tests/e2e/`. This is the recommended approach for interactive testing.

Alternatively, run specific end-to-end tests from the command line:
.. code-block:: bash

    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 -m pytest tests/e2e/e2e_test.py::TestE2E::test_social_dr_success_d

Ensure test scenarios are included in the `scenarios/` directory and documented in the pull request.

### Simulation-Based Testing
For simulation-based tests, use the NAMOSIM simulator to validate planner behavior in realistic scenarios. Create test cases in the `scenarios/` directory, specifying:

- Environment configuration (e.g., polygon shapes, obstacle positions).
- Robot start and goal positions (using ROS message types like `geometry_msgs/Point`).
- Expected outcomes (e.g., successful navigation, obstacle movement).

Run simulation tests using:
.. code-block:: bash

    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python -m namosim.simulate --test scenarios/test_scenario1.yaml

### Continuous Integration
NAMOSIM uses GitHub Actions for continuous integration (CI), running on the `humble` branch and all pull requests. The CI pipeline:

- Uses the `osrf/ros:humble-desktop-full` container.
- Installs dependencies (`swig`, `curl`, `python3-pip`, `ros-humble-grid-map-msgs`, etc.).
- Builds the project with `colcon build`.
- Runs tests with `colcon test`, `test_unit.sh`, `test_e2e.sh`, and `test_types.sh`.
- Ensures all tests pass and reports results.

Before submitting a pull request, verify locally that your changes pass the CI pipeline's testing steps. The CI pipeline will automatically run on your pull request, and maintainers will review the results.

Code Review Process
-------------------

Once you submit a pull request, it will be reviewed by the NAMOSIM maintainers. The review will focus on:

- Code quality and adherence to style guidelines.
- Test coverage and correctness.
- Compatibility with existing functionality and ROS Humble.
- Clarity of documentation and commit messages.

You may be asked to make revisions before your pull request is merged. Please respond promptly to review comments.

Reporting Issues
---------------

If you encounter bugs or have feature suggestions, please open an issue on the GitHub repository. Include:

- A clear description of the issue or feature.
- Steps to reproduce (for bugs).
- Expected and actual behavior.
- Any relevant logs or screenshots.

Community Guidelines
-------------------

We strive to maintain a welcoming and inclusive community. Please adhere to the following:

- Be respectful and constructive in all interactions.
- Follow the project's code of conduct (available in the repository).
- Provide clear and actionable feedback in issues and pull requests.

Thank you for contributing to NAMOSIM! Your efforts help advance research and development in navigation among movable obstacles.