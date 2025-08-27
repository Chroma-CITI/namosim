Creating Custom Agents
=======================

This guide explains how to create custom agents for the NAMO planner by extending the `Agent` base class. Agents in NAMO are responsible for navigating and interacting within a simulation environment, using sensors, navigation goals, and path planning. The `NavigationOnlyAgent` class, which focuses solely on navigation tasks, serves as a practical example.

Overview
--------

The `Agent` base class, defined in `namosim/agents/agent.py <https://gitlab.inria.fr/chroma/namo/namosim/-/blob/dev/namosim/agents/agent.py?ref_type=heads>`_, provides the core functionality for all agents. To create a custom agent, you must:

1. Inherit from the `Agent` class.
2. Implement the abstract `think` method to define the agent's decision-making logic.
3. Optionally override other methods (e.g., `init`, `sense`) to customize initialization, sensor updates, or other behaviors.
4. Define a configuration model specific to your agent's behavior.

The `NavigationOnlyAgent` is a simple example that navigates to predefined goals using A* path planning, avoiding obstacles without manipulating them.

Defining the Custom Agent Class
-------------------------------

To create a custom agent, start by defining a new class that inherits from `Agent`. Below is the structure of the `NavigationOnlyAgent` class as an example:

.. code-block:: python
   :caption: Example: NavigationOnlyAgent class definition

   from shapely.geometry import Polygon
   from typing_extensions import Self
   import namosim.world.world as w
   from namosim.agents.agent import Agent, ThinkResult
   from namosim.data_models import NavigationOnlyBehaviorConfigModel, Pose2D
   from namosim.utils import utils

   class NavigationOnlyAgent(Agent):
       def __init__(
           self,
           *,
           navigation_goals: t.List[Goal],
           config: NavigationOnlyBehaviorConfigModel,
           logs_dir: str,
           uid: str,
           polygon: Polygon,
           pose: Pose2D,
           sensors: t.List[OmniscientSensor],
           style: AgentStyle,
           logger: utils.NamosimLogger,
           cell_size: float,
       ):
           Agent.__init__(
               self,
               uid=uid,
               navigation_goals=navigation_goals,
               config=config,
               logs_dir=logs_dir,
               polygon=polygon,
               pose=pose,
               sensors=sensors,
               style=style,
               logger=logger,
               cell_size=cell_size,
           )
           self.config = config
           self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD
           self.robot_max_inflation_radius = utils.get_circumscribed_radius(self.polygon)

This constructor initializes the agent with parameters such as its unique ID, navigation goals, configuration, and geometry (represented by a `Polygon`). The `config` parameter should be an instance of a custom configuration model (e.g., `NavigationOnlyBehaviorConfigModel`), which defines behavior-specific settings.

Implementing the `init` Method
-------------------------------

The `init` method is called to set up the agent with a reference to the simulation world. In `NavigationOnlyAgent`, it creates an inflated map to account for the robot's size during path planning:

.. code-block:: python
   :caption: Example: init method

   def init(self, world: "w.World"):
       super().init(world)
       self.robot_inflated_static_map = copy.deepcopy(
           world.map
       ).inflate_map_destructive(self.robot_max_inflation_radius)

You may override `init` to perform additional setup, such as initializing internal state or precomputing data structures.

Implementing the `think` Method
-------------------------------

The `think` method is the core of the agent's decision-making process, determining the next action based on the current state. It returns a `ThinkResult` object containing the plan, next action, and other metadata. Below is the `think` method from `NavigationOnlyAgent`:

.. code-block:: python
   :caption: Example: think method

   def think(
       self,
       ros_publisher: t.Optional["rp.RosPublisher"] = None,
       input: t.Optional[Input] = None,
   ) -> ThinkResult:
       if self._goal is None:
           if self._navigation_goals:
               self._goal = self._navigation_goals.pop(0)
               self._p_opt = nav_plan.Plan(agent_id=self.uid, goal=self._goal.pose)
           else:
               return ThinkResult(
                   plan=None,
                   next_action=ba.GoalsFinished(),
                   goal_pose=None,
                   did_replan=False,
                   agent_id=self.uid,
               )

       if self._p_opt is None:
           raise Exception("No plan")

       if self.is_goal_reached(
           robot_pose=self.world.dynamic_entities[self.uid].pose,
           goal_pose=self._goal.pose,
       ):
           result = ThinkResult(
               plan=None,
               next_action=ba.GoalSuccess(goal=self._goal.pose),
               goal_pose=self._goal.pose,
               did_replan=False,
               agent_id=self.uid,
           )
           self._goal = None
           return result

       if not self._p_opt.is_empty():
           return ThinkResult(
               plan=self._p_opt,
               goal_pose=self._goal.pose,
               did_replan=False,
               agent_id=self.uid,
           )

       path = self.find_path(
           robot_pose=self.world.dynamic_entities[self.uid].pose,
           goal_pose=self._goal.pose,
           robot_inflated_grid=self.robot_inflated_static_map,
           robot_polygon=self.world.dynamic_entities[self.uid].polygon,
       )

       if path is None:
           return ThinkResult(
               plan=None,
               next_action=ba.GoalFailed(self._goal.pose),
               goal_pose=self._goal.pose,
               did_replan=False,
               agent_id=self.uid,
           )

       self._p_opt = nav_plan.Plan(
           paths=[path], goal=self._goal.pose, agent_id=self.uid
       )
       self.goal_to_plans[self._goal] = self._p_opt

       return ThinkResult(
           plan=self._p_opt,
           next_action=None,
           goal_pose=self._goal.pose,
           did_replan=True,
           agent_id=self.uid,
       )

This method:
1. Checks if a goal is set; if not, it selects the next goal from the list.
2. Verifies if the current goal is reached using `is_goal_reached`.
3. Reuses an existing plan if valid, or computes a new path using `find_path` (A* algorithm).
4. Returns appropriate `ThinkResult` objects for success, failure, or ongoing navigation.

For a custom agent, implement `think` to define your agent's behavior, such as handling movable obstacles, integrating sensor data, or using alternative path-planning algorithms.

Implementing the `copy` Method
-------------------------------

The `copy` method creates a deep copy of the agent, ensuring independent instances for simulations. Here's the implementation from `NavigationOnlyAgent`:

.. code-block:: python
   :caption: Example: copy method

   def copy(self) -> Self:
       return NavigationOnlyAgent(
           navigation_goals=copy.deepcopy(self._navigation_goals),
           config=self.config,
           logs_dir=self.logs_dir,
           uid=self.uid,
           polygon=copy.deepcopy(self.polygon),
           style=copy.deepcopy(self.agent_style),
           pose=copy.deepcopy(self.pose),
           sensors=copy.deepcopy(self.sensors),
           cell_size=self.cell_size,
           logger=self.logger,
       )

Ensure all relevant attributes are copied, especially those that are mutable (e.g., lists, polygons).

Configuration Model
-------------------------------

Define a configuration model for your agent to specify behavior parameters. For `NavigationOnlyAgent`, the `NavigationOnlyBehaviorConfigModel` is used, defined in `namosim/data_models.py <https://github.com/Chroma-CITI/namosim/blob/humble/namosim/data_models.py?ref_type=heads>`_. Create a similar model using Pydantic or a similar library to validate and manage settings.

.. code-block:: python
   :caption: Example: Configuration model (conceptual)

   from pydantic import BaseModel

   class CustomAgentConfigModel(BaseBehaviorConfigModel):
       type: t.Literal["custom_behavior"] = "custom_behavior"
       max_speed: float = 1.0
       rotation_speed: float = 0.5
       # Add other parameters as needed

Integrating with Scenarios
-------------------------------

To use your custom agent in a NAMO scenario, configure it in the `<namo_config>` element for SVG scenarios or the NAMO config YAML file for ROS map scenarios. Additionally, update the `World` class methods `construct_agent_from_xml()` and `construct_agent_from_yaml()` to support your custom agent.

**SVG Scenario Configuration**

In an SVG scenario, the `<namo_config>` element, defined by `NamoConfigModel`, specifies agents and their behaviors. Include your custom agent by setting the `type` attribute in the `behavior` element to match your behavior configuration (e.g., `custom_behavior`). Below is an example:

.. code-block:: xml
   :caption: Example: namo_config for CustomAgent in SVG scenario

   <namo_config cell_size_cm="10" random_seed="10" generate_report="true">
       <agent agent_id="robot_0">
           <goal goal_id="goal_0"/>
           <behavior type="custom_behavior"/>
       </agent>
   </namo_config>

Ensure the `agent_id` matches the `id` of an `<svg:path>` element in the SVG file, and the `goal_id` corresponds to a goal path. The `behavior` element's `type` attribute must match the `type` defined in your custom behavior configuration model (e.g., `CustomAgentConfigModel`).

**Updating construct_agent_from_xml**

The `World.construct_agent_from_xml()` method instantiates agents based on the behavior type specified in the XML configuration. To support your custom agent, add a condition to check for your behavior type and instantiate your `CustomAgent` class. Below is an example of how to update this method:

.. code-block:: python
   :caption: Example: Updating construct_agent_from_xml for CustomAgent

   @classmethod
   def construct_agent_from_xml_model(
       cls,
       *,
       agent_config: AgentConfigXmlModel,
       goals: t.List[Goal],
       logs_dir: str,
       robot_polygon: Polygon,
       agent_style: AgentStyle,
       init_pose: Pose2D,
       cell_size: float,
       collision_margin: float,
       logger: utils.NamosimLogger,
   ) -> agts.Agent:
       if agent_config.behavior.type == "stilman_2005_behavior":
           new_robot = agts.Stilman2005Agent(
               navigation_goals=goals,
               config=agent_config.behavior,
               logs_dir=logs_dir,
               uid=agent_config.agent_id,
               polygon=robot_polygon,
               style=agent_style,
               pose=init_pose,
               sensors=[OmniscientSensor()],
               cell_size=cell_size,
               collision_margin=collision_margin,
               logger=logger,
           )
       elif agent_config.behavior.type == "stilman_rrt_star_behavior":
           new_robot = agts.StilmanRRTStarAgent(
               navigation_goals=goals,
               config=agent_config.behavior,
               logs_dir=logs_dir,
               uid=agent_config.agent_id,
               polygon=robot_polygon,
               style=agent_style,
               pose=init_pose,
               sensors=[OmniscientSensor()],
               cell_size=cell_size,
               collision_margin=collision_margin,
               logger=logger,
           )
       elif agent_config.behavior.type == "navigation_only_behavior":
           new_robot = agts.NavigationOnlyAgent(
               navigation_goals=goals,
               config=agent_config.behavior,
               logs_dir=logs_dir,
               uid=agent_config.agent_id,
               polygon=robot_polygon,
               style=agent_style,
               pose=init_pose,
               sensors=[OmniscientSensor()],
               cell_size=cell_size,
               logger=logger,
           )
       elif agent_config.behavior.type == "rrt":
           new_robot = agts.RRTAgent(
               navigation_goals=goals,
               config=agent_config.behavior,
               logs_dir=logs_dir,
               uid=agent_config.agent_id,
               polygon=robot_polygon,
               style=agent_style,
               pose=init_pose,
               sensors=[OmniscientSensor()],
               cell_size=cell_size,
               logger=logger,
           )
       elif agent_config.behavior.type == "teleop_behavior":
           new_robot = agts.TeleopAgent(
               navigation_goals=goals,
               config=agent_config.behavior,
               logs_dir=logs_dir,
               uid=agent_config.agent_id,
               polygon=robot_polygon,
               style=agent_style,
               pose=init_pose,
               sensors=[OmniscientSensor()],
               cell_size=cell_size,
               logger=logger,
           )
       elif agent_config.behavior.type == "custom_behavior":
           new_robot = agts.CustomAgent(
               navigation_goals=goals,
               config=agent_config.behavior,
               logs_dir=logs_dir,
               uid=agent_config.agent_id,
               polygon=robot_polygon,
               style=agent_style,
               pose=init_pose,
               sensors=[OmniscientSensor()],
               cell_size=cell_size,
               logger=logger,
           )
       else:
           raise NotImplementedError(
               "You tried to associate entity '{agent_name}' with a behavior named"
               "'{b_name}' that is not implemented yet."
               "Maybe you mispelled something ?".format(
                   agent_name=agent_config.agent_id, b_name=agent_config.behavior.type
               )
           )
       return new_robot

Add an `elif` clause for your custom agent's behavior type (e.g., `custom_behavior`) and instantiate your `CustomAgent` class with the appropriate parameters. Ensure the `CustomAgent` class is imported into the `World` module (e.g., `import namosim.agents as agts`).

**ROS Map Scenario Configuration**

For ROS map scenarios, the NAMO config YAML file, defined by `NamoConfigYamlModel`, links the SVG geometry to the ROS map and configures agents. Below is an example:

.. code-block:: yaml
   :caption: Example: NAMO config YAML for CustomAgent

   map_yaml: "map.yaml"
   svg_file: "scenario.svg"
   collision_margin: 0.1
   agents:
     - id: robot_0
       initial_pose: [1.0, 1.0, 90]
       radius: 0.5
       push_only: false
       behavior: custom_behavior

The `id` must match an `<svg:path>` in the SVG file's `robots_layer`, and the `initial_pose` defines the starting position and orientation in meters and degrees. The `radius` specifies the agent's size, and `push_only` is specific to manipulation behaviors (set to `false` for navigation-only agents). The `behavior` field must match the `type` defined in your custom behavior configuration model. Goals are defined in the SVG file's `goals_layer`.

**Updating construct_agent_from_yaml**

The `World.construct_agent_from_yaml_model()` method instantiates agents based on the behavior type specified in the YAML configuration. To support your custom agent, add a condition to check for your behavior type and instantiate your `CustomAgent` class. Below is an example of how to update this method:

.. code-block:: python
   :caption: Example: Updating construct_agent_from_yaml_model for CustomAgent

   @classmethod
   def construct_agent_from_yaml_model(
       cls,
       *,
       agent: AgentConfigYamlModel,
       goals: t.List[Goal],
       logs_dir: str,
       robot_polygon: Polygon,
       agent_style: AgentStyle,
       init_pose: Pose2D,
       cell_size: float,
       collision_margin: float,
       logger: utils.NamosimLogger,
   ) -> agts.Agent:
       if agent.behavior.type == "stilman_2005_behavior":
           return agts.Stilman2005Agent(
               navigation_goals=goals,
               config=agent.behavior,
               logs_dir=logs_dir,
               uid=agent.id,
               polygon=robot_polygon,
               style=agent_style,
               pose=init_pose,
               sensors=[OmniscientSensor()],
               cell_size=cell_size,
               collision_margin=collision_margin,
               logger=logger,
           )
       elif agent.behavior.type == "custom_behavior":
           return agts.CustomAgent(
               navigation_goals=goals,
               config=agent.behavior,
               logs_dir=logs_dir,
               uid=agent.id,
               polygon=robot_polygon,
               style=agent_style,
               pose=init_pose,
               sensors=[OmniscientSensor()],
               cell_size=cell_size,
               logger=logger,
           )
       else:
           raise NotImplementedError(
               "You tried to associate entity '{agent_name}' with a behavior named"
               "'{b_name}' that is not implemented yet."
               "Maybe you mispelled something ?".format(
                   agent_name=agent.id, b_name=agent.behavior.type
               )
           )

Add an `elif` clause for your custom agent's behavior type (e.g., `custom_behavior`) and instantiate your `CustomAgent` class with the appropriate parameters. Ensure the `CustomAgent` class is imported into the `World` module.

Testing Your Agent
----------------------------

Test your agent in a simulation environment using the NAMO simulator. Create a minimal SVG scenario (see :ref:`Creating an SVG Scenario <creating_svg_scenario>`) with a robot, goal, and obstacles. Run the simulation to verify the agent's navigation behavior. Use `Inkscape <https://inkscape.org/>`_ to create or edit scenario files.

.. note::
   Ensure your agent's `find_path` method (if overridden) handles the `BinaryOccupancyGrid` correctly, as it uses a grid-based representation of the environment.

For advanced debugging, enable logging via the `NamosimLogger` passed to the agent, and inspect the logs in the specified `logs_dir`.

Additional Considerations
-------------------------

- **Sensors**: The `NavigationOnlyAgent` uses an `OmniscientSensor` for simplicity. For realistic scenarios, implement custom sensors in `namosim/world/sensors/` to simulate limited fields of view or noisy data.
- **Path Planning**: The example uses A* (`graph_search.real_to_grid_search_a_star`). You can implement alternative algorithms like RRT or potential fields by overriding `find_path`.
- **Movability**: To handle movable objects, extend the `think` method to detect and manipulate objects of type `Movability.MOVABLE`, using actions like `Push` or `Pull` from `namosim.navigation.basic_actions`.
- **Thread Safety**: Ensure thread-safe operations if your agent interacts with ROS publishers or external systems.

Refer to the NAMO source code, particularly `namosim/agents/agent.py` and `namosim/data_models.py`, for detailed specifications and additional methods.