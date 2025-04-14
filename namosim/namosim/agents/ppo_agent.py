# pyright: reportConstantRedefinition=false

import copy
import math
import random
import typing as t

import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as transforms
from PIL import Image
from shapely.geometry import Point, Polygon
from shapely import affinity
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from typing_extensions import Self

import namosim.display.ros2_publisher as rp
import namosim.navigation.action_result as ar
import namosim.navigation.basic_actions as ba
from namosim.svg_styles import AgentStyle
import namosim.world.world as w
from namosim.agents.agent import Agent, RLThinkResult
from namosim.agents.models import DEIT_MODEL_CHECKPOINT, PPOActor
from namosim.algorithms import graph_search
from namosim.data_models import GridCellModel, PoseModel, PPOAgentConfigModel
from namosim.input import Input
from namosim.log import logger
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import (
    BinaryOccupancyGrid,
)
from namosim.world.entity import Movability
from namosim.world.goal import Goal
from namosim.world.obstacle import Obstacle
from namosim.world.sensors.omniscient_sensor import OmniscientSensor


class State:
    def __init__(
        self,
        *,
        grid: npt.NDArray[np.float32],
        goal_state_grid: npt.NDArray[np.float32],
        robot_pose: PoseModel,
        normalized_robot_pose: npt.NDArray[np.float32],
        goal_pose: npt.NDArray[np.float32],
        normalized_goal_pose: npt.NDArray[np.float32],
        goal_reached: bool,
    ):
        self.grid = grid
        self.goal_state_grid = goal_state_grid
        self.robot_pose = robot_pose
        self.normalized_robot_pose = normalized_robot_pose
        self.goal_pose = goal_pose
        self.normalized_goal_pose = normalized_goal_pose
        self.goal_reached = goal_reached


class RLAgentStepResult:
    def __init__(
        self,
        *,
        state: State,
        goal_reached: bool,
        action: ba.Action,
        action_idx: int,
        action_log_prob: float,
    ):
        self.state = state
        self.goal_reached = goal_reached
        self.action_idx = action_idx
        self.action_log_prob = action_log_prob
        self.action = action


class PPOAgent(Agent):
    def __init__(
        self,
        *,
        navigation_goals: t.List[Goal],
        config: PPOAgentConfigModel,
        logs_dir: str,
        uid: str,
        full_geometry_acquired: bool,
        polygon: Polygon,
        pose: PoseModel,
        sensors: t.List[OmniscientSensor],
        logger: utils.NamosimLogger,
        cell_size: float,
        style: AgentStyle | None = None,
    ):
        Agent.__init__(
            self,
            uid=uid,
            navigation_goals=navigation_goals,
            config=config,
            logs_dir=logs_dir,
            full_geometry_acquired=full_geometry_acquired,
            polygon=polygon,
            pose=pose,
            sensors=sensors,  # type: ignore
            style=style,
            logger=logger,
            cell_size=cell_size,
        )
        self.config = config
        self.robot_max_inflation_radius = utils.get_circumscribed_radius(self.polygon)
        self.steps_per_goal = 0
        self.step_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor: PPOActor | None = None
        self.dist_to_goal: t.Dict[GridCellModel, float] = {}
        self.deit_image_processor = AutoImageProcessor.from_pretrained(
            DEIT_MODEL_CHECKPOINT
        )

    def init(self, world: "w.World"):
        super().init(world)
        self._compute_occupancy_grids()
        self.word_diagonal = math.sqrt(
            self.static_obstacle_grid.width**2 + self.static_obstacle_grid.height**2
        )

    def action_idx_to_action(self, idx: int) -> ba.Action:
        if idx == 0:
            action = ba.Wait()
        elif idx == 1:
            action = self._grab() or ba.Wait()
        elif idx == 2:
            action = self._release() or ba.Wait()
        elif idx == 3:
            action = ba.Advance(distance=self.cell_size)
        elif idx == 4:
            action = ba.Advance(distance=-self.cell_size)
        elif idx == 5:
            action = ba.Rotation(angle=30)
        elif idx == 6:
            action = ba.Rotation(angle=-30)
        else:
            raise Exception("Invalid action index")
        return action

    def set_actor(self, actor: PPOActor):
        self.actor = actor

    def get_state(self) -> State:
        if self.goal_pose is None:
            raise Exception("No goal pose")

        # img = self.world.to_image(grayscale=True, width=224, ignore_goal=False)  # type: ignore
        # img_transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Resize((224, 224)),
        #         transforms.Grayscale(),
        #         transforms.Normalize(0.0, 1.0),
        #     ]
        # )
        # img: npt.NDArray[np.float32] = img_transform(img).squeeze().numpy()

        img = self.world.to_image(grayscale=False, width=224)
        img = self.deit_image_processor(
            img,
            return_tensors="pt",
            do_center_crop=False,
            size={"height": 224, "width": 224},
        )["pixel_values"]
        img = img.squeeze().detach().cpu().numpy()

        goal_reached = (
            utils.euclidean_distance(self.pose, self.goal_pose) < self.cell_size
        )
        return State(
            grid=img,  # type: ignore
            goal_state_grid=self.goal_state.grid,
            robot_pose=self.pose,
            normalized_robot_pose=np.array(self.get_normalized_robot_pose()),
            goal_pose=np.array((self.goal_pose[0], self.goal_pose[1])),
            normalized_goal_pose=np.array(self.get_normalized_goal_pose()),
            goal_reached=goal_reached,
        )

    def get_goal_state(self) -> State:
        if self.goal_pose is None:
            raise Exception("No goal pose")

        # save original
        original_pose = self.pose
        original_polygon = self.polygon

        self.pose = self.goal_pose
        current_centroid = self.polygon.centroid
        translate_x = self.pose[0] - current_centroid.x
        translate_y = self.pose[1] - current_centroid.y
        self.polygon = affinity.translate(
            self.polygon,
            xoff=translate_x,
            yoff=translate_y,
        )

        img = self.world.to_image(grayscale=True, width=224, ignore_goal=False)  # type: ignore
        img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Grayscale(),
                transforms.Normalize(0.0, 1.0),
            ]
        )
        img: npt.NDArray[np.float32] = img_transform(img).squeeze().numpy()  # type: ignore

        goal_state = State(
            grid=img,  # type: ignore
            goal_state_grid=img,
            robot_pose=self.pose,
            normalized_robot_pose=np.array(self.get_normalized_robot_pose()),
            goal_pose=np.array((self.goal_pose[0], self.goal_pose[1])),
            normalized_goal_pose=np.array(self.get_normalized_goal_pose()),
            goal_reached=True,
        )

        # restore original
        self.pose = original_pose
        self.polygon = original_polygon

        return goal_state

    def get_normalized_robot_pose(self) -> PoseModel:
        """Scales the robot pose so that x, y, and theta all lie in the range [0, 1]"""
        return (
            self.pose[0] / self.world.map.width,
            self.pose[1] / self.world.map.height,
            (utils.normalize_angle_degrees(self.pose[2]) + 180) / 360,
        )

    def get_normalized_goal_pose(self) -> t.Tuple[float, float]:
        """Scales the goal pose so that x and y lie in the range [0, 1]"""
        if self.goal_pose is None:
            raise Exception("No goal pose")

        return (
            self.goal_pose[0] / self.world.map.width,
            self.goal_pose[1] / self.world.map.height,
        )

    def _compute_occupancy_grids(self):
        static_obs_polygons = {
            uid: entity.polygon
            for uid, entity in self.world.dynamic_entities.items()
            if (isinstance(entity, Obstacle) or entity.movability == Movability.STATIC)
        }
        self.static_obstacle_grid = self.world.map
        self.robot_inflated_grid = copy.deepcopy(
            self.world.map
        ).inflate_map_destructive(self.circumscribed_radius)

    def sense(
        self, ref_world: "w.World", last_action_result: ar.ActionResult, step_count: int
    ):
        self._world = ref_world
        self.step_count = step_count

    def get_reward(
        self,
        *,
        state: State,
        next_state: State,
        action_result: ar.ActionResult,
        action: ba.Action | None,
    ) -> float:
        next_pose = next_state.robot_pose
        next_cell = self.robot_inflated_grid.pose_to_cell(next_pose[0], next_pose[1])
        pose = state.robot_pose
        cell = self.robot_inflated_grid.pose_to_cell(pose[0], pose[1])

        if next_cell not in self.dist_to_goal or cell not in self.dist_to_goal:
            self.debug_dist_to_goal(prev_robot_cell=cell, robot_cell=next_cell)
            raise Exception(
                f"Robot cell not in dist to goal, action={action}, prev_pose={pose}, pose={next_pose}, prev_cell={cell}, cell={next_cell}"
            )

        assert np.allclose(state.goal_pose, next_state.goal_pose)
        # reward = (-self.dist_to_goal[robot_cell]) / self.word_diagonal
        # d = utils.euclidean_distance(state.robot_pose, state.goal_pose)  # type: ignore
        # d_next = utils.euclidean_distance(next_state.robot_pose, next_state.goal_pose)  # type: ignore
        # reward = 1e-3 * (d - d_next) / self.cell_size - 5e-4

        d_next = utils.euclidean_distance(next_state.normalized_robot_pose, next_state.normalized_goal_pose)  # type: ignore
        reward = -(d_next**2)
        if isinstance(action_result, ar.ActionFailure):
            logger.warn("Action failed")
            reward -= 10
        elif next_state.goal_reached:
            reward += 100

        return reward

    def start_new_episode(self, ref_world: "w.World"):
        self._world = ref_world
        self._compute_occupancy_grids()
        goal = self.sample_random_goal()
        self._goal = goal

        goal_cell = self.robot_inflated_grid.pose_to_cell(goal.pose[0], goal.pose[1])
        self.dist_to_goal = self.run_dijkstra(
            robot_cell=goal_cell, static_grid=self.static_obstacle_grid
        )
        self.dist_to_goal_grid = self.get_dist_to_goal_grid()

        # set robot pose to an accessible cell
        candidate_robot_start_cells = list(
            self.run_dijkstra(
                robot_cell=goal_cell, static_grid=self.robot_inflated_grid
            ).keys()
        )
        self.pose = self.sample_random_pose_from_cells(
            cells=candidate_robot_start_cells
        )

        # center robot polygon on the robot pose
        current_centroid = self.polygon.centroid
        translate_x = self.pose[0] - current_centroid.x
        translate_y = self.pose[1] - current_centroid.y
        self.polygon = affinity.translate(
            self.polygon,
            xoff=translate_x,
            yoff=translate_y,
        )

        self.steps_per_goal = 0
        self._release()
        self.goal_state = self.get_goal_state()

    def step(
        self,
    ) -> RLAgentStepResult:
        if self._goal is None:
            self.start_new_episode(self.world)
        if self._goal is None:
            raise Exception("No goal")
        if self.actor is None:
            raise Exception("No actor")

        goal_pose = self._goal.pose
        state = self.get_state()

        # check for goal success
        if utils.euclidean_distance(self.pose, goal_pose) < self.cell_size:
            return RLAgentStepResult(
                state=state,
                goal_reached=True,
                action_idx=0,
                action=ba.GoalSuccess(goal_pose),
                action_log_prob=0.0,
            )

        self.steps_per_goal += 1

        with torch.no_grad():
            curr_state_grids = torch.Tensor(state.grid).to(self.device).unsqueeze(0)
            goal_state_grids = (
                torch.Tensor(self.goal_state.grid).to(self.device).unsqueeze(0)
            )
            action_probs, _, _ = self.actor(curr_state_grids)

        action_dist = action_probs.cpu().squeeze().numpy()
        action_i = np.random.choice(list(range(7)), p=action_dist)
        log_prob = np.log(action_dist[action_i])

        return RLAgentStepResult(
            state=state,
            goal_reached=False,
            action_log_prob=log_prob,
            action_idx=action_i,
            action=self.action_idx_to_action(action_i),
        )

    def think(
        self,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        input: t.Optional[Input] = None,
    ) -> RLThinkResult:
        if self._goal is None:
            self.start_new_episode(self.world)
        if self._goal is None:
            raise Exception("No goal")
        goal_pose = self._goal.pose
        step_result = self.step()
        action_i = step_result.action_idx

        if action_i == 0:
            action = ba.Wait()
        elif action_i == 1:
            action = self._grab()
        elif action_i == 2:
            action = self._release()
        elif action_i == 3:
            action = ba.Advance(distance=self.cell_size)
        elif action_i == 4:
            action = ba.Advance(distance=-self.cell_size)
        elif action_i == 5:
            action = ba.Rotation(angle=30)
        elif action_i == 6:
            action = ba.Rotation(angle=-30)
        else:
            raise Exception("Invalid action index")

        return RLThinkResult(
            goal_pose=goal_pose,
            next_action=action,
            did_replan=False,
            agent_id=self.uid,
            action_idx=action_i,
        )

    def _grab(self) -> ba.Grab | None:
        movables = self.world.get_movable_obstacles()
        for m in movables:
            d = m.polygon.distance(self.polygon)

            if d > self.circumscribed_radius:
                continue

            angle = utils.get_angle_to_turn(self.pose, m.pose)
            if np.abs(angle) > 10:
                continue

            return ba.Grab(m.uid)

    def _release(self) -> ba.Release | None:
        if self.world.is_holding_obstacle(self.uid):
            return ba.Release(self.world.entity_to_agent.inverse[self.uid], distance=0)

    def copy(self) -> Self:
        return PPOAgent(
            uid=self.uid,
            navigation_goals=copy.deepcopy(self._navigation_goals),
            config=self.config,
            logs_dir=self.logs_dir,
            full_geometry_acquired=self.full_geometry_acquired,
            polygon=copy.deepcopy(self.polygon),
            style=copy.deepcopy(self.agent_style),
            pose=copy.deepcopy(self.pose),
            sensors=copy.deepcopy(self.sensors),  # type: ignore
            cell_size=self.cell_size,
            logger=self.logger,
        )

    def sample_random_robot_pose(
        self,
    ) -> PoseModel:
        accessible_cells: t.Set[GridCellModel] = set()
        for i in range(self.robot_inflated_grid.d_width):
            for j in range(self.robot_inflated_grid.d_height):
                if self.robot_inflated_grid.grid[i][j] == 0:
                    accessible_cells.add((i, j))

        if len(accessible_cells) == 0:
            raise Exception("No accessible cells")

        rand_cell = random.choice(tuple(accessible_cells))
        cell_center = self.robot_inflated_grid.get_cell_center(rand_cell)
        rand_pose = (
            cell_center[0],
            cell_center[1],
            random.uniform(0.0, 360.0),
        )

        return rand_pose

    def sample_random_pose_from_cells(self, cells: t.List[GridCellModel]) -> PoseModel:
        rand_cell = random.choice(cells)
        cell_center = self.robot_inflated_grid.get_cell_center(rand_cell)
        rand_pose = (
            cell_center[0],
            cell_center[1],
            random.uniform(0.0, 360.0),
        )
        return rand_pose

    def sample_random_goal(self) -> Goal:
        rand_pose = self.sample_random_robot_pose()
        return Goal(
            uid="rand",
            polygon=Point(rand_pose[0], rand_pose[1]).buffer(self.circumscribed_radius),
            pose=rand_pose,
        )

    def run_dijkstra(
        self, robot_cell: GridCellModel, static_grid: BinaryOccupancyGrid
    ) -> t.Dict[GridCellModel, float]:
        def get_neighbors(
            current: GridCellModel,
            gscore: t.Dict[GridCellModel, float],
            close_set: t.Set[GridCellModel],
            open_queue: t.List[GridCellModel],
            came_from: t.Dict[GridCellModel, GridCellModel | None],
        ) -> t.Tuple[t.List[GridCellModel], t.List[float]]:
            grid = static_grid.grid
            neighbors, tentative_gscores = [], []

            current_gscore = gscore[current]
            for i, j in utils.CHESSBOARD_NEIGHBORHOOD:
                neighbor = current[0] + i, current[1] + j
                if neighbor in close_set or not utils.is_in_matrix(
                    cell=neighbor,
                    width=static_grid.d_width,
                    height=static_grid.d_height,
                ):
                    continue

                if grid[neighbor[0]][neighbor[1]] != 0:
                    continue

                    # Check if grid cell is completed coveraged by one or more obstacle polygons.
                    # In this case, the robot could never enter the cell.
                    # cell_polygon = static_grid.cell_to_polygon(neighbor)
                    # obstacles = [self.world.entities[obs_id].polygon for obs_id in static_grid.cell_to_obstacle_ids(neighbor)]
                    # obstacles_union: Polygon = shapely.unary_union(obstacles)
                    # if obstacles_union.contains(cell_polygon):
                    #     continue

                neighbors.append(neighbor)
                tentative_gscores.append(current_gscore + math.sqrt(i**2 + j**2))

            return neighbors, tentative_gscores

        def exit_condition(current: t.Any):
            return False

        _, _, _, _, gscore, _ = graph_search.new_generic_dijkstra(
            exit_condition=exit_condition,
            start=robot_cell,
            get_neighbors=get_neighbors,
        )  # type: ignore

        return gscore

    def debug_dist_to_goal(
        self, prev_robot_cell: GridCellModel, robot_cell: GridCellModel
    ):
        # debug images
        self.static_obstacle_grid.to_image().save("static.png")
        self.robot_inflated_grid.to_image().save("inflated.png")
        self.world.to_image(width=self.static_obstacle_grid.grid.shape[0]).save(
            "world.png"
        )
        self.dist_to_goal_to_image(
            prev_robot_cell=prev_robot_cell, robot_cell=robot_cell
        ).save("dist_to_goal.png")

    def dist_to_goal_to_image(
        self, prev_robot_cell: GridCellModel, robot_cell: GridCellModel
    ) -> Image.Image:
        grid = np.zeros_like(self.static_obstacle_grid.grid)
        for cell, value in self.dist_to_goal.items():
            grid[cell[0]][cell[1]] = value

        # grid = np.flipud(self.grid)
        grid = grid.astype(np.float32)
        grid[grid == -1] = 1
        grid = grid - np.min(grid)
        grid /= np.max(grid)
        grid *= 255
        grid = grid.astype(np.uint8)
        grid = np.expand_dims(
            grid, axis=2
        )  # Add a new axis to make it (height, width, 1)
        grid = np.tile(grid, (1, 1, 3))
        grid[robot_cell[0]][robot_cell[1]][0] = 255
        grid[robot_cell[0]][robot_cell[1]][1] = 0
        grid[robot_cell[0]][robot_cell[1]][2] = 0
        grid[prev_robot_cell[0]][prev_robot_cell[1]][0] = 0
        grid[prev_robot_cell[0]][prev_robot_cell[1]][1] = 255
        grid[prev_robot_cell[0]][prev_robot_cell[1]][2] = 0
        img = Image.fromarray(grid, "RGB")
        return img

    def get_dist_to_goal_grid(self) -> npt.NDArray[np.float32]:
        grid = np.zeros_like(self.static_obstacle_grid.grid)
        for cell, value in self.dist_to_goal.items():
            grid[cell[0]][cell[1]] = value

        grid = grid.astype(np.float32)
        grid = grid - np.min(grid)
        grid /= np.max(grid)
        grid = np.transpose(grid)  # (x, y) -> (y, x)
        return grid
