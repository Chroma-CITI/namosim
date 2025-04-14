import sys
import concurrent
import gc
import json
import multiprocessing
import multiprocessing.pool
import os
import queue as pyqueue
import random
import time
import typing as t
from multiprocessing.pool import AsyncResult

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import transformers
from PIL import Image
from shapely.geometry import Point
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

import wandb
from namosim.agents.models import PPOActor, PPOCritic
from namosim.agents.ppo_agent import PPOAgent, State
from namosim.data_models import PPOAgentConfigModel
from namosim.log import logger
from namosim.navigation.action_result import ActionResult, ActionSuccess
from namosim.navigation.basic_actions import Action
from namosim.utils import utils
from namosim.world.sensors.omniscient_sensor import OmniscientSensor
from namosim.world.world import World

mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy("file_system")

transformers.utils.logging.set_verbosity_error()
torch.set_num_threads(1)

GAMMA = 0.97
TRAJECTORIES_PER_BATCH = 50
UPDATES_PER_LEARNING_STEP = 3
MAX_STEPS_PER_EPISODE = 100
ENTROPY_LOSS_COEF = 0.5
V_LOSS_CEOF = 0.5
SSL_LOSS_COEF = 0
RECONSTRUCTION_LOSS_COEF = 1e-1
ACTOR_LOSS_COEF = 1
CLIP = 0.2
LR = 1e-4
MINI_BATCH_SIZE = 64


class Transition(t.NamedTuple):
    state: State
    action: int
    action_obj: Action
    action_result: ActionResult
    log_prob: float
    next_state: State
    reward: float
    goal_reached: bool


class TrajectorySample(t.NamedTuple):
    state: State
    action: int
    log_prob: float
    next_state: State
    reward: float
    future_rewards: float
    goal_reached: bool


class Example(t.NamedTuple):
    start_state: State
    goal_grid: npt.NDArray[np.float32]
    action: int
    action_log_prob: float
    G: float  # sum of discounted future rewards


class TrajectoryExamples(t.NamedTuple):
    true_trajectory_examples: t.List[Example]
    pseudo_trajectory_examples: t.List[Example]


class Trajectory:
    def __init__(self):
        self.transitions: t.List[Transition] = []
        self.total_reward = 0.0

    def add_transition(self, transition: Transition):
        self.transitions.append(transition)
        self.total_reward += transition.reward

    def to_samples(self) -> t.List[TrajectorySample]:
        result: t.List[TrajectorySample] = []

        n = len(self.transitions)
        gammas = GAMMA ** np.arange(0, n)
        rewards = np.array([trans.reward for trans in self.transitions])
        for i, trans in enumerate(self.transitions):
            future_rewards: float = float(np.sum(gammas[: n - i] * rewards[i:]))
            result.append(
                TrajectorySample(
                    state=trans.state,
                    action=trans.action,
                    log_prob=trans.log_prob,
                    next_state=trans.next_state,
                    reward=trans.reward,
                    future_rewards=future_rewards,
                    goal_reached=trans.goal_reached,
                )
            )
        return result

    def to_examples(self) -> TrajectoryExamples:
        true_trajectory_examples: t.List[Example] = []
        pseudo_trajectory_examples: t.List[Example] = []
        n = len(self.transitions)
        gammas = GAMMA ** np.arange(0, n, dtype=np.float32)
        rewards = np.array([trans.reward for trans in self.transitions])

        # gather true trajectory examples
        for i, trans in enumerate(self.transitions):
            G = float(np.sum(gammas[: n - i] * rewards[i:]))
            true_trajectory_examples.append(
                Example(
                    start_state=trans.state,
                    goal_grid=trans.state.goal_state_grid,
                    action=trans.action,
                    action_log_prob=trans.log_prob,
                    G=G,
                )
            )

        # gather pseudo trajectory examples
        # last_non_repeated_idx = 0
        # start_pose = utils.real_pose_to_fixed_precision_pose(
        #     self.transitions[0].state.robot_pose, trans_mult=10, rot_mult=1
        # )
        # visited_poses: t.Set[PoseModel] = set([start_pose])

        # for i, trans_i in enumerate(self.transitions):
        #     next_pose = utils.real_pose_to_fixed_precision_pose(
        #         trans_i.next_state.robot_pose, trans_mult=10, rot_mult=1
        #     )
        #     if next_pose in visited_poses:
        #         break
        #     visited_poses.add(next_pose)
        #     last_non_repeated_idx = i

        # last_non_repeated_trans = self.transitions[last_non_repeated_idx]
        # start_transition = self.transitions[0]

        # for i in range(last_non_repeated_idx + 1):
        #     G = -np.sum(gammas[: last_non_repeated_idx - i]) + gammas[last_non_repeated_idx - i] * 100  # type: ignore
        #     pseudo_trajectory_examples.append(
        #         Example(
        #             start_state=self.transitions[i].state,
        #             goal_grid=last_non_repeated_trans.next_state.grid,
        #             action=start_transition.action,
        #             action_log_prob=self.transitions[i].log_prob,
        #             G=G,
        #         )
        #     )

        # if len(pseudo_trajectory_examples) < 4:
        #     pseudo_trajectory_examples = []

        return TrajectoryExamples(
            true_trajectory_examples=true_trajectory_examples,
            pseudo_trajectory_examples=pseudo_trajectory_examples,
        )
        # for i, trans_i in enumerate(self.transitions):
        #     visited_poses: t.Set[PoseModel] = set()
        #     start_pose = utils.real_pose_to_fixed_precision_pose(
        #         trans_i.state.robot_pose, trans_mult=10, rot_mult=1
        #     )
        #     visited_poses.add(start_pose)

        #     for j, trans_j in enumerate(self.transitions[i:]):
        #         end_pose = utils.real_pose_to_fixed_precision_pose(
        #             trans_j.next_state.robot_pose, trans_mult=10, rot_mult=1
        #         )
        #         if end_pose in visited_poses:
        #             continue
        #         else:
        #             visited_poses.add(end_pose)

        #         G = -np.sum(gammas[:j]) + gammas[j] * 100

        #         result.append(
        #             Example(
        #                 start_state=trans_i.state,
        #                 goal_state=trans_j.next_state,
        #                 action=trans_i.action,
        #                 action_log_prob=trans_i.log_prob,
        #                 G=G,
        #             )
        #         )
        # return result


def render_trajectory(images: t.List[npt.NDArray[np.float32]]):
    # Create a window
    cv2.namedWindow("NAMO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NAMO", 500, 500)
    for np_img in images:
        # Convert RGB to BGR (OpenCV uses BGR color order)
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # Display the image in the window
        cv2.imshow("NAMO", bgr_img)

        # Delay between frames (in milliseconds). Adjust as needed.
        cv2.waitKey(16)  # Change the value to adjust frame rate

    cv2.destroyAllWindows()


def get_trajectory(idx: int, actor: PPOActor) -> Trajectory | None:
    try:
        now = int(time.time() + idx)
        np.random.seed(now)
        random.seed(now)
        logger.info("Generating world")
        # robot_radius = 60
        # map = MapGen(height=20, width=20, init_open=0.42)
        # map.gen_map()
        # world = map.to_namo_world(robot_radius_cm=robot_radius)

        robot_radius = 30
        world = World.load_from_svg("tests/experiments/scenarios/tiny_square_world.svg")

        logger.info("Creating agent")

        agent_polygon = Point(0, 0).buffer(robot_radius)
        agent = PPOAgent(
            config=PPOAgentConfigModel(type="ppo_agent"),
            navigation_goals=[],
            logs_dir="namo_logs",
            uid="robot_0",
            full_geometry_acquired=True,
            polygon=agent_polygon,
            pose=(0, 0, 0),
            sensors=[OmniscientSensor()],
            logger=utils.NamosimLogger(),
            cell_size=world.map.cell_size,
        )

        logger.info("Initializing agent")

        logger.info("Initializing agent world")
        world.add_agent(agent)
        agent.init(world)

        logger.info("Setting agent actor model")

        agent.set_actor(actor=actor)

        logger.info("Staring a new episode")
        agent.start_new_episode(ref_world=world)

        traj = Trajectory()
        agent.sense(
            ref_world=world,
            last_action_result=ActionSuccess(),
            step_count=0,
        )
        logger.info("Generating trajectory")

        for step_count in range(1, MAX_STEPS_PER_EPISODE):
            step_result = agent.step()

            action = step_result.action
            action_results = world.step(
                actions={agent.uid: action},
                step_count=step_count,
            )

            agent.sense(
                ref_world=world,
                last_action_result=ActionSuccess(),
                step_count=step_count,
            )

            next_state = agent.get_state()

            traj.add_transition(
                Transition(
                    state=step_result.state,
                    action=step_result.action_idx,
                    action_obj=step_result.action,
                    log_prob=step_result.action_log_prob,
                    next_state=next_state,
                    reward=agent.get_reward(
                        next_state=next_state,
                        state=step_result.state,
                        action=action,
                        action_result=action_results[agent.uid],
                    ),
                    action_result=action_results[agent.uid],
                    goal_reached=next_state.goal_reached,
                )
            )

            if next_state.goal_reached:
                break

        torch.cuda.empty_cache()

        if len(traj.transitions) > 1:
            return traj
        return None
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        return None
    finally:
        torch.cuda.empty_cache()


def log_trajectory(traj: Trajectory, training_iter: int):
    traj_examples = traj.to_examples()

    true_traj_video = examples_to_video(traj_examples.true_trajectory_examples)
    save_video(true_traj_video, "true_traj.mp4", training_iter)

    # if len(traj_examples.pseudo_trajectory_examples) > 0:
    #     pseudo_traj_video = examples_to_video(traj_examples.pseudo_trajectory_examples)
    #     save_video(pseudo_traj_video, f"pseudo_traj.mp4", training_iter)


def floating_img_to_uint8(img: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    min_value, max_value = img.min(), img.max()
    img = (img - min_value) / (max_value - min_value)
    return (img * 255).astype(np.uint8)


def examples_to_video(examples: t.List[Example]):
    frames: t.List[npt.NDArray[np.uint8]] = []
    for ex in examples:
        state_img = floating_img_to_uint8(ex.start_state.grid)
        # goal_img = floating_img_to_uint8(ex.goal_grid)
        # frame = np.concatenate((state_img, goal_img), axis=1)
        # frame = np.stack((frame,) * 3, axis=0)  # gray to rgb

        frames.append(state_img)

    video = np.stack(frames, axis=0)  # (n, c, h, w)
    return video


def save_video(video: npt.NDArray[np.uint8], filename: str, training_iter: int):
    wandb.log({filename: wandb.Video(video, fps=16), "training_iter": training_iter})
    out = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        16,
        (video.shape[3], video.shape[2]),
    )
    for frame in video:
        frame = frame.transpose(1, 2, 0)  # (h, w, c)
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)  # frame is a numpy.ndarray with shape (w, h, 3)
    out.release()


def get_trajectory_with_timeout(idx: int, actor: PPOActor) -> Trajectory | None:
    torch.cuda.empty_cache()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(get_trajectory, idx, actor)
        try:
            traj = future.result(timeout=300 if __debug__ else 30)
            return traj
        except concurrent.futures.TimeoutError:
            logger.warn("Function execution timed out")
        except Exception as e:
            logger.exception("An error occurred: %s", str(e))


class TransitionBatch(t.NamedTuple):
    batch_total_reward: float
    state_grids: torch.Tensor
    state_robot_poses: torch.Tensor
    state_goal_poses: torch.Tensor
    R: torch.Tensor
    actions: torch.Tensor
    T: torch.Tensor
    log_probs: torch.Tensor


class ExampleBatch(t.NamedTuple):
    batch_total_reward: float
    start_state_grids: torch.Tensor
    goal_state_grids: torch.Tensor
    state_robot_poses: torch.Tensor
    state_goal_poses: torch.Tensor
    actions: torch.Tensor
    action_log_probs: torch.Tensor
    T: torch.Tensor
    G: torch.Tensor  # sum of discounted future rewards


class TransitionBatchDataset(Dataset[TransitionBatch]):  # type: ignore
    def __init__(self, trajectories: t.List[Trajectory]):
        self.trajectories = trajectories
        self.batch = self.trajectories_to_batch()

    def __len__(self):
        return len(self.batch.T)

    def trajectories_to_batch(self) -> TransitionBatch:
        batch_total_reward = 0
        T_ls = []
        R_ls = []
        state_grids_ls = []
        state_robot_poses_ls = []
        state_goal_poses_ls = []

        log_probs_ls = []
        actions_ls = []
        for traj in self.trajectories:
            batch_total_reward += traj.total_reward
            samples = traj.to_samples()
            for t_step, samp in enumerate(samples):
                T_ls.append(t_step)
                R_ls.append(samp.future_rewards)
                state_grids_ls.append(samp.state.grid)
                state_robot_poses_ls.append(samp.state.normalized_robot_pose)
                state_goal_poses_ls.append(samp.state.normalized_goal_pose)
                actions_ls.append(samp.action)
                log_probs_ls.append(samp.log_prob)

        state_grids = torch.tensor(state_grids_ls).float().unsqueeze(1)
        state_robot_poses = torch.tensor(state_robot_poses_ls).float()
        state_goal_poses = torch.tensor(state_goal_poses_ls).float()
        R = torch.tensor(R_ls).float().unsqueeze(-1)
        actions = torch.tensor(actions_ls).long().unsqueeze(-1)
        T = torch.tensor(T_ls).float().unsqueeze(-1)
        log_probs = torch.tensor(log_probs_ls).float().unsqueeze(-1)
        return TransitionBatch(
            batch_total_reward,
            state_grids,
            state_robot_poses,
            state_goal_poses,
            R,
            actions,
            T,
            log_probs,
        )

    def __getitem__(self, idx: int):
        item = (
            self.batch.state_grids[idx],
            self.batch.state_robot_poses[idx],
            self.batch.state_goal_poses[idx],
            self.batch.R[idx],
            self.batch.actions[idx],
            self.batch.T[idx],
            self.batch.log_probs[idx],
        )
        return item


class ExampleBatchDataset(Dataset[ExampleBatch]):  # type: ignore
    def __init__(self, trajectories: t.List[Trajectory]):
        self.trajectories = trajectories
        self.batch = self.trajectories_to_batch()

    def __len__(self):
        return self.batch.start_state_grids.shape[0]

    def trajectories_to_batch(self) -> ExampleBatch:
        start_state_grids_ls = []
        goal_state_grids_ls = []
        state_robot_poses_ls = []
        state_goal_poses_ls = []
        log_probs_ls = []
        actions_ls = []
        T_ls = []
        G_ls = []
        batch_total_reward = 0
        for traj in self.trajectories:
            traj_examples = traj.to_examples()
            examples = (
                traj_examples.true_trajectory_examples
                # + traj_examples.pseudo_trajectory_examples
            )
            batch_total_reward += traj.total_reward
            for i, ex in enumerate(examples):
                start_state_grids_ls.append(ex.start_state.grid)
                goal_state_grids_ls.append(ex.goal_grid)
                state_robot_poses_ls.append(ex.start_state.normalized_robot_pose)
                state_goal_poses_ls.append(ex.start_state.normalized_goal_pose)
                actions_ls.append(ex.action)
                log_probs_ls.append(ex.action_log_prob)
                T_ls.append(i)
                G_ls.append(ex.G)

        start_state_grids = torch.tensor(np.array(start_state_grids_ls)).float()
        goal_state_grids = torch.tensor(np.array(goal_state_grids_ls)).float()
        state_robot_poses = torch.tensor(np.array(state_robot_poses_ls)).float()
        state_goal_poses = torch.tensor(np.array(state_goal_poses_ls)).float()
        actions = torch.tensor(np.array(actions_ls)).long().unsqueeze(-1)
        action_log_probs = torch.tensor(np.array(log_probs_ls)).float().unsqueeze(-1)
        T = torch.tensor(np.array(T_ls)).float().unsqueeze(-1)
        G = torch.tensor(np.array(G_ls)).float().unsqueeze(-1)

        return ExampleBatch(
            batch_total_reward,
            start_state_grids,
            goal_state_grids,
            state_robot_poses,
            state_goal_poses,
            actions,
            action_log_probs,
            T,
            G,
        )

    def __getitem__(self, idx: int):
        item = (
            self.batch.start_state_grids[idx],
            self.batch.goal_state_grids[idx],
            self.batch.state_robot_poses[idx],
            self.batch.state_goal_poses[idx],
            self.batch.actions[idx],
            self.batch.action_log_probs[idx],
            self.batch.T[idx],
            self.batch.G[idx],
        )
        return item


class Trainer:
    def __init__(self, num_workers: int = 24):
        self.max_iters = int(1e7)
        self.max_steps_per_trajectory = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PPOActor(action_size=7).to(self.device)
        self.critic = PPOCritic().to(self.device)
        self.actor.share_memory()
        self.optimizer = optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LR,
            amsgrad=True,
        )
        self.learning_step_count = 0
        self.best_batch_total_reward = float("-inf")
        self.checkpoint: str = "ppo_parallel.pt"
        self.workers: t.List[mp.Process] = []
        self.num_workers = num_workers

        if os.path.exists(self.checkpoint):
            data = torch.load(self.checkpoint)
            self.actor.load_state_dict(data["actor_state_dict"])
            self.critic.load_state_dict(data["critic_state_dict"])

        wandb.init(
            project="namo_ppo_parallel",
            config={
                "LR": LR,
                "GAMMA": GAMMA,
                "ENTROPY_LOSS_COEF": ENTROPY_LOSS_COEF,
                "V_LOSS_CEOF": V_LOSS_CEOF,
                "TRAJECTORIES_PER_BATCH": TRAJECTORIES_PER_BATCH,
            },
        )

    def memory_stats(self):
        print("GPU Memory Allocated: ", torch.cuda.memory_allocated() / 1024**2)
        print("GPU Memory Cached: ", torch.cuda.memory_cached() / 1024**2)

    def get_trajectories_parallel(
        self, queue: pyqueue.Queue[Trajectory]
    ) -> t.List[Trajectory]:
        result = []
        while len(result) < TRAJECTORIES_PER_BATCH:
            try:
                logger.info(
                    f"Getting trajectory from queue. {len(result)}/{TRAJECTORIES_PER_BATCH}"
                )
                traj = queue.get(timeout=1.0)
                logger.info("Got trajectory from queue.")
                result.append(traj)
            except pyqueue.Empty:
                logger.info("Waiting to get trajectory from queue.")
        return result

    def get_batch(self, pool: multiprocessing.pool.Pool) -> t.List[Trajectory]:
        torch.cuda.empty_cache()
        trajectories = []
        futures: t.List[AsyncResult] = []
        for i in range(TRAJECTORIES_PER_BATCH):
            futures.append(
                pool.apply_async(get_trajectory_with_timeout, args=(i, self.actor))
            )

        for f in futures:
            try:
                traj = f.get(timeout=30)
                if traj is not None:
                    trajectories.append(traj)
            except mp.TimeoutError:
                logger.exception("Timeout while getting trajectory")
                continue

        return trajectories

    def train(self):
        with mp.Pool(self.num_workers) as pool:
            for training_iter in range(self.max_iters):
                torch.cuda.empty_cache()
                gc.collect()
                trajectories = self.get_batch(pool)
                logger.info(f"Get batch of {len(trajectories)} trajectories.")
                for i, traj in enumerate(trajectories):
                    if i < 3:
                        log_trajectory(traj, training_iter=training_iter)
                    # if traj.transitions[-1].goal_reached:
                    #     inspect_trajectory(traj)
                torch.cuda.empty_cache()
                self.learn(trajectories, training_iter=training_iter)
                del trajectories
                # pool.close()
                # pool.terminate()
                # pool.join()
                # del pool
                # torch.cuda.empty_cache()

    def train_sequential(self):
        for training_iter in range(self.max_iters):
            torch.cuda.empty_cache()
            trajectories = []
            for i in range(TRAJECTORIES_PER_BATCH):
                traj = get_trajectory(i, self.actor)
                if traj:
                    trajectories.append(traj)

            torch.cuda.empty_cache()
            self.learn(trajectories, training_iter)

    def learn(self, trajectories: t.List[Trajectory], training_iter: int):
        torch.cuda.empty_cache()
        # assert len(trajectories) == TRAJECTORIES_PER_BATCH
        dataset = ExampleBatchDataset(trajectories=trajectories)
        dataloader = DataLoader(
            dataset,
            batch_size=MINI_BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda x: tuple(x_ for x_ in default_collate(x)),  # type: ignore
            num_workers=0,
        )

        if dataset.batch.batch_total_reward > self.best_batch_total_reward:
            self.best_batch_total_reward = dataset.batch.batch_total_reward
            print(
                f"New best batch total reward: {dataset.batch.batch_total_reward}. Saving models."
            )
            torch.save(
                {
                    "actor_state_dict": self.actor.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "best_batch_total_reward": self.best_batch_total_reward,
                },
                self.checkpoint,
            )

        print(f"Learning on {len(trajectories)} trajectories")

        for i in range(UPDATES_PER_LEARNING_STEP):
            total_loss = 0.0
            total_action_loss = 0.0
            total_entropy_loss = 0.0
            total_ssl_loss = 0.0
            total_critic_loss = 0.0

            for batch in dataloader:
                (
                    start_state_grids,
                    goal_state_grids,
                    state_robot_poses,
                    state_goal_poses,
                    actions,
                    log_probs,
                    T,
                    G,
                ) = batch

                start_state_grids = start_state_grids.to(self.device)
                goal_state_grids = goal_state_grids.to(self.device)
                state_robot_poses = state_robot_poses.to(self.device)
                state_goal_poses = state_goal_poses.to(self.device)
                actions = actions.to(self.device)
                log_probs = log_probs.to(self.device).detach()
                T = T.to(self.device)  # type: ignore
                G = G.to(self.device)  # type: ignore

                all_action_probs, pred_robot_poses, pred_goal_poses = self.actor(
                    start_state_grids
                )
                action_probs = all_action_probs.gather(-1, actions)
                curr_log_probs = torch.log(action_probs + 1e-10)

                ratios = torch.exp(curr_log_probs - log_probs)

                V = self.critic(start_state_grids, T)
                A = G - V.detach()
                A = (A - A.mean()) / (A.std(correction=0) + 1e-10)  # type: ignore
                surr1 = ratios * A
                surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * A

                critic_loss = nn.MSELoss()(G, V)
                actor_loss = -torch.min(surr1, surr2)
                actor_loss = actor_loss.mean()
                ssl_loss = SSL_LOSS_COEF * (
                    nn.MSELoss()(state_robot_poses, pred_robot_poses)
                    + nn.MSELoss()(state_goal_poses, pred_goal_poses)
                )

                # actor_loss = -torch.mean(curr_log_probs * G)

                entropy_loss = ENTROPY_LOSS_COEF * torch.mean(
                    torch.log(all_action_probs + 1e-10) * all_action_probs
                )

                loss = actor_loss + entropy_loss + ssl_loss + critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad.clip_grad_value_(self.actor.parameters(), 1)
                # torch.nn.utils.clip_grad.clip_grad_value_(self.critic.parameters(), 1)
                self.optimizer.step()

                total_loss += loss.item()
                total_action_loss += actor_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_ssl_loss += ssl_loss.item()
                total_critic_loss += critic_loss.item()

                print(
                    f"loss: {loss}, actor_loss: {actor_loss}, ssl_loss: {ssl_loss}, critic_loss: {critic_loss}"
                )

            if i == 0:
                wandb.log(
                    {
                        "total_loss": total_loss,
                        "total_action_loss": total_action_loss,
                        "total_entropy_loss": total_entropy_loss,
                        "total_ssl_loss": total_ssl_loss,
                        "total_critic_loss": total_critic_loss,
                        "training_iter": training_iter,
                        "batch_total_reward": dataset.batch.batch_total_reward,
                    }
                )

        self.learning_step_count += 1


def inspect_trajectory(traj: Trajectory):
    """Logs transition in a trajectory to a local folder for debugging purposes."""
    log_dir = "debug_trajectory"
    os.makedirs(log_dir, exist_ok=False)

    # Save transition images to the log directory
    for i, trans in enumerate(traj.transitions):
        transition_dir = os.path.join(log_dir, f"{i}".zfill(6))
        os.makedirs(transition_dir, exist_ok=True)

        # save state image
        state_img = floating_img_to_uint8(trans.state.grid)
        state_img = np.transpose(state_img, (1, 2, 0))
        Image.fromarray(state_img, mode="RGB").save(
            os.path.join(transition_dir, "state.png")
        )

        # save next_state image
        next_state_img = floating_img_to_uint8(trans.next_state.grid)
        next_state_img = np.transpose(next_state_img, (1, 2, 0))
        Image.fromarray(next_state_img, mode="RGB").save(
            os.path.join(transition_dir, "next_state.png")
        )

        with open(os.path.join(transition_dir, "info.json"), "w") as f:
            json.dump(
                {
                    "action": str(trans.action_obj),
                    "action_idx": int(trans.action),
                    "action_result": str(trans.action_result),
                    "reward": float(trans.reward),
                    "action_log_prob": float(trans.log_prob),
                    "goal_reached": trans.goal_reached,
                    "robot_pose": str(trans.state.robot_pose),
                    "goal_pose": str(trans.state.goal_pose),
                    "next_robot_pose": str(trans.next_state.robot_pose),
                    "next_goal_pose": str(trans.next_state.goal_pose),
                },
                f,
                indent=2,
            )
