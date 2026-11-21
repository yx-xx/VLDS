from __future__ import annotations

import collections
import contextlib
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys

import cv2
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.simulation_viewer import InteractiveViewer
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task_ids: list[int] | None = None
    max_steps: int | None = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    viewer: bool = False
    camera_preview: bool = False

    seed: int = 7  # Random Seed (for reproducibility)


class CameraPreview:
    def __init__(self):
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise ValueError("Live preview requires a desktop DISPLAY; omit --viewer for headless evaluation.")
        self.window_name = "OpenPI Franka Panda - scene / wrist"
        self.enabled = True
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1024, 608)
        except cv2.error as error:
            raise RuntimeError("Cannot open live preview; check OpenCV GUI support or omit --viewer.") from error

    def show(self, observation, *, task_id: int, episode_idx: int, step: int, status: str) -> None:
        if not self.enabled:
            return
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        if visible < 1:
            self.close()
            return
        images = np.concatenate(
            (observation["agentview_image"][::-1, ::-1], observation["robot0_eye_in_hand_image"][::-1, ::-1]), axis=1
        )
        frame = cv2.copyMakeBorder(cv2.cvtColor(images, cv2.COLOR_RGB2BGR), 44, 0, 0, 0, cv2.BORDER_CONSTANT)
        labels = [
            f"Task {task_id} | Trial {episode_idx + 1} | Step {step} | {status}",
            "Scene / wrist | Q / Esc: hide | Ctrl+C in terminal: stop",
        ]
        for row, label in enumerate(labels):
            cv2.putText(frame, label, (8, 16 + row * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            self.close()

    def close(self) -> None:
        if self.enabled:
            self.enabled = False
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                logging.debug("Preview window already closed")


def eval_libero(args: Args) -> None:
    with contextlib.ExitStack() as resources:
        viewer = resources.enter_context(contextlib.closing(InteractiveViewer())) if args.viewer else None
        camera_preview = resources.enter_context(contextlib.closing(CameraPreview())) if args.camera_preview else None
        _eval_libero(args, viewer, camera_preview)


def _eval_libero(args: Args, viewer: InteractiveViewer | None, camera_preview: CameraPreview | None = None) -> None:
    def show(observation, *, task_id, episode_idx, step, status):
        if viewer is not None:
            viewer.sync()
        if camera_preview is not None:
            camera_preview.show(observation, task_id=task_id, episode_idx=episode_idx, step=step, status=status)

    if args.num_trials_per_task <= 0 or args.replan_steps <= 0 or args.num_steps_wait < 0:
        raise ValueError("Trials and replan steps must be positive; wait steps must be nonnegative.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = list(range(num_tasks_in_suite)) if args.task_ids is None else args.task_ids
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be nonempty and contain no duplicates.")
    if any(task_id < 0 or task_id >= num_tasks_in_suite for task_id in task_ids):
        raise ValueError(f"task_ids must be between 0 and {num_tasks_in_suite - 1}.")
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.max_steps is not None:
        max_steps = min(max_steps, args.max_steps)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    episodes = []
    errors = []
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        if args.num_trials_per_task > len(initial_states):
            raise ValueError(f"Task {task_id} has only {len(initial_states)} initial states.")

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        try:
            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")

                # Reset environment
                if viewer is not None:
                    viewer.detach()
                env.reset()
                action_plan = collections.deque()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])
                if viewer is not None:
                    viewer.attach(env)

                # Setup
                t = 0
                done = False
                replay_images = []

                logging.info(f"Starting episode {task_episodes + 1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            if viewer is not None:
                                viewer.wait_for_step()
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            show(obs, task_id=task_id, episode_idx=episode_idx, step=0, status="Settling")
                            t += 1
                            continue

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                            }

                            # Query model to get action
                            show(
                                obs,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                step=t - args.num_steps_wait,
                                status="Policy inference (first call may compile)",
                            )
                            action_chunk = np.asarray(client.infer(element)["actions"])
                            if (
                                action_chunk.ndim != 2
                                or action_chunk.shape[1] != 7
                                or not np.isfinite(action_chunk).all()
                            ):
                                raise ValueError("LIBERO requires a finite action chunk with shape (horizon, 7).")
                            if len(action_chunk) < args.replan_steps:
                                raise ValueError(
                                    f"Replanning needs {args.replan_steps} steps, but policy returned {len(action_chunk)}."
                                )
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Execute action in environment
                        if viewer is not None:
                            viewer.wait_for_step()
                        obs, reward, done, info = env.step(action.tolist())
                        show(
                            obs,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            step=t - args.num_steps_wait + 1,
                            status="SUCCESS" if done else "Running",
                        )
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.exception("Rollout failed")
                        errors.append(f"Task {task_id}, episode {episode_idx}: {e}")
                        break

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                video_path = pathlib.Path(args.video_out_path) / (
                    f"rollout_{task_id:03d}_{episode_idx:03d}_{task_segment}_{suffix}.mp4"
                )
                if replay_images:
                    imageio.mimwrite(video_path, [np.asarray(image) for image in replay_images], fps=10)
                episodes.append(
                    {
                        "task_id": task_id,
                        "episode_id": episode_idx,
                        "instruction": task_description,
                        "success": bool(done),
                        "video": str(video_path) if replay_images else None,
                    }
                )

                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        finally:
            if viewer is not None:
                viewer.detach()
            env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    results = {
        "robot": "Panda",
        "task_suite": args.task_suite_name,
        "seed": args.seed,
        "viewer": args.viewer,
        "camera_preview": args.camera_preview,
        "max_steps": max_steps,
        "replan_steps": args.replan_steps,
        "num_trials_per_task": args.num_trials_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "episodes": episodes,
        "errors": errors,
    }
    results_path = pathlib.Path(args.video_out_path) / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    logging.info("Results saved to %s", results_path)
    if errors:
        raise RuntimeError(f"{len(errors)} rollout(s) failed; see {results_path}.")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "robots": ["Panda"],
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
