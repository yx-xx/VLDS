"""One persistent inspection scene. JSON commands arrive on stdin; no dataset evaluation loop."""

# ruff: noqa: RUF001 -- Chinese interface text uses Chinese punctuation.

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import queue
import signal
import sys
import threading
import time
import traceback

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from openpi_client.embedded_viewer import EmbeddedViewer
from openpi_client.simulation_viewer import InteractiveViewer

EVENT_PREFIX = "@simulation "


def emit(**event):
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False), flush=True)


def validate_actions(actions, dimensions):
    chunk = np.asarray(actions)
    if chunk.ndim != 2 or not len(chunk) or chunk.shape[1] != dimensions or not np.isfinite(chunk).all():
        raise ValueError(f"模型动作须为有限数值数组 (horizon, {dimensions})，实际为 {chunk.shape}。")
    return chunk


class LiberoScene:
    action_dimensions = 7

    def __init__(self, args):
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[args.suite]()
        task = suite.get_task(args.task_id)
        self.instruction = task.language
        self.initial_states = suite.get_task_init_states(args.task_id)
        self.initial_count = len(self.initial_states)
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            robots=["Panda"],
            camera_heights=256,
            camera_widths=256,
        )
        self.seed = args.seed
        self.max_steps = {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
            "libero_90": 400,
        }[args.suite]

    def reset(self, index):
        if not 0 <= index < self.initial_count:
            raise ValueError(f"初始状态编号应为 0 至 {self.initial_count - 1}。")
        self.env.seed(self.seed)
        self.env.reset()
        self.obs = self.env.set_init_state(self.initial_states[index])
        for _ in range(10):
            self.obs, _, _, _ = self.env.step([0.0] * 6 + [-1.0])

    def attach(self, viewer):
        viewer.attach(self.env)

    def observation(self):
        obs = self.obs
        quaternion = np.array(obs["robot0_eef_quat"], copy=True)
        w = np.clip(quaternion[3], -1.0, 1.0)
        denominator = math.sqrt(1.0 - w * w)
        rotation = (
            np.zeros(3) if math.isclose(denominator, 0.0) else quaternion[:3] * (2.0 * math.acos(w) / denominator)
        )
        return {
            "observation/image": self._image(obs["agentview_image"]),
            "observation/wrist_image": self._image(obs["robot0_eye_in_hand_image"]),
            "observation/state": np.concatenate((obs["robot0_eef_pos"], rotation, obs["robot0_gripper_qpos"])),
            "prompt": self.instruction,
        }

    @staticmethod
    def _image(image):
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(image[::-1, ::-1]), 224, 224)
        )

    def step(self, action):
        self.obs, _, done, _ = self.env.step(action.tolist())
        return bool(done)

    def close(self):
        self.env.close()


class AlohaScene:
    action_dimensions = 14
    initial_count = None
    max_steps = 300

    def __init__(self, args):
        import gym_aloha  # noqa: F401
        import gymnasium

        task = "AlohaTransferCube-v0" if args.suite == "transfer" else "AlohaInsertion-v0"
        self.instruction = "Transfer cube" if args.suite == "transfer" else "Insert the peg into the socket"
        self.env = gymnasium.make(
            f"gym_aloha/{task}", obs_type="pixels_agent_pos", observation_width=224, observation_height=224
        )
        # gym-aloha exposes its dm_control physics through the wrapped environment.
        self.physics = self.env.unwrapped._env.physics  # noqa: SLF001
        self.seed = args.seed

    def reset(self, index):
        self.obs, _ = self.env.reset(seed=(self.seed + index) % 2**32)

    def attach(self, viewer):
        viewer.attach_native(self.physics.model.ptr, self.physics.data.ptr, lookat=(0.0, 0.5, 0.1), distance=1.6)

    def observation(self):
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(self.obs["pixels"]["top"], 224, 224))
        return {
            "state": self.obs["agent_pos"],
            "images": {"cam_high": image.transpose(2, 0, 1)},
            "prompt": self.instruction,
        }

    def step(self, action):
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        return bool(terminated or truncated)

    def close(self):
        self.env.close()
        self.physics.free()


def run(args):
    commands = queue.Queue()

    def read_commands():
        for line in sys.stdin:
            try:
                commands.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        commands.put({"command": "stop"})

    threading.Thread(target=read_commands, daemon=True).start()
    scene = None
    viewer = EmbeddedViewer(args.render_buffer) if args.render_buffer else InteractiveViewer()
    plan = collections.deque()
    client = None
    step = 0
    done = False
    stopped = False
    last_status = None
    latency = None
    try:
        emit(status="正在打开三维场景", paused=True, step=0)
        scene = LiberoScene(args) if args.dataset == "libero" else AlohaScene(args)
        scene.reset(args.initial_state)
        scene.attach(viewer)
        viewer.paused = True
        emit(
            status="场景已就绪，等待加载模型",
            instruction=scene.instruction,
            initial_count=scene.initial_count,
            scene_ready=True,
        )
        while viewer.enabled and not stopped:
            reset = None
            while not commands.empty():
                command = commands.get_nowait()
                action = command.get("command")
                if args.render_buffer and viewer.command(command):
                    continue
                if action == "stop":
                    stopped = True
                elif action == "policy_ready":
                    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", int(command["port"]))
                    plan.clear()
                    viewer.paused = True
                    viewer.single_step = False
                elif action == "policy_disconnect":
                    client = None
                    plan.clear()
                    viewer.paused = True
                    viewer.single_step = False
                elif action == "play" and not done:
                    viewer.paused = False
                elif action == "pause":
                    viewer.paused = True
                    viewer.single_step = False
                elif action == "step" and not done:
                    viewer.paused = True
                    viewer.single_step = True
                elif action == "reset":
                    reset = int(command.get("initial_state", args.initial_state))
            if stopped:
                break
            if reset is not None:
                if reset < 0 or (scene.initial_count is not None and reset >= scene.initial_count):
                    emit(error="初始状态编号超出范围。")
                else:
                    viewer.detach()
                    scene.reset(reset)
                    scene.attach(viewer)
                    viewer.paused = True
                    viewer.single_step = False
                    args.initial_state = reset
                    step, done = 0, False
                    plan.clear()
                    latency = None
            viewer.sync()
            if not viewer.enabled:
                break
            status = "本次场景已完成" if done else ("已暂停" if viewer.paused else "正在推理")
            if client is None:
                status = "场景已就绪，等待加载模型"
            current = (status, viewer.paused, step, args.initial_state, latency)
            if current != last_status:
                emit(
                    status=status,
                    paused=viewer.paused,
                    step=step,
                    initial_state=args.initial_state,
                    latency_ms=latency,
                    done=done,
                )
                last_status = current
            if client is None or done or (viewer.paused and not viewer.single_step):
                time.sleep(0.02)
                continue
            if not plan:
                emit(status="模型推理中，仍可旋转视角", paused=viewer.paused)
                started = time.monotonic()
                try:
                    chunk = validate_actions(client.infer(scene.observation())["actions"], scene.action_dimensions)
                except Exception as error:
                    viewer.paused = True
                    viewer.single_step = False
                    client = None
                    emit(error=f"推理失败，已暂停并保留现场：{error}", model_ready=False, paused=True)
                    continue
                latency = round((time.monotonic() - started) * 1000, 1)
                plan.extend(chunk[:5])
                # Process pause/reset/stop requests received while inference was in flight.
                continue
            viewer.single_step = False
            done = scene.step(plan.popleft())
            step += 1
            viewer.sync()
            if done or step >= scene.max_steps:
                viewer.paused = True
                done = True
                emit(
                    status="任务已完成，保留现场" if step < scene.max_steps else "已到本次步数上限，保留现场",
                    done=True,
                    paused=True,
                    step=step,
                )
                # Keep this result until reset; no automatic next task or repeated trials.
                last_status = ("本次场景已完成", True, step, args.initial_state, latency)
    finally:
        viewer.close()
        if scene is not None:
            scene.close()
        emit(status="场景已关闭", scene_ready=False, paused=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("libero", "aloha"), required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-buffer", help="Render into the inspector instead of a separate window.")
    args = parser.parse_args()

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        emit(error=str(error), scene_ready=False)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
