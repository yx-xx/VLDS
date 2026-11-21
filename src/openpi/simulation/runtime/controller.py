"""Manage one inspection scene and one reusable policy server for the desktop panel."""

# ruff: noqa: RUF001 -- Chinese interface text uses Chinese punctuation.

import contextlib
import json
import os
import pathlib
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from openpi.simulation import catalog
from openpi.simulation import paths

EVENT_PREFIX = "@simulation "


def simulation_environment(selection: dict) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    if selection["software_rendering"]:
        # robosuite preserves "glx" while dm_control accepts "glfw".
        backend = "glx" if selection["dataset"] == "libero" else "glfw"
        env.update(MUJOCO_GL=backend, __GLX_VENDOR_LIBRARY_NAME="mesa", LIBGL_ALWAYS_SOFTWARE="1")
        env.pop("PYOPENGL_PLATFORM", None)
    else:
        env.setdefault("MUJOCO_GL", "egl")
        if env["MUJOCO_GL"] == "egl":
            env.setdefault("PYOPENGL_PLATFORM", "egl")
    if selection["dataset"] == "libero":
        config_dir = paths.ROOT / "examples/libero/.cache/config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "benchmark_root": paths.LIBERO_ROOT,
            "bddl_files": paths.LIBERO_ROOT / "bddl_files",
            "init_states": paths.LIBERO_ROOT / "init_files",
            "datasets": paths.ROOT / "data/datasets/raw/libero",
            "assets": paths.LIBERO_ROOT / "assets",
        }
        (config_dir / "config.yaml").write_text(
            "".join(f"{key}: {json.dumps(str(value))}\n" for key, value in config.items())
        )
        env["LIBERO_CONFIG_PATH"] = str(config_dir)
    return env


class InspectorController:
    def __init__(self):
        self.events = queue.Queue()
        self.catalog = catalog.build_catalog()
        self.selection = None
        self.worker = None
        self.policy = None
        self.policy_key = None
        self.policy_ready = False
        self.policy_port = None
        self.want_policy = False
        self.session = 0
        self.closed = False
        self.operation_lock = threading.RLock()

    def publish(self, **event):
        self.events.put(event)

    def background(self, operation, *args, **kwargs):
        def run():
            try:
                with self.operation_lock:
                    if not self.closed:
                        operation(*args, **kwargs)
            except Exception as error:
                self.publish(error=str(error))

        threading.Thread(target=run, daemon=True).start()

    def open_scene(self, selection: dict, *, load_model: bool):
        selected = catalog.validate_selection(selection, self.catalog)
        self.background(self._open_scene, selected, load_model=load_model)

    def _open_scene(self, selected: dict, *, load_model: bool):
        self.session += 1
        session = self.session
        self._stop_worker()
        self.selection = selected
        self.want_policy = load_model
        self.publish(
            scene_ready=False,
            status="正在打开三维场景",
            step=0,
            done=False,
            paused=True,
            instruction=selected["task_name"],
        )
        python = (
            paths.ROOT / "examples/libero/.venv/bin/python"
            if selected["dataset"] == "libero"
            else pathlib.Path(sys.executable)
        )
        command = [str(python), "-u", str(paths.WORKERS_ROOT / "scene.py")]
        for key in ("dataset", "suite", "task_id", "initial_state", "seed"):
            command.extend(["--" + key.replace("_", "-"), str(selected[key])])
        self.worker = subprocess.Popen(
            command,
            cwd=paths.ROOT,
            env=simulation_environment(selected),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        threading.Thread(target=self._read_worker, args=(self.worker, session), daemon=True).start()
        if load_model:
            self._load_policy()
        else:
            self.publish(model_status="当前只观察场景；点击“加载模型”后可推理", model_ready=False)

    def _read_worker(self, process, session):
        for line in process.stdout:
            if session != self.session or self.closed:
                continue
            if line.startswith(EVENT_PREFIX):
                try:
                    event = json.loads(line[len(EVENT_PREFIX) :])
                    self.publish(session=session, **event)
                except json.JSONDecodeError:
                    self.publish(log=line.rstrip())
            else:
                self.publish(log=line.rstrip())
        returncode = process.wait()
        if session == self.session and not self.closed:
            self.publish(scene_ready=False, paused=True)
            if returncode:
                self.publish(error=f"场景进程退出（{returncode}），详细原因见运行日志。")
        process.stdout.close()

    def send(self, command: str, **arguments):
        process = self.worker
        if process is None or process.poll() is not None:
            raise ValueError("请先打开一个场景。")
        try:
            process.stdin.write(json.dumps({"command": command, **arguments}) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise ValueError("场景已经关闭，请重新打开。") from error

    def load_policy(self, model: str, checkpoint: str):
        def load():
            if self.selection is None:
                raise ValueError("请先打开一个场景。")
            selected = catalog.validate_selection(
                {**self.selection, "model": model, "checkpoint": checkpoint}, self.catalog
            )
            catalog.policy_spec(selected)
            self.send("policy_disconnect")
            self.selection = selected
            self.publish(model_ready=False)
            self._load_policy()

        self.background(load)

    def _load_policy(self):
        if self.selection is None or self.worker is None or self.worker.poll() is not None:
            raise ValueError("请先打开一个场景。")
        key = catalog.policy_spec(self.selection)
        self.want_policy = True
        if self.policy is not None and self.policy.poll() is None and self.policy_key == key:
            if self.policy_ready:
                self.send("policy_ready", port=self.policy_port)
                self.publish(model_status=f"{key[0]} 已就绪 · 复用已加载权重", model_ready=True)
            else:
                self.publish(model_status=f"正在加载 {key[0]}", model_ready=False)
            return
        self._stop_policy()
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            self.policy_port = port.getsockname()[1]
        config, checkpoint = key
        out_dir = paths.ROOT / "data/sim_inspector"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{config}_{time.time_ns()}.log"
        server_env = os.environ.copy()
        server_env.pop("PYTHONPATH", None)
        server_env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
        server_env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            str(paths.ROOT / "scripts/serve_policy.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.policy_port),
            "policy:checkpoint",
            f"--policy.config={config}",
            f"--policy.dir={checkpoint}",
        ]
        with log_path.open("w") as log:
            self.policy = subprocess.Popen(
                command, cwd=paths.ROOT, env=server_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        self.policy_key = key
        self.policy_ready = False
        self.publish(
            model_status=f"正在加载 {config}；首次使用可能下载权重", model_ready=False, log=f"模型日志：{log_path}"
        )
        threading.Thread(
            target=self._wait_for_policy, args=(self.policy, self.policy_port, key, log_path), daemon=True
        ).start()

    def _wait_for_policy(self, process, port, key, log_path):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 3600
        while not self.closed and self.policy is process and time.monotonic() < deadline:
            if process.poll() is not None:
                self.publish(
                    model_status="模型加载失败", model_ready=False, error=f"模型服务已退出，查看日志：{log_path}"
                )
                return
            try:
                with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                    ready = response.status == 200
            except (urllib.error.URLError, TimeoutError):
                ready = False
            if ready:
                with self.operation_lock:
                    if self.policy is not process or self.closed:
                        return
                    self.policy_ready = True
                    try:
                        selected_key = catalog.policy_spec(self.selection) if self.selection else None
                    except ValueError:
                        selected_key = None
                    if self.want_policy and selected_key == key:
                        try:
                            self.send("policy_ready", port=port)
                        except ValueError:
                            return
                        self.publish(model_status=f"{key[0]} 已就绪", model_ready=True)
                return
            time.sleep(0.5)
        if self.policy is process and not self.closed:
            self.publish(model_status="模型加载超时", model_ready=False, error=f"查看模型日志：{log_path}")

    @staticmethod
    def _terminate(process, *, worker=False):
        if process is None:
            return
        if process.poll() is None:
            if worker and process.stdin is not None:
                try:
                    process.stdin.write('{"command":"stop"}\n')
                    process.stdin.flush()
                    process.wait(timeout=5)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        if worker and process.stdin is not None:
            process.stdin.close()

    def _stop_worker(self):
        process, self.worker = self.worker, None
        self._terminate(process, worker=True)

    def _stop_policy(self):
        process, self.policy = self.policy, None
        self.policy_ready = False
        self.policy_key = None
        self._terminate(process)

    def stop_scene(self):
        def stop():
            self.session += 1
            self._stop_worker()
            self.want_policy = False
            self.publish(scene_ready=False, status="场景已关闭，模型保留以便下次使用", paused=True)

        self.background(stop)

    def close(self):
        with self.operation_lock:
            self.closed = True
            self.session += 1
            self._stop_worker()
            self._stop_policy()
