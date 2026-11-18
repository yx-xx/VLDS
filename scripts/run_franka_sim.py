import argparse
import datetime
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = {
    "pi0": "pi0_libero",
    "pi0_fast": "pi0_fast_libero",
    "pi05": "pi05_libero",
}
TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run openpi models sequentially on a simulated Franka Panda.")
    parser.add_argument(
        "--mode",
        choices=("single", "full", "view"),
        default="single",
        help="full: all suites, 50 trials per task; view: 3 suites, 3 tasks each, with live preview.",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["pi05"])
    parser.add_argument("--suite", choices=TASK_SUITES)
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--viewer", action="store_true", help="Show live scene and wrist cameras on the local desktop.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--sim-python", type=pathlib.Path, default=ROOT / "examples/libero/.venv/bin/python")
    parser.add_argument("--checkpoint", help="Override the checkpoint for a single selected model.")
    parser.add_argument("--startup-timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    mode_suites = {"single": TASK_SUITES[:1], "full": TASK_SUITES, "view": TASK_SUITES[:3]}
    args.suites = [args.suite] if args.suite is not None else list(mode_suites[args.mode])
    args.suite = args.suites[0]
    args.all_tasks = args.all_tasks or args.mode == "full"
    args.viewer = args.viewer or args.mode == "view"
    if args.trials is None:
        args.trials = 50 if args.mode == "full" else 1
    if args.task_ids is None:
        args.task_ids = [0, 1, 2] if args.mode == "view" else [0]
    if args.output_dir is None:
        args.output_dir = ROOT / "data/franka_sim"
        if args.mode != "single":
            timestamp = datetime.datetime.now(datetime.UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
            args.output_dir /= f"{args.mode}_{timestamp}"
    if args.trials <= 0 or args.startup_timeout <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error("Trials, startup timeout, and max steps must be positive.")
    if any(task_id < 0 for task_id in args.task_ids) or len(set(args.task_ids)) != len(args.task_ids):
        parser.error("Task IDs must be nonnegative and unique.")
    if len(set(args.models)) != len(args.models):
        parser.error("Models must be unique.")
    if args.checkpoint and len(args.models) != 1:
        parser.error("--checkpoint requires exactly one model.")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    if args.viewer and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("--viewer requires a desktop DISPLAY; omit it for headless evaluation.")
    return args


def simulation_env() -> dict[str, str]:
    libero_root = ROOT / "examples/libero/.cache/LIBERO"
    benchmark_root = libero_root / "libero/libero"
    if not (benchmark_root / "envs").is_dir():
        raise FileNotFoundError("LIBERO is missing. Run: bash examples/libero/setup_franka.sh")
    config_dir = ROOT / "examples/libero/.cache/config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": libero_root / "libero/datasets",
        "assets": benchmark_root / "assets",
    }
    (config_dir / "config.yaml").write_text(
        "".join(f"{key}: {json.dumps(str(value))}\n" for key, value in config.items())
    )
    env = os.environ.copy()
    env["LIBERO_CONFIG_PATH"] = str(config_dir)
    env.setdefault("MUJOCO_GL", "egl")
    if env["MUJOCO_GL"] == "egl":
        env.setdefault("PYOPENGL_PLATFORM", "egl")
    return env


def client_command(args: argparse.Namespace, output_dir: pathlib.Path) -> list[str]:
    command = [
        str(args.sim_python),
        str(ROOT / "examples/libero/main.py"),
        "--args.host",
        "127.0.0.1",
        "--args.port",
        str(args.port),
        "--args.task-suite-name",
        args.suite,
        "--args.num-trials-per-task",
        str(args.trials),
        "--args.seed",
        str(args.seed),
        "--args.video-out-path",
        str(output_dir),
    ]
    if not args.all_tasks:
        command.extend(["--args.task-ids", *map(str, args.task_ids)])
    if args.max_steps is not None:
        command.extend(["--args.max-steps", str(args.max_steps)])
    if args.viewer:
        command.append("--args.viewer")
    return command


def wait_for_server(server: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Policy server exited with status {server.returncode}; see server.log.")
        try:
            with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("Policy server startup timed out; check server.log and checkpoint download progress.")


def run_model(args: argparse.Namespace, model: str, env: dict[str, str]) -> dict:
    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", args.port))
    output_dir = args.output_dir.resolve() / model / args.suite
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").unlink(missing_ok=True)
    checkpoint = args.checkpoint or f"gs://openpi-assets/checkpoints/{MODELS[model]}"
    command = [
        sys.executable,
        str(ROOT / "scripts/serve_policy.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "policy:checkpoint",
        f"--policy.config={MODELS[model]}",
        f"--policy.dir={checkpoint}",
    ]
    server_env = os.environ.copy()
    server_env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    server_env["PYTHONUNBUFFERED"] = "1"
    print(f"Starting {model} on Franka Panda. Server log: {output_dir / 'server.log'}", flush=True)
    with (output_dir / "server.log").open("w") as log_file:
        server = subprocess.Popen(command, cwd=ROOT, env=server_env, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            wait_for_server(server, args.port, args.startup_timeout)
            subprocess.run(client_command(args, output_dir), cwd=ROOT, env=env, check=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    results = json.loads((output_dir / "results.json").read_text())
    results.update(model=model, checkpoint=checkpoint)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"{model}: {results['total_successes']}/{results['total_episodes']} successful episodes", flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.sim_python.is_file():
        raise FileNotFoundError("Simulation Python is missing. Run: bash examples/libero/setup_franka.sh")
    env = simulation_env()
    results = []
    summary_path = args.output_dir.resolve() / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("[]\n")
    for model in args.models:
        for suite in args.suites:
            args.suite = suite
            results.append(run_model(args, model, env))
            summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Videos and results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
