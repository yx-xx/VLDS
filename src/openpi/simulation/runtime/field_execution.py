"""Prepare reproducible full-environment field diagnostics and isolate LIBERO."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from filelock import FileLock

from openpi.datasets.libero.reader import ProcessedLiberoDataset
from openpi.datasets.libero.reader import atomic_json
from openpi.fields import primitive
from openpi.simulation import paths
from openpi.simulation.runtime.controller import simulation_environment

BUDGETS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def run_worker(command, *, cwd, env, output, timeout):
    """A worker has no descendants; subprocess.run kills and reaps it on timeout."""
    output = Path(output)
    started = time.monotonic()
    status = {
        "status": "running",
        "command": shlex.join(command),
        "timeout_seconds": timeout,
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    atomic_json(output / "worker.json", status)
    try:
        with (output / "worker.log").open("w") as log:
            result = subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False
            )
        status.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
    except subprocess.TimeoutExpired:
        status.update(status="timeout", returncode=None)
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - started
        atomic_json(output / "worker.json", status)
        result_file = output / "result.json"
        if status["status"] != "complete":
            result = json.loads(result_file.read_text()) if result_file.exists() else {"training_ready": False}
            result.update(status=status["status"], infrastructure_failure=True, worker_status=status)
            atomic_json(result_file, result)
    return status


def execute_fields(args):
    if args.audit_stride < 0:
        raise ValueError("audit_stride must be nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".execution.lock"), timeout=0):
        if (output / "request.json").exists():
            raise ValueError("An execution already exists here; choose a new output directory")
        dataset = ProcessedLiberoDataset(args.dataset)
        data = dataset.field_input(args.episode)
        control = primitive.DEFAULT_CONTROL
        sequences = []
        if not args.source_actions:
            if args.fields is None:
                raise ValueError("Field execution requires an offline episode result")
            result = json.loads(args.fields.read_text())
            if result["source_episode"]["signature"] != data["entry"]["signature"]:
                raise ValueError("Field result belongs to a different source trajectory")
            run = json.loads((args.fields.parent / "run.json").read_text())
            if run["status"] != "complete" or run["fingerprint"] != result["fingerprint"]:
                raise ValueError("Offline field run is incomplete or mismatched")
            if (
                hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
                != run["config"]["source_manifest_sha256"]
            ):
                raise ValueError("Field result and execution source manifests differ")
            control = primitive.ControlConfig(**result.get("control", run["config"]["control"]))
            sequences = [
                [result["records"][i]["primitive"] for i in s["record_indices"]]
                for s in result["sequences"][: args.max_sequences]
            ]
            if not sequences:
                raise ValueError("No complete accepted offline sequence; record that failure before simulation")
        source_stat = data["source_file"].stat()
        request = {
            "mode": "source_actions" if args.source_actions else "fields",
            "gripper_feedback": args.gripper_feedback,
            "audit_stride": args.audit_stride,
            "source_episode_index": args.episode,
            "source_episode": data["entry"],
            "source_file": str(data["source_file"]),
            "source_group": data["source_group"],
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_initial_position": data["trajectory"]["position"][0].tolist(),
            "body_to_control_rotation": data["body_to_control_rotation"].tolist(),
            "seed": args.seed,
            "sequences": sequences,
            "max_steps": int(BUDGETS[data["entry"]["suite"]] * args.budget_multiplier),
            "control": dataclasses.asdict(control),
            "training_ready": False,
            "source_provenance": data["source_provenance"],
            "source_manifest_sha256": hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest(),
            "offline_result_sha256": hashlib.sha256(args.fields.read_bytes()).hexdigest() if args.fields else None,
            "teacher_source": result.get("teacher_source") if not args.source_actions else None,
        }
        atomic_json(output / "request.json", request)
        env = simulation_environment({"dataset": "libero", "software_rendering": False})
        env.update(PYTHONPATH=str(paths.ROOT / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        env["MUJOCO_GL"], env["PYOPENGL_PLATFORM"] = "egl", "egl"
        command = [
            str(paths.ROOT / "examples/libero/.venv/bin/python"),
            "-m",
            "openpi.simulation.workers.fields",
            str(output / "request.json"),
            "--output",
            str(output),
        ]
        status = run_worker(command, cwd=paths.ROOT, env=env, output=output, timeout=args.timeout)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        if status["status"] == "complete":
            print((output / "result.json").read_text())
        return 0 if status["status"] == "complete" else 1
