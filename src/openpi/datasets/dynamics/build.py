"""Resumable offline build for processed LIBERO and native LeRobot states."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import datetime
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from openpi.datasets.libero.reader import ProcessedLiberoDataset
from openpi.datasets.libero.reader import atomic_json
from openpi.fields import sequence

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "vlds.dynamics.episode.v1"
_DATASETS = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(root, kind, index, entry):
    """Return states in known native coordinates, without reading camera arrays."""
    root = Path(root)
    if kind == "processed_libero":
        dataset = _DATASETS.setdefault(str(root), None)
        if dataset is None:
            dataset = _DATASETS[str(root)] = ProcessedLiberoDataset(root)
        data = dataset.field_input(index)
        position = data["trajectory"]["position"]
        rotations = (
            Rotation.from_quat(data["trajectory"]["quaternion_xyzw"]).as_matrix() @ data["body_to_control_rotation"]
        )
        return {
            "position": position,
            "rotations": rotations,
            "time": data["time"],
            "intervals": data["command_intervals"],
            "tolerance": np.full(3, 0.01),
            "coordinates": {
                "space": "world Cartesian",
                "units": "m",
                "axes": ["x", "y", "z"],
                "control_point": "gripper0_grip_site",
                "position_tolerance": 0.01,
                "rotation_tolerance_rad": np.pi / 36,
            },
            "source_key": str(data["source_file"].relative_to(ROOT)) + "::" + str(data["source_group"]),
            "source_sha256": entry["signature"],
            "source_entry": entry,
        }
    path = root / entry["path"]
    state_key = "state" if kind == "lerobot_libero" else "observation.state"
    columns = [state_key, "timestamp", "actions"] if kind == "lerobot_libero" else [state_key, "timestamp"]
    table = pq.read_table(path, columns=columns)
    state = np.asarray(table[state_key].to_pylist(), dtype=float)
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=float)
    rotations = None
    if kind == "lerobot_libero":
        position, rotations = state[:, :3], Rotation.from_rotvec(state[:, 3:6]).as_matrix()
        tolerance = np.full(3, 0.01)
        coordinates = {
            "space": "recorded end effector Cartesian",
            "units": "m",
            "axes": ["x", "y", "z"],
            "rotation": "rotation vector from source state, SO(3) metric",
            "position_tolerance": 0.01,
        }
    elif kind == "pusht":
        position, tolerance = state, np.full(2, 5.0)
        coordinates = {"space": "recorded image plane", "units": "pixel", "axes": ["x", "y"], "position_tolerance": 5.0}
    elif kind == "xarm":
        position, tolerance = state, np.array([0.01, 0.01, 0.01, 0.05])
        coordinates = {
            "space": "recorded end effector xyz and gripper",
            "units": "m for xyz; native gripper scalar",
            "axes": ["x", "y", "z", "gripper"],
            "position_tolerance": 0.01,
            "gripper_tolerance": 0.05,
        }
    elif kind == "aloha":
        position, tolerance = state, np.full(14, 0.05)
        coordinates = {
            "space": "dual arm joint and gripper state",
            "units": "rad for joints; normalized grippers",
            "axes": [
                "left_waist",
                "left_shoulder",
                "left_elbow",
                "left_roll",
                "left_wrist",
                "left_rotate",
                "left_gripper",
                "right_waist",
                "right_shoulder",
                "right_elbow",
                "right_roll",
                "right_wrist",
                "right_rotate",
                "right_gripper",
            ],
            "joint_tolerance_rad": 0.05,
            "gripper_tolerance": 0.05,
        }
    else:
        raise ValueError(f"Unknown state adapter {kind}")
    intervals = [{"start": 0, "end": len(state) - 1, "command_interval_id": 0, "command_label": "MOVE"}]
    if kind == "lerobot_libero":
        commands = np.asarray(table["actions"].to_pylist(), dtype=float)[:-1, -1]
        if not np.isin(commands, [-1, 1]).all():
            raise ValueError("LeRobot LIBERO action gripper must be an explicit binary command")
        boundaries = np.r_[0, np.flatnonzero(np.diff(commands)) + 1, len(commands)]
        intervals = [
            {
                "start": int(a),
                "end": int(b),
                "command_interval_id": i,
                "command": int(commands[a]),
                "command_label": "OPEN" if commands[a] == -1 else "CLOSE",
            }
            for i, (a, b) in enumerate(itertools.pairwise(boundaries))
        ]
    return {
        "position": position,
        "rotations": rotations,
        "time": timestamp,
        "intervals": intervals,
        "tolerance": tolerance,
        "coordinates": coordinates,
        "source_key": entry["path"],
        "source_sha256": digest(path),
        "source_entry": entry,
    }


def build_one(root, kind, index, entry, destination, fingerprint):
    destination = Path(destination)
    output = destination / f"episode_{index:05d}.json"
    trace_path = output.with_suffix(".npz")
    data = inputs(root, kind, index, entry)
    if output.exists() and trace_path.exists():
        previous = json.loads(output.read_text())
        if (
            previous.get("fingerprint") == fingerprint
            and previous.get("trace_sha256") == digest(trace_path)
            and previous["source_sha256"] == data["source_sha256"]
        ):
            return {
                "index": index,
                "file": output.name,
                "source_key": previous["source_key"],
                **previous["summary"],
                "resumed": True,
            }
    calibrated_speed = None
    if kind == "processed_libero":
        dt = float(np.median(np.diff(data["time"])))
        measured = float(np.max(np.linalg.norm(np.diff(data["position"] / data["tolerance"], axis=0), axis=1)) / dt)
        calibrated_speed = max(30.0, measured * 1.05)
    for margin in (0.9, 0.5, 0.1, 0.0):
        records, summary, nominal, limits = sequence.fit_sequence(
            data["position"],
            data["time"],
            data["intervals"],
            data["tolerance"],
            rotations=data["rotations"],
            maximum_speed=calibrated_speed,
            config=sequence.SequenceConfig(fit_margin=margin),
        )
        if summary["passed"]:
            break
    summary["fit_margin"] = margin
    summary["source_observations"] = len(data["time"])
    summary["terminal_action_target_available"] = kind == "processed_libero"
    if kind == "processed_libero":
        summary["source_tracking_max_speed_m_per_second"] = (
            max(r["maximum_speed_scaled_per_second"] for r in records) * 0.01
        )
        summary["production_speed_cap_0_3_met"] = summary["source_tracking_max_speed_m_per_second"] <= 0.3
    arrays = {"position": data["position"], "time": data["time"], "nominal": nominal, "tolerance": data["tolerance"]}
    if data["rotations"] is not None:
        arrays["rotation"] = data["rotations"]
    with trace_path.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(trace_path.with_suffix(".npz.tmp"), trace_path)
    result = {
        "schema": SCHEMA,
        "model": sequence.VERSION,
        "fingerprint": fingerprint,
        "source_episode_index": index,
        "source_key": data["source_key"],
        "source_sha256": data["source_sha256"],
        "source_entry": entry,
        "coordinates": data["coordinates"],
        "tolerance": data["tolerance"].tolist(),
        "limits": limits,
        "source_intervals": data["intervals"],
        "records": records,
        "summary": summary,
        "trace_file": trace_path.name,
        "trace_sha256": digest(trace_path),
        "scope": "offline nominal source-time playback; not arrival-switched robot execution or training supervision",
    }
    atomic_json(output, result)
    return {"index": index, "file": output.name, "source_key": data["source_key"], **summary, "resumed": False}


def catalogue(root):
    root = Path(root).resolve()
    if (root / "manifest.json").exists():
        dataset = ProcessedLiberoDataset(root)
        return "processed_libero", dataset.episodes, digest(root / "manifest.json")
    info = json.loads((root / "meta/info.json").read_text())
    kinds = {
        "libero": "lerobot_libero",
        "pusht": "pusht",
        "xarm_lift_medium": "xarm",
        "aloha_sim_transfer_cube_human": "aloha",
    }
    kind = kinds.get(root.name)
    if kind is None:
        raise ValueError(f"No verified state adapter for {root.name}")
    records = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    entries = []
    for record in records:
        index = record["episode_index"]
        path = root / info["data_path"].format(
            episode_index=index, episode_chunk=index // info.get("chunks_size", 1000)
        )
        if not path.exists():
            raise ValueError(f"Missing declared source episode: {path}")
        entries.append({"path": str(path.relative_to(root)), "episode_index": index, "frames": record.get("length")})
    return (
        kind,
        entries,
        hashlib.sha256(
            (root / "meta/info.json").read_bytes() + (root / "meta/episodes.jsonl").read_bytes()
        ).hexdigest(),
    )


def build(root, output, *, workers=8, indices=None, register=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    kind, entries, source_hash = catalogue(root)
    chosen = list(range(len(entries))) if indices is None else list(indices)
    if not chosen or len(chosen) != len(set(chosen)) or any(i < 0 or i >= len(entries) for i in chosen):
        raise ValueError("Select distinct existing source episodes")
    implementation = [
        Path(__file__),
        Path(sequence.__file__),
        Path(sys.modules[ProcessedLiberoDataset.__module__].__file__),
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in implementation}
    config = {
        "schema": "vlds.dynamics.run.v1",
        "model": sequence.VERSION,
        "kind": kind,
        "source_manifest_sha256": source_hash,
        "source_episodes": len(entries),
        "indices": chosen,
        "implementation_sha256": hashes,
        "profile": "source-calibrated nominal tracking",
        "sequence": dataclasses_asdict(sequence.SequenceConfig()),
        "versions": {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "h5py", "pyarrow")},
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    with FileLock(str(output / ".build.lock"), timeout=0):
        run_path = output / "run.json"
        if run_path.exists() and json.loads(run_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Output configuration/source/code differs; choose a new output directory")
        for p in implementation:
            copy = output / "source_snapshot" / p.relative_to(ROOT)
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(p.read_bytes())
        run = {
            "fingerprint": fingerprint,
            "config": config,
            "source_relative": os.path.relpath(root, output),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "command": sys.orig_argv,
            "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "running",
        }
        atomic_json(run_path, run)
        started, results, errors = time.monotonic(), [], []
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending = {
                    pool.submit(build_one, str(root), kind, i, entries[i], str(output), fingerprint): i for i in chosen
                }
                for future in as_completed(pending):
                    i = pending[future]
                    try:
                        value = future.result()
                        results.append(value)
                    except Exception as error:
                        errors.append({"index": i, "error": repr(error)})
                    if len(results) % 100 == 0 or len(results) + len(errors) == len(chosen):
                        print(
                            json.dumps(
                                {
                                    "dataset": root.name,
                                    "finished": len(results),
                                    "errors": len(errors),
                                    "passed": sum(r["passed"] for r in results),
                                    "seconds": round(time.monotonic() - started, 1),
                                }
                            ),
                            flush=True,
                        )
                        atomic_json(
                            output / "progress.json", {"finished": len(results), "total": len(chosen), "errors": errors}
                        )
            results.sort(key=lambda r: r["index"])
            index = {
                "schema": "vlds.dynamics.index.v1",
                "fingerprint": fingerprint,
                "episodes": [
                    {"index": r["index"], "file": r["file"], "source_key": r["source_key"], "passed": r["passed"]}
                    for r in results
                ],
            }
            atomic_json(output / "index.json", index)
            summary = {
                "episodes": len(results),
                "expected_episodes": len(entries),
                "selected_episodes": len(chosen),
                "passed_episodes": sum(r["passed"] for r in results),
                "segments": sum(r["segments"] for r in results),
                "accepted_segments": sum(r["accepted_segments"] for r in results),
                "transitions_covered": sum(r["actions_covered"] for r in results),
                "single_transition_segments": sum(r["single_transition_segments"] for r in results),
                "full_collection": indices is None and not errors and len(results) == len(entries),
                "errors": errors,
                "seconds": time.monotonic() - started,
                "maximum_scaled_error": max((r["maximum_scaled_error"] for r in results), default=None),
            }
            run.update(status="complete" if not errors else "failed", summary=summary)
            atomic_json(output / "summary.json", summary)
            if register and summary["full_collection"]:
                registry = root / ".vlds/dynamics.json"
                registry.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    registry,
                    {
                        "schema": "vlds.dynamics.pointer.v1",
                        "directory": os.path.relpath(output, registry.parent),
                        "fingerprint": fingerprint,
                    },
                )
        except BaseException as error:
            run.update(status="interrupted", error=repr(error))
            raise
        finally:
            run["seconds"] = time.monotonic() - started
            atomic_json(run_path, run)
    return run


def dataclasses_asdict(value):
    import dataclasses

    return dataclasses.asdict(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--episodes", nargs="+", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    result = build(args.dataset, args.output, workers=args.workers, indices=args.episodes)
    print(json.dumps(result.get("summary", {}), ensure_ascii=False), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
