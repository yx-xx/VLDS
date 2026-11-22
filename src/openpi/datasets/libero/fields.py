"""Finite field fitting and k-best DAG selection over aligned LIBERO episodes.

Outputs are an offline diagnostic pool, never vlds.fields.v1 training samples.
Operation confirmation boundaries can be supplied only with verified evidence.
Absent such evidence, semantic completeness and training_ready remain false.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import hashlib
import importlib.metadata
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time as clock

from filelock import FileLock
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.datasets.libero import replay_teacher
from openpi.datasets.libero import trajectory
from openpi.datasets.libero.reader import ProcessedLiberoDataset
from openpi.datasets.libero.reader import atomic_json
from openpi.fields import primitive as primitives
from openpi.fields import seds

SCHEMA = "vlds.libero.offline_fields.v1"
ROOT = Path(__file__).resolve().parents[4]


@dataclasses.dataclass(frozen=True)
class SelectionConfig:
    initial_candidates: int = 128
    expanded_candidates: int = 256
    field_penalty: float = 0.1
    max_sequences: int = 3
    grid_seconds: float = 0.5
    minimum_seconds: float = 0.5

    def __post_init__(self):
        if self.initial_candidates < 1 or self.expanded_candidates < self.initial_candidates:
            raise ValueError("Invalid finite candidate capacities")
        if any(not np.isfinite(v) or v <= 0 for v in (self.field_penalty, self.grid_seconds, self.minimum_seconds)):
            raise ValueError("Selection times and costs must be finite and positive")
        if self.field_penalty <= 0 or not 1 <= self.max_sequences <= 3:
            raise ValueError("Positive field penalty and at most three sequences required")


DEFAULT_SELECTION = SelectionConfig()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_files():
    return [
        Path(__file__),
        Path(seds.__file__),
        Path(primitives.__file__),
        Path(trajectory.__file__),
        Path(replay_teacher.__file__),
        Path(sys.modules[ProcessedLiberoDataset.__module__].__file__),
    ]


def implementation_hashes():
    return {str(path.relative_to(ROOT)): _hash(path) for path in implementation_files()}


def save_source_snapshot(output, extra=()):
    for path in [*implementation_files(), *extra]:
        destination = Path(output) / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())


def candidates_for_episode(data, config=DEFAULT_SELECTION, *, confirmed_boundaries=()):
    """Enumerate a fixed, nested set; reserve full mandatory spans before optional edges."""
    times = data["time"]
    source_config = trajectory.ProcessingConfig(
        max_candidates=10**9, grid_seconds=config.grid_seconds, minimum_motion_seconds=config.minimum_seconds
    )
    proposals, _ = trajectory.propose_candidates(
        data["command_intervals"], times, data["trajectory"]["linear_velocity"], source_config
    )
    confirmations = set()
    for item in confirmed_boundaries:
        if not item.get("verified") or not item.get("evidence") or not item.get("observable"):
            raise ValueError("Confirmation boundary requires verified, observable source evidence")
        index = item.get("index")
        if not isinstance(index, int) or not 0 < index < len(times) - 1:
            raise ValueError("Confirmation boundary is outside the episode")
        confirmations.add(index)
    required, optional = {}, {}
    for interval in data["command_intervals"]:
        start, end = interval["start"], interval["end"]
        forced = [start, *sorted(i for i in confirmations if start < i < end), end]
        points = sorted(
            {start, end, *forced}
            | {
                p[key]
                for p in proposals
                if p["command_interval_id"] == interval["command_interval_id"]
                for key in ("start", "end")
            }
        )
        full = {
            "start": start,
            "end": end,
            "command": interval["command"],
            "command_interval_id": interval["command_interval_id"],
            "required": True,
            "selectable": len(forced) == 2,
            "basis": "source_complete_command_interval",
        }
        required[start, end] = full
        for a, b in pairwise(forced):
            required.setdefault(
                (a, b), {**full, "start": a, "end": b, "selectable": True, "basis": "adjacent_mandatory_boundaries"}
            )
        for a_index, a in enumerate(points[:-1]):
            for b in points[a_index + 1 :]:
                if (a, b) in required or any(a < i < b for i in confirmations):
                    continue
                adjacent = b == points[a_index + 1]
                if times[b] - times[a] < config.minimum_seconds - 1e-9 and not adjacent:
                    continue
                optional[a, b] = {
                    **full,
                    "start": a,
                    "end": b,
                    "required": False,
                    "selectable": True,
                    "basis": "short_adjacent_boundary_span"
                    if times[b] - times[a] < config.minimum_seconds - 1e-9
                    else "finite_motion_boundaries",
                }
    # Adjacent spans are first so truncation does not spend the entire budget on long overlapping edges.
    rank = {}
    for interval in data["command_intervals"]:
        points = sorted({x for key in required | optional for x in key if interval["start"] <= x <= interval["end"]})
        rank.update({(interval["command_interval_id"], p): i for i, p in enumerate(points)})
    ordered = sorted(
        optional.values(),
        key=lambda p: (
            rank[p["command_interval_id"], p["end"]] - rank[p["command_interval_id"], p["start"]],
            p["command_interval_id"],
            p["start"],
            p["end"],
        ),
    )
    return list(required.values()) + ordered, len(required)


def k_best_sequences(records, action_count, config=DEFAULT_SELECTION):
    """Exact k-best paths over this finite DAG, with deterministic ties and full coverage."""
    outgoing = collections.defaultdict(list)
    for index, record in enumerate(records):
        if record["accepted"] and record["candidate"]["selectable"]:
            candidate = record["candidate"]
            a, b = candidate["start"], candidate["end"]
            cost = record["cost"]
            if not 0 <= a < b <= action_count or cost is None or not math.isfinite(cost) or cost < 0:
                raise ValueError("Invalid accepted edge")
            outgoing[a].append((b, float(cost), index))
    paths = {0: [(0.0, ())]}
    for start in sorted(outgoing):
        for current_cost, sequence in paths.get(start, []):
            for end, edge_cost, index in outgoing[start]:
                available = paths.setdefault(end, [])
                available.append((current_cost + edge_cost, (*sequence, index)))
                available.sort()
                del available[config.max_sequences :]
    result = []
    for cost, indices in paths.get(action_count, []):
        start = 0
        for i in indices:
            edge = records[i]["candidate"]
            if edge["start"] != start:
                raise ValueError("Selected sequence omits or repeats source actions")
            start = edge["end"]
        if start != action_count:
            raise ValueError("Incomplete source action coverage")
        result.append({"cost": cost, "record_indices": list(indices), "actions_covered": action_count})
    return result


def construct_episode(
    data,
    field_config=seds.DEFAULT_CONFIG,
    control=primitives.DEFAULT_CONTROL,
    selection=DEFAULT_SELECTION,
    *,
    confirmed_boundaries=(),
    representation="fitted",
    cached=None,
    publish=None,
):
    required_candidates, required_count = candidates_for_episode(
        data, selection, confirmed_boundaries=confirmed_boundaries
    )
    positions = data["trajectory"]["position"]
    velocities = data["trajectory"]["linear_velocity"]
    hand = Rotation.from_quat(data["trajectory"]["quaternion_xyzw"]).as_matrix()
    control_rotations = hand @ data["body_to_control_rotation"]
    intervals = {item["command_interval_id"]: item for item in data["command_intervals"]}
    cache = {(r["candidate"]["start"], r["candidate"]["end"]): r for r in (cached or [])}
    records, sequences, capacities = [], [], []
    started = clock.monotonic()
    for capacity in (selection.initial_candidates, selection.expanded_candidates):
        actual_capacity = max(capacity, required_count)
        candidates = required_candidates[:actual_capacity]
        capacities.append(
            {
                "nominal": capacity,
                "actual": actual_capacity,
                "required": required_count,
                "required_overage": max(0, required_count - capacity),
            }
        )
        for candidate in candidates[len(records) :]:
            start, end = candidate["start"], candidate["end"]
            cached_record = cache.get((start, end))
            if cached_record is not None and cached_record["candidate"] == candidate:
                records.append(cached_record)
                continue
            tick = clock.monotonic()
            try:
                field, report = primitives.fit_primitive(
                    positions[start : end + 1],
                    velocities[start : end + 1],
                    control_rotations[start : end + 1],
                    data["time"][start : end + 1],
                    candidate["command"],
                    field_config,
                    control,
                    representation=representation,
                )
                record = {"candidate": candidate, **report, "primitive": field.record() if field else None}
                if report["accepted"]:
                    nominal = report["nominal"]
                    interval = intervals[candidate["command_interval_id"]]
                    fraction = (data["time"][end] - data["time"][start]) / (
                        data["time"][interval["end"]] - data["time"][interval["start"]]
                    )
                    record["cost"] = float(
                        fraction
                        * (
                            (nominal["position_rms_m"] / control.position_rms_tolerance) ** 2
                            + (nominal["rotation_rms_rad"] / control.rotation_rms_tolerance) ** 2
                        )
                        / 2
                        + selection.field_penalty
                    )
                else:
                    record["cost"] = None
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                record = {
                    "candidate": candidate,
                    "accepted": False,
                    "reasons": ["numerical_error"],
                    "error": str(error),
                    "primitive": None,
                    "cost": None,
                }
            record["fit_wall_seconds"] = clock.monotonic() - tick
            records.append(record)
            if publish is not None:
                publish(records)
        sequences = k_best_sequences(records, len(data["actions"]), selection)
        if sequences or capacity == selection.expanded_candidates:
            break
    failure_counts = collections.Counter(reason for r in records for reason in r.get("reasons", []))
    return {
        "schema": SCHEMA,
        "source_episode": data["entry"],
        "records": records,
        "sequences": sequences,
        "candidate_capacity_rounds": capacities,
        "available_candidates": len(required_candidates),
        "summary": {
            "evaluated_candidates": len(records),
            "accepted_candidates": sum(r["accepted"] for r in records),
            "complete_offline_sequence": bool(sequences),
            "failure_counts": dict(failure_counts),
            "wall_seconds_this_invocation": clock.monotonic() - started,
            "total_fitting_wall_seconds": sum(r["fit_wall_seconds"] for r in records),
        },
        "teacher_representation": representation,
        "confirmed_boundaries": list(confirmed_boundaries),
        "semantic_confirmation_status": "provided_boundaries_only" if confirmed_boundaries else "not_constructed",
        "continuous_execution_status": "not_run",
        "training_ready": False,
        "scope": "offline fitting and nominal acceptance; operation evidence, causal observer and robot execution remain separate",
    }


def _run_locked(args):
    root, output = args.dataset.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Field outputs must be outside the source trajectory dataset")
    output.mkdir(parents=True, exist_ok=True)
    dataset = ProcessedLiberoDataset(root)
    if args.max_episodes < 1:
        raise ValueError("max-episodes must be positive")
    indices = args.episodes
    if indices is None:
        indices = [
            i
            for i, e in enumerate(dataset.episodes)
            if e["split"] == args.split and (not args.suite or e["suite"] == args.suite)
        ][: args.max_episodes]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Select distinct existing source episodes")
    if any(not 0 <= i < len(dataset) or dataset.episodes[i]["split"] != args.split for i in indices):
        raise ValueError("Explicit episode indices must match the requested source split")
    field_config = seds.SEDSConfig(stability_margin=args.stability_margin)
    control = dataclasses.replace(primitives.ControlConfig(), max_speed=args.max_speed)
    selection = SelectionConfig(grid_seconds=args.grid_seconds, minimum_seconds=args.grid_seconds)
    hashes = implementation_hashes()
    config = {
        "schema": SCHEMA,
        "source_manifest_sha256": _hash(root / "manifest.json"),
        "implementation_sha256": hashes,
        "field": dataclasses.asdict(field_config),
        "control": dataclasses.asdict(control),
        "selection": dataclasses.asdict(selection),
        "episodes": indices,
        "split": args.split,
        "versions": {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "torch", "h5py")},
    }
    config["representation"] = args.representation
    teacher_source = None
    if args.source_replay:
        if len(indices) != 1 or args.split == "test":
            raise ValueError("Replay-derived teacher requires one training/validation episode")
        teacher_source = replay_teacher.prepare(
            dataset.field_input(indices[0]),
            args.source_replay,
            output / "teacher_trajectory.pending.npz",
            config["source_manifest_sha256"],
        )
        teacher_source["file"] = "teacher_trajectory.npz"
        config["teacher_source"] = teacher_source
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    previous_file = output / "run.json"
    if previous_file.exists() and json.loads(previous_file.read_text())["fingerprint"] != fingerprint:
        (output / "teacher_trajectory.pending.npz").unlink(missing_ok=True)
        raise ValueError("Output fingerprint differs; use a new output directory")
    if teacher_source:
        os.replace(output / "teacher_trajectory.pending.npz", output / teacher_source["file"])
    save_source_snapshot(output)
    now = datetime.datetime.now(datetime.UTC).isoformat()
    run = {
        "fingerprint": fingerprint,
        "config": config,
        "started_at_utc": now,
        "status": "running",
        "command": shlex.join(sys.orig_argv),
        "source_dataset_relative": os.path.relpath(root, output),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "training_ready": False,
    }
    atomic_json(previous_file, run)
    started, results = clock.monotonic(), []
    try:
        for index in indices:
            data = replay_teacher.apply(dataset.field_input(index), output, teacher_source)
            episode_file = output / f"episode_{index:05d}.json"
            cache_file = output / f"episode_{index:05d}.cache.json"
            cached = []
            if cache_file.exists():
                cache = json.loads(cache_file.read_text())
                if cache["fingerprint"] != fingerprint:
                    raise ValueError("Candidate cache fingerprint differs from the current run")
                cached = cache["records"]

            def publish(records, path=cache_file):
                atomic_json(path, {"fingerprint": fingerprint, "records": records})

            result = construct_episode(
                data,
                field_config,
                control,
                selection,
                representation=args.representation,
                cached=cached,
                publish=publish,
            )
            result.update(
                fingerprint=fingerprint,
                source_episode_index=index,
                source_provenance=data["source_provenance"],
                teacher_source=teacher_source,
            )
            atomic_json(episode_file, result)
            results.append({"episode_index": index, "file": episode_file.name, **result["summary"]})
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        run.update(
            status="complete",
            episodes=results,
            seconds=clock.monotonic() - started,
            complete_offline_sequences=sum(r["complete_offline_sequence"] for r in results),
        )
    except BaseException as error:
        run.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
            seconds=clock.monotonic() - started,
            episodes=results,
        )
        raise
    finally:
        atomic_json(previous_file, run)
    return 0


def fit_fields(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".fields.lock"), timeout=0):
        return _run_locked(args)
