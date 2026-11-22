"""Offline operation evidence, isolated from deployed observation and control.

Sustained contact, lift and co-motion only propose review candidates. They do
not certify held / placed, camera observability, or fitting boundaries.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.datasets.libero.reader import atomic_json


@dataclasses.dataclass(frozen=True)
class EvidenceConfig:
    dwell_seconds: float = 0.15
    lift_metres: float = 0.02
    relative_position_range_metres: float = 0.01
    relative_rotation_range_radians: float = math.radians(10)
    placement_speed_metres_per_second: float = 0.02

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in dataclasses.asdict(self).values()):
            raise ValueError("Evidence thresholds must be finite and positive")


DEFAULT_EVIDENCE = EvidenceConfig()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runs(mask):
    indices = np.flatnonzero(mask)
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(indices) else []


def _window_start(times, end, seconds):
    return int(np.searchsorted(times, times[end] - seconds + 1e-9, side="right") - 1)


def summarize_trajectory(path, config=DEFAULT_EVIDENCE):
    """Read a diagnostic log, including legacy logs without predicate traces."""
    with h5py.File(path) as archive:
        log = archive["control_log"][:]
        columns = json.loads(archive.attrs["control_log_columns"])
        if len(log) < 2 or not np.isfinite(log).all():
            raise ValueError("At least two finite control samples are required")
        if len(columns) != log.shape[1] or len(set(columns)) != len(columns):
            raise ValueError("Malformed control log columns")
        at = {name: i for i, name in enumerate(columns)}
        times, steps = log[:, at["simulation_time"]], log[:, at["step"]].astype(int)
        if np.any(np.diff(times) <= 0) or np.any(np.diff(steps) != 1):
            raise ValueError("Audit requires increasing time and consecutive control steps")
        audit = archive["privileged_audit"]
        if not audit.attrs.get("not_policy_input", False):
            raise ValueError("Operation evidence must be isolated from policy input")
        poses, contacts = audit["object_pose"][:], audit["bilateral_finger_contact"][:].astype(bool)
        names = json.loads(audit.attrs["object_names"])
        if poses.shape != (len(log), len(names), 7) or contacts.shape != (len(log), len(names)):
            raise ValueError("Object audit shape differs from control log")
        if not np.isfinite(poses).all() or not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1, atol=1e-4):
            raise ValueError("Invalid object poses")
        control_p = log[:, [at[k] for k in ("px", "py", "pz")]]
        control_r = Rotation.from_quat(log[:, [at[k] for k in ("qx", "qy", "qz", "qw")]])
        closing = log[:, at["gripper"]] > 0
        result = json.loads(archive.attrs["result"])
        image_steps = sorted(int(s) for s in archive.get("audit_observations", {}))
        candidates, objects = [], []

        def candidate(kind, name, end, start, evidence):
            prior_images = [s for s in image_steps if s <= steps[end]]
            image_step = prior_images[-1] if prior_images else None
            return {
                "kind": kind,
                "object": name,
                "observation_index": int(steps[end]),
                "control_step": int(steps[end]),
                "simulation_time": float(times[end]),
                "field_index": int(log[end, at["field_index"]]),
                "window_start_step": int(steps[start]),
                "verified": False,
                "observable": False,
                "review_status": "requires image and factual review; not a fitting boundary",
                "evidence": evidence,
                "causal_image_group": f"audit_observations/{image_step}" if image_step is not None else None,
                "causal_image_age_seconds": (
                    float(times[end] - times[np.searchsorted(steps, image_step)]) if image_step is not None else None
                ),
            }

        for i, name in enumerate(names):
            p = poses[:, i, :3]
            relative_p = control_r.inv().apply(p - control_p)
            relative_r = control_r.inv() * Rotation.from_quat(poses[:, i, 3:])
            intervals, qualified = [], []
            for run in _runs(contacts[:, i]):
                first, last = int(run[0]), int(run[-1])
                intervals.append(
                    {
                        "start_step": int(steps[first]),
                        "end_step": int(steps[last]),
                        "sample_count": len(run),
                        "sample_span_seconds": float(times[last] - times[first]),
                    }
                )
                for end in run:
                    start = _window_start(times, end, config.dwell_seconds)
                    if start < first or times[end] - times[start] < config.dwell_seconds - 1e-9:
                        continue
                    window = slice(start, end + 1)
                    lift = p[window, 2] - p[0, 2]
                    relative_p_range = np.linalg.norm(relative_p[window] - relative_p[start], axis=1).max()
                    relative_r_range = (relative_r[start].inv() * relative_r[window]).magnitude().max()
                    if (
                        closing[window].all()
                        and lift.min() >= config.lift_metres
                        and relative_p_range <= config.relative_position_range_metres
                        and relative_r_range <= config.relative_rotation_range_radians
                    ):
                        qualified.append(int(end))
                        candidates.append(
                            candidate(
                                "lifted_contact_candidate",
                                name,
                                int(end),
                                start,
                                {
                                    "basis": "privileged sustained bilateral contact, lift, and relative pose consistency",
                                    "minimum_lift_metres": float(lift.min()),
                                    "relative_position_range_metres": float(relative_p_range),
                                    "relative_rotation_range_radians": float(relative_r_range),
                                    "contact_run_start_step": int(steps[first]),
                                },
                            )
                        )
                        break
            objects.append(
                {
                    "object": name,
                    "bilateral_contact_samples": int(contacts[:, i].sum()),
                    "contact_intervals": intervals,
                    "height_range_metres": float(np.ptp(p[:, 2])),
                    "maximum_lift_from_first_sample_metres": float(np.max(p[:, 2] - p[0, 2])),
                    "maximum_displacement_metres": float(np.linalg.norm(p - p[0], axis=1).max()),
                    "lifted_contact_candidate_steps": [int(steps[k]) for k in qualified],
                }
            )
        predicate_trace_available = "goal_predicate_values" in audit
        if predicate_trace_available:
            predicates = json.loads(audit.attrs["goal_predicates"])
            values = audit["goal_predicate_values"][:].astype(bool)
            if values.shape != (len(log), len(predicates)):
                raise ValueError("Predicate audit shape differs from control log")
            for j, predicate in enumerate(predicates):
                if len(predicate) != 3 or predicate[0] not in ("in", "on") or predicate[1] not in names:
                    continue
                i = names.index(predicate[1])
                speed = np.r_[np.inf, np.linalg.norm(np.diff(poses[:, i, :3], axis=0), axis=1) / np.diff(times)]
                mask = values[:, j] & ~closing & ~contacts[:, i] & (speed <= config.placement_speed_metres_per_second)
                # A placed fact persists while another object is manipulated. Emit once
                # per predicate-true episode, rather than again at every open command.
                for run in _runs(values[:, j]):
                    for end in run:
                        start = _window_start(times, end, config.dwell_seconds)
                        if (
                            start >= run[0]
                            and times[end] - times[start] >= config.dwell_seconds - 1e-9
                            and mask[start : end + 1].all()
                        ):
                            candidates.append(
                                candidate(
                                    "placement_candidate",
                                    predicate[1],
                                    int(end),
                                    start,
                                    {
                                        "basis": "privileged predicate, open command, no bilateral contact, low object speed",
                                        "predicate": predicate,
                                        "maximum_object_speed_metres_per_second": float(speed[start : end + 1].max()),
                                    },
                                )
                            )
                            break
        return {
            "trajectory_file": Path(path).name,
            "trajectory_sha256": _digest(path),
            "trial": result["trial"],
            "status": result["status"],
            "task_success": result["task_success"],
            "completed_fields": result["completed_fields"],
            "control_steps": result["control_steps"],
            "final_goal_predicates": result.get("final_goal_predicates"),
            "baseline_pose_basis": "first post-action sample; no exact pre-action object pose claimed",
            "predicate_trace_available": predicate_trace_available,
            "objects": objects,
            "candidates": sorted(candidates, key=lambda c: (c["control_step"], c["object"])),
        }


def summarize_execution(directory, config=DEFAULT_EVIDENCE):
    directory = Path(directory)
    request = json.loads((directory / "request.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Only terminal, completed executions may supply operation evidence")
    return {
        "directory": directory.name,
        "mode": result["mode"],
        "source_signature": request["source_episode"]["signature"],
        "source_episode_index": request["source_episode_index"],
        "seed": request["seed"],
        "source_manifest_sha256": request["source_manifest_sha256"],
        "initial_position": request["source_initial_position"],
        "request_sha256": _digest(directory / "request.json"),
        "result_sha256": _digest(directory / "result.json"),
        "trials": [summarize_trajectory(directory / t["trajectory_file"], config) for t in result["sequence_results"]],
    }


def plot_audit(execution_dirs, output):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(execution_dirs), 1, figsize=(10, 3.4 * len(execution_dirs)), squeeze=False)
    for ax, directory in zip(axes[:, 0], execution_dirs, strict=True):
        with h5py.File(Path(directory) / "trial_0.hdf5") as archive:
            log = archive["control_log"][:]
            audit = archive["privileged_audit"]
            names = json.loads(audit.attrs["object_names"])
            goals = json.loads(archive.attrs["result"]).get("final_goal_predicates", [])
            relevant = {goal["predicate"][1] for goal in goals if len(goal["predicate"]) == 3}
            t = log[:, 2] - log[0, 2]
            for i, name in enumerate(names):
                if relevant and name not in relevant:
                    continue
                height = audit["object_pose"][:, i, 2] - audit["object_pose"][0, i, 2]
                (line,) = ax.plot(t, height, label=name)
                contact = audit["bilateral_finger_contact"][:, i].astype(bool)
                ax.scatter(t[contact], height[contact], color=line.get_color(), s=7)
            ax.set(
                title=Path(directory).name, xlabel="Time since first post-action sample (s)", ylabel="Object lift (m)"
            )
            ax.legend()
            ax.grid(alpha=0.2)
    fig.suptitle("Offline operation audit: dots indicate bilateral contact, not held labels")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def audit_fields(args):
    config = EvidenceConfig()
    current = summarize_execution(args.execution, config)
    source = summarize_execution(args.compare_source, config) if args.compare_source else None
    if source:
        if source["mode"] != "source_actions":
            raise ValueError("Comparison must use source-action replay")
        for key in ("source_signature", "source_manifest_sha256", "seed", "initial_position"):
            if current[key] != source[key]:
                raise ValueError(f"Comparison source differs: {key}")
    report = {
        "schema": "vlds.libero.operation_evidence.v1",
        "training_ready": False,
        "thresholds": dataclasses.asdict(config),
        "collector_sha256": _digest(__file__),
        "scope": "evaluation-only privileged evidence; candidates require factual and camera review; no validated observer",
        "execution": current,
        "source_replay": source,
    }
    atomic_json(args.output, report)
    if args.figure:
        plot_audit([d for d in (args.compare_source, args.execution) if d], args.figure)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "trials": len(current["trials"]),
                "candidate_counts": [len(t["candidates"]) for t in current["trials"]],
                "source_candidate_counts": [len(t["candidates"]) for t in source["trials"]] if source else None,
                "verified_boundaries": 0,
                "training_ready": False,
            },
            ensure_ascii=False,
        )
    )
    return 0
