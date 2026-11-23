"""Length-constrained nonlinear motion fitting, with honest unresolved spans.

Run against saved source observations. Interpolation or slower display does not
increase the observation count. Failed long spans remain failed rather than
being replaced by singleton fields. Binary command events remain explicit.
"""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese configuration labels.
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.datasets.libero.reader import atomic_json
from openpi.fields import contractive

ROOT = Path(__file__).resolve().parents[4]
PROFILE = "coherent"


@dataclasses.dataclass(frozen=True)
class SegmentationConfig:
    minimum_source_seconds: float = 0.5
    minimum_transitions: int = 10
    refinement_candidates: int = 2
    rollout_iterations: int = 120
    maximum_candidate_factor: int = 16
    fine_endpoint_radius: int = 4
    fine_rollout_iterations: int = 240
    goal_lookahead_steps: int = 0


DEFAULT_CONFIG = SegmentationConfig()


def _velocity_score(path, reference, tolerance):
    """Score source-time increments, including the exit from both start states.

    This is a fitting objective, not a proof of velocity continuity between
    different fields. No interpolation or display-only smoothing is applied.
    """
    delta = np.diff(reference, axis=0) / tolerance
    actual = np.diff(path, axis=0) / tolerance
    error = np.sum((actual - delta[:, None]) ** 2, axis=-1)
    return float(np.mean(error) + 2 * np.mean(error[-1]))


def _fit_position(position, start, end, current, tolerance, dt, speed, config, fit_config):
    reference = position[start : end + 1]
    field, path, maximum = contractive.fit(
        reference, tolerance, dt, speed, carried=current, refine=False, config=fit_config
    )
    if config.goal_lookahead_steps == 0 or end == len(position) - 1:
        return field, path, maximum, end
    # Keep a true stop or a reversal at the observed boundary. A forward target
    # is considered only if the source itself continues in the same direction.
    incoming = (position[end] - position[end - 1]) / tolerance
    outgoing = (position[end + 1] - position[end]) / tolerance
    if np.dot(incoming, outgoing) <= 0 or min(np.linalg.norm(incoming), np.linalg.norm(outgoing)) < 1e-8:
        return field, path, maximum, end
    candidates = [(field, path, maximum, end)]
    offsets = sorted({1, 2, 4, config.goal_lookahead_steps})
    for offset in offsets:
        if offset > config.goal_lookahead_steps or end + offset >= len(position):
            continue
        target = end + offset
        future = (position[target] - position[end]) / tolerance
        if np.dot(incoming, future) <= 0:
            continue
        trial, trace, error = contractive.fit(
            reference,
            tolerance,
            dt,
            speed,
            carried=current,
            goal=position[target],
            # Goal search is cheap. Only the bounded outer endpoint refinement
            # budget performs closed-loop gradient optimization.
            refine=False,
            config=fit_config,
        )
        candidates.append((trial, trace, error, target))
    feasible = [
        c
        for c in candidates
        if c[2] <= 1.0 + 1e-8
        and np.linalg.norm((c[0].rollout(c[1][-1], round(2.0 / dt))[-1] - c[0].goal) / tolerance, axis=-1).max()
        <= 0.5 + 1e-8
    ]
    if feasible:
        return min(feasible, key=lambda c: _velocity_score(c[1], reference, tolerance))
    return min(candidates, key=lambda c: c[2])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def length_quality(start, end, source_time, config=DEFAULT_CONFIG, *, event=False):
    seconds = float(source_time[end] - source_time[start])
    eligible = end - start >= config.minimum_transitions and seconds >= config.minimum_source_seconds - 1e-7
    return {
        "source_seconds": seconds,
        "observations": end - start + 1,
        "transitions": end - start,
        "eligible_length": bool(eligible),
        "short_command_event": bool(event and not eligible),
        "minimum_source_seconds": config.minimum_source_seconds,
        "minimum_transitions": config.minimum_transitions,
        "parameter_supervision": False,
    }


def _rotation_checks(coordinates, path, angular_speed, dt):
    angles = Rotation.from_matrix((path[1:] @ path[:-1].transpose(0, 1, 3, 2)).reshape(-1, 3, 3)).magnitude()
    return {
        "chart_valid": bool(np.linalg.norm(coordinates, axis=-1).max() < np.pi - 1e-6),
        "angular_speed_valid": bool(angles.max(initial=0) / dt <= angular_speed + 1e-7),
    }


def _orientation_coordinates(rotations, goal):
    return Rotation.from_matrix(rotations @ goal.T).as_rotvec()


def _candidate(position, rotations, start, end, current, orientation, dt, tolerance, speed, angular_speed, config):
    fit_config = dataclasses.replace(contractive.DEFAULT_CONFIG, rollout_iterations=config.rollout_iterations)
    field, path, maximum, goal_index = _fit_position(
        position, start, end, current, tolerance, dt, speed, config, fit_config
    )
    rotation_field, rotation_path, angular_error = None, None, 0.0
    checks = {"chart_valid": True, "angular_speed_valid": True}
    if rotations is not None:
        goal = rotations[end]
        reference = _orientation_coordinates(rotations[start : end + 1], goal)
        carried = _orientation_coordinates(orientation[None], goal)[0]
        rotation_field, rotation_path, _ = contractive.fit(
            reference,
            np.full(3, np.pi / 36),
            dt,
            angular_speed / (np.pi / 36),
            carried=carried,
            refine=False,
            config=fit_config,
        )
        coordinates = rotation_path.copy()
        rotation_path = Rotation.from_rotvec(coordinates.reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3) @ goal
        checks = _rotation_checks(coordinates, rotation_path, angular_speed, dt)
        errors = rotations[start : end + 1, None] @ rotation_path.transpose(0, 1, 3, 2)
        angular_error = float(Rotation.from_matrix(errors.reshape(-1, 3, 3)).magnitude().max())
    return {
        "start": start,
        "end": end,
        "field": field,
        "goal_source_index": goal_index,
        "path": path,
        "maximum": maximum,
        "rotation_field": rotation_field,
        "rotation_path": rotation_path,
        "angular_error": angular_error,
        "angular_speed": angular_speed,
        **checks,
    }


def _refine_candidate(value, position, rotations, current, orientation, config):
    start, end = value["start"], value["end"]
    fit_config = dataclasses.replace(contractive.DEFAULT_CONFIG, rollout_iterations=config.rollout_iterations)
    if 0.9 < value["maximum"] < 4.0:
        value["field"] = contractive.refine_field(value["field"], position[start : end + 1], current, fit_config)
        value["maximum"], value["path"] = contractive.maximum_error(value["field"], position[start : end + 1], current)
    if rotations is not None and value["maximum"] <= 1.0 and value["angular_error"] > np.pi / 36 * 0.9:
        goal = rotations[end]
        reference = _orientation_coordinates(rotations[start : end + 1], goal)
        carried = _orientation_coordinates(orientation[None], goal)[0]
        value["rotation_field"] = contractive.refine_field(value["rotation_field"], reference, carried, fit_config)
        path = value["rotation_field"].rollout(np.stack([reference[0], carried]), end - start)
        value["rotation_path"] = Rotation.from_rotvec(path.reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3) @ goal
        value.update(_rotation_checks(path, value["rotation_path"], value["angular_speed"], value["field"].dt))
        errors = rotations[start : end + 1, None] @ value["rotation_path"].transpose(0, 1, 3, 2)
        value["angular_error"] = float(Rotation.from_matrix(errors.reshape(-1, 3, 3)).magnitude().max())
    return value


def _reasons(value, dt, tolerance):
    reasons = []
    if value["maximum"] > 1.0 + 1e-8:
        reasons.append("position_maximum")
    if not value["chart_valid"]:
        reasons.append("orientation_log_chart")
    if not value["angular_speed_valid"]:
        reasons.append("angular_speed")
    if value["angular_error"] > np.pi / 36 + 1e-8:
        reasons.append("rotation_maximum")
    tail = value["field"].rollout(value["path"][-1], round(2.0 / dt))
    if np.linalg.norm((tail[-1] - value["field"].goal) / tolerance, axis=-1).max() > 0.5 + 1e-8:
        reasons.append("nominal_arrival_timeout")
    return reasons


def _refine_slow_exit(value, position, current, dt, tolerance, speed, config):
    """Spend at most one additional optimization on an actually selected seam."""
    start, end = value["start"], value["end"]
    if (
        not config.goal_lookahead_steps
        or not config.rollout_iterations
        or not config.refinement_candidates
        or end == len(position) - 1
        or _reasons(value, dt, tolerance)
    ):
        return value
    incoming = (position[end] - position[end - 1]) / tolerance
    outgoing = (position[end + 1] - position[end]) / tolerance
    actual = (value["path"][-1, 1] - value["path"][-2, 1]) / tolerance
    if (
        min(np.linalg.norm(incoming), np.linalg.norm(outgoing)) < 1e-8
        or np.dot(incoming, outgoing) <= 0
        or np.linalg.norm(actual) >= 0.65 * np.linalg.norm(incoming)
    ):
        return value
    goal_index = min(end + config.goal_lookahead_steps, len(position) - 1)
    if np.dot((position[goal_index] - position[end]) / tolerance, incoming) <= 0:
        return value
    fit_config = dataclasses.replace(contractive.DEFAULT_CONFIG, rollout_iterations=config.rollout_iterations)
    reference = position[start : end + 1]
    field, path, maximum = contractive.fit(
        reference, tolerance, dt, speed, carried=current, goal=position[goal_index], refine=True, config=fit_config
    )
    trial = {**value, "field": field, "path": path, "maximum": maximum, "goal_source_index": goal_index}
    if not _reasons(trial, dt, tolerance) and _velocity_score(path, reference, tolerance) < _velocity_score(
        value["path"], reference, tolerance
    ):
        return trial
    return value


def fit_sequence(position, source_time, intervals, tolerance, limits, *, rotations=None, config=DEFAULT_CONFIG):
    position, source_time, tolerance = np.asarray(position), np.asarray(source_time), np.asarray(tolerance)
    if (
        position.ndim != 2
        or len(position) < 2
        or source_time.shape != (len(position),)
        or tolerance.shape != (position.shape[1],)
        or not all(np.isfinite(a).all() for a in (position, source_time, tolerance))
        or np.any(tolerance <= 0)
        or config.minimum_source_seconds <= 0
        or config.minimum_transitions < 1
        or config.refinement_candidates < 0
        or config.rollout_iterations < 0
        or config.fine_endpoint_radius < 0
        or config.fine_rollout_iterations < 0
        or not isinstance(config.goal_lookahead_steps, int)
        or config.goal_lookahead_steps < 0
    ):
        raise ValueError("Invalid source observations or segmentation configuration")
    cursor = 0
    for i, parent in enumerate(intervals):
        if parent["start"] != cursor or not cursor < parent["end"] < len(position):
            raise ValueError("Command intervals must exactly cover observations")
        if parent.get("command_interval_id", i) != i:
            raise ValueError("Command interval IDs must follow source order")
        cursor = parent["end"]
    if cursor != len(position) - 1:
        raise ValueError("Command intervals must exactly cover observations")
    if rotations is not None:
        rotations = np.asarray(rotations)
        if (
            rotations.shape != (len(position), 3, 3)
            or not np.isfinite(rotations).all()
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6)
        ):
            raise ValueError("Invalid SO(3) observations")
    dt = float(np.median(np.diff(source_time)))
    if dt <= 0 or not np.allclose(np.diff(source_time), dt, rtol=1e-3, atol=1e-5):
        raise ValueError("Uniform positive source times required")
    minimum = max(config.minimum_transitions, int(np.ceil((config.minimum_source_seconds - 1e-7) / dt)))
    current = position[0].copy()
    orientation = None if rotations is None else rotations[0].copy()
    nominal, orientations = [current.copy()], [] if rotations is None else [orientation.copy()]
    records = []
    for parent_id, parent in enumerate(intervals):
        start, stop = parent["start"], parent["end"]
        while start < stop:
            remaining = stop - start
            short_event = remaining < minimum and start == parent["start"]
            if remaining < 2 * minimum:
                ends = [stop]
            else:
                spans = {
                    minimum,
                    round(1.5 * minimum),
                    2 * minimum,
                    3 * minimum,
                    4 * minimum,
                    6 * minimum,
                    8 * minimum,
                    config.maximum_candidate_factor * minimum,
                }
                ends = sorted({stop, *(start + span for span in spans if start + span <= stop - minimum)}, reverse=True)
            values = [
                _candidate(
                    position,
                    rotations,
                    start,
                    end,
                    current,
                    orientation,
                    dt,
                    tolerance,
                    limits["maximum_speed_scaled_per_second"],
                    limits["maximum_angular_speed_rad_per_second"],
                    config,
                )
                for end in ends
            ]
            successful = [v for v in values if not _reasons(v, dt, tolerance)]
            longest = max((v["end"] for v in successful), default=start)
            refine = sorted(
                (v for v in values if v["end"] > longest and v["maximum"] < 4.0 and v["angular_error"] < np.pi / 9),
                key=lambda v: (max(v["maximum"], v["angular_error"] / (np.pi / 36)), -v["end"]),
            )[: config.refinement_candidates]
            for value in refine:
                _refine_candidate(value, position, rotations, current, orientation, config)
            successful = [v for v in values if not _reasons(v, dt, tolerance)]
            if not successful and config.fine_endpoint_radius > 0:
                # Sparse coarse lengths can miss an otherwise feasible nearby endpoint.
                # Keep enough observations for both this field and the remaining tail.
                fine_ends = {
                    end + shift
                    for end in ends
                    for shift in range(-config.fine_endpoint_radius, config.fine_endpoint_radius + 1)
                    if start + minimum <= end + shift <= stop and (end + shift == stop or stop - end - shift >= minimum)
                } - set(ends)
                extra = [
                    _candidate(
                        position,
                        rotations,
                        start,
                        end,
                        current,
                        orientation,
                        dt,
                        tolerance,
                        limits["maximum_speed_scaled_per_second"],
                        limits["maximum_angular_speed_rad_per_second"],
                        config,
                    )
                    for end in sorted(fine_ends, reverse=True)
                ]
                values.extend(extra)
                successful = [v for v in values if not _reasons(v, dt, tolerance)]
                if not successful:
                    fine_config = dataclasses.replace(config, rollout_iterations=config.fine_rollout_iterations)
                    for value in sorted(
                        (v for v in extra if v["maximum"] < 4.0 and v["angular_error"] < np.pi / 9),
                        key=lambda v: max(v["maximum"], v["angular_error"] / (np.pi / 36)),
                    )[: config.refinement_candidates]:
                        _refine_candidate(value, position, rotations, current, orientation, fine_config)
                    successful = [v for v in values if not _reasons(v, dt, tolerance)]
            # A failed minimum-length span stays visible. Never fall back to a singleton.
            selected = max(successful, key=lambda v: v["end"]) if successful else min(values, key=lambda v: v["end"])
            selected = _refine_slow_exit(
                selected, position, current, dt, tolerance, limits["maximum_speed_scaled_per_second"], config
            )
            end = selected["end"]
            reasons = _reasons(selected, dt, tolerance)
            local_error = selected["path"][:, 0] - position[start : end + 1]
            continuous_error = selected["path"][:, 1] - position[start : end + 1]
            report = {
                "passed": not reasons,
                "reasons": reasons,
                "maximum_scaled_error": selected["maximum"],
                "position_rms_native": float(np.sqrt(np.mean(np.sum(local_error**2, axis=-1)))),
                "continuous_position_rms_native": float(np.sqrt(np.mean(np.sum(continuous_error**2, axis=-1)))),
                "maximum_rotation_error_rad": selected["angular_error"],
            }
            field = selected["field"]
            record = {
                "start": start,
                "end": end,
                "command": parent.get("command"),
                "command_interval_id": parent.get("command_interval_id", parent_id),
                "command_label": parent.get("command_label", "MOVE"),
                "goal": field.goal.tolist(),
                "goal_rotation": None if rotations is None else rotations[end].tolist(),
                "rate_per_second": (1 - field.radius) / dt,
                "maximum_speed_scaled_per_second": field.speed,
                "field": field.record(),
                "orientation_field": None
                if selected["rotation_field"] is None
                else selected["rotation_field"].record(),
                "quality": length_quality(start, end, source_time, config, event=short_event),
                "accepted": not reasons,
                "report": report,
            }
            if config.goal_lookahead_steps:
                record["goal_source_index"] = selected["goal_source_index"]
                record["boundary_position"] = position[end].tolist()
                report["velocity_fit_score"] = _velocity_score(selected["path"], position[start : end + 1], tolerance)
            nominal.extend(selected["path"][1:, 1])
            current = selected["path"][-1, 1].copy()
            if rotations is not None:
                path = selected["rotation_path"]
                # Rotation.from_matrix accepts a flat batch across SciPy versions.
                errors = rotations[start : end + 1, None] @ path.transpose(0, 1, 3, 2)
                angles = Rotation.from_matrix(errors.reshape(-1, 3, 3)).magnitude().reshape(-1, 2)
                report["rotation_rms_rad"] = float(np.sqrt(np.mean(angles[:, 0] ** 2)))
                report["continuous_rotation_rms_rad"] = float(np.sqrt(np.mean(angles[:, 1] ** 2)))
                orientations.extend(path[1:, 1])
                orientation = path[-1, 1].copy()
            records.append(record)
            start = end
    nominal = np.asarray(nominal)
    errors = np.linalg.norm(nominal - position, axis=-1)
    quality = [r["quality"] for r in records]
    summary = {
        "complete_coverage": records[0]["start"] == 0 and records[-1]["end"] == len(position) - 1,
        "actions_covered": len(position) - 1,
        "segments": len(records),
        "accepted_segments": sum(r["accepted"] for r in records),
        "passed": all(r["accepted"] for r in records),
        "position_rms_native": float(np.sqrt(np.mean(errors**2))),
        "maximum_scaled_error": max(r["report"]["maximum_scaled_error"] for r in records),
        "single_transition_segments": sum(r["end"] - r["start"] == 1 for r in records),
        "length_eligible_segments": sum(q["eligible_length"] for q in quality),
        "short_command_events": sum(q["short_command_event"] for q in quality),
        "new_short_motion_segments": sum(not q["eligible_length"] and not q["short_command_event"] for q in quality),
        "eligible_accepted_segments": sum(r["accepted"] and r["quality"]["eligible_length"] for r in records),
        "source_segment_seconds_median": float(np.median([q["source_seconds"] for q in quality])),
        "parameter_supervision": False,
        "physical_execution_validated": False,
        "training_ready": False,
    }
    if rotations is not None:
        angles = Rotation.from_matrix(rotations @ np.asarray(orientations).transpose(0, 2, 1)).magnitude()
        summary["rotation_rms_rad"] = float(np.sqrt(np.mean(angles**2)))
    if config.goal_lookahead_steps:
        summary["pass_through_segments"] = sum(r["goal_source_index"] > r["end"] for r in records)
        summary["flow_fallback"] = False
        if not summary["passed"]:
            # A locally better exit velocity can leave the next long span
            # infeasible. Validate the complete carried-state rollout before
            # accepting this change to the sequence. Refit the endpoint model
            # from the original initial state; never reset individual seams.
            fallback, checked, trace, rotation_trace = fit_sequence(
                position,
                source_time,
                intervals,
                tolerance,
                limits,
                rotations=rotations,
                config=dataclasses.replace(config, goal_lookahead_steps=0),
            )
            if checked["passed"]:
                for record in fallback:
                    a, b = record["start"], record["end"]
                    record["goal_source_index"] = b
                    record["boundary_position"] = position[b].tolist()
                    path = contractive.ContractiveField(record["field"]).rollout(
                        np.stack([position[a], trace[a]]), b - a
                    )
                    record["report"]["velocity_fit_score"] = _velocity_score(path, position[a : b + 1], tolerance)
                checked.update(
                    pass_through_segments=0,
                    flow_fallback=True,
                    flow_attempt_maximum_scaled_error=summary["maximum_scaled_error"],
                    flow_attempt_accepted_segments=summary["accepted_segments"],
                    flow_attempt_segments=summary["segments"],
                )
                return fallback, checked, trace, rotation_trace
    return records, summary, nominal, None if rotations is None else np.asarray(orientations)


def _one(arguments):
    base_directory, destination, entry, fingerprint, config = arguments
    base_directory, destination = Path(base_directory), Path(destination)
    source_path = base_directory / entry["file"]
    base = json.loads(source_path.read_text())
    trace_path = base_directory / base["trace_file"]
    if digest(trace_path) != base["trace_sha256"]:
        raise ValueError("Base observation cache hash mismatch")
    output = destination / source_path.name
    if output.exists():
        saved = json.loads(output.read_text())
        if (
            saved["fingerprint"] == fingerprint
            and saved["base_episode_sha256"] == digest(source_path)
            and digest(destination / saved["trace_file"]) == saved["trace_sha256"]
        ):
            return {"index": entry["index"], "file": output.name, "source_key": base["source_key"], **saved["summary"]}
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = dict(archive)
    records, summary, nominal, orientations = fit_sequence(
        arrays["position"],
        arrays["time"],
        base["source_intervals"],
        arrays["tolerance"],
        base["limits"],
        rotations=arrays.get("rotation"),
        config=config,
    )
    arrays["nominal"] = nominal
    arrays["source_time"] = arrays["time"].copy()
    if orientations is not None:
        arrays["nominal_rotation"] = orientations
    trace = output.with_suffix(".npz")
    np.savez_compressed(trace, **arrays)
    result = {
        **base,
        "model": contractive.VERSION,
        "fingerprint": fingerprint,
        "base_fingerprint": base["fingerprint"],
        "base_episode_sha256": digest(source_path),
        "profile": "coherent_flow" if config.goal_lookahead_steps else PROFILE,
        "profile_label": "长片段/过点稳定场（实验）" if config.goal_lookahead_steps else "长片段/非线性稳定场（实验）",
        "records": records,
        "summary": summary,
        "trace_file": trace.name,
        "trace_sha256": digest(trace),
        "limits": {
            **base["limits"],
            "dt": float(np.median(np.diff(arrays["time"]))),
            "integration_substeps": 1,
            "time_scale": 1,
            "segmentation": dataclasses.asdict(config),
            "orientation_model": "goal-relative SO(3) log chart, learned contractive map",
        },
    }
    atomic_json(output, result)
    return {"index": entry["index"], "file": output.name, "source_key": base["source_key"], **summary}


def build(base_directory, *, workers=8, indices=None, output=None, config=DEFAULT_CONFIG):
    if (
        workers < 1
        or config.minimum_transitions < 1
        or config.minimum_source_seconds <= 0
        or not isinstance(config.goal_lookahead_steps, int)
        or config.goal_lookahead_steps < 0
    ):
        raise ValueError("Positive workers and length thresholds required")
    base_directory = Path(base_directory).resolve()
    base_run = json.loads((base_directory / "run.json").read_text())
    base_index = json.loads((base_directory / "index.json").read_text())
    by_index = {e["index"]: e for e in base_index["episodes"]}
    if indices is not None and (len(indices) != len(set(indices)) or any(i not in by_index for i in indices)):
        raise ValueError("Episode indices must be distinct and present in the base collection")
    chosen = base_index["episodes"] if indices is None else [by_index[i] for i in indices]
    if base_run["status"] != "complete" or not base_run["summary"]["full_collection"]:
        raise ValueError("A complete base collection is required")
    profile = "coherent_flow" if config.goal_lookahead_steps else PROFILE
    label = "长片段/过点稳定场（实验）" if config.goal_lookahead_steps else "长片段/非线性稳定场（实验）"
    destination = base_directory / "profiles" / profile if output is None else Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    implementation = {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), Path(contractive.__file__))}
    fit_config = {
        "model": contractive.VERSION,
        "base_fingerprint": base_run["fingerprint"],
        "segmentation": dataclasses.asdict(config),
        "implementation_sha256": implementation,
        "field_fit": dataclasses.asdict(contractive.DEFAULT_CONFIG),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")},
    }
    fingerprint = hashlib.sha256(json.dumps(fit_config, sort_keys=True).encode()).hexdigest()
    run_path = destination / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Implementation/configuration changed; use a new output directory")
    for relative in implementation:
        path = destination / "source_snapshot" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    source_relative = base_run.get("source_relative", base_run["config"].get("source_relative"))
    run = {
        "fingerprint": fingerprint,
        "config": {
            **fit_config,
            "base_directory": os.path.relpath(base_directory, destination),
            "source_relative": os.path.relpath((base_directory / source_relative).resolve(), destination),
            "source_manifest_sha256": base_run["config"]["source_manifest_sha256"],
            "kind": base_run["config"]["kind"],
            "source_episodes": base_run["config"]["source_episodes"],
            "profile": {"name": profile, "label": label},
        },
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": sys.orig_argv,
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "running",
    }
    atomic_json(run_path, run)
    results, errors = [], []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_one, (str(base_directory), str(destination), entry, fingerprint, config)): entry["index"]
            for entry in chosen
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({"index": futures[future], "error": repr(error)})
            if (len(results) + len(errors)) % 25 == 0 or len(results) + len(errors) == len(chosen):
                progress = {
                    "finished": len(results) + len(errors),
                    "total": len(chosen),
                    "errors": len(errors),
                    "passed_episodes": sum(r["passed"] for r in results),
                    "seconds": round(time.monotonic() - started, 1),
                }
                atomic_json(destination / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    results.sort(key=lambda r: r["index"])
    counts = Counter()
    for result in results:
        for key in (
            "segments",
            "accepted_segments",
            "actions_covered",
            "single_transition_segments",
            "length_eligible_segments",
            "short_command_events",
            "new_short_motion_segments",
            "eligible_accepted_segments",
        ):
            counts[key] += result[key]
    summary = {
        "episodes": len(results),
        "expected_episodes": len(chosen),
        "passed_episodes": sum(r["passed"] for r in results),
        **counts,
        "errors": errors,
        "full_collection": not errors and len(results) == base_run["config"]["source_episodes"],
        "seconds": time.monotonic() - started,
        "training_ready": False,
        "physical_execution_validated": False,
        "parameter_supervision": False,
    }
    atomic_json(destination / "index.json", {"fingerprint": fingerprint, "episodes": results})
    atomic_json(destination / "summary.json", summary)
    run.update(status="complete" if not errors else "error", summary=summary)
    atomic_json(run_path, run)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--fine-iterations", type=int, default=240)
    parser.add_argument("--goal-lookahead-steps", type=int, default=0)
    args = parser.parse_args(argv)
    result = build(
        args.base,
        workers=args.workers,
        indices=args.episodes,
        output=args.output,
        config=dataclasses.replace(
            DEFAULT_CONFIG,
            rollout_iterations=args.iterations,
            fine_rollout_iterations=args.fine_iterations,
            goal_lookahead_steps=args.goal_lookahead_steps,
        ),
    )
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
