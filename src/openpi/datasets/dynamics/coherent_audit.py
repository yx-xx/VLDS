"""Independent direct reconstruction of length-constrained Gaussian LPV fields."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.datasets.libero.reader import atomic_json


def _validate(field, dt, tolerance, goal):
    if field["version"] != "gaussian_contractive_map_v1":
        raise ValueError("Unsupported Gaussian map")
    arrays = {
        key: np.asarray(field[key], dtype=float)
        for key in ("goal", "tolerance", "transform", "means", "variance", "matrices")
    }
    d, k = len(goal), len(arrays["means"])
    shapes = {
        "goal": (d,),
        "tolerance": (d,),
        "transform": (d, d),
        "means": (k, d),
        "variance": (d,),
        "matrices": (k, d, d),
    }
    if not k or any(a.shape != shapes[key] or not np.isfinite(a).all() for key, a in arrays.items()):
        raise ValueError("Invalid field dimensions or values")
    if (
        np.any(arrays["tolerance"] <= 0)
        or np.any(arrays["variance"] <= 0)
        or not np.allclose(arrays["goal"], goal, atol=1e-12, rtol=0)
        or not np.array_equal(arrays["tolerance"], tolerance)
        or not np.isfinite([field["dt"], field["maximum_speed_scaled_per_second"]]).all()
        or abs(field["dt"] - dt) > 1e-9
        or field["maximum_speed_scaled_per_second"] <= 0
    ):
        raise ValueError("Field goal, metric or time mismatch")
    radius = np.linalg.svd(arrays["matrices"], compute_uv=False).max()
    if (not np.isfinite(radius) or radius >= 1.0) or np.linalg.cond(arrays["transform"]) ** 2 > 1000.0 + 1e-5:
        raise ValueError("Invalid common Lyapunov certificate")
    if (
        abs(field["maximum_contraction"] - radius) > 1e-9
        or abs(field["metric_condition"] - np.linalg.cond(arrays["transform"]) ** 2) > 1e-6
    ):
        raise ValueError("Saved certificate bounds differ from reconstructed parameters")
    return arrays


def _step(position, field, arrays):
    # Deliberately independent of the fitter and its field class.
    z = ((position - arrays["goal"]) / arrays["tolerance"]) @ arrays["transform"].T
    log_weights = -0.5 * (((z[:, None] - arrays["means"]) ** 2) / arrays["variance"]).sum(-1)
    weights = np.exp(log_weights - log_weights.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    mapped = np.zeros_like(z)
    for k, matrix in enumerate(arrays["matrices"]):
        mapped += weights[:, k, None] * (z @ matrix.T)
    delta = np.linalg.solve(arrays["transform"], (mapped - z).T).T
    speed = np.linalg.norm(delta, axis=-1) / field["dt"]
    alpha = np.minimum(1.0, field["maximum_speed_scaled_per_second"] / np.maximum(speed, 1e-30))
    result = position + alpha[:, None] * delta * arrays["tolerance"]
    next_z = ((result - arrays["goal"]) / arrays["tolerance"]) @ arrays["transform"].T
    if np.any(np.sum(next_z**2, axis=-1) > np.sum(z**2, axis=-1) + 1e-8):
        raise ValueError("Actual capped step violates the saved Lyapunov metric")
    return result


def audit_episode(result, arrays):
    ref, nominal, times, tolerance = (arrays[k] for k in ("position", "nominal", "time", "tolerance"))
    dt = result["limits"]["dt"]
    source_times = arrays["source_time"]
    if (
        not np.isfinite(dt)
        or dt <= 0
        or ref.ndim != 2
        or tolerance.shape != (ref.shape[1],)
        or np.any(tolerance <= 0)
        or source_times.shape != (len(ref),)
        or len(ref) < 2
        or nominal.shape != ref.shape
        or len(times) != len(ref)
        or not np.allclose(times, source_times, atol=1e-10, rtol=0)
        or not np.allclose(np.diff(times), dt, atol=1e-5, rtol=1e-3)
        or any(not np.isfinite(a).all() for a in arrays.values())
    ):
        raise ValueError("Invalid source observations, time or trace")
    cursor = 0
    for i, parent in enumerate(result["source_intervals"]):
        if parent["start"] != cursor or not cursor < parent["end"] < len(ref):
            raise ValueError("Invalid source command coverage")
        if parent.get("command_interval_id", i) != i:
            raise ValueError("Invalid command interval identity")
        cursor = parent["end"]
    if cursor != len(ref) - 1 or not np.array_equal(result["tolerance"], tolerance):
        raise ValueError("Invalid source coverage or tolerance")
    rotation = arrays.get("rotation")
    if rotation is not None and (
        rotation.shape != (len(ref), 3, 3)
        or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-6)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError("Invalid source SO(3) observations")
    current = ref[0].copy()
    carried_rotation = None if rotation is None else rotation[0].copy()
    actual, orientations = [current], [] if rotation is None else [carried_rotation]
    cursor, accepted, quality_accepted = 0, 0, 0
    worst, angular_worst = 0.0, 0.0
    rejected, counts = Counter(), Counter()
    config = result["limits"]["segmentation"]
    for record in result["records"]:
        start, end = record["start"], record["end"]
        if start != cursor or not start < end < len(ref):
            raise ValueError("Missing or duplicated observed transitions")
        parent = result["source_intervals"][record["command_interval_id"]]
        if not parent["start"] <= start < end <= parent["end"] or record["command"] != parent.get("command"):
            raise ValueError("Crossed or changed a binary command event")
        field = record["field"]
        goal_index = record.get("goal_source_index", end)
        lookahead = config.get("goal_lookahead_steps", 0)
        if (
            not isinstance(goal_index, int)
            or not end <= goal_index <= min(end + lookahead, len(ref) - 1)
            or (lookahead and not np.array_equal(record.get("boundary_position"), ref[end]))
        ):
            raise ValueError("Invalid pass-through target or observed boundary")
        if (
            not np.allclose(record["goal"], ref[goal_index], atol=1e-12, rtol=0)
            or abs(record["maximum_speed_scaled_per_second"] - field["maximum_speed_scaled_per_second"]) > 1e-9
        ):
            raise ValueError("Displayed parameters differ from the actual field")
        decoded = _validate(field, dt, tolerance, ref[goal_index])
        if field["maximum_speed_scaled_per_second"] > result["limits"]["maximum_speed_scaled_per_second"] + 1e-9:
            raise ValueError("Position speed cap exceeds base profile")
        position = np.stack([ref[start], current])
        path = [position.copy()]
        angular_maximum = 0.0
        chart_valid, angular_speed_valid = True, True
        if rotation is not None:
            goal_rotation = np.asarray(record["goal_rotation"])
            if not np.allclose(goal_rotation, rotation[end], atol=1e-12, rtol=0):
                raise ValueError("Rotation target changed")
            rotation_field = record["orientation_field"]
            rotation_decoded = _validate(rotation_field, dt, np.full(3, np.pi / 36), np.zeros(3))
            if (
                rotation_field["maximum_speed_scaled_per_second"] * (np.pi / 36)
                > result["limits"]["maximum_angular_speed_rad_per_second"] + 1e-9
            ):
                raise ValueError("Orientation field cap exceeds the base profile")
            orient = np.stack([rotation[start], carried_rotation])
            error = Rotation.from_matrix(orient @ goal_rotation.T).as_rotvec()
            chart_valid = np.linalg.norm(error, axis=-1).max() < np.pi - 1e-6
            orientation_trace = [orient.copy()]
        for _ in range(end - start):
            position = _step(position, field, decoded)
            path.append(position.copy())
            if rotation is not None:
                next_error = _step(error, rotation_field, rotation_decoded)
                if np.linalg.norm(next_error, axis=-1).max() >= np.pi - 1e-6:
                    chart_valid = False
                next_orientation = Rotation.from_rotvec(next_error).as_matrix() @ goal_rotation
                velocity = Rotation.from_matrix(next_orientation @ orient.transpose(0, 2, 1)).magnitude() / dt
                if velocity.max() > result["limits"]["maximum_angular_speed_rad_per_second"] + 1e-7:
                    angular_speed_valid = False
                error = next_error
                orient = next_orientation
                orientation_trace.append(orient.copy())
        path = np.asarray(path)
        for key, column in (("position_rms_native", 0), ("continuous_position_rms_native", 1)):
            rms = float(np.sqrt(np.mean(np.sum((path[:, column] - ref[start : end + 1]) ** 2, axis=-1))))
            if abs(record["report"][key] - rms) > 1e-8:
                raise ValueError("Saved RMS differs from independent reconstruction")
        error = np.linalg.norm((path - ref[start : end + 1, None]) / tolerance, axis=-1)
        maximum = float(error.max())
        if lookahead:
            expected_delta = np.diff(ref[start : end + 1], axis=0) / tolerance
            generated_delta = np.diff(path, axis=0) / tolerance
            delta_error = np.sum((generated_delta - expected_delta[:, None]) ** 2, axis=-1)
            score = float(np.mean(delta_error) + 2 * np.mean(delta_error[-1]))
            if abs(record["report"]["velocity_fit_score"] - score) > 1e-8:
                raise ValueError("Saved velocity fit score differs from independent reconstruction")
        reasons = []
        if not chart_valid:
            reasons.append("orientation_log_chart")
        if not angular_speed_valid:
            reasons.append("angular_speed")
        if maximum > 1.0 + 1e-8:
            reasons.append("position_maximum")
        tail = position.copy()
        for _ in range(round(2.0 / dt)):
            tail = _step(tail, field, decoded)
        if np.linalg.norm((tail - ref[goal_index]) / tolerance, axis=-1).max() > 0.5 + 1e-8:
            reasons.append("nominal_arrival_timeout")
        if rotation is not None:
            orientation_trace = np.asarray(orientation_trace)
            errors = rotation[start : end + 1, None] @ orientation_trace.transpose(0, 1, 3, 2)
            angular_maximum = float(Rotation.from_matrix(errors.reshape(-1, 3, 3)).magnitude().max())
            if angular_maximum > np.pi / 36 + 1e-8:
                reasons.append("rotation_maximum")
            orientations.extend(orientation_trace[1:, 1])
            carried_rotation = orient[1].copy()
        if (
            bool(record["accepted"]) != (not reasons)
            or bool(record["report"]["passed"]) != (not reasons)
            or set(record["report"]["reasons"]) != set(reasons)
        ):
            raise ValueError("Saved acceptance differs from independent step feedback")
        if abs(record["report"]["maximum_scaled_error"] - maximum) > 1e-7:
            raise ValueError("Saved error differs from independent recomputation")
        quality = record["quality"]
        seconds = float(source_times[end] - source_times[start])
        eligible = end - start >= config["minimum_transitions"] and seconds >= config["minimum_source_seconds"] - 1e-7
        event = not eligible and start == parent["start"] and end == parent["end"]
        if (
            quality["transitions"] != end - start
            or quality["eligible_length"] != eligible
            or quality["observations"] != end - start + 1
            or abs(quality["source_seconds"] - seconds) > 1e-8
            or quality["short_command_event"] != event
        ):
            raise ValueError("Saved observed length/quality mismatch")
        if not eligible and not event:
            raise ValueError("Fitter manufactured a short motion segment")
        counts.update(
            segments=1,
            transitions=end - start,
            length_eligible_segments=int(eligible),
            short_command_events=int(event),
            accepted_segments=int(not reasons),
        )
        accepted += int(not reasons)
        quality_accepted += int(not reasons and eligible)
        rejected.update(reasons)
        worst, angular_worst = max(worst, maximum), max(angular_worst, angular_maximum)
        actual.extend(path[1:, 1])
        current = position[1].copy()
        cursor = end
    difference = float(np.abs(np.asarray(actual) - nominal).max())
    if cursor != len(ref) - 1 or difference > 1e-9:
        raise ValueError("Incomplete coverage or saved continuous trace mismatch")
    if rotation is not None and np.abs(np.asarray(orientations) - arrays["nominal_rotation"]).max() > 1e-9:
        raise ValueError("Saved continuous orientation trace mismatch")
    counts["actions_covered"] = counts["transitions"]
    counts["single_transition_segments"] = sum(r["end"] - r["start"] == 1 for r in result["records"])
    counts["new_short_motion_segments"] = 0
    counts["eligible_accepted_segments"] = quality_accepted
    if any(
        result["summary"].get(key) != counts[key]
        for key in (
            "actions_covered",
            "segments",
            "accepted_segments",
            "single_transition_segments",
            "length_eligible_segments",
            "short_command_events",
            "new_short_motion_segments",
            "eligible_accepted_segments",
        )
    ):
        raise ValueError("Saved summary counts differ from independent reconstruction")
    if bool(result["summary"]["passed"]) != (not rejected):
        raise ValueError("Saved episode acceptance mismatch")
    counts["eligible_accepted_segments"] = quality_accepted
    return {
        "counts": dict(counts),
        "rejected": dict(rejected),
        "maximum_scaled_error": worst,
        "maximum_rotation_error_rad": angular_worst,
        "saved_vs_recomputed": difference,
        "nominal_passed": not rejected,
    }


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _audit_one(arguments):
    directory, run, entry = arguments
    directory = Path(directory)
    try:
        path = directory / entry["file"]
        result = json.loads(path.read_text())
        base = (directory / run["config"]["base_directory"]).resolve()
        base_path = base / entry["file"]
        if _digest(base_path) != result["base_episode_sha256"]:
            raise ValueError("Base episode hash mismatch")
        original = json.loads(base_path.read_text())
        source = (directory / run["config"]["source_relative"]).resolve()
        if (
            run["config"]["kind"] != "processed_libero"
            and _digest(source / result["source_key"]) != result["source_sha256"]
        ):
            raise ValueError("Original source episode file changed")
        if (
            result["base_fingerprint"] != run["config"]["base_fingerprint"]
            or result["source_sha256"] != original["source_sha256"]
            or result["source_intervals"] != original["source_intervals"]
            or result["source_episode_index"] != entry["index"]
            or result["limits"]["segmentation"] != run["config"]["segmentation"]
            or result["profile"] != run["config"]["profile"]["name"]
        ):
            raise ValueError("Base source identity mismatch")
        trace = directory / result["trace_file"]
        trace_hash = _digest(trace)
        if trace_hash != result["trace_sha256"] or result["fingerprint"] != run["fingerprint"]:
            raise ValueError("Trace hash or fingerprint mismatch")
        source_trace = base / original["trace_file"]
        if _digest(source_trace) != original["trace_sha256"]:
            raise ValueError("Base trace hash mismatch")
        with np.load(trace, allow_pickle=False) as data, np.load(source_trace, allow_pickle=False) as source:
            for key in ("position", "time", "tolerance", "rotation"):
                if key in source and not np.array_equal(data[key], source[key]):
                    raise ValueError("Original observations were changed")
            checked = audit_episode(result, dict(data))
        for key, value in checked["counts"].items():
            if key in result["summary"] and entry.get(key) != value:
                raise ValueError("Index counts differ from reconstructed episode")
        checked["artifact_hash"] = _digest(path) + trace_hash
        return checked, None
    except (ValueError, KeyError, IndexError, OSError, TypeError, np.linalg.LinAlgError) as error:
        return None, {"index": entry["index"], "error": str(error)}


def validate_saved_episode(directory, run, entry):
    """Recheck a manually selected pilot against its base observations and field."""
    checked, error = _audit_one((str(directory), run, entry))
    if error is not None:
        raise ValueError("样例独立重算失败: " + error["error"])
    return checked


def audit(directory, *, workers=1):
    directory = Path(directory)
    run, index = (json.loads((directory / name).read_text()) for name in ("run.json", "index.json"))
    failures, records, counts, rejected = [], [], Counter(), Counter()
    chain = hashlib.sha256()
    if run["status"] != "complete" or run["fingerprint"] != index["fingerprint"]:
        failures.append({"error": "Run incomplete or fingerprint mismatch"})
    audit_bytes = Path(__file__).read_bytes()
    audit_snapshot = directory / "validation_snapshot/coherent_audit.py"
    audit_snapshot.parent.mkdir(parents=True, exist_ok=True)
    audit_snapshot.write_bytes(audit_bytes)
    source = (directory / run["config"]["source_relative"]).resolve()
    metadata = (
        (source / "manifest.json").read_bytes()
        if run["config"]["kind"] == "processed_libero"
        else ((source / "meta/info.json").read_bytes() + (source / "meta/episodes.jsonl").read_bytes())
    )
    if hashlib.sha256(metadata).hexdigest() != run["config"]["source_manifest_sha256"]:
        failures.append({"error": "Source catalogue hash mismatch"})
    for relative, expected in run["config"]["implementation_sha256"].items():
        if _digest(directory / "source_snapshot" / relative) != expected:
            failures.append({"error": "Archived implementation hash mismatch", "file": relative})
    arguments = [(str(directory), run, entry) for entry in index["episodes"]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(_audit_one, arguments, chunksize=4))
    else:
        values = list(map(_audit_one, arguments))
    for checked, error in values:
        if error:
            failures.append(error)
            continue
        records.append(checked)
        counts.update(checked["counts"])
        rejected.update(checked["rejected"])
        chain.update(bytes.fromhex(checked["artifact_hash"]))
    coverage = sorted(e["index"] for e in index["episodes"]) == list(range(run["config"]["source_episodes"]))
    for key, value in counts.items():
        if key in run["summary"] and run["summary"][key] != value:
            failures.append({"error": "Run summary mismatch", "key": key})
    result = {
        "schema": "vlds.dynamics.coherent_audit.v1",
        "fingerprint": run["fingerprint"],
        "audit_implementation_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "episodes": len(records),
        "counts": dict(counts),
        "failures": failures,
        "rejected_segments_by_reason": dict(rejected),
        "full_source_episode_coverage": coverage,
        "integrity_passed": not failures,
        "all_nominal_passed": not failures and coverage and not rejected,
        "passed_episodes": sum(r["nominal_passed"] for r in records),
        "maximum_saved_vs_stepwise_error": max((r["saved_vs_recomputed"] for r in records), default=0),
        "artifact_sha256_chain": chain.hexdigest(),
        "parameter_supervision": False,
        "physical_execution_validated": False,
        "training_ready": False,
    }
    atomic_json(directory / "audit.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    result = audit(args.directory, workers=args.workers)
    print(json.dumps(result, indent=2))
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
