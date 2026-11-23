"""Independently reconstruct every control step, acceptance decision and source interval."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from openpi.datasets.libero.reader import atomic_json


def _rollout(starts, rotations, record, limits, tolerance, steps):
    """Step feedback directly; do not call the fitter or its closed-form integrator."""
    position = starts.copy()
    positions, orientations = [position.copy()], [] if rotations is None else [rotations.copy()]
    goal, dt = np.asarray(record["goal"]), limits["dt"]
    for _ in range(steps):
        velocity = record["rate_per_second"] * (goal - position)
        norm = np.linalg.norm(velocity / tolerance, axis=-1)
        velocity *= np.minimum(1.0, record["maximum_speed_scaled_per_second"] / np.maximum(norm, 1e-30))[:, None]
        position += dt * velocity
        positions.append(position.copy())
        if rotations is not None:
            error = Rotation.from_matrix(np.asarray(record["goal_rotation"]) @ rotations.transpose(0, 2, 1)).as_rotvec()
            omega = limits["rotation_gain_per_second"] * error
            norm = np.linalg.norm(omega, axis=-1)
            omega *= np.minimum(1.0, limits["maximum_angular_speed_rad_per_second"] / np.maximum(norm, 1e-30))[:, None]
            rotations = Rotation.from_rotvec(dt * omega).as_matrix() @ rotations
            orientations.append(rotations.copy())
    return np.asarray(positions), None if rotations is None else np.asarray(orientations)


def _episode(result, arrays, profile_config):
    reference, expected, time = arrays["position"], arrays["nominal"], arrays["time"]
    tolerance, rotations = arrays["tolerance"], arrays.get("rotation")
    limits = result["limits"]
    dt, substeps = limits["dt"], limits.get("integration_substeps", 1)
    if not isinstance(substeps, int) or substeps < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid control period or substeps")
    if (
        reference.ndim != 2
        or len(reference) < 2
        or expected.shape != reference.shape
        or len(time) != len(reference)
        or not np.allclose(np.diff(time), dt * substeps, atol=1e-5, rtol=1e-3)
    ):
        raise ValueError("Timestamp/state count or control period mismatch")
    if any(not np.isfinite(value).all() for value in arrays.values()) or np.any(tolerance <= 0):
        raise ValueError("Nonfinite trace or invalid tolerance")
    if isinstance(profile_config, dict):
        speed = profile_config["maximum_speed_native_per_second"]
        if speed is not None and any(
            r["maximum_speed_scaled_per_second"] * np.max(tolerance) > speed + 1e-9 for r in result["records"]
        ):
            raise ValueError("Profile position speed cap exceeded")
        if (
            abs(limits["maximum_angular_speed_rad_per_second"] - profile_config["maximum_angular_speed_rad_per_second"])
            > 1e-9
        ):
            raise ValueError("Profile angular speed cap mismatch")
        if abs(limits["rotation_gain_per_second"] - profile_config["rotation_gain_per_second"]) > 1e-9:
            raise ValueError("Profile rotation gain mismatch")
    source_time = arrays.get("source_time", time)
    scale = limits.get("time_scale", 1)
    if not np.allclose(time - time[0], (source_time - source_time[0]) * scale, atol=1e-6, rtol=1e-5):
        raise ValueError("Source-to-execution time mapping mismatch")
    if isinstance(profile_config, dict) and profile_config.get("maximum_time_scale", 1) > 1:
        if scale != substeps or not 1 <= scale <= profile_config["maximum_time_scale"]:
            raise ValueError("Retiming must use one original control step per substep")
        if not np.allclose(np.diff(source_time), dt, atol=1e-5, rtol=1e-3):
            raise ValueError("Retiming changed the original control period")
    query = np.arange((len(time) - 1) * substeps + 1) / substeps
    dense = np.stack([np.interp(query, np.arange(len(time)), axis) for axis in reference.T], axis=1)
    dense_rotations = (
        None if rotations is None else Slerp(np.arange(len(time)), Rotation.from_matrix(rotations))(query).as_matrix()
    )
    current, rotation = reference[0].copy(), None if rotations is None else rotations[0].copy()
    actual, orientations = [current.copy()], [] if rotation is None else [rotation.copy()]
    cursor, maximum_error, maximum_rotation, accepted, rejected = 0, 0.0, 0.0, 0, Counter()
    for record in result["records"]:
        start, end = record["start"], record["end"]
        if start != cursor or not start < end < len(reference):
            raise ValueError("Incomplete, overlapping or invalid segmentation")
        if (
            not np.isfinite([record["rate_per_second"], record["maximum_speed_scaled_per_second"]]).all()
            or record["rate_per_second"] <= 0
            or record["maximum_speed_scaled_per_second"] <= 0
            or record["rate_per_second"] * dt > 1 + 1e-9
        ):
            raise ValueError("Invalid stability or discrete integration bound")
        parent = result["source_intervals"][record["command_interval_id"]]
        if not parent["start"] <= start < end <= parent["end"] or record.get("command") != parent.get("command"):
            raise ValueError("Dynamics crosses a command boundary")
        if not np.allclose(record["goal"], reference[end], atol=1e-12, rtol=0):
            raise ValueError("Field goal differs from its source endpoint")
        initial_rotations = None if rotations is None else np.stack([rotations[start], rotation])
        path, orientation = _rollout(
            np.stack([reference[start], current]),
            initial_rotations,
            record,
            limits,
            tolerance,
            (end - start) * substeps,
        )
        selection = slice(start * substeps, end * substeps + 1)
        maximum = float(np.linalg.norm((path - dense[selection, None]) / tolerance, axis=-1).max())
        maximum_error = max(maximum_error, maximum)
        reasons = []
        if maximum > 1 + 1e-8:
            reasons.append("position_maximum")
        extra = round(limits["extra_convergence_seconds"] / dt)
        tail, tail_orientation = _rollout(
            path[-1], None if orientation is None else orientation[-1], record, limits, tolerance, extra
        )
        dwell = max(1, int(np.ceil(limits.get("arrival_dwell_seconds", 0) / dt)))
        distance = np.linalg.norm((tail[-dwell:] - np.asarray(record["goal"])) / tolerance, axis=-1)
        arrived = distance.max() <= limits["arrival_fraction"] + 1e-8
        speed_gate = limits.get("arrival_speed_scaled_per_second")
        if speed_gate is not None:
            tail_speed = np.minimum(record["rate_per_second"] * distance, record["maximum_speed_scaled_per_second"])
            arrived &= tail_speed.max() <= speed_gate + 1e-8
        angle_maximum = 0.0
        if rotations is not None:
            if not np.allclose(record["goal_rotation"], rotations[end], atol=1e-12, rtol=0):
                raise ValueError("Rotation goal differs from its source endpoint")
            errors = dense_rotations[selection, None] @ orientation.transpose(0, 1, 3, 2)
            angles = Rotation.from_matrix(errors.reshape(-1, 3, 3)).magnitude()
            angle_maximum = float(angles.max())
            maximum_rotation = max(maximum_rotation, angle_maximum)
            angular_gate = limits.get("arrival_angular_speed_rad_per_second")
            if angular_gate is not None:
                errors = np.asarray(record["goal_rotation"]) @ tail_orientation[-dwell:].transpose(0, 1, 3, 2)
                angles = Rotation.from_matrix(errors.reshape(-1, 3, 3)).magnitude()
                arrived &= angles.max() <= limits["rotation_tolerance_rad"] + 1e-8
                arrived &= (
                    min(
                        limits["maximum_angular_speed_rad_per_second"],
                        limits["rotation_gain_per_second"] * angles.max(),
                    )
                    <= angular_gate + 1e-8
                )
        if not arrived:
            reasons.append("nominal_arrival_timeout")
        if angle_maximum > limits["rotation_tolerance_rad"] + 1e-8:
            reasons.append("rotation_maximum")
        if bool(record["accepted"]) != (not reasons) or set(record["report"]["reasons"]) != set(reasons):
            raise ValueError("Saved acceptance/reasons disagree with independent continuous reconstruction")
        if abs(record["report"]["maximum_scaled_error"] - maximum) > 1e-7:
            raise ValueError("Saved local/continuous maximum error mismatch")
        rejected.update(reasons)
        accepted += int(not reasons)
        actual.extend(path[substeps::substeps, 1])
        current = path[-1, 1].copy()
        if orientation is not None:
            orientations.extend(orientation[substeps::substeps, 1])
            rotation = orientation[-1, 1].copy()
        cursor = end
    if cursor != len(reference) - 1:
        raise ValueError("Missing final observed transition")
    actual = np.asarray(actual)
    discrepancy = float(np.abs(actual - expected).max())
    if discrepancy > 1e-9:
        raise ValueError(f"Saved rollout differs from stepwise feedback: {discrepancy}")
    if "nominal_rotation" in arrays and np.max(np.abs(arrays["nominal_rotation"] - orientations)) > 1e-9:
        raise ValueError("Saved orientation rollout differs from stepwise feedback")
    nominal_passed = not rejected
    if bool(result["summary"]["passed"]) != nominal_passed:
        raise ValueError("Saved whole-episode acceptance mismatch")
    rotation_rms = None
    if rotations is not None:
        angles = Rotation.from_matrix(rotations @ np.asarray(orientations).transpose(0, 2, 1)).magnitude()
        rotation_rms = float(np.sqrt(np.mean(angles**2)))
    return {
        "maximum_error": maximum_error,
        "maximum_rotation": maximum_rotation,
        "discrepancy": discrepancy,
        "position_rms": float(np.sqrt(np.mean(np.sum((actual - reference) ** 2, axis=-1)))),
        "rotation_rms": rotation_rms,
        "nominal_passed": nominal_passed,
        "rejected": rejected,
        "counts": {
            "episodes": 1,
            "segments": len(result["records"]),
            "accepted_segments": accepted,
            "transitions": len(reference) - 1,
            "control_steps": (len(reference) - 1) * substeps,
            "single_transition_segments": sum(r["end"] - r["start"] == 1 for r in result["records"]),
        },
    }


def _read_entry(arguments):
    directory, entry, fingerprint, configured_profile = arguments
    try:
        path = Path(directory) / entry["file"]
        result = json.loads(path.read_text())
        trace = Path(directory) / result["trace_file"]
        trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
        digest = path.name.encode() + hashlib.sha256(path.read_bytes()).digest() + bytes.fromhex(trace_hash)
        if result["fingerprint"] != fingerprint or trace_hash != result["trace_sha256"]:
            raise ValueError("Fingerprint/trace hash mismatch")
        with np.load(trace, allow_pickle=False) as data:
            arrays = dict(data)
        return {
            "value": _episode(result, arrays, configured_profile),
            "digest": digest,
            "source_speed": result["summary"].get("source_tracking_max_speed_m_per_second"),
        }
    except (ValueError, KeyError, IndexError, OSError, TypeError) as error:
        return {"error": {"index": entry["index"], "error": str(error)}}


def audit(directory, *, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    directory = Path(directory)
    run, index = (json.loads((directory / name).read_text()) for name in ("run.json", "index.json"))
    counts, failures, rejected, chain = Counter(), [], Counter(), hashlib.sha256()
    worst, maximum_difference, worst_rotation, passed_episodes = 0.0, 0.0, 0.0, 0
    position_rms, rotation_rms, source_speed = [], [], []
    configured_profile = run.get("config", {}).get("profile", "replay")
    profile_name = configured_profile["name"] if isinstance(configured_profile, dict) else "replay"
    profile_label = configured_profile["label"] if isinstance(configured_profile, dict) else str(configured_profile)
    if run["status"] != "complete" or index["fingerprint"] != run["fingerprint"]:
        failures.append({"error": "Run incomplete or index fingerprint mismatch"})
    arguments = [(str(directory), entry, run["fingerprint"], configured_profile) for entry in index["episodes"]]
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        values = map(_read_entry, arguments) if pool is None else pool.map(_read_entry, arguments, chunksize=8)
        for read in values:
            if "error" in read:
                failures.append(read["error"])
                continue
            chain.update(read["digest"])
            value = read["value"]
            counts.update(value["counts"])
            rejected.update(value["rejected"])
            worst = max(worst, value["maximum_error"])
            worst_rotation = max(worst_rotation, value["maximum_rotation"])
            maximum_difference = max(maximum_difference, value["discrepancy"])
            passed_episodes += value["nominal_passed"]
            position_rms.append(value["position_rms"])
            if value["rotation_rms"] is not None:
                rotation_rms.append(value["rotation_rms"])
            if read["source_speed"] is not None:
                source_speed.append(read["source_speed"])
    finally:
        if pool is not None:
            pool.shutdown()
    full_coverage = sorted(entry["index"] for entry in index["episodes"]) == list(
        range(run["config"]["source_episodes"])
    )
    output = {
        "schema": "vlds.dynamics.audit.v2",
        "fingerprint": run["fingerprint"],
        "counts": dict(counts),
        "failures": failures,
        "profile": profile_name,
        "profile_label": profile_label,
        "rejected_segments_by_reason": dict(sorted(rejected.items())),
        "full_source_episode_coverage": full_coverage,
        "maximum_scaled_error": worst,
        "maximum_rotation_error_rad": worst_rotation,
        "maximum_saved_vs_stepwise_error": maximum_difference,
        "artifact_sha256_chain": chain.hexdigest(),
        "position_rms_native_p50_p95_max": np.percentile(position_rms, [50, 95, 100]).tolist() if position_rms else [],
        "rotation_rms_rad_p50_p95_max": np.percentile(rotation_rms, [50, 95, 100]).tolist() if rotation_rms else [],
        "episodes_with_position_speed_cap_over_0_3": sum(v > 0.3 + 1e-9 for v in source_speed),
        "maximum_position_speed_cap_m_per_second": max(source_speed, default=None),
        "profile_passed_episodes": passed_episodes,
        "integrity_passed": not failures and full_coverage,
        "all_nominal_passed": not failures and full_coverage and passed_episodes == len(index["episodes"]),
        "physical_execution_validated": False,
        "training_ready": False,
    }
    output["passed"] = output["integrity_passed"] and (profile_name != "replay" or output["all_nominal_passed"])
    atomic_json(directory / "audit.json", output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    result = audit(args.directory, workers=args.workers)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
