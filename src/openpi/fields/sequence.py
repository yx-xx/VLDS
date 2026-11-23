"""Stable isotropic fields fitted to rollouts, with continuous-state subdivision.

This reduced family is deliberately distinct from the nonlinear SEDS estimator.
Every segment has a fixed target and f(x)=-rate*(x-target), rate>0. Diagonal
coordinate scaling is a metric for fitting, not a change to the native field.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

VERSION = "stable_isotropic_rollout_v1"


@dataclasses.dataclass(frozen=True)
class SequenceConfig:
    maximum_position_error: float = 1.0  # norm in explicitly recorded tolerance units
    minimum_rate: float = 0.5
    extra_seconds: float = 2.0
    arrival_fraction: float = 0.5
    fit_margin: float = 0.9
    arrival_speed_scaled_per_second: float | None = None
    arrival_angular_speed_rad_per_second: float | None = None
    arrival_dwell_seconds: float = 0.0


DEFAULT_SEQUENCE = SequenceConfig()


def integrate(position, goal, rate, steps, dt, speed):
    """Euler feedback with a scalar norm cap; 0 < rate*dt <= 1 prevents overshoot."""
    position, goal, rate = np.asarray(position, dtype=float), np.asarray(goal), np.asarray(rate)
    direction = position - goal
    distance, rate, speed = np.broadcast_arrays(np.linalg.norm(direction, axis=-1), rate, speed)
    direction = np.broadcast_to(direction, (*distance.shape, position.shape[-1]))
    capped = np.ceil(np.maximum(0, distance - speed / rate) / (speed * dt)).astype(int)
    n = np.arange(steps + 1)
    linear_steps = np.minimum(n, capped[..., None])
    remaining = (distance[..., None] - linear_steps * speed[..., None] * dt) * np.maximum(
        0, 1 - rate[..., None] * dt
    ) ** np.maximum(0, n - capped[..., None])
    ratio = np.divide(remaining, distance[..., None], out=np.zeros_like(remaining), where=distance[..., None] > 0)
    return goal + ratio[..., None] * direction[..., None, :]


def integrate_rotation(initial, goal, steps, dt, maximum_speed, gain=None):
    """Fixed SO(3) target with a configurable gain and bounded angular velocity."""
    return _integrate_rotation(initial, goal, steps, dt, maximum_speed, 1 / dt if gain is None else gain)


def _integrate_rotation(initial, goal, steps, dt, maximum_speed, gain):
    if not np.isfinite([gain, dt, maximum_speed]).all() or min(gain, dt, maximum_speed) <= 0 or gain * dt > 1 + 1e-9:
        raise ValueError("Rotation feedback requires positive finite limits and gain*dt <= 1")
    initial = np.asarray(initial)
    error = Rotation.from_matrix(goal @ initial.T).as_rotvec()
    angle = np.linalg.norm(error)
    remaining = integrate(np.array([angle]), np.zeros(1), gain, steps, dt, maximum_speed)[:, 0]
    fractions = 1 - remaining / max(angle, 1e-30)
    return Rotation.from_rotvec(fractions[:, None] * error).as_matrix() @ initial


def angular_errors(reference, actual):
    return Rotation.from_matrix(reference @ actual.transpose(0, 2, 1)).magnitude()


def fit_sequence(
    position,
    time,
    intervals,
    tolerance,
    *,
    rotations=None,
    rotation_tolerance=np.pi / 36,
    maximum_speed=None,
    maximum_angular_speed=None,
    rotation_gain=None,
    integration_substeps=1,
    config=DEFAULT_SEQUENCE,
):
    """Partition every transition; assess both local starts and propagated starts.

    Tolerance is in the source's native units. Speed limits are in tolerance
    units/s; all values and any calibrated limits are persisted by the caller.
    Source timestamps must be uniform. No terminal observation is invented.
    """
    position, time, tolerance = np.asarray(position), np.asarray(time), np.asarray(tolerance)
    if position.ndim != 2 or len(position) != len(time) or len(time) < 2:
        raise ValueError("Expected at least two aligned state/time observations")
    if not np.isfinite(position).all() or not np.isfinite(time).all() or np.any(tolerance <= 0):
        raise ValueError("Nonfinite trajectory or invalid metric")
    sample_dt = float(np.median(np.diff(time)))
    if sample_dt <= 0 or not np.allclose(np.diff(time), sample_dt, atol=1e-5, rtol=1e-3):
        raise ValueError("Uniform positive sample times required")
    if not isinstance(integration_substeps, int | np.integer) or integration_substeps < 1:
        raise ValueError("integration_substeps must be a positive integer")
    dt = sample_dt / integration_substeps
    cursor = 0
    for interval in intervals:
        if interval["start"] != cursor or not cursor < interval["end"] < len(position):
            raise ValueError("Intervals must partition every observed transition")
        cursor = interval["end"]
    if cursor != len(position) - 1:
        raise ValueError("Intervals omit source transitions")
    scaled = position / tolerance
    coordinates = np.arange((len(time) - 1) * integration_substeps + 1) / integration_substeps
    dense = np.stack([np.interp(coordinates, np.arange(len(time)), axis) for axis in scaled.T], axis=1)
    dense_rotations = (
        None
        if rotations is None
        else Slerp(np.arange(len(time)), Rotation.from_matrix(rotations))(coordinates).as_matrix()
    )
    observed_speed = float(np.max(np.linalg.norm(np.diff(scaled, axis=0), axis=1)) / sample_dt)
    # The offline source tracking profile is not a robot safety/speed certificate.
    speed = maximum_speed if maximum_speed is not None else observed_speed * 1.05 + 1 / dt
    angular_observed = (
        0.0 if rotations is None else float(np.max(angular_errors(rotations[1:], rotations[:-1])) / sample_dt)
    )
    angular_speed = (
        maximum_angular_speed
        if maximum_angular_speed is not None
        else max(0.5, angular_observed * 1.05 + rotation_tolerance / dt)
    )
    rotation_gain = 1 / dt if rotation_gain is None else float(rotation_gain)
    if rotation_gain <= 0:
        raise ValueError("rotation_gain must be positive")
    rates = np.unique(np.clip([0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 20, 1 / dt], config.minimum_rate, 1 / dt))
    carried, carried_rotation = scaled[0].copy(), None if rotations is None else rotations[0].copy()
    traces, orientations, records = [position[0].copy()], ([] if rotations is None else [rotations[0].copy()]), []

    def evaluate(start, end, parent):
        nonlocal carried, carried_rotation
        selection = slice(start * integration_substeps, end * integration_substeps + 1)
        reference = dense[selection]
        starts = np.stack([reference[0], carried])
        goal = reference[-1]
        steps = (end - start) * integration_substeps
        average_speed = np.linalg.norm(reference[-1] - reference[0]) / (steps * dt)
        speeds = np.unique(
            np.clip([average_speed * 0.7, average_speed, average_speed * 1.4, average_speed * 2, speed], 1e-8, speed)
        )
        trial_rates, trial_speeds = np.meshgrid(rates, speeds, indexing="ij")
        trial_rates, trial_speeds = trial_rates.ravel(), trial_speeds.ravel()
        paths = integrate(starts[None, :, :], goal, trial_rates[:, None], steps, dt, trial_speeds[:, None])
        # rate x (local, carried) x time x coordinates
        errors = np.linalg.norm(paths - reference, axis=-1)
        losses = np.mean(errors**2, axis=(1, 2))
        maxima = np.max(errors, axis=(1, 2))
        arrival = integrate(
            paths[:, :, -1], goal, trial_rates[:, None], round(config.extra_seconds / dt), dt, trial_speeds[:, None]
        )
        dwell = max(1, int(np.ceil(config.arrival_dwell_seconds / dt)))
        arrival_distances = np.linalg.norm(arrival[:, :, -dwell:] - goal, axis=-1)
        arrival_error = arrival_distances.max(axis=(1, 2))
        arrival_speed = np.minimum(trial_rates[:, None, None] * arrival_distances, trial_speeds[:, None, None]).max(
            axis=(1, 2)
        )
        arrived = arrival_error <= config.arrival_fraction
        if config.arrival_speed_scaled_per_second is not None:
            arrived &= arrival_speed <= config.arrival_speed_scaled_per_second
        valid = (maxima <= config.fit_margin) & arrived
        rotation_report, rotation_paths = {}, None
        if rotations is not None:
            rotation_paths = [
                _integrate_rotation(r, rotations[end], steps, dt, angular_speed, rotation_gain)
                for r in (rotations[start], carried_rotation)
            ]
            angles = np.stack([angular_errors(dense_rotations[selection], r) for r in rotation_paths])
            rotation_report = {
                "rotation_rms_rad": float(np.sqrt(np.mean(angles[0] ** 2))),
                "continuous_rotation_rms_rad": float(np.sqrt(np.mean(angles[1] ** 2))),
                "maximum_rotation_error_rad": float(angles.max()),
            }
            if config.arrival_angular_speed_rad_per_second is not None:
                tail_angles = [
                    angular_errors(
                        np.broadcast_to(rotations[end], (round(config.extra_seconds / dt) + 1, 3, 3)),
                        _integrate_rotation(
                            path[-1], rotations[end], round(config.extra_seconds / dt), dt, angular_speed, rotation_gain
                        ),
                    )[-dwell:]
                    for path in rotation_paths
                ]
                tail_angle = float(np.max(tail_angles))
                tail_speed = min(angular_speed, rotation_gain * tail_angle)
                rotation_report.update(
                    arrival_rotation_error_after_extra_rad=tail_angle, arrival_angular_speed_rad_per_second=tail_speed
                )
                orientation_arrived = (
                    tail_angle <= rotation_tolerance and tail_speed <= config.arrival_angular_speed_rad_per_second
                )
                arrived &= orientation_arrived
                valid &= orientation_arrived
            valid &= angles.max() <= rotation_tolerance * config.fit_margin
        choices = np.flatnonzero(valid)
        best = int(choices[np.argmin(losses[choices])]) if len(choices) else int(np.argmin(losses))
        if not valid[best] and end - start > 1:
            middle = (start + end) // 2
            evaluate(start, middle, parent)
            evaluate(middle, end, parent)
            return
        path = paths[best]
        reasons = []
        if maxima[best] > config.maximum_position_error:
            reasons.append("position_maximum")
        if not arrived[best]:
            reasons.append("nominal_arrival_timeout")
        if rotations is not None and rotation_report["maximum_rotation_error_rad"] > rotation_tolerance:
            reasons.append("rotation_maximum")
        local_errors = (path[0] - reference) * tolerance
        continuous_errors = (path[1] - reference) * tolerance
        report = {
            "passed": not reasons,
            "reasons": reasons,
            "maximum_scaled_error": float(maxima[best]),
            "position_rms_native": float(np.sqrt(np.mean(np.sum(local_errors**2, axis=1)))),
            "continuous_position_rms_native": float(np.sqrt(np.mean(np.sum(continuous_errors**2, axis=1)))),
            "arrival_scaled_error_after_extra": float(arrival_error[best]),
            "arrival_speed_scaled_per_second": float(arrival_speed[best]),
            **rotation_report,
        }
        records.append(
            {
                "start": start,
                "end": end,
                "command": parent.get("command"),
                "command_interval_id": parent.get("command_interval_id", 0),
                "command_label": parent.get("command_label", "MOVE"),
                "rate_per_second": float(trial_rates[best]),
                "maximum_speed_scaled_per_second": float(trial_speeds[best]),
                "goal": position[end].tolist(),
                "goal_rotation": None if rotations is None else rotations[end].tolist(),
                "accepted": not reasons,
                "report": report,
            }
        )
        traces.extend(path[1, integration_substeps::integration_substeps] * tolerance)
        carried = path[1, -1].copy()
        if rotations is not None:
            orientations.extend(rotation_paths[1][integration_substeps::integration_substeps])
            carried_rotation = rotation_paths[1][-1].copy()

    for interval in intervals:
        evaluate(interval["start"], interval["end"], interval)
    trace = np.asarray(traces)
    scaled_error = np.linalg.norm((trace - position) / tolerance, axis=1)
    errors = np.linalg.norm(trace - position, axis=1)
    report = {
        "complete_coverage": records[0]["start"] == 0 and records[-1]["end"] == len(time) - 1,
        "actions_covered": len(time) - 1,
        "segments": len(records),
        "accepted_segments": sum(r["accepted"] for r in records),
        "position_rms_native": float(np.sqrt(np.mean(errors**2))),
        "maximum_position_error_native": float(errors.max()),
        "maximum_scaled_error": max(r["report"]["maximum_scaled_error"] for r in records),
        "single_transition_segments": sum(r["end"] - r["start"] == 1 for r in records),
        "minimum_stability_margin_per_second": min(r["rate_per_second"] for r in records),
        "maximum_matrix_norm_per_second": max(r["rate_per_second"] for r in records),
        "switching": "source timestamps, continuous state, no resets",
        "physical_execution_validated": False,
        "parameter_supervision": False,
    }
    if rotations is not None:
        angle = angular_errors(rotations, np.asarray(orientations))
        report.update(
            rotation_rms_rad=float(np.sqrt(np.mean(angle**2))),
            maximum_rotation_error_rad=max(r["report"]["maximum_rotation_error_rad"] for r in records),
        )
    report["passed"] = bool(
        report["complete_coverage"]
        and all(r["accepted"] for r in records)
        and scaled_error.max() <= config.maximum_position_error
    )
    limits = {
        "dt": dt,
        "sample_dt": sample_dt,
        "integration_substeps": int(integration_substeps),
        "maximum_speed_scaled_per_second": float(speed),
        "observed_speed_scaled_per_second": observed_speed,
        "maximum_angular_speed_rad_per_second": float(angular_speed),
        "observed_angular_speed_rad_per_second": angular_observed,
        "rotation_gain_per_second": rotation_gain,
        "rotation_tolerance_rad": float(rotation_tolerance),
        "extra_convergence_seconds": config.extra_seconds,
        "arrival_fraction": config.arrival_fraction,
        "arrival_speed_scaled_per_second": config.arrival_speed_scaled_per_second,
        "arrival_angular_speed_rad_per_second": config.arrival_angular_speed_rad_per_second,
        "arrival_dwell_seconds": config.arrival_dwell_seconds,
    }
    return records, report, trace, limits
