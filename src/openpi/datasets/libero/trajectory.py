"""Trajectory semantics shared by LIBERO preprocessing and downstream adapters.

All intervals use observation indices [start, end], covering actions [start, end).
Proposed boundaries are evidence to be checked by a fitter, never fitted-field labels.
"""

from __future__ import annotations

import dataclasses
import hashlib
from itertools import pairwise
import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation


@dataclasses.dataclass(frozen=True)
class ProcessingConfig:
    seed: int = 20261121
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    command_hold_seconds: float = 0.15
    feedback_hold_seconds: float = 0.15
    gripper_speed_threshold: float = 0.01  # metres / second, finger separation
    gripper_response_threshold: float = 0.002  # metres
    coupled_translation_threshold: float = 0.005  # metres
    coupled_rotation_threshold: float = math.pi / 36  # radians
    smoothing_seconds: float = 0.15
    low_speed_threshold: float = 0.02  # metres / second; candidate proposal only
    grid_seconds: float = 0.5
    minimum_motion_seconds: float = 0.5
    direction_change_radians: float = math.pi / 4
    max_candidates: int = 128
    alignment_position_tolerance: float = 0.01  # metres
    alignment_rotation_tolerance: float = math.pi / 36
    alignment_samples: int = 8
    horizons: tuple[int, ...] = (1, 3, 4, 5)

    def __post_init__(self):
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1 - self.train_fraction:
            raise ValueError("Split fractions must leave nonempty train, validation, and test ranges")
        for name in (
            "command_hold_seconds",
            "feedback_hold_seconds",
            "gripper_speed_threshold",
            "gripper_response_threshold",
            "coupled_translation_threshold",
            "coupled_rotation_threshold",
            "smoothing_seconds",
            "low_speed_threshold",
            "grid_seconds",
            "minimum_motion_seconds",
            "direction_change_radians",
            "alignment_position_tolerance",
            "alignment_rotation_tolerance",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_candidates < 1 or self.alignment_samples < 2:
            raise ValueError("Candidate budget must be positive; alignment requires at least two samples")
        if not self.horizons or any(type(h) is not int or h < 1 for h in self.horizons):
            raise ValueError("Horizons must be positive integers")


def episode_split(signature: str, config: ProcessingConfig) -> str:
    """Content-based assignment keeps exact duplicated trajectories in the same split."""
    digest = hashlib.sha256(f"{config.seed}:{signature}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < config.train_fraction:
        return "train"
    return "validation" if value < config.train_fraction + config.validation_fraction else "test"


def continuous_quaternion(rotvec: np.ndarray) -> np.ndarray:
    quat = Rotation.from_rotvec(rotvec).as_quat()  # xyzw
    if quat[0, 3] < 0:
        quat[0] *= -1
    for i in range(1, len(quat)):
        if quat[i - 1] @ quat[i] < 0:
            quat[i] *= -1
    return quat


def measured_derivatives(position, quaternion, time):
    """Differentiate measured states, including SO(3), rather than command vectors."""
    time = np.asarray(time, dtype=np.float64)
    if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Observation timestamps must be finite and strictly increasing")
    linear = np.gradient(position, time, axis=0, edge_order=1)
    rotations = Rotation.from_quat(quaternion)
    # Relative rotations R(t+dt) R(t)^-1 express angular velocity in the world frame.
    increments = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / np.diff(time)[:, None]
    angular = np.empty((len(time), 3), dtype=np.float64)
    angular[0], angular[-1] = increments[0], increments[-1]
    if len(time) > 2:
        dt = np.diff(time)
        angular[1:-1] = (increments[:-1] * dt[1:, None] + increments[1:] * dt[:-1, None]) / (
            dt[:-1, None] + dt[1:, None]
        )
    return linear, angular


def _held_end(time, start, duration, stop):
    # time[stop] is available (T actions have T+1 observations).
    end = int(np.searchsorted(time, time[start] + duration - 1e-9, side="left"))
    return end if end <= stop else None


def extract_events(actions, time, position, quaternion, gripper, config):
    """Preserve every binary command change; persistence and settling are separate evidence."""
    raw = np.asarray(actions)[:, -1]
    if not len(raw) or not np.isfinite(raw).all() or not np.all(np.isclose(np.abs(raw), 1, atol=1e-6, rtol=0)):
        raise ValueError("This binary LIBERO profile requires gripper commands -1 (OPEN) or +1 (CLOSE)")
    commands = np.where(raw > 0, 1, -1).astype(np.int8)
    changes = np.flatnonzero(np.diff(commands)) + 1
    width = np.abs(gripper[:, 0] - gripper[:, 1])
    speed = np.abs(np.gradient(width, time))
    rotations = Rotation.from_quat(quaternion)
    events = []
    for index, change in enumerate(changes):
        onset = int(change)
        command = int(commands[onset])
        limit = int(changes[index + 1]) if index + 1 < len(changes) else len(raw)
        confirmed = _held_end(time, onset, config.command_hold_seconds, limit)
        settled = None
        response = False
        for point in range(onset + 1, limit + 1):
            directional_change = (width[point] - width[onset]) * (-command)
            response = bool(response or directional_change >= config.gripper_response_threshold)
            end = _held_end(time, point, config.feedback_hold_seconds, limit)
            if response and end is not None and np.all(speed[point : end + 1] <= config.gripper_speed_threshold):
                settled = end
                break
        end = settled if settled is not None else limit
        translation = float(np.max(np.linalg.norm(position[onset : end + 1] - position[onset], axis=1)))
        rotation = float(np.max((rotations[onset : end + 1] * rotations[onset].inv()).magnitude()))
        events.append(
            {
                "event": "CLOSE" if command == 1 else "OPEN",
                "command": command,
                "onset": onset,
                "command_confirmed": confirmed,
                "command_persistent": confirmed is not None,
                "feedback_settled": settled,
                "response_observed": response,
                "feedback_check_end": end,
                "translation_during_operation_m": translation,
                "rotation_during_operation_rad": rotation,
                "coupled_motion": translation > config.coupled_translation_threshold
                or rotation > config.coupled_rotation_threshold,
                "grasp_success": None,
                "operation_success": None,
            }
        )
    return commands, events


def command_intervals(commands, events):
    """Partition all actions at command onset, independently of feedback settling."""
    boundaries = [0, *(int(i) for i in np.flatnonzero(np.diff(commands)) + 1), len(commands)]
    return [
        {
            "command_interval_id": index,
            "start": start,
            "end": end,
            "command": int(commands[start]),
            "command_label": "CLOSE" if commands[start] == 1 else "OPEN",
            "start_event": index - 1,
            "end_event": index if index < len(events) else -1,
        }
        for index, (start, end) in enumerate(pairwise(boundaries))
    ]


def propose_phases(events, action_count):
    """Keep every action, including operation waits and unsupported coupled motion."""
    phases = []
    cursor = 0
    for event_index, event in enumerate(events):
        onset = event["onset"]
        if onset > cursor:
            phases.append({"start": cursor, "end": onset, "kind": "motion", "event_after": event_index})
        end = max(onset + 1, event["feedback_check_end"])
        end = min(end, action_count)
        phases.append(
            {
                "start": onset,
                "end": end,
                "kind": "coupled_operation" if event["coupled_motion"] else "operation",
                "event_index": event_index,
                "event_after": -1,
                "feedback_verified": event["feedback_settled"] is not None,
            }
        )
        cursor = end
    if cursor < action_count:
        phases.append({"start": cursor, "end": action_count, "kind": "motion", "event_after": -1})
    return phases


def propose_candidates(intervals, time, linear_velocity, config):
    """Stops, direction changes, and a grid suggest endpoints; no fitting is implied."""
    dt = float(np.median(np.diff(time)))
    window = max(1, round(config.smoothing_seconds / dt))
    velocity = uniform_filter1d(linear_velocity, size=window, axis=0, mode="nearest")
    speed = np.linalg.norm(velocity, axis=1)
    proposals = []
    mandatory = []
    for interval in intervals:
        start, end = interval["start"], interval["end"]
        endpoints = {start: "command_start", end: "command_end"}
        for timestamp in np.arange(time[start] + config.grid_seconds, time[end], config.grid_seconds):
            point = int(np.searchsorted(time, timestamp))
            if start < point < end:
                endpoints.setdefault(point, "grid")
        low = speed[start : end + 1] <= config.low_speed_threshold
        edges = np.diff(np.r_[False, low, False].astype(np.int8))
        for low_start, low_end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            a, b = start + int(low_start), start + int(low_end) - 1
            if time[b] - time[a] >= config.feedback_hold_seconds - 1e-9:
                endpoints.setdefault((a + b) // 2, "low_speed")
        stride = max(1, round(config.grid_seconds / dt))
        for point in range(start + stride, end - stride, stride):
            before, after = velocity[point - stride], velocity[point + stride]
            scale = np.linalg.norm(before) * np.linalg.norm(after)
            if scale > config.low_speed_threshold**2:
                angle = math.acos(float(np.clip(before @ after / scale, -1, 1)))
                if angle >= config.direction_change_radians:
                    endpoints.setdefault(point, "direction_change")
        points = sorted(endpoints)
        for a_index, a in enumerate(points[:-1]):
            for b in points[a_index + 1 :]:
                full = a == start and b == end
                if not full and time[b] - time[a] < config.minimum_motion_seconds - 1e-9:
                    continue
                item = {
                    "start": a,
                    "end": b,
                    "command_interval_id": interval["command_interval_id"],
                    "command": interval["command"],
                    "start_reason": endpoints[a],
                    "end_reason": endpoints[b],
                    "terminal_low_speed": bool(speed[b] <= config.low_speed_threshold),
                    "fit_status": "not_run",
                    "nominal_status": "not_run",
                    "recovery_status": "not_run",
                    "handoff_status": "not_run",
                }
                (mandatory if full else proposals).append(item)
    # Every command interval, including a short pulse or stationary wait, has a full-length candidate.
    proposals.sort(
        key=lambda p: (
            p["end_reason"] == "grid",
            -(time[p["end"]] - time[p["start"]]),
            p["command_interval_id"],
            p["start"],
        )
    )
    if len(mandatory) > config.max_candidates:
        raise ValueError("Candidate budget is smaller than the number of complete command intervals")
    selected = mandatory + proposals[: config.max_candidates - len(mandatory)]
    selected.sort(key=lambda p: (p["command_interval_id"], p["start"], p["end"]))
    return selected, len(mandatory) + len(proposals) - len(selected)


def proposal_windows(intervals, position, quaternion, horizon):
    """Diagnostic command-interval windows; endpoints are not fitted-field labels."""
    count = len(intervals)
    goals = np.zeros((count, horizon, 7), dtype=np.float64)
    commands = np.zeros((count, horizon), dtype=np.int8)  # OPEN=-1, CLOSE=+1, padding=0
    mask = np.zeros((count, horizon), dtype=bool)
    interval_refs = np.full((count, horizon), -1, dtype=np.int32)
    for row in range(count):
        for column, interval in enumerate(intervals[row : row + horizon]):
            end = interval["end"]
            goals[row, column] = np.r_[position[end], quaternion[end]]
            commands[row, column] = interval["command"]
            interval_refs[row, column] = interval["command_interval_id"]
            mask[row, column] = True
    return {
        "query_observation": np.array([p["start"] for p in intervals], dtype=np.int32),
        "command_interval_refs": interval_refs,
        "goal_candidates": goals,
        "gripper_commands": commands,
        "proposal_mask": mask,
        "valid_length": mask.sum(axis=1).astype(np.int32),
        "source_end_in_window": np.arange(count) + horizon >= count,
        "field_supervision_mask": np.zeros(count, dtype=bool),
        "goal_supervision_mask": np.zeros_like(mask),
        "terminal_label_mask": np.zeros(count, dtype=bool),
    }


class RunningMoments:
    """Mergeable population moments; callers supply training episodes only."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None
        self.minimum = None
        self.maximum = None

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Moments require nonempty finite matrices")
        count = len(values)
        mean = values.mean(axis=0)
        m2 = np.sum((values - mean) ** 2, axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
            self.minimum, self.maximum = values.min(axis=0), values.max(axis=0)
            return
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta**2 * (self.count * count / total)
        self.mean += delta * count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def result(self):
        if not self.count:
            return None
        std = np.sqrt(np.maximum(self.m2 / self.count, 0))
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "scale": np.maximum(std, 1e-6).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "transform": "(x - mean) / scale",
            "inverse": "x_normalized * scale + mean",
        }
