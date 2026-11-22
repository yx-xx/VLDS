"""Pose feedback, MOVE/HOLD dwell, and nominal field acceptance (no dynamics)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.fields import seds


@dataclasses.dataclass(frozen=True)
class ControlConfig:
    dt: float = 0.05
    reference_seconds: float = 0.05
    max_speed: float = 0.1
    max_angular_speed: float = 0.5
    rotation_gain: float = 2.0
    position_tolerance: float = 0.005
    rotation_tolerance: float = math.pi / 36
    speed_tolerance: float = 0.02
    angular_speed_tolerance: float = 0.1
    arrival_seconds: float = 0.1
    extra_convergence_seconds: float = 2.0
    position_rms_tolerance: float = 0.01
    rotation_rms_tolerance: float = math.pi / 36
    hold_position_range: float = 0.002
    hold_rotation_range: float = math.pi / 90

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in dataclasses.asdict(self).values()):
            raise ValueError("Control configuration must be finite and strictly positive")


DEFAULT_CONTROL = ControlConfig()


def rotation_matrix(value):
    matrix = seds.finite_array(value, (3, 3))
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6):
        raise ValueError("Invalid control-point rotation")
    return matrix


def rotation_6d(matrix):
    return rotation_matrix(matrix)[:, :2].T.ravel()


def from_rotation_6d(value):
    columns = seds.finite_array(value, (6,)).reshape(2, 3)
    norm = np.linalg.norm(columns[0])
    if norm < 1e-6:
        raise ValueError("Degenerate first 6D rotation column")
    first = columns[0] / norm
    second = columns[1] - np.dot(columns[1], first) * first
    norm = np.linalg.norm(second)
    if norm < 1e-6:
        raise ValueError("Degenerate second 6D rotation column")
    second /= norm
    return np.column_stack((first, second, np.cross(first, second)))


@dataclasses.dataclass
class Primitive:
    field: seds.PositionField
    rotation: np.ndarray
    command: int
    kind: str = "MOVE"
    hold_seconds: float = 0.0

    def __post_init__(self):
        self.rotation = rotation_matrix(self.rotation).copy()
        if self.command not in (-1, 1) or self.kind not in ("MOVE", "HOLD"):
            raise ValueError("Expected binary gripper command and MOVE/HOLD")
        if not np.isfinite(self.hold_seconds) or self.hold_seconds < 0:
            raise ValueError("Invalid HOLD duration")
        if (self.kind == "MOVE" and self.hold_seconds != 0) or (self.kind == "HOLD" and self.hold_seconds <= 0):
            raise ValueError("Only HOLD has a positive dwell duration")

    def record(self):
        return {
            **self.field.record(),
            "goal_rotation_6d": rotation_6d(self.rotation).tolist(),
            "goal_quaternion_xyzw": Rotation.from_matrix(self.rotation).as_quat().tolist(),
            "command": self.command,
            "primitive_kind": self.kind,
            "hold_seconds": self.hold_seconds,
            "frame": "world",
            "control_point": "gripper0_grip_site",
        }

    @classmethod
    def from_record(cls, record):
        return cls(
            seds.PositionField.from_record(record),
            from_rotation_6d(record["goal_rotation_6d"]),
            record["command"],
            record["primitive_kind"],
            record["hold_seconds"],
        )


def feedback(primitive, position, rotation, config=DEFAULT_CONTROL):
    """A single positive scalar caps speed and bounds nominal Euler Delta V."""
    position = seds.finite_array(position, (3,))
    rotation = rotation_matrix(rotation)
    velocity = primitive.field.velocity(position)
    error = position - primitive.field.goal
    squared_speed = float(velocity @ velocity)
    alpha = 1.0
    if squared_speed > 0:
        radial = float(error @ velocity)
        if radial >= 0:
            raise ValueError("Nonzero field fails numerical descent check")
        alpha = min(
            1.0,
            config.max_speed / np.sqrt(squared_speed),
            -radial / (max(config.dt, config.reference_seconds) * squared_speed),
        )
        velocity = alpha * velocity
    angular_error = Rotation.from_matrix(primitive.rotation @ rotation.T).as_rotvec()
    omega = config.rotation_gain * angular_error
    magnitude = np.linalg.norm(omega)
    if magnitude > config.max_angular_speed:
        omega *= config.max_angular_speed / magnitude
    return velocity, omega, float(alpha)


class Arrival:
    """HOLD pauses its dwell outside tolerance; MOVE requires consecutive arrival."""

    def __init__(self):
        self.consecutive_seconds = 0.0
        self.hold_seconds = 0.0

    def update(self, primitive, position, rotation, velocity, omega, config=DEFAULT_CONTROL):
        position = seds.finite_array(position, (3,))
        rotation = rotation_matrix(rotation)
        velocity, omega = seds.finite_array(velocity, (3,)), seds.finite_array(omega, (3,))
        inside = (
            np.linalg.norm(position - primitive.field.goal) <= config.position_tolerance
            and Rotation.from_matrix(primitive.rotation @ rotation.T).magnitude() <= config.rotation_tolerance
            and np.linalg.norm(velocity) <= config.speed_tolerance
            and np.linalg.norm(omega) <= config.angular_speed_tolerance
        )
        self.consecutive_seconds = self.consecutive_seconds + config.dt if inside else 0.0
        if inside and primitive.kind == "HOLD":
            self.hold_seconds += config.dt
        ready = self.consecutive_seconds >= config.arrival_seconds - 1e-9
        if primitive.kind == "HOLD":
            ready = ready and self.hold_seconds >= primitive.hold_seconds - 1e-9
        return bool(ready)


def nominal_replay(primitive, position, rotations, time, config=DEFAULT_CONTROL, *, keep_trace=False):
    position = seds.finite_array(position)
    time = seds.finite_array(time)
    rotations = seds.finite_array(rotations)
    if position.shape != (len(time), 3) or rotations.shape != (len(time), 3, 3):
        raise ValueError("Nominal replay requires aligned positions and control-point rotations")
    if len(time) < 2 or not np.allclose(np.diff(time), config.dt, atol=1e-6, rtol=1e-5):
        raise ValueError("design-v1 nominal replay requires 20 Hz source timestamps")
    duration = float(time[-1] - time[0])
    steps = round((duration + config.extra_convergence_seconds) / config.dt)
    p, r = position[0].copy(), rotation_matrix(rotations[0]).copy()
    positions, orientation = [p.copy()], [r.copy()]
    arrival, arrival_time = Arrival(), None
    limited, minimum_alpha = 0, 1.0
    for step in range(steps):
        v, omega, alpha = feedback(primitive, p, r, config)
        p = p + config.dt * v
        r = Rotation.from_rotvec(config.dt * omega).as_matrix() @ r
        positions.append(p.copy())
        orientation.append(r.copy())
        limited += int(alpha < 1 - 1e-9)
        minimum_alpha = min(minimum_alpha, alpha)
        # Nominal post-step velocity is evaluated at the post-step state.
        next_v, next_omega, _ = feedback(primitive, p, r, config)
        if arrival.update(primitive, p, r, next_v, next_omega, config) and arrival_time is None:
            arrival_time = (step + 1) * config.dt
    positions, orientation = np.array(positions), np.array(orientation)
    weights = seds.time_weights(time)
    errors = positions[: len(time)] - position
    rotation_errors = Rotation.from_matrix(rotations @ orientation[: len(time)].transpose(0, 2, 1)).magnitude()
    rms_p = float(np.sqrt(np.average(np.sum(errors**2, axis=1), weights=weights)))
    rms_r = float(np.sqrt(np.average(rotation_errors**2, weights=weights)))
    reasons = []
    if rms_p > config.position_rms_tolerance:
        reasons.append("position_rms")
    if rms_r > config.rotation_rms_tolerance:
        reasons.append("rotation_rms")
    if arrival_time is None:
        reasons.append("nominal_arrival_timeout")
    report = {
        "passed": not reasons,
        "reasons": reasons,
        "position_rms_m": rms_p,
        "rotation_rms_rad": rms_r,
        "source_seconds": duration,
        "arrival_seconds": arrival_time,
        "extra_seconds": max(0.0, arrival_time - duration) if arrival_time is not None else None,
        "limited_steps": limited,
        "minimum_alpha": minimum_alpha,
        "scope": "time-aligned nominal pose integration only; no robot dynamics or operation facts",
    }
    if keep_trace:
        report["trace"] = {"position": positions, "rotation": orientation}
    return report


def fit_primitive(
    position,
    velocity,
    rotations,
    time,
    command,
    field_config=seds.DEFAULT_CONFIG,
    control=DEFAULT_CONTROL,
    *,
    representation="fitted",
):
    if representation not in {"fitted", "linear"}:
        raise ValueError("Unknown teacher representation")
    position, rotations = seds.finite_array(position), seds.finite_array(rotations)
    velocity, time = seds.finite_array(velocity), seds.finite_array(time)
    seds.time_weights(time)
    if position.shape != (len(time), 3) or velocity.shape != position.shape or rotations.shape != (len(time), 3, 3):
        raise ValueError("Primitive fitting requires aligned physical pose, velocity and time samples")
    goal_rotation = rotation_matrix(rotations[-1])
    # All-pairs range prevents a midpoint excursion from being mistaken for HOLD.
    position_range = float(np.linalg.norm(position[:, None] - position[None, :], axis=-1).max())
    rotation_range = max(float(Rotation.from_matrix(rotations @ r.T).magnitude().max()) for r in rotations)
    if position_range <= control.hold_position_range and rotation_range <= control.hold_rotation_range:
        field = seds.PositionField(seds.linear_factors(1.0, field_config), position[-1], field_config)
        duration = round((time[-1] - time[0]) / control.dt) * control.dt
        primitive = Primitive(field, goal_rotation, command, "HOLD", duration)
        fit_report = {"kind": "analytic_hold", "optimizer_converged": None, "parameter_supervision": False}
    else:
        fit = seds.fit if representation == "fitted" else seds.fit_linear
        field, fit_report = fit(position, velocity, time, field_config)
        if field is None:
            return None, {"fit": fit_report, "accepted": False, "reasons": ["optimizer_or_constraints"]}
        primitive = Primitive(field, goal_rotation, command)
    replay = nominal_replay(primitive, position, rotations, time, control)
    return primitive, {"fit": fit_report, "nominal": replay, "accepted": replay["passed"], "reasons": replay["reasons"]}


class CommandFeedback:
    """Proprioceptive settling only; this does not infer held/placed facts."""

    def __init__(self, finger_position, *, speed_tolerance=0.01, settle_seconds=0.15):
        position = seds.finite_array(finger_position, (2,))
        self.previous_separation = abs(position[0] - position[1])
        self.still_seconds = 0.0
        self.speed_tolerance, self.settle_seconds = speed_tolerance, settle_seconds

    def update(self, finger_position, dt):
        position = seds.finite_array(finger_position, (2,))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Feedback dt must be positive")
        separation = abs(position[0] - position[1])
        speed = abs(separation - self.previous_separation) / dt
        self.previous_separation = separation
        self.still_seconds = self.still_seconds + dt if speed <= self.speed_tolerance else 0.0
        return self.still_seconds >= self.settle_seconds - 1e-9
