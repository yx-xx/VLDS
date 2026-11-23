import dataclasses
import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.fields import primitive
from openpi.fields import seds
from openpi.fields import sequence


def test_closed_form_matches_actual_feedback_with_speed_cap():
    dt, rate, speed = 0.05, 8.0, 0.12
    start, goal = np.array([0.17, 0.02, -0.04]), np.zeros(3)
    actual = [start.copy()]
    field = primitive.Primitive(seds.PositionField(seds.linear_factors(rate), goal), np.eye(3), 1)
    control = dataclasses.replace(primitive.DEFAULT_CONTROL, max_speed=speed)
    for _ in range(60):
        velocity, _, _ = primitive.feedback(field, actual[-1], np.eye(3), control)
        actual.append(actual[-1] + dt * velocity)
    np.testing.assert_allclose(sequence.integrate(start, goal, rate, 60, dt, speed), actual, atol=1e-14)


def test_partition_covers_reversal_and_checks_continuous_state():
    position = np.column_stack([np.r_[np.linspace(0, 0.1, 21), np.linspace(0.1, 0, 21)[1:]], np.zeros((41, 2))])
    time = np.arange(len(position)) * 0.05
    intervals = [{"start": 0, "end": 20}, {"start": 20, "end": 40}]
    records, report, trace, _ = sequence.fit_sequence(position, time, intervals, np.full(3, 0.01))
    assert report["passed"]
    assert len(records) > 1
    assert records[0]["start"] == 0
    assert records[-1]["end"] == 40
    assert all(a["end"] == b["start"] for a, b in itertools.pairwise(records))
    assert all(not (r["start"] < 20 < r["end"]) for r in records)
    assert np.linalg.norm(trace - position, axis=1).max() <= 0.01
    # Reconstruct independently, propagating state across every boundary.
    current = position[0].copy()
    for r in records:
        rollout = (
            sequence.integrate(
                current / 0.01,
                np.asarray(r["goal"]) / 0.01,
                r["rate_per_second"],
                r["end"] - r["start"],
                0.05,
                r["maximum_speed_scaled_per_second"],
            )
            * 0.01
        )
        np.testing.assert_allclose(rollout, trace[r["start"] : r["end"] + 1], atol=1e-14)
        current = rollout[-1]


def test_impossible_speed_limit_retains_rejected_segment():
    position = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    records, report, _, _ = sequence.fit_sequence(
        position, [0, 0.05], [{"start": 0, "end": 1}], np.full(3, 0.01), maximum_speed=1.0
    )
    assert not report["passed"]
    assert not records[0]["accepted"]
    assert "position_maximum" in records[0]["report"]["reasons"]


def test_rotation_uses_so3_across_pi_branch():
    rotations = Rotation.from_euler("z", [179, -179, -177], degrees=True).as_matrix()
    p = np.zeros((3, 3))
    _, report, _, _ = sequence.fit_sequence(
        p, np.arange(3) * 0.05, [{"start": 0, "end": 2}], np.full(3, 0.01), rotations=rotations
    )
    assert report["passed"]
    assert report["maximum_rotation_error_rad"] < np.deg2rad(5)


def test_rotation_closed_form_matches_bounded_low_gain_feedback():
    initial = Rotation.from_euler("xyz", [0.3, -0.4, 2.9]).as_matrix()
    goal = Rotation.from_euler("xyz", [-0.2, 0.5, -2.9]).as_matrix()
    current, dt = initial.copy(), 0.05
    actual = [current.copy()]
    for _ in range(80):
        omega = 2 * Rotation.from_matrix(goal @ current.T).as_rotvec()
        omega *= min(1, 0.5 / max(np.linalg.norm(omega), 1e-30))
        current = Rotation.from_rotvec(dt * omega).as_matrix() @ current
        actual.append(current.copy())
    expected = sequence.integrate_rotation(initial, goal, 80, dt, 0.5, gain=2)
    np.testing.assert_allclose(expected, actual, atol=1e-14)
