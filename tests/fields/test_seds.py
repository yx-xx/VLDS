import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.fields import primitive
from openpi.fields import seds


def test_stable_field_global_radial_descent_and_serialization():
    rng = np.random.default_rng(42)
    field = seds.PositionField(rng.normal(size=(4, 19)), np.array([0.2, -0.1, 0.5]))
    positions = rng.normal(size=(200, 3)) * 0.2 + field.goal
    velocities = field.velocity(positions)
    errors = positions - field.goal
    assert np.all(np.sum(errors * velocities, axis=1) <= -0.01 * np.sum(errors**2, axis=1) + 1e-12)
    np.testing.assert_array_equal(field.velocity(field.goal), np.zeros(3))
    copy = seds.PositionField.from_record(field.record())
    np.testing.assert_array_equal(copy.velocity(positions), velocities)
    pose = primitive.Primitive(field, Rotation.from_rotvec([0.3, -0.2, 1]).as_matrix(), 1)
    restored = primitive.Primitive.from_record(pose.record())
    np.testing.assert_allclose(restored.rotation, pose.rotation, atol=1e-15)


def test_untrusted_network_factors_are_rejected():
    raw = seds.linear_factors(1)
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        seds.PositionField(raw, np.zeros(3))
    raw = seds.linear_factors(1)
    raw[0, 13] = 30
    with pytest.raises(ValueError, match="limit"):
        seds.PositionField(raw, np.zeros(3))
    raw = seds.linear_factors(1)
    raw[0, 4] = 1000
    with pytest.raises(ValueError, match="limit"):
        seds.PositionField(raw, np.zeros(3))


def test_autograd_matches_numerical_velocity_objective_gradient():
    import torch

    rng = np.random.default_rng(12)
    y, velocity = rng.normal(size=(2, 16, 3))
    raw = rng.normal(scale=0.3, size=(4, 19)).ravel()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        objective = seds.velocity_objective(y, velocity, np.ones(len(y)), seds.DEFAULT_CONFIG)
        loss, gradient = objective(raw)
        assert np.isfinite(loss)
        for index in (0, 1, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 56, 75):
            delta = np.zeros_like(raw)
            delta[index] = 1e-5
            difference = (objective(raw + delta)[0] - objective(raw - delta)[0]) / 2e-5
            assert gradient[index] == pytest.approx(difference, abs=1e-6, rel=1e-5)
    finally:
        torch.set_num_threads(threads)


def test_fitter_recovers_measured_linear_flow_without_faking_zero_terminal_speed():
    time = np.arange(81) * 0.05
    position = np.array([0.03, 0.01, -0.02])[None, :] * np.exp(-2 * time[:, None])
    velocity = -2 * position
    field, report = seds.fit(position, velocity, time)
    assert field is not None
    assert report["optimizer_converged"]
    assert len(report["restarts"]) == 3
    assert report["velocity_mse"] < 1e-8
    np.testing.assert_array_equal(field.goal, position[-1])
    assert np.linalg.norm(velocity[-1]) > 0
    replay = primitive.nominal_replay(
        primitive.Primitive(field, np.eye(3), -1), position, np.tile(np.eye(3), (len(time), 1, 1)), time
    )
    assert replay["passed"], replay


def test_short_motion_uses_behavior_only_linear_fit():
    time = np.arange(6) * 0.05
    position = np.array([0.02, 0, 0])[None, :] * np.exp(-time[:, None])
    field, report = seds.fit(position, -position, time)
    assert field is not None
    assert report["kind"] == "short_linear"
    assert not report["parameter_supervision"]
    assert not report["restarts"]


def test_hold_dwell_pauses_and_pose_motion_does_not_disappear():
    field = seds.PositionField(seds.linear_factors(1), np.zeros(3))
    hold = primitive.Primitive(field, np.eye(3), 1, "HOLD", 0.2)
    arrival = primitive.Arrival()
    results = [
        arrival.update(hold, np.zeros(3) if inside else np.ones(3), np.eye(3), np.zeros(3), np.zeros(3))
        for inside in (True, True, False, False, True, True)
    ]
    assert results == [False, False, False, False, False, True]
    time = np.arange(5) * 0.05
    p = np.zeros((5, 3))
    r = np.tile(np.eye(3), (5, 1, 1))
    fitted, report = primitive.fit_primitive(p, p, r, time, 1)
    assert fitted.kind == "HOLD"
    assert fitted.hold_seconds == pytest.approx(0.2)
    assert report["accepted"]
    assert report["fit"]["kind"] == "analytic_hold"
    # A rotation excursion with identical endpoints must not be classified as waiting.
    r[2] = Rotation.from_rotvec([0.5, 0, 0]).as_matrix()
    fitted, report = primitive.fit_primitive(p, p, r, time, 1)
    assert fitted.kind == "MOVE"
    assert not report["accepted"]
    assert "rotation_rms" in report["reasons"]


def test_nominal_reference_step_decreases_distance_with_strong_skew_and_rotation():
    raw = seds.linear_factors(0.1)
    raw[:, 10:13] = [10, -5, 2]
    field = seds.PositionField(raw, np.zeros(3))
    pose = primitive.Primitive(field, Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix(), -1)
    p = np.array([0.1, -0.1, 0.1])
    v, omega, alpha = primitive.feedback(pose, p, np.eye(3))
    assert 0 < alpha < 1
    assert np.linalg.norm(p + 0.05 * v) <= np.linalg.norm(p)
    assert np.linalg.norm(v) <= 0.1
    assert np.linalg.norm(omega) <= 0.5 + 1e-12


def test_invalid_rotation_and_source_clock_rejected():
    with pytest.raises(ValueError, match="Degenerate"):
        primitive.from_rotation_6d(np.zeros(6))
    with pytest.raises(ValueError, match="Degenerate"):
        primitive.from_rotation_6d([1, 0, 0, 2, 0, 0])
    field = seds.PositionField(seds.linear_factors(1), np.zeros(3))
    with pytest.raises(ValueError, match="rotation"):
        primitive.Primitive(field, np.eye(3) * 2, 1)
    with pytest.raises(ValueError, match="20 Hz"):
        primitive.nominal_replay(
            primitive.Primitive(field, np.eye(3), -1), np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)), [0, 0.1]
        )
    with pytest.raises(ValueError, match="increasing"):
        seds.fit(np.zeros((2, 3)), np.zeros((2, 3)), [0, 0])
    with pytest.raises(ValueError, match="configuration"):
        dataclasses.replace(seds.DEFAULT_CONFIG, stability_margin=0)


def test_corrupted_physical_archive_and_missing_hold_velocity_are_rejected():
    field = seds.PositionField(seds.linear_factors(1), np.zeros(3))
    record = field.record()
    record["field_physical"]["matrices"][0][0][0] = 9
    with pytest.raises(ValueError, match="disagree"):
        seds.PositionField.from_record(record)
    with pytest.raises(ValueError, match="finite"):
        primitive.fit_primitive(
            np.zeros((2, 3)), np.ones((2, 3)) * np.nan, np.tile(np.eye(3), (2, 1, 1)), [0, 0.05], -1
        )


def test_command_feedback_settling_restarts_when_fingers_move():
    feedback = primitive.CommandFeedback([0.04, -0.04])
    values = [[0.04, -0.04], [0.04, -0.04], [0.03, -0.03], [0.03, -0.03], [0.03, -0.03], [0.03, -0.03]]
    assert [feedback.update(value, 0.05) for value in values] == [False, False, False, False, False, True]


@pytest.mark.parametrize("reference_seconds", [0.001, 0.05, 0.16329931618554522])
def test_both_reference_and_integrated_step_descend_for_any_positive_offset_time(reference_seconds):
    config = dataclasses.replace(primitive.DEFAULT_CONTROL, reference_seconds=reference_seconds)
    raw = seds.linear_factors(0.02)
    raw[:, 10:13] = [5, -4, 3]
    target = primitive.Primitive(seds.PositionField(raw, np.zeros(3)), np.eye(3), -1)
    position = np.array([0.0001, 0.0002, -0.0002])
    velocity, _, alpha = primitive.feedback(target, position, np.eye(3), config)
    assert 0 < alpha <= 1
    for step in [config.dt, config.reference_seconds]:
        assert np.linalg.norm(position + step * velocity) <= np.linalg.norm(position) + 1e-14
