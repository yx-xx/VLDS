import dataclasses

import numpy as np
import pytest

from openpi.fields import contractive


def record():
    return {
        "version": contractive.VERSION,
        "goal": [0, 0],
        "tolerance": [0.1, 0.3],
        "transform": [[2.0, 1.0], [0.0, 0.6]],
        "means": [[-1, 0], [1, 0]],
        "variance": [1, 2],
        "matrices": [[[0.8, -0.3], [0.3, 0.8]], [[0.6, 0.1], [-0.1, 0.6]]],
        "dt": 0.05,
        "maximum_speed_scaled_per_second": 0.2,
    }


def test_common_metric_descends_under_gaussian_gating_and_active_speed_cap():
    field = contractive.ContractiveField(record())
    x = np.random.default_rng(7).normal(size=(1000, 2))
    next_x = field.step(x)
    before = ((x - field.goal) / field.tolerance) @ field.transform.T
    after = ((next_x - field.goal) / field.tolerance) @ field.transform.T
    assert np.all(np.sum(after**2, axis=1) < np.sum(before**2, axis=1))
    assert np.linalg.norm((next_x - x) / field.tolerance, axis=1).max() <= field.speed * field.dt + 1e-12
    np.testing.assert_array_equal(field.step(field.goal), field.goal)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("matrices", [np.eye(2).tolist()]),
        ("transform", [[1, 0], [0, 0]]),
        ("dt", float("nan")),
        ("tolerance", [0.1, float("nan")]),
        ("variance", [-1, 1]),
    ],
)
def test_rejects_invalid_certificate_and_nonfinite_parameters(key, value):
    data = record()
    data[key] = value
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        contractive.ContractiveField(data)


def test_fits_curved_stable_demonstration_in_closed_loop():
    transition = np.array([[0.94, -0.12], [0.04, 0.96]])
    reference = [np.array([0.8, -0.2])]
    for _ in range(65):
        reference.append(transition @ reference[-1])
    reference = np.asarray(reference)
    field, trace, error = contractive.fit(
        reference,
        np.full(2, 0.05),
        0.05,
        100,
        config=dataclasses.replace(contractive.DEFAULT_CONFIG, rollout_iterations=80),
    )
    assert error <= 1
    assert trace.shape == (66, 2, 2)
    assert field.radius < 1
    # Curved velocities differ from a scalar attraction toward the final point.
    v = field.velocity(reference[0])
    towards = field.goal - reference[0]
    assert abs(v[0] * towards[1] - v[1] * towards[0]) > 1e-3


@pytest.mark.parametrize(("time", "speed"), [(0, 1), (0.1, float("nan")), (-1, 1)])
def test_invalid_fit_limits_are_rejected(time, speed):
    with pytest.raises(ValueError, match="Invalid reference"):
        contractive.fit(np.array([[1.0, 0], [0, 0]]), np.ones(2), time, speed)


def test_forward_goal_is_the_actual_equilibrium_and_not_a_display_offset():
    reference = np.c_[np.linspace(0, 0.1, 21), np.zeros(21)]
    goal = np.array([0.13, 0.0])
    field, trace, error = contractive.fit(reference, np.full(2, 0.01), 0.05, 100, goal=goal, refine=False)
    endpoint, old_trace, _ = contractive.fit(reference, np.full(2, 0.01), 0.05, 100, refine=False)
    np.testing.assert_array_equal(field.goal, goal)
    np.testing.assert_array_equal(field.step(goal), goal)
    np.testing.assert_allclose(trace[:, 0], field.rollout(reference[0], 20), atol=1e-14)
    assert error < 1
    assert field.radius < 1
    assert np.linalg.norm(trace[-1, 0] - trace[-2, 0]) > np.linalg.norm(old_trace[-1, 0] - old_trace[-2, 0])
    np.testing.assert_array_equal(endpoint.goal, reference[-1])


@pytest.mark.parametrize("goal", [[0.0], [float("nan"), 0.0]])
def test_invalid_forward_goal_is_rejected(goal):
    with pytest.raises(ValueError, match="goal"):
        contractive.fit(np.array([[0.0, 0.0], [0.1, 0.0]]), np.ones(2), 0.05, 100, goal=goal)
