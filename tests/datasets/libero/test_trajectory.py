import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.datasets.libero import trajectory


def test_measured_so3_derivative_crosses_axis_angle_wrap():
    time = np.linspace(0, 1, 21)
    position = np.c_[time, time * 2, time * 0]
    rotation = np.c_[time * 0, time * 0, np.pi - 0.1 + time * 0.2]
    quaternion = trajectory.continuous_quaternion(Rotation.from_rotvec(rotation).as_rotvec())
    linear, angular = trajectory.measured_derivatives(position, quaternion, time)
    np.testing.assert_allclose(linear, np.tile([1, 2, 0], (21, 1)), atol=1e-10)
    np.testing.assert_allclose(angular, np.tile([0, 0, 0.2], (21, 1)), atol=1e-10)
    assert np.all(np.sum(quaternion[1:] * quaternion[:-1], axis=1) > 0)


def test_command_intervals_retain_short_commands_coupled_motion_and_full_coverage():
    time = np.arange(61) * 0.05
    action = np.zeros((60, 7))
    action[:, -1] = -1
    action[5, -1] = 1  # Short pulses remain data unless independently identified as noise.
    action[15:40, -1] = 1
    position = np.c_[time * 0.02, time * 0, time * 0]
    quaternion = np.tile([0, 0, 0, 1], (61, 1))
    width = np.full(61, 0.08)
    width[16:26] = np.linspace(0.075, 0.02, 10)
    width[26:41] = 0.02
    width[41:51] = np.linspace(0.025, 0.08, 10)
    gripper = np.c_[width / 2, -width / 2]
    config = trajectory.ProcessingConfig()
    commands, events = trajectory.extract_events(action, time, position, quaternion, gripper, config)
    np.testing.assert_array_equal(commands, action[:, -1])
    assert [event["event"] for event in events] == ["CLOSE", "OPEN", "CLOSE", "OPEN"]
    assert not events[0]["command_persistent"]
    assert events[0]["command_confirmed"] is None
    assert all(event["feedback_settled"] is not None for event in events[2:])
    assert all(event["coupled_motion"] for event in events[2:])
    assert all(event["grasp_success"] is None for event in events)
    coverage = np.zeros(60, dtype=int)
    phases = trajectory.propose_phases(events, 60)
    for phase in phases:
        coverage[phase["start"] : phase["end"]] += 1
    np.testing.assert_array_equal(coverage, 1)
    intervals = trajectory.command_intervals(commands, events)
    assert [(i["start"], i["end"]) for i in intervals] == [(0, 5), (5, 6), (6, 15), (15, 40), (40, 60)]
    candidates, _ = trajectory.propose_candidates(intervals, time, np.gradient(position, time, axis=0), config)
    candidate_coverage = np.zeros(60, dtype=bool)
    for candidate in candidates:
        parent = intervals[candidate["command_interval_id"]]
        assert parent["start"] <= candidate["start"] < candidate["end"] <= parent["end"]
        np.testing.assert_array_equal(action[candidate["start"] : candidate["end"], -1], candidate["command"])
        candidate_coverage[candidate["start"] : candidate["end"]] = True
    assert candidate_coverage.all()


def test_short_final_command_is_not_confirmed():
    action = np.zeros((8, 7))
    action[:, -1] = -1
    action[-1, -1] = 1
    commands, events = trajectory.extract_events(
        action,
        np.arange(9) * 0.05,
        np.zeros((9, 3)),
        np.tile([0, 0, 0, 1], (9, 1)),
        np.tile([0.04, -0.04], (9, 1)),
        trajectory.ProcessingConfig(),
    )
    assert len(events) == 1
    assert events[0]["onset"] == 7
    assert events[0]["command_confirmed"] is None
    assert commands[-1] == 1
    intervals = trajectory.command_intervals(commands, events)
    assert intervals[-1]["start"] == 7
    assert intervals[-1]["end"] == 8


@pytest.mark.parametrize("horizon", [1, 3, 4, 5])
def test_proposal_windows_never_claim_verified_supervision(horizon):
    intervals = [{"start": i, "end": i + 1, "command_interval_id": i, "command": (-1) ** i} for i in range(5)]
    result = trajectory.proposal_windows(intervals, np.ones((6, 3)), np.tile([0, 0, 0, 1], (6, 1)), horizon)
    np.testing.assert_array_equal(result["gripper_commands"][:, 0], [i["command"] for i in intervals])
    np.testing.assert_array_equal(result["gripper_commands"][~result["proposal_mask"]], 0)
    assert result["valid_length"].tolist() == [min(horizon, 5 - i) for i in range(5)]
    assert result["source_end_in_window"].tolist() == [i + horizon >= 5 for i in range(5)]
    assert not result["goal_supervision_mask"].any()
    assert not result["terminal_label_mask"].any()
    assert not result["field_supervision_mask"].any()
    np.testing.assert_array_equal(result["goal_candidates"][~result["proposal_mask"]], 0)


def test_train_moments_are_mergeable_and_invertible():
    random = np.random.default_rng(5)
    values = random.normal(size=(55, 4))
    moments = trajectory.RunningMoments()
    moments.update(values[:20])
    moments.update(values[20:])
    result = moments.result()
    np.testing.assert_allclose(result["mean"], values.mean(axis=0))
    np.testing.assert_allclose(result["std"], values.std(axis=0))
    normalized = (values - result["mean"]) / result["scale"]
    np.testing.assert_allclose(normalized * result["scale"] + result["mean"], values, atol=1e-15)
    config = trajectory.ProcessingConfig()
    assert trajectory.episode_split("same trajectory", config) == trajectory.episode_split("same trajectory", config)


@pytest.mark.parametrize(
    "kwargs", [{"grid_seconds": 0}, {"train_fraction": 0.95}, {"horizons": (0,)}, {"max_candidates": 0}]
)
def test_bad_processing_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError, match="positive|Split|Horizons|budget"):
        trajectory.ProcessingConfig(**kwargs)


def test_stationary_wait_has_a_complete_fitting_candidate():
    intervals = [{"start": 0, "end": 2, "command_interval_id": 0, "command": -1}]
    candidates, pruned = trajectory.propose_candidates(
        intervals, np.arange(3) * 0.05, np.zeros((3, 3)), trajectory.ProcessingConfig(max_candidates=1)
    )
    assert len(candidates) == 1
    assert pruned == 0
    assert (candidates[0]["start"], candidates[0]["end"]) == (0, 2)


@pytest.mark.parametrize("command", [0.0, 0.6, float("nan")])
def test_unknown_or_nonbinary_commands_are_not_silently_interpreted(command):
    action = np.zeros((2, 7))
    action[:, -1] = command
    with pytest.raises(ValueError, match="binary LIBERO"):
        trajectory.extract_events(
            action,
            np.arange(3) * 0.05,
            np.zeros((3, 3)),
            np.tile([0, 0, 0, 1], (3, 1)),
            np.tile([0.04, -0.04], (3, 1)),
            trajectory.ProcessingConfig(),
        )
