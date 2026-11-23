import dataclasses
import json

import numpy as np
import pytest
from test_build import source_fixture

from openpi.datasets.dynamics.coherent import DEFAULT_CONFIG
from openpi.datasets.dynamics.coherent import build
from openpi.datasets.dynamics.coherent import fit_sequence
from openpi.datasets.dynamics.coherent_audit import audit
from openpi.datasets.dynamics.coherent_audit import audit_episode
from openpi.datasets.dynamics.playback import load_for_episode

LIMITS = {"maximum_speed_scaled_per_second": 100, "maximum_angular_speed_rad_per_second": 1.0}
CONFIG = dataclasses.replace(DEFAULT_CONFIG, rollout_iterations=0)


def test_long_segments_preserve_short_command_event_and_continuous_state():
    time = np.arange(51) * 0.05
    position = np.c_[0.1 * (1 - np.exp(-time)), np.zeros((51, 1))]
    intervals = [
        {"start": a, "end": b, "command": c, "command_interval_id": i}
        for i, (a, b, c) in enumerate([(0, 25, -1), (25, 27, 1), (27, 50, -1)])
    ]
    records, summary, nominal, _ = fit_sequence(position, time, intervals, np.full(2, 0.02), LIMITS, config=CONFIG)
    assert summary["actions_covered"] == 50
    assert summary["new_short_motion_segments"] == 0
    events = [r for r in records if r["quality"]["short_command_event"]]
    assert [(r["start"], r["end"], r["command"]) for r in events] == [(25, 27, 1)]
    result = {
        "limits": {**LIMITS, "dt": 0.05, "segmentation": dataclasses.asdict(CONFIG)},
        "tolerance": [0.02, 0.02],
        "source_intervals": intervals,
        "records": records,
        "summary": summary,
    }
    arrays = {
        "position": position,
        "nominal": nominal,
        "time": time,
        "source_time": time,
        "tolerance": np.full(2, 0.02),
    }
    assert audit_episode(result, arrays)["saved_vs_recomputed"] < 1e-12
    records[0]["quality"]["observations"] += 1
    with pytest.raises(ValueError, match="length/quality"):
        audit_episode(result, arrays)


def test_unidentifiable_motion_stays_long_and_rejected():
    time = np.arange(41) * 0.05
    position = np.c_[np.arange(41) % 2, np.zeros(41)]
    records, summary, _, _ = fit_sequence(
        position, time, [{"start": 0, "end": 40}], np.full(2, 0.01), LIMITS, config=CONFIG
    )
    assert not summary["passed"]
    assert summary["new_short_motion_segments"] == 0
    assert all(r["end"] - r["start"] >= 10 for r in records)
    assert any("position_maximum" in r["report"]["reasons"] for r in records)


@pytest.mark.parametrize("intervals", [[{"start": 1, "end": 20}], [{"start": 0, "end": 10}], []])
def test_source_coverage_is_required(intervals):
    with pytest.raises(ValueError, match="cover"):
        fit_sequence(np.zeros((21, 2)), np.arange(21) * 0.1, intervals, np.ones(2), LIMITS, config=CONFIG)


def test_build_independent_audit_and_forgery_detection(tmp_path):
    from openpi.datasets.dynamics.build import build as build_base

    root, _, episode = source_fixture(tmp_path)
    base = tmp_path / "base"
    build_base(root, base, workers=1)
    run = build(base, workers=1, config=CONFIG)
    output = base / "profiles/coherent"
    assert run["summary"]["full_collection"]
    from openpi.datasets.dynamics.playback import available_profiles
    from openpi.simulation.data.reader import EpisodeReader

    assert "coherent" not in available_profiles(episode)
    assert audit(output)["integrity_passed"]
    assert "coherent" in available_profiles(episode)
    reader = EpisodeReader(episode)
    try:
        assert reader.segmentation["profile"] == "coherent"
    finally:
        reader.close()
    bundle = load_for_episode(episode, "coherent")
    assert bundle["result"]["model"] == "gaussian_contractive_map_v1"
    path = output / "episode_00000.json"
    data = json.loads(path.read_text())
    data["records"][0]["report"]["continuous_position_rms_native"] += 0.1
    path.write_text(json.dumps(data))
    assert "RMS" in audit(output)["failures"][0]["error"]
    data["records"][0]["report"]["continuous_position_rms_native"] -= 0.1
    data["records"][0]["accepted"] = not data["records"][0]["accepted"]
    path.write_text(json.dumps(data))
    checked = audit(output)
    assert not checked["integrity_passed"]
    assert "acceptance" in checked["failures"][0]["error"]


def test_saved_gaussian_preview_and_native_projection_use_actual_field(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openpi.datasets.dynamics.playback import native_plane_velocity
    from openpi.datasets.dynamics.playback import preview
    from openpi.fields import contractive
    from openpi.fields import primitive

    field = contractive.ContractiveField(
        {
            "version": contractive.VERSION,
            "goal": [0, 0, 0],
            "tolerance": [0.01] * 3,
            "transform": np.eye(3),
            "means": [[0, 0, 0]],
            "variance": [1, 1, 1],
            "matrices": [[[0.8, -0.2, 0], [0.2, 0.8, 0], [0, 0, 0.7]]],
            "dt": 0.05,
            "maximum_speed_scaled_per_second": 100,
        }
    )
    demo = field.rollout(np.array([0.03, 0.01, 0.02]), 10)
    nominal = field.rollout(demo[0] + [0.001, 0, 0], 10)
    arrays = {
        "position": demo,
        "nominal": nominal,
        "time": np.arange(11) * 0.05,
        "tolerance": np.full(3, 0.01),
        "rotation": np.tile(np.eye(3), (11, 1, 1)),
        "nominal_rotation": np.tile(np.eye(3), (11, 1, 1)),
    }
    record = {
        "start": 0,
        "end": 10,
        "command": -1,
        "goal": [0, 0, 0],
        "field": field.record(),
        "accepted": False,
        "report": {"reasons": ["position_maximum"], "continuous_position_rms_native": 0.001},
    }
    result = {
        "model": contractive.VERSION,
        "records": [record],
        "coordinates": {"control_point": "gripper0_grip_site"},
        "limits": {"maximum_angular_speed_rad_per_second": 1},
        "summary": {"position_rms_native": 0.001, "passed": False},
        "source_entry": {},
    }
    bundle = {"result": result, "arrays": arrays, "path": str(tmp_path / "saved.json")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Browsing must only sample saved contractive fields")

    monkeypatch.setattr(contractive, "fit", forbidden)
    monkeypatch.setattr(primitive, "feedback", forbidden)
    values, info = preview(bundle, 0, SimpleNamespace(path=tmp_path / "demo.hdf5", group="demo"), density=3)
    np.testing.assert_allclose(values["velocity"], field.velocity(values["grid"]))
    np.testing.assert_array_equal(values["nominal"], nominal)
    assert not info["report"]["passed"]
    assert info["control"]["rotation_gain"] is None
    plane = np.array([[0.01, 0.02], [-0.01, 0]])
    points = np.tile(demo[4], (2, 1))
    points[:, [0, 2]] = plane
    np.testing.assert_allclose(native_plane_velocity(bundle, 0, [0, 2], plane, 4), field.velocity(points)[:, [0, 2]])


def test_fine_endpoint_search_recovers_feasible_boundary_without_short_tail(monkeypatch):
    from openpi.datasets.dynamics import coherent

    original = coherent._candidate  # noqa: SLF001 -- Exercise failed-boundary search independently of fitting.

    def candidate(*args):
        value = original(*args)
        if value["start"] == 0 and value["end"] != 12:
            value["maximum"] = 2.0
        return value

    monkeypatch.setattr(coherent, "_candidate", candidate)
    config = dataclasses.replace(CONFIG, refinement_candidates=0, fine_endpoint_radius=2)
    records, summary, _, _ = coherent.fit_sequence(
        np.zeros((41, 2)), np.arange(41) * 0.05, [{"start": 0, "end": 40}], np.ones(2), LIMITS, config=config
    )
    assert records[0]["end"] == 12
    assert summary["passed"]
    assert summary["new_short_motion_segments"] == 0


def test_orientation_principal_chart_failure_is_preserved_and_auditable():
    from scipy.spatial.transform import Rotation

    time = np.arange(11) * 0.1
    position = np.zeros((11, 2))
    rotation = Rotation.from_euler("z", np.linspace(np.pi, 0, 11)).as_matrix()
    intervals = [{"start": 0, "end": 10}]
    config = dataclasses.replace(CONFIG, refinement_candidates=0, fine_endpoint_radius=0)
    limits = {**LIMITS, "maximum_angular_speed_rad_per_second": 100}
    records, summary, nominal, orientations = fit_sequence(
        position, time, intervals, np.ones(2), limits, rotations=rotation, config=config
    )
    assert "orientation_log_chart" in records[0]["report"]["reasons"]
    assert not summary["passed"]
    result = {
        "limits": {**limits, "dt": 0.1, "segmentation": dataclasses.asdict(config)},
        "source_intervals": intervals,
        "records": records,
        "summary": summary,
        "tolerance": [1, 1],
    }
    arrays = {
        "position": position,
        "nominal": nominal,
        "rotation": rotation,
        "nominal_rotation": orientations,
        "time": time,
        "source_time": time,
        "tolerance": np.ones(2),
    }
    checked = audit_episode(result, arrays)
    assert not checked["nominal_passed"]
    assert checked["rejected"]["orientation_log_chart"] == 1


def test_pass_through_goals_reduce_artificial_boundary_slowdown_and_are_audited():
    time = np.arange(41) * 0.05
    position = np.c_[0.1 * time, np.zeros(41)]
    intervals = [{"start": 0, "end": 20}, {"start": 20, "end": 40}]
    config = dataclasses.replace(CONFIG, goal_lookahead_steps=4)
    records, summary, nominal, _ = fit_sequence(position, time, intervals, np.full(2, 0.01), LIMITS, config=config)
    _, _, old, _ = fit_sequence(position, time, intervals, np.full(2, 0.01), LIMITS, config=CONFIG)
    assert summary["passed"]
    assert records[0]["goal_source_index"] > 20
    assert records[-1]["goal_source_index"] == 40
    assert nominal[20, 0] - nominal[19, 0] > old[20, 0] - old[19, 0]
    assert records[0]["end"] == records[1]["start"] == 20
    result = {
        "limits": {**LIMITS, "dt": 0.05, "segmentation": dataclasses.asdict(config)},
        "tolerance": [0.01, 0.01],
        "source_intervals": intervals,
        "records": records,
        "summary": summary,
    }
    arrays = {
        "position": position,
        "nominal": nominal,
        "time": time,
        "source_time": time,
        "tolerance": np.full(2, 0.01),
    }
    assert audit_episode(result, arrays)["nominal_passed"]
    records[0]["goal_source_index"] = 30
    with pytest.raises(ValueError, match="pass-through target"):
        audit_episode(result, arrays)


def test_pass_through_does_not_move_a_true_stop_or_reversal_goal():
    from openpi.datasets.dynamics.coherent import _candidate

    config = dataclasses.replace(CONFIG, goal_lookahead_steps=4)
    for tail in (np.full(10, 0.1), np.linspace(0.095, 0.05, 10)):
        position = np.c_[np.r_[np.linspace(0, 0.1, 21), tail], np.zeros(31)]
        value = _candidate(position, None, 0, 20, position[0], None, 0.05, np.full(2, 0.01), 100, 1, config)
        assert value["goal_source_index"] == 20
        np.testing.assert_array_equal(value["field"].goal, position[20])


def test_pass_through_profile_is_separate_and_available_after_full_audit(tmp_path):
    from openpi.datasets.dynamics.build import build as build_base
    from openpi.datasets.dynamics.playback import available_profiles

    root, _, episode = source_fixture(tmp_path)
    base = tmp_path / "base"
    build_base(root, base, workers=1)
    config = dataclasses.replace(CONFIG, goal_lookahead_steps=4)
    build(base, workers=1, config=config)
    assert "coherent_flow" not in available_profiles(episode)
    assert audit(base / "profiles/coherent_flow")["integrity_passed"]
    assert "coherent_flow" in available_profiles(episode)
    bundle = load_for_episode(episode, "coherent_flow")
    assert bundle["result"]["profile"] == "coherent_flow"
    assert not (base / "profiles/coherent").exists()


def test_pass_through_falls_back_to_complete_endpoint_sequence_if_a_later_span_fails(monkeypatch):
    from openpi.datasets.dynamics import coherent

    original = coherent._candidate  # noqa: SLF001 -- Force a complete-sequence failure independently of fitting.

    def reject_forward_sequence(*args):
        value = original(*args)
        if args[-1].goal_lookahead_steps:
            value["maximum"] = 2.0
        return value

    monkeypatch.setattr(coherent, "_candidate", reject_forward_sequence)
    time = np.arange(41) * 0.05
    position = np.c_[0.1 * time, np.zeros(41)]
    intervals = [{"start": 0, "end": 20}, {"start": 20, "end": 40}]
    config = dataclasses.replace(CONFIG, goal_lookahead_steps=4, refinement_candidates=0, fine_endpoint_radius=0)
    records, summary, nominal, _ = fit_sequence(position, time, intervals, np.full(2, 0.01), LIMITS, config=config)
    assert summary["passed"]
    assert summary["flow_fallback"]
    assert summary["flow_attempt_maximum_scaled_error"] == 2
    assert summary["pass_through_segments"] == 0
    assert all(r["goal_source_index"] == r["end"] for r in records)
    result = {
        "limits": {**LIMITS, "dt": 0.05, "segmentation": dataclasses.asdict(config)},
        "tolerance": [0.01, 0.01],
        "source_intervals": intervals,
        "records": records,
        "summary": summary,
    }
    arrays = {
        "position": position,
        "nominal": nominal,
        "time": time,
        "source_time": time,
        "tolerance": np.full(2, 0.01),
    }
    assert audit_episode(result, arrays)["nominal_passed"]


def test_selected_slow_exit_can_be_improved_without_reset_or_changing_the_boundary():
    from openpi.datasets.dynamics.coherent import _refine_slow_exit
    from openpi.fields import contractive

    position = np.c_[np.arange(21) * 0.01, np.zeros(21)]
    field = contractive.ContractiveField(
        {
            "version": contractive.VERSION,
            "goal": position[10],
            "tolerance": [0.1, 0.1],
            "transform": np.eye(2),
            "means": [[0, 0]],
            "variance": [1, 1],
            "matrices": [np.eye(2) * 0.2],
            "dt": 0.05,
            "maximum_speed_scaled_per_second": 100,
        }
    )
    path = field.rollout(np.stack([position[0], position[0]]), 10)
    maximum = float(np.linalg.norm((path - position[:11, None]) / 0.1, axis=-1).max())
    value = {
        "start": 0,
        "end": 10,
        "goal_source_index": 10,
        "field": field,
        "path": path,
        "maximum": maximum,
        "chart_valid": True,
        "angular_speed_valid": True,
        "angular_error": 0,
    }
    config = dataclasses.replace(CONFIG, goal_lookahead_steps=4, rollout_iterations=1)
    improved = _refine_slow_exit(value, position, position[0], 0.05, np.full(2, 0.1), 100, config)
    assert improved["goal_source_index"] > 10
    assert improved["start"] == 0
    assert improved["end"] == 10
    np.testing.assert_array_equal(improved["path"][0], path[0])
    assert improved["maximum"] <= 1
    assert np.linalg.norm(improved["path"][-1, 1] - improved["path"][-2, 1]) > np.linalg.norm(path[-1, 1] - path[-2, 1])
    assert improved["field"].radius < 1
