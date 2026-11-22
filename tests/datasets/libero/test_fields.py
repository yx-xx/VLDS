import numpy as np
import pytest

from openpi.datasets.libero import fields


def data_example():
    return {
        "time": np.arange(41) * 0.05,
        "actions": np.zeros((40, 7)),
        "trajectory": {"linear_velocity": np.tile([0.04, 0, 0], (41, 1))},
        "command_intervals": [
            {"command_interval_id": 0, "start": 0, "end": 20, "command": -1},
            {"command_interval_id": 1, "start": 20, "end": 40, "command": 1},
        ],
    }


def test_mandatory_spans_survive_capacity_and_confirmation_blocks_crossing():
    data = data_example()
    config = fields.SelectionConfig(initial_candidates=1, expanded_candidates=2)
    candidates, required = fields.candidates_for_episode(
        data,
        config,
        confirmed_boundaries=[{"index": 10, "verified": True, "observable": True, "evidence": "state-audit"}],
    )
    mandatory = {(c["start"], c["end"]): c for c in candidates[:required]}
    assert required == 4
    assert {(0, 20), (0, 10), (10, 20), (20, 40)} == set(mandatory)
    assert not mandatory[0, 20]["selectable"]
    assert mandatory[0, 10]["selectable"]
    for candidate in candidates[required:]:
        assert not candidate["start"] < 10 < candidate["end"]
    assert all(c["end"] <= 20 or c["start"] >= 20 for c in candidates)
    with pytest.raises(ValueError, match="evidence"):
        fields.candidates_for_episode(data, confirmed_boundaries=[{"index": 10, "verified": True}])


def edge(start, end, cost, *, accepted=True):
    return {"candidate": {"start": start, "end": end, "selectable": True}, "accepted": accepted, "cost": cost}


def test_k_best_dynamic_programming_preserves_full_coverage_and_reports_failure():
    records = [
        edge(0, 10, 0.4),
        edge(10, 20, 0.5),
        edge(0, 20, 1.2),
        edge(20, 40, 0.1),
        edge(10, 40, 0.1, accepted=False),
    ]
    result = fields.k_best_sequences(records, 40)
    assert [r["record_indices"] for r in result] == [[0, 1, 3], [2, 3]]
    assert result[0]["cost"] == pytest.approx(1.0)
    assert all(r["actions_covered"] == 40 for r in result)
    assert fields.k_best_sequences(records, 41) == []
    with pytest.raises(ValueError, match="edge"):
        fields.k_best_sequences([edge(0, 20, np.nan)], 20)


def test_complete_offline_pipeline_keeps_semantics_and_training_unverified():
    from scipy.spatial.transform import Rotation

    from openpi.datasets.libero.calibration import recheck_episode
    from openpi.fields import primitive

    time = np.arange(11) * 0.05
    movement = 0.02 * np.exp(-2 * time[:6])
    position = np.zeros((11, 3))
    position[:, 0] = np.r_[movement, np.repeat(movement[-1], 5)]
    control_rotation = Rotation.from_rotvec([0, 0, 0.4]).as_matrix()
    data = {
        "entry": {"split": "train", "signature": "synthetic"},
        "time": time,
        "actions": np.r_[np.tile([0] * 6 + [-1], (5, 1)), np.tile([0] * 6 + [1], (5, 1))],
        "trajectory": {
            "position": position,
            "linear_velocity": np.gradient(position, 0.05, axis=0),
            "quaternion_xyzw": np.tile([0, 0, 0, 1], (11, 1)),
        },
        "body_to_control_rotation": control_rotation,
        "command_intervals": [
            {"start": 0, "end": 5, "command": -1, "command_interval_id": 0},
            {"start": 5, "end": 10, "command": 1, "command_interval_id": 1},
        ],
    }
    published = []
    result = fields.construct_episode(data, publish=lambda records: published.append(len(records)))
    assert result["summary"]["complete_offline_sequence"]
    assert result["sequences"][0]["actions_covered"] == 10
    assert result["semantic_confirmation_status"] == "not_constructed"
    assert result["continuous_execution_status"] == "not_run"
    assert not result["training_ready"]
    assert published == [1, 2]
    records = result["records"]
    assert [r["primitive"]["command"] for r in records] == [-1, 1]
    assert records[1]["primitive"]["primitive_kind"] == "HOLD"
    np.testing.assert_allclose(primitive.Primitive.from_record(records[0]["primitive"]).rotation, control_rotation)
    baseline = recheck_episode(result, data, primitive.DEFAULT_CONTROL, representation="linear")
    assert baseline["summary"]["complete_offline_sequence"]
    assert baseline["records"][0]["fit"]["kind"] == "linear_baseline"
    assert result["records"][0]["fit"]["kind"] == "short_linear"  # no mutation of the fitted pool


def test_speed_lower_bound_detects_impossible_tracking_without_fitting():
    from openpi.datasets.libero.calibration import reachability_lower_bound

    position = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    assert reachability_lower_bound(position, np.array([0.0, 1.0]), 1.0) == pytest.approx(9 / np.sqrt(2))
    assert reachability_lower_bound(position, np.array([0.0, 1.0]), 20.0) == 0
