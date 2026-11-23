"""Automatic previews must preserve action coverage and reject incompatible archives."""

import itertools
from types import SimpleNamespace

from openpi.datasets.libero import field_preview_sequence as sequence
from openpi.fields.primitive import DEFAULT_CONTROL


def test_interval_boundary_uses_following_field_and_keeps_terminal():
    intervals = [{"start": 0, "end": 5}, {"start": 5, "end": 11}]
    assert sequence.interval_at(intervals, 0) == 0
    assert sequence.interval_at(intervals, 5) == 1
    assert sequence.interval_at(intervals, 11) == 1
    assert sequence.interval_at(intervals, -1) is None
    assert sequence.interval_at(intervals, 12) is None


def test_adaptive_partition_preserves_rejected_leaves_and_exact_coverage():
    attempts = []

    def evaluate(start, end):
        attempts.append((start, end))
        return {}, {"interval": {"start": start, "end": end}, "report": {"passed": False}}

    values = sequence.adaptive_partition(36, 85, evaluate)
    spans = [info["interval"] for _, info in values]
    assert spans[0]["start"] == 36
    assert spans[-1]["end"] == 85
    assert all(a["end"] == b["start"] for a, b in itertools.pairwise(spans))
    assert all(5 <= item["end"] - item["start"] < 10 for item in spans)
    assert len(attempts) == 2 * len(values) - 1
    assert not any(info["report"]["passed"] for _, info in values)


def test_saved_sequence_requires_full_contiguous_original_demo_and_control(tmp_path, monkeypatch):
    (tmp_path / "episode_00000.json").touch()
    records = [
        {"accepted": True, "primitive": {"valid": True}, "candidate": {"start": a, "end": b}}
        for a, b in [(0, 5), (6, 10)]
    ]
    result = {"records": records, "sequences": [{"record_indices": [0, 1]}]}
    monkeypatch.setattr(sequence, "load_result", lambda *_: (result, DEFAULT_CONTROL))
    dataset = SimpleNamespace(episodes=[{"actions": 10}])
    assert sequence.saved_sequence(dataset, 0, DEFAULT_CONTROL, tmp_path) is None
    records[1]["candidate"]["start"] = 5
    assert sequence.saved_sequence(dataset, 0, DEFAULT_CONTROL, tmp_path) is not None
    result["teacher_source"] = {"kind": "other-reference"}
    assert sequence.saved_sequence(dataset, 0, DEFAULT_CONTROL, tmp_path) is None
    result.pop("teacher_source")
    records[1]["accepted"] = False
    assert sequence.saved_sequence(dataset, 0, DEFAULT_CONTROL, tmp_path) is None


def test_continuous_report_does_not_reset_to_demo_at_boundaries():
    import dataclasses

    import numpy as np

    from openpi.fields import primitive
    from openpi.fields import seds

    control = dataclasses.replace(DEFAULT_CONTROL, max_speed=0.3)
    values = []
    for start, end, goal in [(0, 2, 0.02), (2, 4, 0.04)]:
        field = seds.PositionField(seds.linear_factors(1.0), [goal, 0, 0])
        model = primitive.Primitive(field, np.eye(3), 1)
        values.append(({}, {"interval": {"start": start, "end": end}, "primitive": model.record()}))
    data = {
        "trajectory": {
            "position": np.c_[np.arange(5) * 0.01, np.zeros((5, 2))],
            "quaternion_xyzw": np.tile([0, 0, 0, 1], (5, 1)),
        },
        "time": np.arange(5) * 0.05,
        "body_to_control_rotation": np.eye(3),
    }
    sequence.attach_continuous_report(values, data, control)
    np.testing.assert_allclose(values[1][0]["sequence_nominal"][:, 0], [0.00195, 0.0038525, 0.005659875])
    assert values[0][1]["continuous_report"] == values[1][1]["continuous_report"]
    assert not values[0][1]["continuous_report"]["time_aligned_passed"]
