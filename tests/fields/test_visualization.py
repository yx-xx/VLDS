import dataclasses
import json

import numpy as np
import pytest

from openpi.fields import primitive
from openpi.fields import seds
from openpi.fields import visualization


def test_sampled_arrows_match_execution_feedback_and_rollout_uses_physical_time(tmp_path):
    control = dataclasses.replace(primitive.DEFAULT_CONTROL, max_speed=0.3)
    field = primitive.Primitive(seds.PositionField(seds.linear_factors(2), np.zeros(3)), np.eye(3), -1)
    time = np.arange(81) * control.dt
    position = np.array([0.03, 0.01, -0.02]) * 0.9 ** np.arange(len(time))[:, None]
    rotations = np.tile(np.eye(3), (len(time), 1, 1))
    arrays, metadata = visualization.sample_field(field, position, rotations, time, control, density=3)
    expected = np.array([primitive.feedback(field, p, np.eye(3), control)[0] for p in arrays["grid"]])
    np.testing.assert_allclose(arrays["velocity"], expected)
    np.testing.assert_allclose(arrays["nominal"][: len(time)], position, atol=1e-16)
    assert metadata["report"]["passed"]
    assert metadata["report"]["position_rms_m"] < 1e-16
    assert np.linalg.norm(arrays["velocity"], axis=1).max() <= control.max_speed
    path = tmp_path / "field.png"
    visualization.plot_field(path, arrays, metadata)
    assert path.stat().st_size > 1000
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["demo"], position)
        assert json.loads(str(archive["metadata"]))["frame"] == "world"
    assert not json.loads(path.with_suffix(".json").read_text())["report"]["reasons"]


def test_incompatible_demonstration_remains_visible_as_rejected_fit():
    field = primitive.Primitive(seds.PositionField(seds.linear_factors(2), np.zeros(3)), np.eye(3), -1)
    time = np.arange(41) * 0.05
    position = np.c_[0.04 + 0.05 * np.sin(np.pi * time / time[-1]), np.zeros((len(time), 2))]
    arrays, metadata = visualization.sample_field(
        field, position, np.tile(np.eye(3), (len(time), 1, 1)), time, density=3
    )
    assert not metadata["report"]["passed"]
    assert "position_rms" in metadata["report"]["reasons"]
    assert arrays["position_error"].max() > 0.04
    np.testing.assert_array_equal(arrays["demo"], position)
    with pytest.raises(ValueError, match="density"):
        visualization.sample_field(field, position, np.tile(np.eye(3), (len(time), 1, 1)), time, density=2)
