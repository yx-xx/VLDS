import numpy as np
import pytest

from openpi.datasets.dynamics.continuity import measure


def test_boundary_diagnostic_distinguishes_generated_slowdown_from_a_source_pause():
    time = np.arange(7, dtype=float)
    position = np.c_[[0, 1, 2, 3, 3, 3, 4], np.zeros(7)]
    nominal = position.copy()
    nominal[1, 0] = 1.9  # Enter frame 2 at 10% of the source speed.
    records = [{"start": 0, "end": 2}, {"start": 2, "end": 4}, {"start": 4, "end": 6}]
    result = measure(position, nominal, time, records, np.ones(2))
    assert result["boundary_count"] == 2
    assert result["moving_boundaries"] == 1
    assert result["slowdown_candidates"] == 1
    assert result["median_end_speed_ratio"] == pytest.approx(0.1)
    assert result["boundaries"][0]["source_continues"]
    assert not result["boundaries"][1]["source_continues"]
    assert not result["boundaries"][1]["slowdown_candidate"]
    assert result["boundaries"][1]["end_speed_ratio"] is None


def test_boundary_diagnostic_rejects_unshared_observations_and_has_no_false_flags_on_source_replay():
    time = np.arange(5, dtype=float)
    position = np.c_[time, np.zeros(5)]
    records = [{"start": 0, "end": 2}, {"start": 2, "end": 4}]
    result = measure(position, position, time, records, np.ones(2))
    assert result["median_end_speed_ratio"] == 1
    assert result["slowdown_candidates"] == 0
    assert result["boundaries"][0]["excess_velocity_jump_scaled"] == 0
    records[1]["start"] = 3
    with pytest.raises(ValueError, match="shared"):
        measure(position, position, time, records, np.ones(2))
