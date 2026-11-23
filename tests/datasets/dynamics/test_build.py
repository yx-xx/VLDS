import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.datasets.dynamics.audit import audit
from openpi.datasets.dynamics.build import build
from openpi.datasets.dynamics.playback import load_for_episode
from openpi.datasets.dynamics.playback import segmentation
from openpi.simulation.data.reader import Episode


def source_fixture(tmp_path):
    root = tmp_path / "pusht"
    (root / "meta").mkdir(parents=True)
    info = {"data_path": "data/episode_{episode_index:06d}.parquet", "chunks_size": 1000}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 8}) + "\n")
    path = root / "data/episode_000000.parquet"
    path.parent.mkdir()
    pq.write_table(
        pa.table({"observation.state": [[float(i), float(i)] for i in range(8)], "timestamp": np.arange(8) / 10}), path
    )
    episode = Episode(path, "episode_000000", "push", 8, 10.0, (), "LeRobot v2.1")
    return root, path, episode


def test_complete_build_resume_validation_and_playback(tmp_path):
    root, source, episode = source_fixture(tmp_path)
    output = tmp_path / "output"
    run = build(root, output, workers=1)
    assert run["summary"]["full_collection"]
    assert run["summary"]["passed_episodes"] == 1
    bundle = load_for_episode(episode)
    assert bundle["result"]["model"] == "stable_isotropic_rollout_v1"
    assert segmentation(bundle)["kind"] == "dynamics"
    assert len(bundle["arrays"]["time"]) == 8  # No synthetic post-terminal observation.
    result_path = output / "episode_00000.json"
    modified = result_path.stat().st_mtime_ns
    build(root, output, workers=1)
    assert result_path.stat().st_mtime_ns == modified
    assert audit(output)["passed"]
    # A changed source must be rejected by playback, and recomputed on resume.
    table = pq.read_table(source).set_column(0, "observation.state", pa.array([[float(i), 0.0] for i in range(8)]))
    pq.write_table(table, source)
    with pytest.raises(ValueError, match="示教文件"):
        load_for_episode(episode)
    build(root, output, workers=1)
    assert result_path.stat().st_mtime_ns != modified
    assert load_for_episode(episode)["result"]["summary"]["passed"]


def test_corrupt_trace_is_rejected_and_rebuilt(tmp_path):
    root, _, episode = source_fixture(tmp_path)
    output = tmp_path / "output"
    build(root, output, workers=1)
    (output / "episode_00000.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="校验"):
        load_for_episode(episode)
    build(root, output, workers=1)
    assert load_for_episode(episode)["result"]["summary"]["passed"]


def test_configuration_mismatch_and_partial_selection(tmp_path):
    root, _, _ = source_fixture(tmp_path)
    output = tmp_path / "output"
    run = build(root, output, workers=1, indices=[0])
    assert not run["summary"]["full_collection"]
    assert not (root / ".vlds/dynamics.json").exists()
    info = json.loads((root / "meta/info.json").read_text())
    info["fps"] = 20
    (root / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="differs"):
        build(root, output, workers=1)
    with pytest.raises(ValueError, match="distinct"):
        build(root, output, indices=[0, 0])


def test_cached_playback_revalidates_changed_source_catalogue(tmp_path):
    root, _, episode = source_fixture(tmp_path)
    build(root, tmp_path / "output", workers=1)
    assert load_for_episode(episode) is not None
    info = json.loads((root / "meta/info.json").read_text())
    info["fps"] = 20
    (root / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="元信息已变化"):
        load_for_episode(episode)


def test_parallel_audit_matches_serial_audit(tmp_path):
    root, _, _ = source_fixture(tmp_path)
    output = tmp_path / "output"
    build(root, output, workers=1)
    assert audit(output, workers=1) == audit(output, workers=2)
