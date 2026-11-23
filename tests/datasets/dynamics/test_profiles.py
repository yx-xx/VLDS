import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.datasets.dynamics.audit import audit
from openpi.datasets.dynamics.build import build
from openpi.datasets.dynamics.profiles import build_profile
from openpi.fields.sequence import fit_sequence


def source_fixture(tmp_path):
    root = tmp_path / "pusht"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps({"data_path": "data/episode_{episode_index:06d}.parquet", "chunks_size": 1000})
    )
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 8}) + "\n")
    path = root / "data/episode_000000.parquet"
    path.parent.mkdir()
    pq.write_table(
        pa.table({"observation.state": [[float(i), float(i)] for i in range(8)], "timestamp": np.arange(8) / 10}),
        path,
    )
    return root


def test_profile_build_preserves_rejected_leaf_and_audits_artifact(tmp_path):
    root = source_fixture(tmp_path)
    base = tmp_path / "base"
    build(root, base, workers=1)
    run_path = base / "run.json"
    run = json.loads(run_path.read_text())
    run["config"]["kind"] = "processed_libero"
    run_path.write_text(json.dumps(run))

    result = build_profile(base, workers=1)
    assert result["status"] == "complete"
    assert result["summary"]["full_collection"]
    assert result["summary"]["failure_reasons"]
    profile_path = base / "profiles" / "production_libero"
    audited = audit(profile_path)
    assert audited["passed"]
    assert audited["profile"] == "production_libero"
    assert audited["profile_passed_episodes"] == result["summary"]["passed_episodes"]


def test_rotation_gain_is_part_of_saved_profile_dynamics():
    position = np.zeros((4, 1))
    time = np.arange(4, dtype=float) / 10
    rotations = np.stack(
        [
            np.eye(3),
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ]
    )
    intervals = [{"start": 0, "end": 3, "command_interval_id": 0, "command_label": "MOVE"}]
    _, _, _, limits = fit_sequence(
        position,
        time,
        intervals,
        np.ones(1),
        rotations=rotations,
        maximum_speed=1.0,
        maximum_angular_speed=0.5,
        rotation_gain=2.0,
    )
    assert limits["rotation_gain_per_second"] == 2.0


def _pose_collection(tmp_path):
    """A real persisted pose fixture, with a fast rotation that needs retiming."""
    import hashlib
    import os

    from scipy.spatial.transform import Rotation

    root, base = tmp_path / "source", tmp_path / "pose_base"
    root.mkdir()
    base.mkdir()
    fingerprint, signature = "base-fingerprint", "episode-signature"
    source_entry = {"signature": signature}
    manifest = {"schema": "vlds.libero.trajectories.v2", "episodes": [source_entry]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    time = np.arange(11) * 0.05
    position = np.c_[np.linspace(0, 0.02, len(time)), np.zeros((len(time), 2))]
    rotations = Rotation.from_euler("z", np.linspace(0, 0.5, len(time))).as_matrix()
    intervals = [{"start": 0, "end": 10, "command_interval_id": 0, "command_label": "OPEN", "command": -1}]
    trace = base / "episode_00000.npz"
    np.savez_compressed(trace, position=position, time=time, tolerance=np.full(3, 0.01), rotation=rotations)
    result = {
        "fingerprint": fingerprint,
        "trace_file": trace.name,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "source_key": "synthetic-pose",
        "source_sha256": signature,
        "source_entry": source_entry,
        "coordinates": {"space": "world Cartesian", "control_point": "gripper0_grip_site", "axes": ["x", "y", "z"]},
        "tolerance": [0.01] * 3,
        "source_intervals": intervals,
    }
    (base / "episode_00000.json").write_text(json.dumps(result))
    (base / "run.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "status": "complete",
                "summary": {"full_collection": True},
                "source_relative": os.path.relpath(root, base),
                "config": {
                    "kind": "processed_libero",
                    "source_episodes": 1,
                    "source_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                },
            }
        )
    )
    (base / "index.json").write_text(json.dumps({"episodes": [{"index": 0, "file": "episode_00000.json"}]}))
    pointer = root / ".vlds/dynamics.json"
    pointer.parent.mkdir()
    pointer.write_text(json.dumps({"directory": os.path.relpath(base, pointer.parent), "fingerprint": fingerprint}))
    from openpi.simulation.data.reader import Episode

    episode = Episode(
        root / "fixture.hdf5", "pose", "pose", len(time), 20, (), "processed", processed_root=root, processed_index=0
    )
    return base, episode


def test_retimed_profile_preserves_control_period_and_continuous_preview(tmp_path):
    from openpi.datasets.dynamics.playback import available_profiles
    from openpi.datasets.dynamics.playback import load_for_episode
    from openpi.datasets.dynamics.playback import preview
    from openpi.datasets.dynamics.playback import segmentation

    base, episode = _pose_collection(tmp_path)
    fixed = build_profile(base, "production_libero", workers=1)
    assert fixed["summary"]["passed_episodes"] == 0
    retimed = build_profile(base, "production_retimed", workers=1)
    assert retimed["summary"]["passed_episodes"] == 1
    bundle = load_for_episode(episode, "production_retimed")
    limits = bundle["result"]["limits"]
    assert np.isclose(limits["dt"], 0.05)
    assert limits["integration_substeps"] == limits["time_scale"] > 1
    np.testing.assert_allclose(bundle["arrays"]["time"], bundle["arrays"]["source_time"] * limits["time_scale"])
    assert "production_retimed" in available_profiles(episode)
    assert segmentation(bundle)["time_scale"] == limits["time_scale"]
    values, metadata = preview(bundle, 0, episode, density=3)
    record = bundle["result"]["records"][0]
    np.testing.assert_array_equal(values["nominal"], bundle["arrays"]["nominal"][record["start"] : record["end"] + 1])
    assert metadata["report"]["passed"] == record["accepted"]
    checked = audit(base / "profiles/production_retimed")
    assert checked["integrity_passed"]
    assert checked["all_nominal_passed"]
    assert checked["counts"]["control_steps"] == 10 * limits["time_scale"]


def test_audit_rejects_forged_accepted_flag_and_preview_preserves_rejection(tmp_path):
    from openpi.datasets.dynamics.playback import load_for_episode
    from openpi.datasets.dynamics.playback import preview

    base, episode = _pose_collection(tmp_path)
    build_profile(base, "production_libero", workers=1)
    bundle = load_for_episode(episode, "production_libero")
    index = next(i for i, r in enumerate(bundle["result"]["records"]) if not r["accepted"])
    _, metadata = preview(bundle, index, episode, density=3)
    assert not metadata["report"]["passed"]
    assert "rotation_maximum" in metadata["report"]["reasons"]
    path = base / "profiles/production_libero/episode_00000.json"
    result = json.loads(path.read_text())
    result["records"][index]["accepted"] = True
    path.write_text(json.dumps(result))
    checked = audit(path.parent)
    assert not checked["integrity_passed"]
    assert "acceptance" in checked["failures"][0]["error"]
