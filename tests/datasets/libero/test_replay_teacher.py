import hashlib
import json

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.datasets.libero import replay_teacher
from openpi.fields import seds


@pytest.fixture
def source_replay(tmp_path):
    n = 12
    actions = np.zeros((n, 7))
    actions[:, -1] = 1
    position = np.zeros((n + 1, 3))
    body_frame = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    body_rotations = Rotation.from_rotvec(np.arange(n)[:, None] * [0.01, 0, 0]).as_matrix()
    log = np.zeros((n, 23))
    log[:, 0] = np.arange(1, n + 1)
    log[:, 1] = -1
    log[:, 2] = 0.25 + np.arange(1, n + 1) * 0.05
    log[:, 3] = np.arange(1, n + 1) * 0.001
    log[:, 6:10] = Rotation.from_matrix(body_rotations @ body_frame).as_quat()
    log[:, -7:] = actions
    data = {
        "entry": {"signature": "source", "split": "train"},
        "actions": actions,
        "time": np.arange(n + 1) * 0.05,
        "body_to_control_rotation": body_frame,
        "trajectory": {"position": position, "quaternion_xyzw": np.tile([0, 0, 0, 1.0], (n + 1, 1))},
    }
    request = {
        "source_episode": data["entry"],
        "source_manifest_sha256": "manifest",
        "control": {"dt": 0.05},
        "source_initial_position": [0, 0, 0],
        "body_to_control_rotation": body_frame.tolist(),
    }
    result = {
        "status": "complete",
        "mode": "source_actions",
        "sequence_results": [
            {
                "task_success": True,
                "status": "complete",
                "control_steps": n,
                "intermediate_resets": 0,
                "trajectory_file": "trial_0.hdf5",
            }
        ],
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    (tmp_path / "result.json").write_text(json.dumps(result))
    with h5py.File(tmp_path / "trial_0.hdf5", "w") as f:
        f["control_log"] = log
        f.attrs["control_log_columns"] = json.dumps(
            ["step", "field_index", "simulation_time", "px", "py", "pz", "qx", "qy", "qz", "qw"]
            + [f"other{i}" for i in range(13)]
        )
        f["privileged_audit/do_not_copy"] = [999]
    return data, tmp_path, body_rotations


def test_derived_teacher_converts_control_frame_and_preserves_source(source_replay):
    data, root, rotations = source_replay
    metadata = replay_teacher.prepare(data, root, root / "teacher.npz", "manifest")
    derived = replay_teacher.apply(data, root, metadata)
    assert metadata["kind"] == "successful_source_action_replay"
    assert not metadata["training_ready"]
    assert derived["actions"] is data["actions"]
    assert derived["entry"] is data["entry"]
    assert not data["trajectory"]["position"].any()
    np.testing.assert_allclose(derived["trajectory"]["position"][:, 0], np.arange(13) * 0.001)
    np.testing.assert_allclose(derived["trajectory"]["linear_velocity"][:, 0], 0.02)
    np.testing.assert_allclose(
        Rotation.from_quat(derived["trajectory"]["quaternion_xyzw"])[1:].as_matrix(), rotations, atol=1e-12
    )
    with np.load(root / "teacher.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"position", "quaternion_xyzw", "linear_velocity", "angular_velocity"}
    (root / "teacher.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        replay_teacher.apply(data, root, metadata)


@pytest.mark.parametrize("reason", ["failed_task", "signature", "manifest", "actions", "time", "frame", "reset"])
def test_invalid_source_replay_cannot_supply_a_teacher(source_replay, reason):
    data, root, _ = source_replay
    if reason in ("failed_task", "reset"):
        value = json.loads((root / "result.json").read_text())
        if reason == "failed_task":
            value["sequence_results"][0]["task_success"] = False
        else:
            value["sequence_results"][0]["intermediate_resets"] = 1
        (root / "result.json").write_text(json.dumps(value))
    elif reason in ("signature", "manifest", "frame"):
        value = json.loads((root / "request.json").read_text())
        if reason == "signature":
            value["source_episode"]["signature"] = "other"
        elif reason == "manifest":
            value["source_manifest_sha256"] = "other"
        else:
            value["body_to_control_rotation"] = np.eye(3).tolist()
        (root / "request.json").write_text(json.dumps(value))
    else:
        with h5py.File(root / "trial_0.hdf5", "r+") as f:
            f["control_log"][4, 16 if reason == "actions" else 2] += 0.1
    with pytest.raises(ValueError, match="Replay|replay"):
        replay_teacher.prepare(data, root, root / "teacher.npz", "manifest")
    assert not (root / "teacher.npz").exists()


def test_linear_teacher_fits_known_rate_without_nonlinear_optimizer(monkeypatch):
    def no_nonlinear_fit(*args, **kwargs):
        pytest.fail("Linear baseline must not run nonlinear optimizer")

    monkeypatch.setattr(seds, "velocity_objective", no_nonlinear_fit)
    position = np.array([[1.0, 0, 0], [0.5, 0, 0], [0, 0, 0]])
    field, report = seds.fit_linear(position, -2 * position, [0, 0.5, 1.0])
    np.testing.assert_allclose(field.velocity(position), -2 * position, atol=1e-12)
    assert report["kind"] == "linear_baseline"
    assert not report["parameter_supervision"]


def test_portable_teacher_copy_verifies_hash_before_use(tmp_path):
    origin, destination = tmp_path / "a", tmp_path / "b"
    origin.mkdir()
    destination.mkdir()
    content = b"immutable teacher"
    (origin / "teacher.npz").write_bytes(content)
    metadata = {"file": "teacher.npz", "sha256": hashlib.sha256(content).hexdigest()}
    replay_teacher.copy_artifact(origin, destination, metadata)
    assert (destination / "teacher.npz").read_bytes() == content
    (origin / "teacher.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale"):
        replay_teacher.copy_artifact(origin, destination, metadata)
