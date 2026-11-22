import dataclasses
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.datasets.libero.operation_audit import EvidenceConfig
from openpi.datasets.libero.operation_audit import audit_fields
from openpi.datasets.libero.operation_audit import summarize_trajectory


@pytest.fixture
def audit_log(tmp_path):
    path = tmp_path / "trial_0.hdf5"
    n = 24
    log = np.zeros((n, 23))
    log[:, 0] = np.arange(1, n + 1)
    log[:, 1] = 3
    log[:, 2] = np.arange(n) * 0.05
    log[:, 9] = 1
    log[:, -1] = -1
    log[4:16, -1] = 1
    pose = np.zeros((n, 1, 7))
    pose[:, 0, 6] = 1
    pose[5:, 0, 2] = 0.03
    log[:, 3:6] = pose[:, 0, :3] + [0, 0, 0.1]
    contact = np.zeros((n, 1), dtype=bool)
    contact[4:16] = True
    values = np.zeros((n, 1), dtype=bool)
    values[17:] = True
    with h5py.File(path, "w") as f:
        f["control_log"] = log
        f.attrs["control_log_columns"] = json.dumps(
            ["step", "field_index", "simulation_time", "px", "py", "pz", "qx", "qy", "qz", "qw"]
            + ["vx", "vy", "vz", "wx", "wy", "wz"]
            + [f"a{i}" for i in range(6)]
            + ["gripper"]
        )
        f.attrs["result"] = json.dumps(
            {
                "trial": 0,
                "status": "geometric_sequence_complete",
                "task_success": True,
                "completed_fields": 4,
                "control_steps": n,
            }
        )
        g = f.create_group("privileged_audit")
        g.attrs["object_names"] = '["box"]'
        g.attrs["not_policy_input"] = True
        g.attrs["goal_predicates"] = '[["in", "box", "basket"]]'
        g["object_pose"], g["bilateral_finger_contact"], g["goal_predicate_values"] = pose, contact, values
        images = f.create_group("audit_observations")
        images.create_group("5")
        images.create_group("10")
    return path


def test_source_operation_candidates_are_unverified_and_use_only_past_images(audit_log):
    result = summarize_trajectory(audit_log)
    lifted, placed = result["candidates"]
    assert lifted["kind"] == "lifted_contact_candidate"
    assert lifted["control_step"] == 9
    assert lifted["causal_image_group"] == "audit_observations/5"
    assert lifted["causal_image_age_seconds"] == pytest.approx(0.2)
    assert placed["kind"] == "placement_candidate"
    assert placed["control_step"] == 21
    assert all(not c["verified"] and not c["observable"] for c in result["candidates"])
    assert result["objects"][0]["bilateral_contact_samples"] == 12


@pytest.mark.parametrize("reason", ["brief", "no_lift", "open", "slip", "rotate", "short_time"])
def test_contact_alone_never_certifies_lifted_candidate(audit_log, reason):
    with h5py.File(audit_log, "r+") as f:
        if reason == "brief":
            f["privileged_audit/bilateral_finger_contact"][7:] = False
        elif reason == "no_lift":
            f["privileged_audit/object_pose"][:, 0, 2] = 0
        elif reason == "open":
            f["control_log"][:, -1] = -1
        elif reason == "slip":
            f["control_log"][:, 3] = np.arange(24) * 0.005
        elif reason == "rotate":
            f["privileged_audit/object_pose"][:, 0, 3:] = Rotation.from_rotvec(
                np.arange(24)[:, None] * np.array([0, 0, 0.1])
            ).as_quat()
        elif reason == "short_time":
            f["control_log"][:, 2] = np.arange(24) * 0.01
    result = summarize_trajectory(audit_log)
    assert not any(c["kind"] == "lifted_contact_candidate" for c in result["candidates"])


def test_legacy_audit_without_predicates_remains_readable(audit_log):
    with h5py.File(audit_log, "r+") as f:
        del f["privileged_audit/goal_predicate_values"]
    result = summarize_trajectory(audit_log)
    assert not result["predicate_trace_available"]
    assert len(result["candidates"]) == 1


def test_rejects_missing_control_steps_and_policy_truth(audit_log):
    with h5py.File(audit_log, "r+") as f:
        f["control_log"][7, 0] = 99
    with pytest.raises(ValueError, match="consecutive"):
        summarize_trajectory(audit_log)
    with h5py.File(audit_log, "r+") as f:
        f["control_log"][7, 0] = 8
        f["privileged_audit"].attrs["not_policy_input"] = False
    with pytest.raises(ValueError, match="isolated"):
        summarize_trajectory(audit_log)


def test_mismatched_source_cannot_be_reported_as_paired_comparison(audit_log, tmp_path):
    paths = []
    for mode, seed in [("fields", 1), ("source_actions", 2)]:
        directory = tmp_path / mode
        directory.mkdir()
        (directory / audit_log.name).write_bytes(audit_log.read_bytes())
        (directory / "request.json").write_text(
            json.dumps(
                {
                    "source_episode": {"signature": "same"},
                    "source_episode_index": 200,
                    "seed": seed,
                    "source_manifest_sha256": "same",
                    "source_initial_position": [0, 0, 0],
                }
            )
        )
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mode": mode,
                    "sequence_results": [{"trajectory_file": audit_log.name}],
                }
            )
        )
        paths.append(directory)
    with pytest.raises(ValueError, match="seed"):
        audit_fields(SimpleNamespace(execution=paths[0], compare_source=paths[1], output=tmp_path / "out.json"))
    assert not (tmp_path / "out.json").exists()


def test_evidence_thresholds_are_positive():
    for name in dataclasses.asdict(EvidenceConfig()):
        with pytest.raises(ValueError, match="positive"):
            EvidenceConfig(**{name: 0})


def test_placed_fact_is_not_reemitted_when_gripper_operates_on_another_object(audit_log):
    with h5py.File(audit_log, "r+") as f:
        f["privileged_audit/goal_predicate_values"][8:] = True
        f["control_log"][8:13, -1] = -1
        f["privileged_audit/bilateral_finger_contact"][8:13] = False
    result = summarize_trajectory(audit_log)
    placed = [c for c in result["candidates"] if c["kind"] == "placement_candidate"]
    assert len(placed) == 1
    assert placed[0]["control_step"] == 12
