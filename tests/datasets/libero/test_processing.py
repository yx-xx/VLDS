import dataclasses
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from openpi.datasets.libero import processing
from openpi.datasets.libero import trajectory
from openpi.datasets.libero.reader import ProcessedLiberoDataset


class FakeKinematics:
    """Independent kinematic relationship for exercising the real HDF5 conversion."""

    def __init__(self, *_args):
        pass

    def observe(self, state):
        elapsed = state[0] - 10
        return {
            "position": np.array([elapsed, 0, 1]),
            "rotvec": np.zeros(3),
            "joint_position": np.full(7, elapsed),
            "gripper_position": np.array([0.04, -0.04]),
            "body_to_control_rotation": np.eye(3),
        }


def make_source(root, *, count=10):
    source = root / "raw/libero_spatial/example_demo.hdf5"
    source.parent.mkdir(parents=True)
    with h5py.File(source, "w") as file:
        data = file.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "move the bowl"})
        data.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        for index in range(3):
            group = data.create_group(f"demo_{index}")
            group.attrs["model_file"] = "<mujoco/>"
            clock = 10 + np.arange(count) * 0.05
            states = np.c_[clock, np.full(count, index), np.ones(count)]
            group["states"] = states
            action = np.zeros((count, 7))
            action[:, 0] = index
            action[:, -1] = -1
            action[4:, -1] = 1
            group["actions"] = action
            elapsed = clock + 0.05 - 10
            group["obs/ee_pos"] = np.c_[elapsed, elapsed * 0, elapsed * 0 + 1]
            group["obs/ee_ori"] = np.zeros((count, 3))
            group["obs/joint_states"] = np.tile(elapsed[:, None], (1, 7))
            group["obs/gripper_states"] = np.tile([0.04, -0.04], (count, 1))
            images = np.broadcast_to(np.arange(count, dtype=np.uint8)[:, None, None, None], (count, 4, 4, 3))
            group["obs/agentview_rgb"] = images
            group["obs/eye_in_hand_rgb"] = images + 10
    return source


def job_for(source, root):
    return {
        "source_file": str(source),
        "source_root": str(source.parent.parent),
        "output_file": str(root / "processed/libero_spatial/example_demo.hdf5"),
        "config": dataclasses.asdict(trajectory.ProcessingConfig()),
        "pipeline_sha256": "test-pipeline",
        "libero_root": "unused",
        "robosuite_root": "unused",
        "source_revision": "test-revision",
        "resource_versions": {},
        "resume": True,
    }


def test_real_hdf5_pipeline_aligns_actions_and_keeps_terminal_state(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    monkeypatch.setattr(processing, "Kinematics", FakeKinematics)
    job = job_for(source, tmp_path)
    result = processing.process_file(job)
    assert not result["rejected"]
    assert len(result["episodes"]) == 3
    run = {"file_errors": [], "config": job["config"], "started_at": "test", "command": "test command"}
    manifest = processing.aggregate(tmp_path / "processed", [result], run)
    assert manifest["counts"]["actions"] == 30
    assert manifest["counts"]["observations"] == 33
    assert manifest["counts"]["alignment_pass"] == 3
    assert manifest["counts"]["command_intervals"] == 6
    assert manifest["candidate_action_coverage_complete"]
    assert not manifest["fitted_field_dataset_ready"]
    dataset = ProcessedLiberoDataset(tmp_path / "processed")
    first = dataset.transition(0, 0, images=True)
    np.testing.assert_allclose(first["before"]["position"], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(first["after"]["position"], [0.05, 0, 1])
    np.testing.assert_allclose(first["before"]["linear_velocity"], [1, 0, 0])
    assert first["before"]["images"]["agentview_rgb"] is None
    assert not first["before"]["image_valid"]
    last = dataset.transition(0, 9, images=True)
    np.testing.assert_allclose(last["after"]["position"], [0.5, 0, 1])
    np.testing.assert_array_equal(last["after"]["images"]["agentview_rgb"], 9)
    assert last["task_success"] is None
    audit = dataset.audit()
    assert audit["passed"], audit
    assert audit["counts"]["candidate_covered_actions"] == 30
    candidates = dataset.candidate(0, 0)
    assert candidates["command_interval"]["command"] == -1
    assert len(candidates["time"]) == len(candidates["actions"]) + 1
    assert not candidates["goal_candidate_verified"]
    assert processing.process_file(job)["resumed"]
    with h5py.File(source, "a") as file:
        file["data/demo_0/actions"][0, 0] = 9
    with pytest.raises(ValueError, match="Raw source has changed"):
        dataset.transition(0, 0)
    assert not processing.process_file(job)["resumed"]


def test_time_gap_is_rejected_with_source_episode_retained(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    with h5py.File(source, "a") as file:
        state = file["data/demo_1/states"][:]
        state[5:, 0] += 0.05
        file["data/demo_1/states"][:] = state
    monkeypatch.setattr(processing, "Kinematics", FakeKinematics)
    result = processing.process_file(job_for(source, tmp_path))
    assert len(result["episodes"]) == 2
    assert result["rejected"][0]["episode"] == "demo_1"
    assert "time gaps" in result["rejected"][0]["reason"]
    with h5py.File(source) as file:
        assert len(file["data"]) == 3  # original remains untouched


def test_missing_linked_source_raises_instead_of_returning_zero_images(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    monkeypatch.setattr(processing, "Kinematics", FakeKinematics)
    job = job_for(source, tmp_path)
    result = processing.process_file(job)
    processing.aggregate(
        tmp_path / "processed",
        [result],
        {"file_errors": [], "config": job["config"], "started_at": "test", "command": "test"},
    )
    source.rename(source.with_suffix(".moved"))
    dataset = ProcessedLiberoDataset(tmp_path / "processed")
    with pytest.raises(FileNotFoundError, match="linked raw source"):
        dataset.transition(0, 0, images=True)
    assert not dataset.audit()["passed"]


def test_normalization_excludes_validation_and_test(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    monkeypatch.setattr(processing, "Kinematics", FakeKinematics)
    job = job_for(source, tmp_path)
    result = processing.process_file(job)
    output = Path(job["output_file"])
    with h5py.File(output, "a") as file:
        for i, summary in enumerate(result["episodes"]):
            split = ["train", "validation", "test"][i]
            summary["split"] = split
            group = file[f"episodes/{summary['episode']}"]
            group.attrs["split"] = split
            group["robot_state/position"][:] = [2 if i == 0 else 99999, 0, 0]
    processing.aggregate(
        tmp_path / "processed",
        [result],
        {"file_errors": [], "config": job["config"], "started_at": "test", "command": "test"},
    )
    norm = json.loads((tmp_path / "processed/normalization.json").read_text())
    assert norm["fields"]["position"]["count"] == 11
    assert norm["fields"]["position"]["mean"] == [2, 0, 0]
    assert len(norm["training_episode_signatures"]) == 1


def test_kinematic_xml_retains_joint_order_and_transforms_without_geometry():
    import mujoco

    xml = """<mujoco>
      <compiler angle="radian"/>
      <default><joint damping="1"/><geom size="0.1" mass="1"/></default>
      <worldbody>
        <body name="floating" pos="0.2 0.3 0.4"><freejoint/>
          <geom type="sphere"/><site name="free_site" pos="0.1 0 0"/>
        </body>
        <body name="arm" pos="0.3 -0.4 1"><joint name="hinge" axis="0 0 1"/>
          <geom type="sphere"/><body pos="0.2 0 0"><joint name="slide" type="slide" axis="1 0 0"/>
            <geom type="sphere"/><site name="tip" pos="0.1 0.2 0.3"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor joint="hinge"/></actuator>
    </mujoco>"""
    full = mujoco.MjModel.from_xml_string(xml)
    reduced = mujoco.MjModel.from_xml_string(processing.kinematic_model_xml(xml))
    assert (full.nq, full.nv, full.na, full.nsite) == (reduced.nq, reduced.nv, reduced.na, reduced.nsite)
    np.testing.assert_array_equal(full.jnt_qposadr, reduced.jnt_qposadr)
    assert reduced.ngeom == 0
    for displacement in (0, 0.1, 0.9):
        full_data, reduced_data = mujoco.MjData(full), mujoco.MjData(reduced)
        values = full.qpos0.copy()
        values[-2:] = [displacement, displacement * 2]
        full_data.qpos[:] = reduced_data.qpos[:] = values
        mujoco.mj_forward(full, full_data)
        mujoco.mj_forward(reduced, reduced_data)
        np.testing.assert_allclose(full_data.site_xpos, reduced_data.site_xpos, atol=1e-14)
        np.testing.assert_allclose(full_data.site_xmat, reduced_data.site_xmat, atol=1e-14)


@pytest.mark.parametrize("corruption", ["missing_candidate", "changed_command", "wrong_window_command"])
def test_audit_detects_incomplete_fitting_inputs(tmp_path, monkeypatch, corruption):
    source = make_source(tmp_path)
    monkeypatch.setattr(processing, "Kinematics", FakeKinematics)
    job = job_for(source, tmp_path)
    result = processing.process_file(job)
    processing.aggregate(
        tmp_path / "processed",
        [result],
        {"file_errors": [], "config": job["config"], "started_at": "test", "command": "test"},
    )
    with h5py.File(job["output_file"], "a") as file:
        episode = file["episodes/demo_0"]
        if corruption == "missing_candidate":
            candidates = json.loads(episode["candidates"][()].decode())
            episode["candidates"][()] = json.dumps([c for c in candidates if c["command_interval_id"] != 0])
        elif corruption == "changed_command":
            episode["gripper_command"][0] = 1
        else:
            episode["proposal_windows/H3/gripper_commands"][0, 0] = 1
    audit = ProcessedLiberoDataset(tmp_path / "processed").audit()
    assert not audit["passed"]
    assert len(audit["errors"]) == 1
