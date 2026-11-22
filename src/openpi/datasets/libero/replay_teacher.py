"""Use a successful source-action replay as an explicitly derived pose teacher.

Original demonstrations, commands, episode signatures and splits stay intact.
Only actual proprioceptive poses / their derivatives enter the derived artifact;
object poses and task predicates are excluded. Task success gates this offline
source selection, without supplying a runtime fact observer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.datasets.libero import trajectory


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(data, execution, output_file, manifest_sha256):
    execution = Path(execution)
    request_file, result_file = execution / "request.json", execution / "result.json"
    request, result = json.loads(request_file.read_text()), json.loads(result_file.read_text())
    if result["status"] != "complete" or result["mode"] != "source_actions":
        raise ValueError("Derived teacher requires a completed source-action replay")
    if request["source_episode"]["signature"] != data["entry"]["signature"]:
        raise ValueError("Replay teacher source signature differs")
    if request["source_manifest_sha256"] != manifest_sha256:
        raise ValueError("Replay teacher source manifest differs")
    trials = result["sequence_results"]
    if len(trials) != 1 or not trials[0]["task_success"] or trials[0]["status"] != "complete":
        raise ValueError("Derived teacher requires exactly one complete successful source replay")
    trial = trials[0]
    if trial["intermediate_resets"] or trial["control_steps"] != len(data["actions"]):
        raise ValueError("Replay teacher must preserve the entire uninterrupted source episode")
    path = execution / trial["trajectory_file"]
    with h5py.File(path) as archive:
        log = archive["control_log"][:]
        columns = json.loads(archive.attrs["control_log_columns"])
    expected = ["step", "field_index", "simulation_time", "px", "py", "pz", "qx", "qy", "qz", "qw"]
    if columns[:10] != expected or log.shape != (len(data["actions"]), 23) or not np.isfinite(log).all():
        raise ValueError("Malformed source replay control log")
    if not np.array_equal(log[:, 0], np.arange(1, len(log) + 1)):
        raise ValueError("Replay teacher samples are not consecutive")
    if not np.allclose(log[:, -7:], data["actions"], atol=1e-12, rtol=0):
        raise ValueError("Replay actions differ from the original demonstration")
    if not np.allclose(request["body_to_control_rotation"], data["body_to_control_rotation"], atol=1e-8, rtol=0):
        raise ValueError("Replay control frame differs from the source")
    dt = request["control"]["dt"]
    times = np.r_[0.0, log[:, 2] - log[0, 2] + dt]
    if not np.allclose(times, data["time"], atol=1e-8, rtol=0):
        raise ValueError("Replay timestamps differ from source action alignment")
    initial_p = np.asarray(request["source_initial_position"])
    if not np.allclose(initial_p, data["trajectory"]["position"][0], atol=1e-8, rtol=0):
        raise ValueError("Replay initial position differs from the source")
    position = np.vstack([initial_p, log[:, 3:6]])
    # Replay uses control-point orientation; the common fitting reader stores hand orientation.
    hand = Rotation.from_quat(log[:, 6:10]).as_matrix() @ data["body_to_control_rotation"].T
    quaternion = np.vstack([data["trajectory"]["quaternion_xyzw"][0], Rotation.from_matrix(hand).as_quat()])
    linear, angular = trajectory.measured_derivatives(position, quaternion, times)
    output_file = Path(output_file)
    arrays = {
        "position": position,
        "quaternion_xyzw": quaternion,
        "linear_velocity": linear,
        "angular_velocity": angular,
    }
    np.savez_compressed(output_file, **arrays)
    difference = np.linalg.norm(position - data["trajectory"]["position"], axis=1)
    return {
        "kind": "successful_source_action_replay",
        "file": output_file.name,
        "sha256": digest(output_file),
        "source_signature": data["entry"]["signature"],
        "request_sha256": digest(request_file),
        "result_sha256": digest(result_file),
        "trajectory_sha256": digest(path),
        "initial_pose_basis": "source FK pose; full-environment worker verifies initial position and frame transform",
        "position_difference_rms_metres": float(np.sqrt(np.mean(difference**2))),
        "position_difference_maximum_metres": float(difference.max()),
        "training_ready": False,
    }


def apply(data, directory, metadata):
    if not metadata:
        return data
    path = Path(directory) / metadata["file"]
    if digest(path) != metadata["sha256"] or metadata["source_signature"] != data["entry"]["signature"]:
        raise ValueError("Derived teacher artifact or source signature mismatch")
    with np.load(path, allow_pickle=False) as archive:
        expected = {"position": 3, "quaternion_xyzw": 4, "linear_velocity": 3, "angular_velocity": 3}
        arrays = {k: archive[k] for k in expected}
        if any(a.shape != (len(data["time"]), expected[k]) or not np.isfinite(a).all() for k, a in arrays.items()):
            raise ValueError("Invalid derived teacher arrays")
    rotations = Rotation.from_quat(arrays["quaternion_xyzw"])
    arrays.update(
        rotvec=rotations.as_rotvec(), orientation_6d=rotations.as_matrix()[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    )
    return {**data, "trajectory": {**data["trajectory"], **arrays}, "teacher_source": metadata}


def field_input(dataset, index, directory):
    run = json.loads((Path(directory) / "run.json").read_text())
    return apply(dataset.field_input(index), directory, run["config"].get("teacher_source"))


def copy_artifact(source, destination, metadata):
    if metadata:
        path = Path(source) / metadata["file"]
        if digest(path) != metadata["sha256"]:
            raise ValueError("Cannot copy a stale derived teacher")
        (Path(destination) / metadata["file"]).write_bytes(path.read_bytes())
