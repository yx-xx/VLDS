"""Plot a field execution against its frozen goals and source demonstration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlds-matplotlib")

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.datasets.libero import replay_teacher
from openpi.datasets.libero.reader import ProcessedLiberoDataset
from openpi.datasets.libero.reader import atomic_json


def inspect_fields(args):
    fields = json.loads(args.fields.read_text())
    request = json.loads((args.execution / "request.json").read_text())
    result = json.loads((args.execution / "result.json").read_text())
    trial = result["sequence_results"][args.trial]
    data = replay_teacher.field_input(
        ProcessedLiberoDataset(args.dataset), fields["source_episode_index"], args.fields.parent
    )
    if request["source_episode"]["signature"] != data["entry"]["signature"]:
        raise ValueError("Execution and source episode mismatch")
    if request["offline_result_sha256"] != hashlib.sha256(args.fields.read_bytes()).hexdigest():
        raise ValueError("Execution used a different offline field artifact")
    with h5py.File(args.execution / trial["trajectory_file"]) as recording:
        log = recording["control_log"][:]
        queries = {
            key: {name: value[:] for name, value in group.items()}
            for key, group in recording["queries"].items()
            if key.isdigit()
        }
    times = log[:, 0] * request["control"]["dt"]
    indices = log[:, 1].astype(int)
    goals = np.array([record["goal_position"] for record in request["sequences"][args.trial]])
    error = np.linalg.norm(log[:, 3:6] - goals[indices], axis=1)
    speed = np.linalg.norm(log[:, 10:13], axis=1)
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.12, wspace=0.20)
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.1])
    spatial = fig.add_subplot(grid[:, 0], projection="3d")
    spatial.plot(
        *data["trajectory"]["position"].T,
        color="0.65",
        label="successful source-action replay" if data.get("teacher_source") else "source demonstration",
    )
    spatial.plot(*log[:, 3:6].T, color="#18794e", label="continuous robot execution")
    spatial.scatter(*goals.T, marker="x", s=35, color="#dc6c2c", label="fixed field goals")
    spatial.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)")
    spatial.legend(fontsize=8, loc="upper left")
    error_axis = fig.add_subplot(grid[0, 1])
    error_axis.plot(times, error * 1000, color="#18794e")
    error_axis.axhline(
        request["control"]["position_tolerance"] * 1000, color="#dc6c2c", ls="--", label="arrival position threshold"
    )
    error_axis.set(ylabel="distance to active goal (mm)", xlabel="execution time (s)")
    error_axis.legend(fontsize=8)
    speed_axis = fig.add_subplot(grid[1, 1], sharex=error_axis)
    speed_axis.plot(times, speed, color="#18794e", label="measured control-point speed")
    speed_axis.axhline(
        request["control"]["max_speed"], color="0.6", ls="--", label="field speed cap (not a measured bound)"
    )
    speed_axis.set(ylabel="speed (m/s)", xlabel="execution time (s)")
    speed_axis.legend(fontsize=8)
    command_axis = fig.add_subplot(grid[2, 1], sharex=error_axis)
    command_axis.step(times, indices, where="post", color="#18794e", label="active field")
    command_axis.set(ylabel="field index", xlabel="execution time (s)")
    second = command_axis.twinx()
    second.step(times, log[:, -1], where="post", color="#dc6c2c", alpha=0.8)
    second.set(ylim=(-1.2, 1.2), yticks=[-1, 1], yticklabels=["OPEN", "CLOSE"])
    fig.suptitle(
        f"Episode {fields['source_episode_index']} / trial {args.trial}: {trial['status']}\n"
        f"task predicate={trial['task_success']}; fields completed={trial['completed_fields']}/{len(goals)}; "
        "operation observer unavailable",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    selected = np.unique(np.linspace(0, len(queries) - 1, min(5, len(queries))).astype(int))
    if len(selected):
        image_fig, axes = plt.subplots(
            2, len(selected), figsize=(3 * len(selected), 5), squeeze=False, layout="constrained"
        )
        for column, index in enumerate(selected):
            query = queries[str(index)]
            for row, camera in enumerate(("agentview_image", "robot0_eye_in_hand_image")):
                axes[row, column].imshow(query[camera][::-1])
                axes[row, column].set_title(f"before field {index}: {camera}", fontsize=8)
                axes[row, column].axis("off")
        image_fig.savefig(args.output.with_name(args.output.stem + "_queries.png"), dpi=150)
        plt.close(image_fig)
    summary = {
        "source_episode_index": fields["source_episode_index"],
        "trial": args.trial,
        "task_success": trial["task_success"],
        "geometric_sequence_status": trial["status"],
        "completed_fields": trial["completed_fields"],
        "planned_fields": len(goals),
        "measured_speed_max_m_per_second": float(speed.max()),
        "field_artifact_sha256": hashlib.sha256(args.fields.read_bytes()).hexdigest(),
        "execution_artifact_sha256": hashlib.sha256(
            (args.execution / trial["trajectory_file"]).read_bytes()
        ).hexdigest(),
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "teacher_source": data.get("teacher_source"),
    }
    atomic_json(args.output.with_suffix(".json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
