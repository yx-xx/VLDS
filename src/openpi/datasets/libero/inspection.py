"""Plot one processed episode's real images, trajectory, events and phase coverage."""

from __future__ import annotations

import collections
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlds-matplotlib")

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.datasets.libero import SEGMENT_COLORS
from openpi.datasets.libero.reader import ProcessedLiberoDataset
from openpi.datasets.libero.reader import atomic_json
from openpi.datasets.libero.reader import read_record


def plot_episode(root, index, output):
    dataset = ProcessedLiberoDataset(root)
    file, episode = dataset._open(index)  # noqa: SLF001 -- single-episode report owns this handle.
    try:
        time = episode["time"][:]
        position = episode["robot_state/position"][:]
        grip = episode["robot_state/gripper_position"][:]
        speed = np.linalg.norm(episode["robot_state/linear_velocity"][:], axis=1)
        events = read_record(episode, "events")
        phases = read_record(episode, "phases")
        indices = sorted({1, len(time) // 2, len(time) - 1})
        fig = plt.figure(figsize=(15, 10), layout="constrained")
        grid = fig.add_gridspec(4, 6, height_ratios=[1.4, 1.4, 1.6, 1.6])
        for row, camera in enumerate(("agentview_rgb", "eye_in_hand_rgb")):
            for column, frame in enumerate(indices):
                ax = fig.add_subplot(grid[row, 2 * column : 2 * column + 2])
                # Display-only OpenGL vertical flip; stored pixels are unchanged.
                ax.imshow(episode[f"images/{camera}"][frame][::-1])
                ax.set_title(f"{camera} / O[{frame}] / {time[frame]:.2f} s", fontsize=10)
                ax.axis("off")
        spatial = fig.add_subplot(grid[2:, :2], projection="3d")
        spatial.plot(*position.T, lw=1.5)
        spatial.scatter(*position[0], color="green", label="start")
        spatial.scatter(*position[-1], color="red", label="final observation")
        spatial.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)")
        spatial.legend(fontsize=8)
        arm = fig.add_subplot(grid[2, 2:])
        for column, axis in enumerate("xyz"):
            arm.plot(time, position[:, column], label=axis)
        arm.set(ylabel="EEF position (m)", xlabel="time (s)")
        arm.legend(loc="upper right", ncols=3)
        gripper = fig.add_subplot(grid[3, 2:])
        gripper.plot(time, np.abs(grip[:, 0] - grip[:, 1]), label="measured finger separation (m)")
        gripper.step(
            time[:-1],
            (episode["actions"][:, -1] + 1) * 0.04,
            where="post",
            label="command (scaled for display)",
            alpha=0.65,
        )
        gripper.set(xlabel="time (s)", ylabel="gripper separation (m)")
        other = gripper.twinx()
        other.plot(time, speed, color="grey", alpha=0.5, label="measured EEF speed")
        other.set_ylabel("EEF speed (m/s)")
        for phase in phases:
            if phase["kind"] != "motion":
                for ax in (arm, gripper):
                    ax.axvspan(time[phase["start"]], time[phase["end"]], color="orange", alpha=0.15)
        for event in events:
            gripper.axvline(time[event["onset"]], color="black", ls="--", lw=0.8)
            gripper.text(time[event["onset"]], 0.01, event["event"], fontsize=8, rotation=90)
        gripper.legend(loc="upper left", fontsize=8)
        entry = dataset.episodes[index]
        fig.suptitle(
            f"{entry['suite']} / {entry['group']} / {entry['split']}\n{entry['language']}\n"
            "Orange: operation / coupled motion (retained). Field and goal supervision unverified.",
            fontsize=12,
        )
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return output
    finally:
        file.close()


def plot_segments(root, index, output):
    """Show complete command intervals; do not draw overlapping candidates as a chosen path."""
    dataset = ProcessedLiberoDataset(root)
    file, episode = dataset._open(index)  # noqa: SLF001 -- report owns the validated file handle.
    try:
        if "command_intervals" not in episode:
            raise ValueError("Command-interval visualization requires the v2 processed dataset")
        intervals = read_record(episode, "command_intervals")
        time = episode["time"][:]
        position = episode["robot_state/position"][:]
        grip = episode["robot_state/gripper_position"][:]
        command = episode["gripper_command"][:]
        fig = plt.figure(figsize=(14, 8), layout="constrained")
        grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.3], height_ratios=[1, 1, 0.65], wspace=0.15)
        spatial = fig.add_subplot(grid[:, 0], projection="3d")
        axes = [fig.add_subplot(grid[row, 1]) for row in range(3)]
        for index_in_episode, interval in enumerate(intervals):
            start, end = interval["start"], interval["end"]
            color = SEGMENT_COLORS[index_in_episode % len(SEGMENT_COLORS)]
            label = f"{index_in_episode + 1}: {interval['command_label']} [{start}, {end})"
            spatial.plot(*position[start : end + 1].T, color=color, lw=2.5, label=label)
            spatial.scatter(*position[start], color=color, s=28)
            spatial.text(*position[start], f" {index_in_episode + 1}", fontsize=9)
            for axis in axes:
                axis.axvspan(time[start], time[end], color=color, alpha=0.16)
                axis.axvline(time[start], color=color, lw=0.8)
            axes[2].plot([time[start], time[end]], [index_in_episode + 1] * 2, color=color, lw=8)
        spatial.scatter(*position[-1], color="#17221f", marker="x", s=50, label="final observation")
        spatial.set(
            xlabel="world x (m)",
            ylabel="world y (m)",
            zlabel="world z (m)",
            title="EEF path colored by command interval",
        )
        spatial.legend(loc="upper left", fontsize=8)
        axes[0].step(time, np.r_[command, command[-1]], where="post", color="#17221f", lw=1.4)
        axes[0].set(
            yticks=[-1, 1], yticklabels=["OPEN", "CLOSE"], title="Original gripper command; short pulses retained"
        )
        axes[1].plot(time, np.abs(grip[:, 0] - grip[:, 1]), color="#17221f", lw=1.4)
        axes[1].set(ylabel="finger separation (m)", title="Measured feedback; settling does not imply grasp success")
        axes[2].set(ylabel="interval", xlabel="time (s)", yticks=range(1, len(intervals) + 1))
        for axis in axes:
            axis.set_xlim(time[0], time[-1])
            axis.grid(alpha=0.15)
        entry = dataset.episodes[index]
        fig.suptitle(
            f"{entry['suite']} / {entry['group']} / {entry['split']}\n{entry['language']}\n"
            "Command intervals cover all source actions. F fitting and final field segmentation have not run.",
            fontsize=12,
        )
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return output
    finally:
        file.close()


def plot_summary(root, output):
    dataset = ProcessedLiberoDataset(root)
    manifest = dataset.manifest
    by_shard = collections.defaultdict(list)
    for entry in dataset.episodes:
        by_shard[entry["shard"]].append(entry)
    phase_actions = collections.Counter()
    feedback = collections.defaultdict(collections.Counter)
    for shard, entries in by_shard.items():
        with h5py.File(dataset.root / shard) as file:
            for entry in entries:
                episode = file[entry["group"]]
                for phase in read_record(episode, "phases"):
                    phase_actions[phase["kind"]] += phase["end"] - phase["start"]
                for event in read_record(episode, "events"):
                    status = "settled" if event["feedback_settled"] is not None else "unconfirmed"
                    feedback[event["event"]][status] += 1
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    suites = sorted(manifest["suite_splits"])
    left = np.zeros(len(suites))
    for split, color in (("train", "#398568"), ("validation", "#82b5a0"), ("test", "#cbded6")):
        values = np.array([manifest["suite_splits"][suite][split] for suite in suites])
        axes[0, 0].barh(suites, values, left=left, label=split, color=color)
        left += values
    axes[0, 0].set(title="Whole-episode split", xlabel="episodes")
    axes[0, 0].legend()
    errors = np.array([e["shifted_position_max_m"] * 1000 for e in dataset.episodes])
    axes[0, 1].hist(errors, bins=40, color="#398568")
    axes[0, 1].set(
        title="Independent state/observation alignment",
        xlabel="per-episode sampled maximum error (mm)",
        ylabel="episodes",
    )
    kinds = ["motion", "coupled_operation", "operation"]
    labels = ["motion intervals", "coupled operation", "stationary operation"]
    shares = np.array([phase_actions[k] / manifest["counts"]["actions"] * 100 for k in kinds])
    bars = axes[1, 0].barh(labels, shares, color=["#398568", "#ddaa54", "#a9bbc5"])
    axes[1, 0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[1, 0].set(title="All source actions retained once", xlabel="share of source actions (%)", xlim=(0, 100))
    bottom = np.zeros(2)
    for status, color in (("settled", "#398568"), ("unconfirmed", "#ddaa54")):
        values = np.array([feedback[event][status] for event in ("OPEN", "CLOSE")])
        axes[1, 1].bar(["OPEN", "CLOSE"], values, bottom=bottom, label=status, color=color)
        bottom += values
    axes[1, 1].set(title="Feedback settling is not grasp/task success", ylabel="command events")
    axes[1, 1].legend()
    counts = manifest["counts"]
    fig.suptitle(
        f"LIBERO preprocessing: {counts['episodes']:,} episodes / {counts['actions']:,} actions\n"
        f"{counts['coupled_operations']:,}/{counts['events']:,} command events involve arm motion; "
        "fitted field supervision remains unverified.",
        fontsize=13,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    atomic_json(
        dataset.root / "quality_analysis.json",
        {
            "phase_action_counts": dict(phase_actions),
            "phase_action_percentages": dict(zip(kinds, shares.tolist(), strict=True)),
            "event_feedback": {key: dict(value) for key, value in feedback.items()},
            "all_actions_accounted_for": sum(phase_actions.values()) == counts["actions"],
            "coupled_event_fraction": counts["coupled_operations"] / counts["events"],
            "scope": "descriptive source-data diagnostics using frozen preprocessing thresholds; not fitted-field performance",
        },
    )
    return output


def inspect(args):
    if args.segments:
        print(plot_segments(args.dataset, args.episode, args.output or args.dataset / "segmentation.png"))
    elif args.summary:
        print(plot_summary(args.dataset, args.output or args.dataset / "quality_summary.png"))
    else:
        print(plot_episode(args.dataset, args.episode, args.output or args.dataset / "preview.png"))
    return 0
