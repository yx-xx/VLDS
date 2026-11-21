"""Lazy reading and independent integrity checks for the derived LIBERO dataset."""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import h5py
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_record(group, name):
    return json.loads(group[name][()].decode())


class ProcessedLiberoDataset:
    """Open one episode at a time; preserve explicit missing observation masks.

    This is a trajectory/proposal reader. It does not advertise unverified field
    proposals as training labels, and never returns future windows as policy input.
    """

    def __init__(self, root, split=None):
        self.root = Path(root).expanduser().resolve()
        run_file = self.root / "run.json"
        if run_file.is_file() and json.loads(run_file.read_text()).get("status") == "running":
            raise ValueError("Conversion is still running; wait for atomic publication of its final manifest")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema") not in ("vlds.libero.trajectories.v1", "vlds.libero.trajectories.v2"):
            raise ValueError("Unsupported processed LIBERO schema")
        if split not in (None, "train", "validation", "test"):
            raise ValueError("Unknown dataset split")
        self.episodes = [e for e in self.manifest["episodes"] if split is None or e["split"] == split]

    def __len__(self):
        return len(self.episodes)

    def _open(self, index):
        entry = self.episodes[index]
        file = h5py.File(self.root / entry["shard"], "r")
        try:
            group = file[entry["group"]]
            source = (Path(file.filename).parent / group.attrs["source_file"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"The linked raw source is missing: {source}")
            provenance = read_record(file, "provenance")
            stat = source.stat()
            if stat.st_size != provenance["source_size"] or stat.st_mtime_ns != provenance["source_mtime_ns"]:
                raise ValueError("Raw source has changed since conversion; rebuild the derived dataset")
            # External/virtual HDF5 links otherwise silently return fill values when broken.
            if "actions" not in group["source"]:
                raise ValueError("Raw source link is invalid")
            return file, group
        except Exception:
            file.close()
            raise

    def transition(self, episode_index, action_index, *, images=False):
        file, episode = self._open(episode_index)
        try:
            if not 0 <= action_index < len(episode["actions"]):
                raise IndexError(action_index)

            def observation(index):
                value = {key: field[index] for key, field in episode["robot_state"].items()}
                value["time"] = float(episode["time"][index])
                value["image_valid"] = bool(episode["image_valid"][index])
                value["source_image_index"] = int(episode["source_image_index"][index])
                if images:
                    value["images"] = {
                        key: image[index] if value["image_valid"] else None for key, image in episode["images"].items()
                    }
                return value

            return {
                "before": observation(action_index),
                "action": episode["actions"][action_index],
                "after": observation(action_index + 1),
                "language": file.attrs["full_language"],
                "split": episode.attrs["split"],
                "task_success": None,
            }
        finally:
            file.close()

    def proposals(self, episode_index, horizon=3):
        file, episode = self._open(episode_index)
        try:
            windows = episode[f"proposal_windows/H{horizon}"]
            return {key: value[:] for key, value in windows.items()}
        finally:
            file.close()

    def candidate(self, episode_index, candidate_index):
        """Return a physical-unit trajectory for F; the endpoint is only a proposal."""
        file, episode = self._open(episode_index)
        try:
            candidates = read_record(episode, "candidates")
            candidate = candidates[candidate_index]
            start, end = candidate["start"], candidate["end"]
            fields = {key: value[start : end + 1] for key, value in episode["robot_state"].items()}
            return {
                "trajectory": fields,
                "time": episode["time"][start : end + 1],
                "actions": episode["actions"][start:end],
                "language": file.attrs["full_language"],
                "candidate": candidate,
                "command_interval": (
                    read_record(episode, "command_intervals")[candidate["command_interval_id"]]
                    if "command_interval_id" in candidate
                    else None
                ),
                "operation_events": read_record(episode, "events"),
                "goal_candidate_verified": False,
                "body_to_control_rotation": episode["body_to_control_rotation"][:],
                "frame": "world",
                "split": episode.attrs["split"],
                "source": {"file": episode.attrs["source_file"], "group": episode.attrs["source_group"]},
            }
        finally:
            file.close()

    def audit(self, *, check_images=True):
        """Check all artifacts, source links, partitions, coverage, and mask invariants."""
        errors = []
        counts = collections.Counter()
        signatures = collections.defaultdict(set)
        normalization = json.loads((self.root / "normalization.json").read_text())["fields"]
        training_sums = {}
        for index, entry in enumerate(self.episodes):
            file = None
            try:
                file, episode = self._open(index)
                if episode.attrs["split"] != entry["split"]:
                    raise ValueError("Stored split disagrees with the manifest")
                count = len(episode["actions"])
                if len(episode["time"]) != count + 1 or np.any(np.diff(episode["time"][:]) <= 0):
                    raise ValueError("T/T+1 timestamp invariant failed")
                for name, field in episode["robot_state"].items():
                    if len(field) != count + 1 or not np.isfinite(field[:]).all():
                        raise ValueError(f"Invalid measured state: {name}")
                if episode["image_valid"][0] or not np.all(episode["image_valid"][1:]):
                    raise ValueError("Missing initial image mask is wrong")
                if episode["simulation_state_valid"][-1] or not np.all(episode["simulation_state_valid"][:-1]):
                    raise ValueError("Missing terminal simulation state mask is wrong")
                phases = read_record(episode, "phases")
                coverage = np.zeros(count, dtype=np.int32)
                for phase in phases:
                    if not 0 <= phase["start"] < phase["end"] <= count:
                        raise ValueError("Invalid phase interval")
                    coverage[phase["start"] : phase["end"]] += 1
                if not np.all(coverage == 1):
                    raise ValueError("Phases overlap or discard source actions")
                candidates = read_record(episode, "candidates")
                if self.manifest["schema"].endswith(".v2") and "command_intervals" not in episode:
                    raise ValueError("Missing command intervals in a v2 shard")
                if "command_intervals" in episode:
                    intervals = read_record(episode, "command_intervals")
                    interval_coverage = np.zeros(count, dtype=np.int32)
                    candidate_coverage = np.zeros(count, dtype=bool)
                    raw_command = episode["actions"][:, -1]
                    if not np.allclose(episode["gripper_command"][:], raw_command, atol=1e-6, rtol=0):
                        raise ValueError("A source gripper command was changed or removed")
                    full_intervals = set()
                    for interval_index, interval in enumerate(intervals):
                        start, end = interval["start"], interval["end"]
                        if not 0 <= start < end <= count or interval["command_interval_id"] != interval_index:
                            raise ValueError("Invalid command interval")
                        if not np.allclose(raw_command[start:end], interval["command"], atol=1e-6, rtol=0):
                            raise ValueError("A command interval crosses a raw command change")
                        interval_coverage[start:end] += 1
                    if not np.all(interval_coverage == 1):
                        raise ValueError("Command intervals overlap or discard source actions")
                    for candidate in candidates:
                        interval = intervals[candidate["command_interval_id"]]
                        start, end = candidate["start"], candidate["end"]
                        if not interval["start"] <= start < end <= interval["end"]:
                            raise ValueError("A candidate crosses a command change")
                        if candidate["command"] != interval["command"]:
                            raise ValueError("Candidate command disagrees with its interval")
                        candidate_coverage[start:end] = True
                        if start == interval["start"] and end == interval["end"]:
                            full_intervals.add(interval["command_interval_id"])
                    if len(full_intervals) != len(intervals) or not np.all(candidate_coverage):
                        raise ValueError("Fitting candidates do not cover all source actions")
                    counts["candidate_covered_actions"] += count
                    counts["command_intervals"] += len(intervals)
                else:
                    for candidate in candidates:
                        phase = phases[candidate["phase_index"]]
                        if (
                            phase["kind"] != "motion"
                            or not phase["start"] <= candidate["start"] < candidate["end"] <= phase["end"]
                        ):
                            raise ValueError("A legacy candidate crosses a gripper operation")
                for windows in episode["proposal_windows"].values():
                    mask = windows["proposal_mask"][:]
                    if np.any(np.diff(mask.astype(int), axis=1) > 0):
                        raise ValueError("Window padding is not a suffix")
                    if not np.array_equal(mask.sum(axis=1), windows["valid_length"][:]):
                        raise ValueError("Window valid length is inconsistent")
                    if "command_intervals" in episode:
                        refs = windows["command_interval_refs"][:]
                        if mask.shape[0] != len(intervals):
                            raise ValueError("Source query count disagrees with command intervals")
                        if not np.array_equal(windows["query_observation"][:], [i["start"] for i in intervals]):
                            raise ValueError("Query observations must precede the new command")
                        expected_refs = np.arange(len(intervals))[:, None] + np.arange(mask.shape[1])
                        expected_mask = expected_refs < len(intervals)
                        if not np.array_equal(mask, expected_mask) or not np.array_equal(
                            refs[mask], expected_refs[mask]
                        ):
                            raise ValueError("Window command interval order is wrong")
                        commands = windows["gripper_commands"][:]
                        if np.any(refs[~mask] != -1) or np.any(commands[~mask] != 0):
                            raise ValueError("Invalid window padding")
                        expected_commands = np.array([i["command"] for i in intervals])[refs[mask]]
                        if not np.array_equal(commands[mask], expected_commands):
                            raise ValueError("Window command disagrees with its interval")
                        targets = windows["goal_candidates"][:]
                        endpoints = np.array([i["end"] for i in intervals])
                        all_goals = np.c_[
                            episode["robot_state/position"][:][endpoints],
                            episode["robot_state/quaternion_xyzw"][:][endpoints],
                        ]
                        if np.any(targets[~mask] != 0) or not np.array_equal(targets[mask], all_goals[refs[mask]]):
                            raise ValueError("Window endpoint does not match the source observation")
                        if not np.array_equal(
                            windows["source_end_in_window"][:], expected_refs[:, -1] >= len(intervals) - 1
                        ):
                            raise ValueError("Source window endpoint marker is wrong")
                    for label in ("field_supervision_mask", "goal_supervision_mask", "terminal_label_mask"):
                        if windows[label][:].any():
                            raise ValueError("Source proposals cannot supply verified field supervision")
                if check_images:
                    source = episode["source"]
                    for name, image in episode["images"].items():
                        for frame in (1, count):
                            if not np.array_equal(image[frame], source[f"obs/{name}"][frame - 1]):
                                raise ValueError("Virtual image mapping is wrong")
                if entry["split"] == "train":
                    for key in normalization:
                        values = episode[key][:] if key == "actions" else episode[f"robot_state/{key}"][:]
                        total, squares, size = training_sums.get(key, (0, 0, 0))
                        training_sums[key] = (
                            total + values.sum(axis=0),
                            squares + (values**2).sum(axis=0),
                            size + len(values),
                        )
                signatures[entry["signature"]].add(entry["split"])
                counts[entry["split"]] += 1
                counts["episodes"] += 1
                counts["actions"] += count
            except (ValueError, OSError, KeyError, IndexError) as error:
                errors.append({"shard": entry["shard"], "group": entry["group"], "reason": str(error)})
            finally:
                if file is not None:
                    file.close()
        if any(len(splits) != 1 for splits in signatures.values()):
            errors.append({"reason": "Duplicate trajectory appears in multiple splits"})
        for key, (total, squares, count) in training_sums.items():
            expected = normalization[key]
            mean = total / count
            variance = np.maximum(squares / count - mean**2, 0)
            if (
                expected["count"] != count
                or not np.allclose(expected["mean"], mean, atol=1e-10, rtol=1e-8)
                or not np.allclose(np.square(expected["std"]), variance, atol=1e-12, rtol=1e-6)
            ):
                errors.append({"reason": f"Training-only normalization disagrees for {key}"})
        return {
            "normalization_check": "independent train-only sums and squared sums",
            "passed": not errors,
            "counts": dict(counts),
            "errors": errors,
            "image_checks": "first and last available frame, both cameras" if check_images else "not_run",
        }


def audit(args):
    dataset = ProcessedLiberoDataset(args.dataset)
    report = dataset.audit(check_images=not args.skip_images)
    atomic_json(dataset.root / "integrity_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1
