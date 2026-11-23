"""Bounded, on-demand segmentation for browsing; never creates training labels."""

import concurrent.futures
import dataclasses
import itertools
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.datasets.libero.reader import ProcessedLiberoDataset
from openpi.datasets.libero.vector_field_inspection import load_result
from openpi.datasets.libero.vector_field_inspection import preview
from openpi.datasets.libero.vector_field_inspection import preview_interval
from openpi.fields import primitive as primitives
from openpi.fields.primitive import DEFAULT_CONTROL


def interval_at(intervals, frame):
    """Shared boundary observations belong to the following command/field."""
    for i, item in enumerate(intervals):
        if item["start"] <= frame < item["end"]:
            return i
    if intervals and frame == intervals[-1]["end"]:
        return len(intervals) - 1
    return None


def adaptive_partition(start, end, evaluate, *, minimum_steps=5):
    """Retain rejected leaves explicitly; bound work to fewer than 2*T/minimum_steps fits."""
    value = evaluate(start, end)
    if value[1]["report"]["passed"] or end - start < 2 * minimum_steps:
        return [value]
    middle = (start + end) // 2
    return adaptive_partition(start, middle, evaluate, minimum_steps=minimum_steps) + adaptive_partition(
        middle, end, evaluate, minimum_steps=minimum_steps
    )


def saved_sequence(dataset, index, control, search_root):
    """Find a compatible original-demo sequence; validate sources before using parameters."""
    if search_root is None or not Path(search_root).is_dir():
        return None
    candidates = []
    # Only exact per-episode filenames, bounded to avoid scanning arbitrary JSON content.
    for path in sorted(Path(search_root).rglob(f"episode_{index:05d}.json"))[:128]:
        try:
            result, saved_control = load_result(dataset, index, path)
            if (
                saved_control != control
                or result.get("teacher_source")
                or result.get("teacher_representation", "fitted") != "fitted"
            ):
                continue
            for sequence in result["sequences"]:
                indices = sequence["record_indices"]
                records = [result["records"][i] for i in indices]
                if not records or any(not r.get("accepted") or not r.get("primitive") for r in records):
                    continue
                spans = [r["candidate"] for r in records]
                if spans[0]["start"] != 0 or spans[-1]["end"] != dataset.episodes[index]["actions"]:
                    continue
                if any(a["end"] != b["start"] for a, b in itertools.pairwise(spans)):
                    continue
                candidates.append((len(indices), str(path), indices))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    return min(candidates, key=lambda item: (item[0], item[1])) if candidates else None


def attach_continuous_report(values, data, control):
    """Check source-time switching without resetting states at fitted boundaries."""
    start = values[0][1]["interval"]["start"]
    end = values[-1][1]["interval"]["end"]
    reference = data["trajectory"]["position"][start : end + 1]
    rotations = Rotation.from_quat(data["trajectory"]["quaternion_xyzw"][start : end + 1]).as_matrix()
    rotations = rotations @ data["body_to_control_rotation"]
    fields = [primitives.Primitive.from_record(info["primitive"]) for _, info in values]
    spans = [info["interval"] for _, info in values]
    position, rotation = reference[0].copy(), rotations[0].copy()
    trace, orientation = [position.copy()], [rotation.copy()]
    for frame in range(start, end):
        field = fields[interval_at(spans, frame)]
        velocity, omega, _ = primitives.feedback(field, position, rotation, control)
        dt = float(data["time"][frame + 1] - data["time"][frame])
        if not np.isclose(dt, control.dt):
            raise ValueError("Preview timeline differs from the configured integration period")
        position = position + dt * velocity
        rotation = Rotation.from_rotvec(dt * omega).as_matrix() @ rotation
        trace.append(position.copy())
        orientation.append(rotation.copy())
    trace = np.asarray(trace)
    error = np.linalg.norm(trace - reference, axis=1)
    angle = Rotation.from_matrix(rotations @ np.asarray(orientation).transpose(0, 2, 1)).magnitude()
    position_rms, rotation_rms = float(np.sqrt(np.mean(error**2))), float(np.sqrt(np.mean(angle**2)))
    report = {
        "start": start,
        "end": end,
        "fields": len(values),
        "position_rms_m": position_rms,
        "rotation_rms_rad": rotation_rms,
        "time_aligned_passed": position_rms <= control.position_rms_tolerance
        and rotation_rms <= control.rotation_rms_tolerance,
        "scope": "source-time switching without state resets; no arrival gating or robot physics",
    }
    for arrays, info in values:
        info["continuous_report"] = report
        first, last = info["interval"]["start"] - start, info["interval"]["end"] - start
        arrays["sequence_nominal"] = trace[first : last + 1]
    return values


def automatic_previews(
    dataset_root,
    index,
    interval_index,
    *,
    max_speed=0.3,
    representation="fitted",
    search_root=None,
    cancelled=lambda: False,
):
    """Reuse verified nominal fits, otherwise subdivide only the viewed command span."""
    dataset = ProcessedLiberoDataset(dataset_root)
    data = dataset.field_input(index)
    interval = data["command_intervals"][interval_index]
    control = dataclasses.replace(DEFAULT_CONTROL, max_speed=max_speed)
    saved = saved_sequence(dataset, index, control, search_root) if representation == "fitted" else None
    if saved:
        _, path, indices = saved
        result, _ = load_result(dataset, index, path)
        chosen = [
            i
            for i in indices
            if interval["start"] <= result["records"][i]["candidate"]["start"]
            and result["records"][i]["candidate"]["end"] <= interval["end"]
        ]
        spans = [result["records"][i]["candidate"] for i in chosen]
        if spans and spans[0]["start"] == interval["start"] and spans[-1]["end"] == interval["end"]:
            values = [preview(dataset_root, index, fields_path=path, record_index=i) for i in chosen]
            if all(info["report"]["passed"] for _, info in values):
                for _, info in values:
                    info["automatic_source"] = "validated_saved_sequence"
                return attach_continuous_report(values, data, control)

    def evaluate(start, end):
        if cancelled():
            raise concurrent.futures.CancelledError("Preview episode changed")
        value = preview_interval(
            dataset,
            data,
            {**interval, "start": start, "end": end},
            control,
            representation=representation,
            label=f"Adaptive {representation} [{start}, {end}]",
        )
        value[1]["automatic_source"] = "adaptive_preview_only"
        return value

    values = adaptive_partition(interval["start"], interval["end"], evaluate)
    return attach_continuous_report(values, data, control)
