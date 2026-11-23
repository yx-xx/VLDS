"""Inspect a command-span fit or a saved candidate without changing training labels."""

import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.datasets.libero import replay_teacher
from openpi.datasets.libero.reader import ProcessedLiberoDataset
from openpi.fields import primitive as primitives
from openpi.fields import visualization


def load_result(dataset, index, path):
    """Require a matching episode and source manifest before displaying saved parameters."""
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    if result.get("schema") != "vlds.libero.offline_fields.v1":
        raise ValueError("Expected an offline-fields episode JSON")
    entry = dataset.episodes[index]
    source = result["source_episode"]
    for key in ("signature", "shard", "group", "split"):
        if source.get(key) != entry.get(key):
            raise ValueError(f"Saved field belongs to another source episode ({key})")
    run = json.loads((path.parent / "run.json").read_text())
    config = run["config"]
    if config["source_manifest_sha256"] != replay_teacher.digest(dataset.root / "manifest.json"):
        raise ValueError("Saved field source manifest differs from the current dataset")
    if result.get("fingerprint") != run.get("fingerprint"):
        raise ValueError("Saved field fingerprint differs from its run")
    if result.get("teacher_source") != config.get("teacher_source"):
        raise ValueError("Saved field reference trajectory differs from its run")
    return result, primitives.ControlConfig(**config["control"])


def preview(
    dataset_root,
    index,
    *,
    interval_index=0,
    fields_path=None,
    record_index=None,
    max_speed=0.3,
    representation="fitted",
    density=7,
):
    dataset = ProcessedLiberoDataset(dataset_root)
    data = dataset.field_input(index)
    fit_report = None
    if fields_path is not None:
        result, control = load_result(dataset, index, fields_path)
        if record_index is None:
            raise ValueError("Choose a saved field candidate")
        record = result["records"][record_index]
        if not record.get("primitive"):
            raise ValueError("This candidate has no numerically valid field")
        data = replay_teacher.apply(data, Path(fields_path).parent, result.get("teacher_source"))
        field = primitives.Primitive.from_record(record["primitive"])
        interval = record["candidate"]
        label = f"Saved candidate {record_index + 1} ({result.get('teacher_representation', 'fitted')})"
        fit_report = record.get("fit")
    else:
        if representation not in ("fitted", "linear"):
            raise ValueError("Unknown field representation")
        interval = data["command_intervals"][interval_index]
        control = dataclasses.replace(primitives.DEFAULT_CONTROL, max_speed=max_speed)
        field = None
        label = f"Command interval {interval_index + 1} ({representation})"
    return preview_interval(
        dataset,
        data,
        interval,
        control,
        field=field,
        fit_report=fit_report,
        label=label,
        fields_path=fields_path,
        representation=representation,
        density=density,
    )


def preview_interval(
    dataset,
    data,
    interval,
    control,
    *,
    field=None,
    fit_report=None,
    label="Adaptive segment",
    fields_path=None,
    representation="fitted",
    density=7,
):
    """Evaluate an explicitly bounded segment against the same physical-time gates."""
    start, end = interval["start"], interval["end"]
    if not 0 <= start < end < len(data["time"]):
        raise ValueError("Field interval is outside the source episode")
    if not np.all(data["actions"][start:end, -1] == interval["command"]):
        raise ValueError("Field interval crosses a gripper-command boundary")
    selection = slice(start, end + 1)
    trajectory = data["trajectory"]
    position, velocity = trajectory["position"][selection], trajectory["linear_velocity"][selection]
    rotations = (
        Rotation.from_quat(trajectory["quaternion_xyzw"][selection]).as_matrix() @ data["body_to_control_rotation"]
    )
    times = data["time"][selection]
    if field is None:
        field, fit_report = primitives.fit_primitive(
            position, velocity, rotations, times, interval["command"], control=control, representation=representation
        )
        if field is None:
            raise ValueError("Fitting produced no numerically valid field: " + str(fit_report.get("reasons")))
    arrays, metadata = visualization.sample_field(field, position, rotations, times, control, density=density)
    arrays["demo_speed"] = np.linalg.norm(velocity, axis=1)
    arrays["demo_velocity"] = velocity
    residual = field.field.velocity(position) - velocity
    radial = np.sum((position - field.field.goal) * velocity, axis=1)
    metadata.update(
        label=label,
        interval={"start": start, "end": end, "command": int(interval["command"])},
        source_episode=data["entry"],
        processed_path=str((dataset.root / data["entry"]["shard"]).resolve()),
        processed_group=data["entry"]["group"],
        source_manifest_sha256=replay_teacher.digest(dataset.root / "manifest.json"),
        fields_file=str(Path(fields_path).resolve()) if fields_path else None,
        teacher_source=data.get("teacher_source"),
        fit=fit_report,
        velocity_rms_m_s=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        radial_outward_samples=int(np.count_nonzero(radial > 1e-6)),
    )
    return arrays, metadata


def inspect_vector_field(args):
    record_index = None
    if args.fields:
        dataset = ProcessedLiberoDataset(args.dataset)
        result, _ = load_result(dataset, args.episode, args.fields)
        record_index = result["sequences"][args.sequence]["record_indices"][args.field]
    arrays, metadata = preview(
        args.dataset,
        args.episode,
        interval_index=args.interval,
        fields_path=args.fields,
        record_index=record_index,
        max_speed=args.max_speed,
        representation=args.representation,
        density=args.density,
    )
    visualization.plot_field(args.output, arrays, metadata)
    print(json.dumps({"output": str(args.output), "report": metadata["report"]}, indent=2))
    return 0
