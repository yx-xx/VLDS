"""Validate saved build identity and adapt dynamics to the dataset inspector."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese diagnostics.
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.fields import contractive
from openpi.fields import primitive
from openpi.fields import seds
from openpi.fields import sequence
from openpi.fields import visualization

_INDEX_CACHE = {}
_ROOT = Path(__file__).resolve().parents[4]


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _profile_ready(directory, run, *, require_full=True):
    if run.get("status") != "complete" or (require_full and not run.get("summary", {}).get("full_collection")):
        return False
    if run.get("config", {}).get("model") == contractive.VERSION:
        path = directory / "audit.json"
        if not path.is_file():
            return False
        audit = json.loads(path.read_text())
        return (
            audit.get("fingerprint") == run["fingerprint"]
            and audit.get("integrity_passed")
            and (not require_full or audit.get("full_source_episode_coverage"))
        )
    return True


def _profile_directory(base_directory, profile_name):
    if profile_name == "replay":
        return base_directory
    directory = base_directory / "profiles" / profile_name
    if not (directory / "run.json").is_file() or not (directory / "index.json").is_file():
        raise ValueError(f"离线动力系统配置 {profile_name} 尚未生成。")
    run = json.loads((directory / "run.json").read_text())
    if not _profile_ready(directory, run):
        raise ValueError(f"离线动力系统配置 {profile_name} 尚未完整生成。")
    return directory


def available_profiles(episode):
    """Return persisted profiles that can be selected without refitting."""
    root = episode.processed_root
    if root is None:
        root = next((p for p in episode.path.parents if (p / ".vlds/dynamics.json").exists()), None)
    if root is None or not (Path(root) / ".vlds/dynamics.json").exists():
        return ()
    pointer = json.loads((Path(root) / ".vlds/dynamics.json").read_text())
    base_directory = (Path(root) / ".vlds" / pointer["directory"]).resolve()
    names = ["replay"]
    profiles = base_directory / "profiles"
    if profiles.is_dir():
        for path in sorted(profiles.iterdir()):
            if not (path / "run.json").is_file() or not (path / "index.json").is_file():
                continue
            run = json.loads((path / "run.json").read_text())
            if _profile_ready(path, run):
                names.append(path.name)
    return tuple(names)


def load_for_episode(episode, profile="replay"):
    """Return None for unbuilt data; fail visibly if a registered result is stale."""
    return _load_for_episode(episode, profile)


def load_saved_episode(episode, path):
    """Explicitly inspect one audited Gaussian-field pilot without registering it."""
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    if result.get("schema") != "vlds.dynamics.episode.v1" or result.get("model") != contractive.VERSION:
        raise ValueError("手动样例载入需要已审计的高斯稳定场；完整配置请从配置列表选择。")
    bundle = _load_for_episode(episode, result["profile"], selected_path=path)
    if bundle is None:
        raise ValueError("当前数据集尚未登记完整基础场，不能核对样例来源。")
    bundle["manual"] = True
    return bundle


def _load_for_episode(episode, profile, *, selected_path=None):
    root = episode.processed_root
    if root is None:
        root = next((p for p in episode.path.parents if (p / ".vlds/dynamics.json").exists()), None)
    if root is None or not (Path(root) / ".vlds/dynamics.json").exists():
        return None
    root = Path(root)
    pointer = json.loads((root / ".vlds/dynamics.json").read_text())
    base_directory = (root / ".vlds" / pointer["directory"]).resolve()
    directory = _profile_directory(base_directory, profile) if selected_path is None else selected_path.parent
    index_path = directory / "index.json"
    current_run = json.loads((directory / "run.json").read_text())
    if selected_path is not None:
        if current_run.get("config", {}).get("model") != contractive.VERSION or not _profile_ready(
            directory, current_run, require_full=False
        ):
            raise ValueError("该样例尚未完成构建和独立完整性复核。")
        if (directory / current_run["config"]["base_directory"]).resolve() != base_directory:
            raise ValueError("该样例不属于当前登记的基础数据集。")
    relative = current_run.get("source_relative", current_run.get("config", {}).get("source_relative"))
    if not relative:
        raise ValueError("离线动力系统运行记录缺少源数据路径。")
    source_root = (directory / relative).resolve()
    source_metadata = (
        [source_root / "manifest.json"]
        if current_run["config"]["kind"] == "processed_libero"
        else [source_root / "meta/info.json", source_root / "meta/episodes.jsonl"]
    )
    key = (
        str(index_path),
        index_path.stat().st_mtime_ns,
        profile,
        pointer["fingerprint"],
        (directory / "run.json").stat().st_mtime_ns,
        tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in source_metadata),
    )
    if key not in _INDEX_CACHE:
        index = json.loads(index_path.read_text())
        run = current_run
        if profile == "replay":
            if index["fingerprint"] != pointer["fingerprint"] or run["fingerprint"] != pointer["fingerprint"]:
                raise ValueError("离线动力系统索引和构建版本不一致，请重新构建。")
        elif (
            index.get("fingerprint") != run.get("fingerprint")
            or run.get("config", {}).get("base_fingerprint") != pointer["fingerprint"]
            or run.get("config", {}).get("profile", {}).get("name") != profile
        ):
            raise ValueError("离线动力系统配置与回放数据版本不一致，请重新生成。")
        source_relative = run.get("source_relative", run.get("config", {}).get("source_relative"))
        if not source_relative:
            raise ValueError("离线动力系统运行记录缺少源数据路径。")
        source_root = (directory / source_relative).resolve()
        if run["config"]["kind"] == "processed_libero":
            source_hash = _digest(source_root / "manifest.json")
        else:
            source_hash = hashlib.sha256(
                (source_root / "meta/info.json").read_bytes() + (source_root / "meta/episodes.jsonl").read_bytes()
            ).hexdigest()
        if source_hash != run["config"]["source_manifest_sha256"]:
            raise ValueError("数据集元信息已变化，离线动力系统结果已过期。")
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = index, run
    index, run = _INDEX_CACHE[key]
    if episode.processed_root is not None:
        match = next((r for r in index["episodes"] if r["index"] == episode.processed_index), None)
        offset = 0
    elif episode.path.suffix == ".parquet":
        relative = str(episode.path.relative_to(root))
        match = next((r for r in index["episodes"] if r["source_key"] == relative), None)
        offset = 0
    else:
        relative = str(episode.path.resolve().relative_to(_ROOT)) + "::" + "/" + episode.group.lstrip("/")
        match = next((r for r in index["episodes"] if r["source_key"] == relative), None)
        offset = 1  # Raw released observations are post-action observations.
    if match is None:
        raise ValueError("该回合不在已登记的离线动力系统索引中。")
    path = directory / match["file"]
    if selected_path is not None and path.resolve() != selected_path:
        raise ValueError("所选结果不是当前播放的回合。")
    result = json.loads(path.read_text())
    expected_fingerprint = pointer["fingerprint"] if profile == "replay" else run["fingerprint"]
    if result["fingerprint"] != expected_fingerprint or result["schema"] != "vlds.dynamics.episode.v1":
        raise ValueError("离线动力系统回合来源/版本不一致。")
    if profile != "replay" and result.get("base_fingerprint") != pointer["fingerprint"]:
        raise ValueError("离线动力系统回合的基础回放版本不一致。")
    if result.get("source_episode_index") != match["index"]:
        raise ValueError("离线动力系统回合身份与索引不一致。")
    if profile != "replay" and _digest(base_directory / match["file"]) != result["base_episode_sha256"]:
        raise ValueError("基础回放场已变化，请重新生成该配置。")
    trace = directory / result["trace_file"]
    if _digest(trace) != result["trace_sha256"]:
        raise ValueError("离线轨迹缓存校验失败。")
    if episode.path.suffix == ".parquet" and _digest(episode.path) != result["source_sha256"]:
        raise ValueError("示教文件已变化，请重新构建动力系统。")
    if episode.processed_root is not None:
        from openpi.datasets.libero.reader import ProcessedLiberoDataset

        entry = ProcessedLiberoDataset(root).episodes[episode.processed_index]
        if entry["signature"] != result["source_sha256"]:
            raise ValueError("示教回合签名与已保存的动力系统不同。")
    with np.load(trace, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if len(arrays["time"]) != episode.frames + offset:
        raise ValueError("离线轨迹与当前播放帧数不同。")
    if selected_path is not None:
        from openpi.datasets.dynamics.coherent_audit import validate_saved_episode

        validate_saved_episode(directory, current_run, match)
    return {"result": result, "arrays": arrays, "path": str(path), "frame_offset": offset, "profile": profile}


def segmentation(bundle):
    result, offset = bundle["result"], bundle["frame_offset"]
    records = [
        {
            **r,
            "quality": record_quality(bundle, i),
            "start": max(0, r["start"] - offset),
            "end": r["end"] - offset,
            "field_index": i,
        }
        for i, r in enumerate(result["records"])
        if r["end"] > offset
    ]
    return {
        "intervals": records,
        "time": bundle["arrays"]["time"][offset:].tolist(),
        "candidate_counts": [1] * len(records),
        "kind": "dynamics",
        "offline_bundle": bundle,
        "profile": bundle.get("profile", "replay"),
        "time_scale": result["limits"].get("time_scale", 1),
        "source_time": bundle["arrays"].get("source_time", bundle["arrays"]["time"])[offset:].tolist(),
    }


def record_quality(bundle, index):
    """Measure original observations even when display/execution time was stretched."""
    result = bundle["result"]
    record = result["records"][index]
    a, b = record["start"], record["end"]
    arrays = bundle.get("arrays", {})
    times = arrays.get("source_time", arrays.get("time"))
    seconds = (
        float(times[b] - times[a])
        if times is not None
        else (b - a) * result["limits"].get("dt", 0) / result["limits"].get("time_scale", 1)
    )
    eligible = b - a >= 10 and seconds >= 0.5 - 1e-7
    parent = next((p for p in result.get("source_intervals", []) if p["start"] == a and p["end"] == b), None)
    return {
        "source_seconds": seconds,
        "observations": b - a + 1,
        "transitions": b - a,
        "eligible_length": eligible,
        "short_command_event": bool(not eligible and parent),
    }


def native_plane_velocity(bundle, index, dimensions, plane, frame):
    """Sample the saved native field, with unplotted coordinates at the source frame."""
    record = bundle["result"]["records"][index]
    arrays = bundle["arrays"]
    state = arrays["position"][min(max(frame, 0), len(arrays["position"]) - 1)]
    points = np.tile(state, (len(plane), 1))
    points[:, dimensions] = plane
    if "field" in record:
        velocity = contractive.ContractiveField(record["field"]).velocity(points)
    else:
        velocity = record["rate_per_second"] * (np.asarray(record["goal"]) - points)
        speed = np.linalg.norm(velocity / arrays["tolerance"], axis=-1)
        velocity *= np.minimum(1.0, record["maximum_speed_scaled_per_second"] / np.maximum(speed, 1e-30))[:, None]
    return velocity[:, dimensions]


def primitive_for_record(result, record):
    if result["coordinates"].get("control_point") != "gripper0_grip_site":
        raise ValueError("Only the processed LIBERO adapter has a verified MuJoCo world/control point")
    if result["model"] == contractive.VERSION:
        raise ValueError("A Gaussian LPV field must be sampled in its saved Lyapunov metric")
    field = seds.PositionField(seds.linear_factors(record["rate_per_second"]), record["goal"])
    dynamics = primitive.Primitive(field, np.asarray(record["goal_rotation"]), int(record["command"]))
    control = dataclasses.replace(
        primitive.DEFAULT_CONTROL,
        dt=result["limits"]["dt"],
        max_speed=record["maximum_speed_scaled_per_second"] * result["tolerance"][0],
        max_angular_speed=result["limits"]["maximum_angular_speed_rad_per_second"],
        rotation_gain=result["limits"]["rotation_gain_per_second"],
    )
    return dynamics, control


def preview(bundle, index, episode, density=5):
    """Reconstruct saved parameters and evaluate; never refit while browsing."""
    result, arrays = bundle["result"], bundle["arrays"]
    record = result["records"][index]
    if result["model"] == contractive.VERSION:
        return _contractive_preview(bundle, index, episode, density)
    field, control = primitive_for_record(result, record)
    a, b = record["start"], record["end"]
    selection = slice(a, b + 1)
    if "nominal_rotation" not in arrays:
        current = arrays["rotation"][0]
        orientations = [current]
        steps_per_observation = result["limits"].get("integration_substeps", 1)
        for saved in result["records"]:
            path = sequence.integrate_rotation(
                current,
                np.asarray(saved["goal_rotation"]),
                (saved["end"] - saved["start"]) * steps_per_observation,
                control.dt,
                control.max_angular_speed,
                control.rotation_gain,
            )
            orientations.extend(path[steps_per_observation::steps_per_observation])
            current = path[-1]
        arrays["nominal_rotation"] = np.asarray(orientations)
    saved_report = {
        **record["report"],
        "passed": record["accepted"],
        "position_rms_m": record["report"]["continuous_position_rms_native"],
        "rotation_rms_rad": record["report"].get("continuous_rotation_rms_rad", 0),
        "scope": "persisted local and continuous fit acceptance; errors display the carried-state rollout",
    }
    values, info = visualization.sample_field(
        field,
        arrays["position"][selection],
        arrays["rotation"][selection],
        arrays["time"][selection],
        control,
        density=density,
        saved_trace={"position": arrays["nominal"][selection], "rotation": arrays["nominal_rotation"][selection]},
        saved_report=saved_report,
    )
    values["sequence_nominal"] = arrays["nominal"][selection]
    info.update(
        label=f"Saved rollout linear field {index + 1}",
        interval={"start": a, "end": b, "command": record["command"]},
        source_episode=result["source_entry"],
        processed_path=str(episode.path.resolve()),
        processed_group=episode.group,
        fields_file=bundle["path"],
        teacher_source=None,
        fit={"kind": "rollout_linear", "parameter_supervision": False},
        continuous_report={
            "fields": len(result["records"]),
            "position_rms_m": result["summary"]["position_rms_native"],
            "rotation_rms_rad": result["summary"].get("rotation_rms_rad", 0),
            "time_aligned_passed": result["summary"]["passed"],
            "scope": "whole episode",
        },
        automatic_source="persisted_complete_dataset_build",
        time_scale=result["limits"].get("time_scale", 1),
        quality=record_quality(bundle, index),
    )
    return values, info


def _contractive_preview(bundle, index, episode, density):
    result, arrays = bundle["result"], bundle["arrays"]
    if result["coordinates"].get("control_point") != "gripper0_grip_site":
        raise ValueError("This field is only available in its native coordinate projection")
    if not 3 <= density <= 11:
        raise ValueError("Field grid density must be between 3 and 11")
    record = result["records"][index]
    field = contractive.ContractiveField(record["field"])
    a, b = record["start"], record["end"]
    demo = arrays["position"][a : b + 1]
    nominal = arrays["nominal"][a : b + 1]
    padding = np.maximum(np.ptp(demo, axis=0) * 0.15, 0.025)
    axes = [np.linspace(lo, hi, density) for lo, hi in zip(demo.min(0) - padding, demo.max(0) + padding, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    velocity = field.velocity(grid)
    arrow_seconds = float(
        max(np.diff(axis).min() for axis in axes) * 0.7 / max(np.linalg.norm(velocity, axis=1).max(), 1e-12)
    )
    angles = Rotation.from_matrix(
        arrays["rotation"][a : b + 1] @ arrays["nominal_rotation"][a : b + 1].transpose(0, 2, 1)
    ).magnitude()
    times = arrays["time"][a : b + 1]
    values = {
        "grid": grid,
        "velocity": velocity,
        "arrows": velocity * arrow_seconds,
        "demo": demo,
        "nominal": nominal,
        "sequence_nominal": nominal,
        "goal": field.goal,
        "time": times - times[0],
        "position_error": np.linalg.norm(nominal - demo, axis=1),
        "rotation_error": angles,
        "field_speed_on_demo": np.linalg.norm(field.velocity(demo), axis=1),
    }
    control = dataclasses.replace(
        primitive.DEFAULT_CONTROL,
        dt=field.dt,
        max_speed=field.speed * max(field.tolerance),
        max_angular_speed=result["limits"]["maximum_angular_speed_rad_per_second"],
    )
    control_info = dataclasses.asdict(control)
    # Pose gains belong to the legacy controller. The saved orientation LPV map
    # supplies its own feedback; do not advertise an unused default gain.
    control_info["rotation_gain"] = None
    info = {
        "schema": "vlds.field_preview.v1",
        "frame": "world",
        "control_point": "gripper0_grip_site",
        "scope": "saved common-metric discrete LPV map; orientation certificate only within visited principal log chart",
        "control": control_info,
        "field": field.record(),
        "orientation_field": record.get("orientation_field"),
        "orientation_certificate_scope": "visited principal SO(3) log chart only",
        "arrow_seconds": arrow_seconds,
        "density": density,
        "label": f"Gaussian LPV field {index + 1}",
        "interval": {"start": a, "end": b, "command": record["command"]},
        "source_episode": result["source_entry"],
        "processed_path": str(episode.path.resolve()),
        "processed_group": episode.group,
        "fields_file": bundle["path"],
        "teacher_source": None,
        "fit": {"kind": "gaussian_contractive", "parameter_supervision": False},
        "report": {
            **record["report"],
            "passed": record["accepted"],
            "position_rms_m": record["report"]["continuous_position_rms_native"],
            "rotation_rms_rad": record["report"].get("continuous_rotation_rms_rad", 0),
        },
        "continuous_report": {
            "fields": len(result["records"]),
            "position_rms_m": result["summary"]["position_rms_native"],
            "rotation_rms_rad": result["summary"].get("rotation_rms_rad", 0),
            "time_aligned_passed": result["summary"]["passed"],
            "scope": "whole episode",
        },
        "quality": record_quality(bundle, index),
        "automatic_source": "persisted_complete_dataset_build",
        "time_scale": 1,
    }
    if "goal_source_index" in record:
        info["goal_source_index"] = record["goal_source_index"]
        info["boundary_position"] = record["boundary_position"]
        info["goal_policy"] = "future source observation; switch at observed boundary without waiting for arrival"
    return values, info
