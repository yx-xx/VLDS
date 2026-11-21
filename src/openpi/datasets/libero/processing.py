"""Reproducible LIBERO release -> aligned trajectories and unverified field proposals."""

# ruff: noqa: RUF001 -- Chinese generated report.
from __future__ import annotations

import collections
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from filelock import FileLock
from filelock import Timeout
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.datasets.libero import trajectory
from openpi.datasets.libero.reader import atomic_json

SCHEMA = "vlds.libero.trajectories.v2"
ROOT = Path(__file__).resolve().parents[4]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
RELEASE_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
NUMERIC_FIELDS = (
    "position",
    "orientation_6d",
    "joint_position",
    "gripper_position",
    "linear_velocity",
    "angular_velocity",
    "actions",
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _dataset(group, name, value):
    value = np.asarray(value)
    kwargs = {"compression": "gzip", "compression_opts": 1, "shuffle": True} if value.size else {}
    return group.create_dataset(name, data=value, **kwargs)


def _record(group, name, value):
    group.create_dataset(name, data=_json(value), dtype=h5py.string_dtype("utf-8"))


def _version(repo):
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def pipeline_fingerprint():
    digest = hashlib.sha256()
    for name in ("processing.py", "trajectory.py", "reader.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def kinematic_model_xml(xml):
    tree = ET.fromstring(xml)
    discarded = {"asset", "geom", "contact", "visual", "sensor", "light", "camera", "tendon", "equality"}
    for parent in tree.iter():
        for child in list(parent):
            if child.tag in discarded:
                parent.remove(child)
        if parent.tag == "site":
            parent.attrib.pop("material", None)
        if (
            parent.tag == "body"
            and any(child.tag in ("joint", "freejoint") for child in parent)
            and parent.find("inertial") is None
        ):
            ET.SubElement(parent, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
    compiler = tree.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(tree, "compiler")
    compiler.set("inertiafromgeom", "false")
    return ET.tostring(tree, encoding="unicode")


class Kinematics:
    """Recover the missing first observation and check the release's one-step lag.

    robosuite's observation orientation is robot0_right_hand, whereas OSC uses
    gripper0_grip_site. These are deliberately not treated as the same frame.
    """

    def __init__(self, xml, source, libero_root, robosuite_root):
        import mujoco

        self.mujoco = mujoco
        # Forward kinematics uses body/joint/site transforms only. Compiling visual
        # meshes for every episode dominated conversion time. Strip rendering and
        # collision geometry, retaining ALL joints and their qpos/qvel ordering.
        # This model must never be used for physical rollout or success checks.
        self.model = mujoco.MjModel.from_xml_string(kinematic_model_xml(xml))
        self.data = mujoco.MjData(self.model)
        self.site = self._id(mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        self.body = self._id(mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
        names = [f"robot0_joint{i}" for i in range(1, 8)]
        names += ["gripper0_finger_joint1", "gripper0_finger_joint2"]
        self.joints = [self.model.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names]

    def _id(self, kind, name):
        value = self.mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"Unsupported robot model: missing {name}")
        return value

    def observe(self, state):
        expected = 1 + self.model.nq + self.model.nv + self.model.na
        if state.shape != (expected,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid simulation state, expected ({expected},), got {state.shape}")
        self.data.time = state[0]
        self.data.qpos[:] = state[1 : 1 + self.model.nq]
        self.data.qvel[:] = state[1 + self.model.nq : 1 + self.model.nq + self.model.nv]
        if self.model.na:
            self.data.act[:] = state[-self.model.na :]
        self.mujoco.mj_forward(self.model, self.data)
        body_rotation = self.data.xmat[self.body].reshape(3, 3)
        site_rotation = self.data.site_xmat[self.site].reshape(3, 3)
        return {
            "position": self.data.site_xpos[self.site].copy(),
            "rotvec": Rotation.from_matrix(body_rotation).as_rotvec(),
            "joint_position": self.data.qpos[self.joints[:7]].copy(),
            "gripper_position": self.data.qpos[self.joints[7:]].copy(),
            "body_to_control_rotation": body_rotation.T @ site_rotation,
        }


def _matrix(group, key, rows, columns):
    if key not in group or group[key].shape != (rows, columns):
        raise ValueError(f"{key} must have shape ({rows}, {columns})")
    values = np.asarray(group[key], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{key} contains nonfinite values")
    return values


def canonicalize(group, env_args, config, kinematics):
    count = len(group["actions"])
    if count < 2:
        raise ValueError("An episode needs at least two actions")
    actions = _matrix(group, "actions", count, 7)
    states = np.asarray(group["states"], dtype=np.float64)
    if states.ndim != 2 or states.shape[0] != count or not np.isfinite(states).all():
        raise ValueError("Simulation states must have one finite row per action")
    fps = float(env_args["env_kwargs"]["control_freq"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("control_freq must be positive")
    dt = 1 / fps
    delta = np.diff(states[:, 0])
    if not np.allclose(delta, dt, atol=1e-6, rtol=1e-4):
        raise ValueError("Source has time gaps or nonmonotone timestamps; missing transitions cannot be invented")
    post = {
        "position": _matrix(group, "obs/ee_pos", count, 3),
        "rotvec": _matrix(group, "obs/ee_ori", count, 3),
        "joint_position": _matrix(group, "obs/joint_states", count, 7),
        "gripper_position": _matrix(group, "obs/gripper_states", count, 2),
    }
    for camera in ("agentview_rgb", "eye_in_hand_rgb"):
        images = group[f"obs/{camera}"]
        if images.ndim != 4 or images.shape[0] != count or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(f"Invalid RGB camera: {camera}")
    initial = kinematics.observe(states[0])
    aligned = {key: np.vstack((initial[key], values)) for key, values in post.items()}
    aligned["quaternion_xyzw"] = trajectory.continuous_quaternion(aligned["rotvec"])
    rotations = Rotation.from_quat(aligned["quaternion_xyzw"])
    aligned["orientation_6d"] = rotations.as_matrix()[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    aligned["time"] = np.r_[states[:, 0], states[-1, 0] + dt]
    aligned["time"] -= aligned["time"][0]
    aligned["linear_velocity"], aligned["angular_velocity"] = trajectory.measured_derivatives(
        aligned["position"], aligned["quaternion_xyzw"], aligned["time"]
    )
    aligned["actions"] = actions
    aligned["body_to_control_rotation"] = initial["body_to_control_rotation"]
    # Compare independently restored states to BOTH alignments. Selecting the convention
    # is not left to whichever comparison happens to be smaller on this episode.
    sample = np.unique(np.linspace(1, count - 1, min(config.alignment_samples, count - 1), dtype=int))
    shifted_position, same_position, shifted_rotation, shifted_joint, shifted_gripper = [], [], [], [], []
    for index in sample:
        observed = kinematics.observe(states[index])
        shifted_position.append(np.linalg.norm(observed["position"] - post["position"][index - 1]))
        same_position.append(np.linalg.norm(observed["position"] - post["position"][index]))
        shifted_rotation.append(
            float(
                (
                    Rotation.from_rotvec(observed["rotvec"]) * Rotation.from_rotvec(post["rotvec"][index - 1]).inv()
                ).magnitude()
            )
        )
        shifted_joint.append(np.max(np.abs(observed["joint_position"] - post["joint_position"][index - 1])))
        shifted_gripper.append(np.max(np.abs(observed["gripper_position"] - post["gripper_position"][index - 1])))
    quality = {
        "alignment_sample_indices": sample.tolist(),
        "shifted_position_max_m": float(np.max(shifted_position)),
        "shifted_position_rms_m": float(np.sqrt(np.mean(np.square(shifted_position)))),
        "same_index_position_rms_m": float(np.sqrt(np.mean(np.square(same_position)))),
        "shifted_rotation_max_rad": float(np.max(shifted_rotation)),
        "shifted_joint_max_rad": float(np.max(shifted_joint)),
        "shifted_gripper_max_m": float(np.max(shifted_gripper)),
        "alignment_pass": bool(
            max(shifted_position) <= config.alignment_position_tolerance
            and max(shifted_rotation) <= config.alignment_rotation_tolerance
        ),
        "initial_image_available": False,
        "terminal_observation_available": True,
        "terminal_simulation_state_available": False,
        "task_success": None,
        "task_success_basis": "not_independently_checked; source dones/rewards are not success labels",
        "initial_observation_basis": "MuJoCo forward kinematics of states[0]; no controller snapshot",
        "post_observation_basis": "source obs after action; simulator states are approximate replay references",
    }
    digest = hashlib.sha256()
    # Omit absolute time so duplicate demonstrations with clock offsets stay grouped.
    for values in (states[:, 1:], actions):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    return aligned, quality, digest.hexdigest()


def _link_images(destination, source_group, source_file, output_file, count):
    relative = os.path.relpath(source_file, output_file.parent)
    images = destination.create_group("images")
    for camera in ("agentview_rgb", "eye_in_hand_rgb"):
        original = source_group[f"obs/{camera}"]
        layout = h5py.VirtualLayout(shape=(count + 1, *original.shape[1:]), dtype=original.dtype)
        origin = h5py.VirtualSource(relative, original.name, shape=original.shape)
        layout[1:] = origin
        images.create_virtual_dataset(camera, layout, fillvalue=0)
    # Zero-filled O_0 images have an explicit missing-data mask, never a copied future image.
    _dataset(destination, "image_valid", np.r_[False, np.ones(count, dtype=bool)])
    _dataset(destination, "source_image_index", np.r_[-1, np.arange(count)].astype(np.int32))


def _write_episode(destination, source, source_file, output_file, arrays, quality, signature, config):
    count = len(arrays["actions"])
    split = trajectory.episode_split(signature, config)
    destination.attrs.update(
        split=split,
        trajectory_sha256=signature,
        source_group=source.name,
        source_file=os.path.relpath(source_file, output_file.parent),
        observation_timing="O[t+1]=source.obs[t]; O[0]=forward_kinematics(source.states[0])",
        query_origin="source_demonstration",
        frame="world",
        orientation_frame="robot0_right_hand",
        control_orientation_frame="gripper0_grip_site",
        task_success_known=False,
        fitted_field_supervision_available=False,
        fps=1 / (arrays["time"][1] - arrays["time"][0]),
    )
    robot = destination.create_group("robot_state")
    for name, values in arrays.items():
        if name in ("time", "actions", "body_to_control_rotation"):
            _dataset(destination, name, values)
        else:
            _dataset(robot, name, values)
    _link_images(destination, source, source_file, output_file, count)
    destination["source"] = h5py.ExternalLink(os.path.relpath(source_file, output_file.parent), source.name)
    _dataset(destination, "simulation_state_index", np.r_[np.arange(count), -1].astype(np.int32))
    _dataset(destination, "simulation_state_valid", np.r_[np.ones(count, dtype=bool), False])
    commands, events = trajectory.extract_events(
        arrays["actions"],
        arrays["time"],
        arrays["position"],
        arrays["quaternion_xyzw"],
        arrays["gripper_position"],
        config,
    )
    intervals = trajectory.command_intervals(commands, events)
    phases = trajectory.propose_phases(events, count)
    candidates, pruned = trajectory.propose_candidates(intervals, arrays["time"], arrays["linear_velocity"], config)
    _dataset(destination, "gripper_command", commands)
    for name, value in (
        ("events", events),
        ("command_intervals", intervals),
        ("phases", phases),
        ("candidates", candidates),
        ("quality", quality),
    ):
        _record(destination, name, value)
    for horizon in config.horizons:
        windows = destination.require_group(f"proposal_windows/H{horizon}")
        windows.attrs.update(query_origin="source_demonstration", labels_verified=False, goal_frame="world")
        for key, values in trajectory.proposal_windows(
            intervals, arrays["position"], arrays["quaternion_xyzw"], horizon
        ).items():
            _dataset(windows, key, values)
    return {
        "episode": destination.name.rsplit("/", 1)[-1],
        "split": split,
        "signature": signature,
        "actions": count,
        "observations": count + 1,
        "duration_seconds": float(arrays["time"][-1]),
        "events": len(events),
        "coupled_operations": sum(e["coupled_motion"] for e in events),
        "unsettled_operations": sum(e["feedback_settled"] is None for e in events),
        "short_command_changes": sum(not e["command_persistent"] for e in events),
        "command_intervals": len(intervals),
        "motion_phases": sum(p["kind"] == "motion" for p in phases),
        "candidates": len(candidates),
        "pruned_candidates": pruned,
        **quality,
    }


def _source_hash(source_root, file):
    manifest = source_root / ".vlds/manifest.json"
    if manifest.is_file():
        items = json.loads(manifest.read_text()).get("siblings", [])
        for item in items:
            if item["rfilename"] == file.relative_to(source_root).as_posix():
                return item.get("lfs", {}).get("sha256", item.get("blobId"))
    return None


def process_file(job):
    started = time.monotonic()
    source_file, output_file, source_root = (Path(job[key]) for key in ("source_file", "output_file", "source_root"))
    config = trajectory.ProcessingConfig(**job["config"])
    source_stat = source_file.stat()
    provenance = {
        "schema": SCHEMA,
        "config": dataclasses.asdict(config),
        "pipeline_sha256": job["pipeline_sha256"],
        "source_relative_file": source_file.relative_to(source_root).as_posix(),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_declared_hash": _source_hash(source_root, source_file),
        "source_revision": job["source_revision"],
        "resource_versions": job["resource_versions"],
    }
    fingerprint = hashlib.sha256(_json(provenance).encode()).hexdigest()
    if output_file.is_file() and job["resume"]:
        with h5py.File(output_file, "r") as existing:
            if existing.attrs.get("fingerprint") == fingerprint and existing.attrs.get("complete", False):
                return json.loads(existing["summary"][()].decode()) | {"resumed": True}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".hdf5.tmp")
    episodes, rejected = [], []
    with h5py.File(source_file, "r") as source, h5py.File(temporary, "w", libver="latest") as output:
        data = source["data"]
        info = json.loads(_text(data.attrs.get("problem_info", "{}")))
        language = info.get("language_instruction")
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"Missing full task language in {source_file}")
        env_args = json.loads(_text(data.attrs["env_args"]))
        output.attrs.update(schema=SCHEMA, fingerprint=fingerprint, complete=False, full_language=language)
        _record(output, "provenance", provenance)
        _record(output, "environment", env_args)
        for name in sorted(data):
            group = data[name]
            if not isinstance(group, h5py.Group):
                continue
            try:
                kinematics = Kinematics(
                    _text(group.attrs["model_file"]), source_file, job["libero_root"], job["robosuite_root"]
                )
                arrays, quality, signature = canonicalize(group, env_args, config, kinematics)
                destination = output.require_group(f"episodes/{name}")
                episodes.append(
                    _write_episode(destination, group, source_file, output_file, arrays, quality, signature, config)
                )
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                if f"episodes/{name}" in output:
                    del output[f"episodes/{name}"]
                rejected.append({"episode": name, "reason": str(error), "exception": type(error).__name__})
        summary = {
            "source": source_file.relative_to(source_root).as_posix(),
            "task": source_file.stem,
            "suite": source_file.parent.name,
            "language": language,
            "episodes": episodes,
            "rejected": rejected,
            "resumed": False,
            "seconds": round(time.monotonic() - started, 3),
            "fingerprint": fingerprint,
        }
        _record(output, "summary", summary)
        output.attrs["complete"] = True
        output.flush()
    os.replace(temporary, output_file)
    return summary


def aggregate(output_root, results, run):
    moments = {key: trajectory.RunningMoments() for key in NUMERIC_FIELDS}
    counts = collections.Counter()
    suite_counts = collections.defaultdict(collections.Counter)
    entries, rejects = [], []
    signatures = collections.defaultdict(list)
    quality_rows = []
    for result in sorted(results, key=lambda r: r["source"]):
        shard = Path(result["suite"]) / (result["task"] + ".hdf5")
        with h5py.File(output_root / shard, "r") as file:
            for summary in result["episodes"]:
                episode = file[f"episodes/{summary['episode']}"]
                split = summary["split"]
                entry = {
                    "shard": shard.as_posix(),
                    "group": episode.name,
                    "suite": result["suite"],
                    "task": result["task"],
                    "language": result["language"],
                    **summary,
                }
                entries.append(entry)
                signatures[summary["signature"]].append(entry)
                quality_rows.append(summary)
                counts["episodes"] += 1
                counts[split] += 1
                suite_counts[result["suite"]][split] += 1
                for key in (
                    "actions",
                    "observations",
                    "events",
                    "coupled_operations",
                    "unsettled_operations",
                    "short_command_changes",
                    "command_intervals",
                    "motion_phases",
                    "candidates",
                    "pruned_candidates",
                ):
                    counts[key] += summary[key]
                counts["alignment_pass"] += summary["alignment_pass"]
                if split == "train":
                    for key in NUMERIC_FIELDS:
                        values = episode[key] if key == "actions" else episode[f"robot_state/{key}"]
                        moments[key].update(values[:])
            rejects.extend({"source": result["source"], **item} for item in result["rejected"])
    duplicates = [[f"{x['shard']}:{x['group']}" for x in group] for group in signatures.values() if len(group) > 1]
    if any(len({x["split"] for x in group}) > 1 for group in signatures.values()):
        raise ValueError("Identical trajectories leaked across splits")
    percentiles = {}
    for key in ("duration_seconds", "shifted_position_max_m", "shifted_rotation_max_rad", "motion_phases"):
        values = np.array([row[key] for row in quality_rows])
        percentiles[key] = dict(
            zip(("min", "p50", "p95", "max"), np.percentile(values, [0, 50, 95, 100]).tolist(), strict=True)
        )
    source_files_failed = run["file_errors"]
    manifest = {
        **run,
        "schema": SCHEMA,
        "counts": dict(counts),
        "suite_splits": {k: dict(v) for k, v in suite_counts.items()},
        "percentiles": percentiles,
        "episodes": entries,
        "rejected_episodes": rejects,
        "duplicate_trajectory_groups": duplicates,
        "status": "complete" if not rejects and not source_files_failed else "complete_with_rejections",
        "processing_complete": not source_files_failed,
        "all_selected_sources_represented": not rejects and not source_files_failed,
        "candidate_action_coverage_complete": True,
        "candidate_basis": "complete constant-command intervals; coupled motion, waits and short commands retained",
        "fitted_field_dataset_ready": False,
        "fitted_field_blocker": "F and verified closed-loop field execution logs are not supplied",
        "time_definition": "T actions and T+1 observations; source obs[t] maps to O[t+1]; dt from simulator clock",
        "coordinates": {
            "position": "world, metres, gripper0_grip_site",
            "orientation": "world <- robot0_right_hand; quaternion xyzw",
            "angular_velocity": "world, radians/second",
            "orientation_6d": "first two rotation-matrix columns, column-major",
            "actions": "original normalized OSC_POSE commands; NOT measured velocities",
        },
        "image_storage": "relative HDF5 virtual datasets; first image missing and masked; raw source required",
        "split_unit": "full source episode, SHA256(states excluding absolute time, actions); duplicates share split",
        "split_scope": "within-task demonstration generalization; not held-out-task or scene generalization",
        "normalization_file": "normalization.json",
        "report_file": "report.md",
    }
    normalization = {
        "schema": SCHEMA,
        "fit_split": "train",
        "seed": run["config"]["seed"],
        "training_episode_signatures": sorted({e["signature"] for e in entries if e["split"] == "train"}),
        "fields": {key: moment.result() for key, moment in moments.items()},
    }
    atomic_json(output_root / "normalization.json", normalization)
    atomic_json(output_root / "manifest.json", manifest)
    report = [
        "# LIBERO 数据处理实测报告",
        "",
        f"生成时间：{run['started_at']}。",
        "",
        f"状态：`{manifest['status']}`；源回合 {counts['episodes'] + len(rejects)}，成功转换 {counts['episodes']}，"
        f"拒绝 {len(rejects)}；文件错误 {len(source_files_failed)}。",
        "",
        f"动作 {counts['actions']:,}，观测 {counts['observations']:,}；每回合保留末次动作后的观测。",
        "",
        "| 套件 | 训练回合 | 验证回合 | 测试回合 |",
        "|---|---:|---:|---:|",
    ]
    for suite, split in sorted(suite_counts.items()):
        report.append(f"| {suite} | {split['train']} | {split['validation']} | {split['test']} |")
    report += [
        "",
        f"独立恢复状态的时序抽检通过 {counts['alignment_pass']}/{counts['episodes']} 回合；"
        f"每回合抽检至多 {run['config']['alignment_samples']} 个动作后状态。",
        f"位移最大误差的 P50/P95/最大值："
        f"{percentiles['shifted_position_max_m']['p50']:.6f} / {percentiles['shifted_position_max_m']['p95']:.6f} / "
        f"{percentiles['shifted_position_max_m']['max']:.6f} m。",
        "",
        f"保留命令事件 {counts['events']:,} 个，其中运动耦合 {counts['coupled_operations']:,} 个，"
        f"未找到夹爪反馈停稳 {counts['unsettled_operations']:,} 个；这些回合和动作均保留。",
        f"夹爪命令区间 {counts['command_intervals']:,} 个；保留短命令变化 {counts['short_command_changes']:,} 个。",
        f"拟合候选 {counts['candidates']:,} 个；候选预算裁剪 {counts['pruned_candidates']:,} 个。",
        "每个命令区间至少保留一个完整候选，拟合输入覆盖所有原始动作，包括耦合运动和等待。",
        f"相同轨迹重复组 {len(duplicates)}；每组严格分配到同一个数据划分。",
        "",
        "## 使用范围",
        "",
        "规范化示教、两路图像引用、实测速度、夹爪命令事件、候选片段和 H=1/3/4/5 的候选窗口已导出。",
        "初始图像缺失，image_valid[0]=false；末帧仿真状态缺失，simulation_state_valid[-1]=false。",
        "首帧本体由录制状态做正向运动学恢复，其余本体与图像取自动作后记录。",
        "夹爪停稳不是抓取成功；原始 dones/rewards 末值不是独立成功证据。所有 task_success 保持未知。",
        "候选窗口来自源示教，不是拟合场执行日志；目标、场参数和终点训练掩码均为 false。",
        "F、单场物理执行、固定场恢复、真实出口接力与整任务回放未执行；尚不能作为完整拟合场训练集。",
        "归一化仅使用 train 回合，按帧加权，提供物理量逆变换；本划分不代表跨任务泛化。",
        "",
        "## 复现",
        "",
        "```bash",
        run["command"],
        "```",
        "",
        "具体源版本、实现快照、配置、每回合质量指标及拒绝原因见 manifest.json。",
        "",
    ]
    (output_root / "report.md").write_text("\n".join(report))
    return manifest


def run_processing(args):
    started = time.monotonic()
    source_root, output_root = args.source.resolve(), args.output.resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Derived data must be outside the raw source tree")
    config = trajectory.ProcessingConfig(seed=args.seed, max_candidates=args.max_candidates)
    suites = args.suites or list(SUITES)
    source_info_file = source_root / "dataset_source.json"
    source_info = json.loads(source_info_file.read_text()) if source_info_file.is_file() else {}
    if args.observation_timing == "release_profile" and (
        source_info.get("revision") != RELEASE_REVISION or not source_info.get("complete")
    ):
        raise ValueError(
            "Release profile needs the verified pinned LIBERO release; explicitly set --observation-timing post_action for another source"
        )
    selected = sorted(file for suite in suites for file in (source_root / suite).glob("*.hdf5"))
    if args.tasks:
        selected = [
            file for file in selected if file.stem in args.tasks or file.stem.removesuffix("_demo") in args.tasks
        ]
    if not selected:
        raise ValueError("No source task files matched the selection")
    if args.limit_tasks:
        selected = selected[: args.limit_tasks]
    output_root.mkdir(parents=True, exist_ok=True)
    # Never mix task selections in a resumed dataset's global normalization and split manifest.
    selection_file = output_root / "selection.json"
    selection = {
        "source": os.path.relpath(source_root, output_root),
        "files": [p.relative_to(source_root).as_posix() for p in selected],
    }
    if selection_file.is_file() and json.loads(selection_file.read_text()) != selection:
        raise ValueError("Output already belongs to a different selection; use a separate output directory")
    atomic_json(selection_file, selection)
    versions = {
        "libero_commit": _version(args.libero_root.parent.parent),
        "robosuite_root_name": args.robosuite_root.name,
        "mujoco": importlib.metadata.version("mujoco"),
        "robosuite_metadata_sha256": hashlib.sha256((args.robosuite_root / "__init__.py").read_bytes()).hexdigest(),
    }
    pipeline_hash = pipeline_fingerprint()
    snapshot = output_root / "implementation"
    snapshot.mkdir(exist_ok=True)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        (snapshot / path.name).write_bytes(path.read_bytes())
    jobs = [
        {
            "source_file": str(file),
            "source_root": str(source_root),
            "output_file": str(output_root / file.relative_to(source_root)),
            "config": dataclasses.asdict(config),
            "pipeline_sha256": pipeline_hash,
            "libero_root": str(args.libero_root.resolve()),
            "robosuite_root": str(args.robosuite_root.resolve()),
            "resource_versions": versions,
            "source_revision": source_info.get("revision"),
            "resume": args.resume,
        }
        for file in selected
    ]
    results, errors = [], []
    run = {
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": "env -u PYTHONPATH .venv/bin/python -m openpi.datasets.libero " + shlex.join(sys.argv[1:]),
        "source": source_info,
        "source_root_relative": os.path.relpath(source_root, output_root),
        "config": dataclasses.asdict(config),
        "code_commit": _version(ROOT),
        "pipeline_sha256": pipeline_hash,
        "resource_versions": versions,
        "observation_timing_profile": args.observation_timing,
        "environment": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py", "mujoco")},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "file_errors": errors,
    }
    atomic_json(output_root / "run.json", run | {"status": "running", "selected_files": len(jobs)})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_file, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{len(results) + len(errors)}/{len(jobs)}] {result['source']}: "
                    f"{len(result['episodes'])} episodes, {len(result['rejected'])} rejected",
                    flush=True,
                )
            except Exception as error:
                errors.append(
                    {"source": Path(job["source_file"]).relative_to(source_root).as_posix(), "reason": str(error)}
                )
                print(f"ERROR {errors[-1]}", file=sys.stderr, flush=True)
    run["seconds"] = round(time.monotonic() - started, 3)
    if not any(result["episodes"] for result in results):
        atomic_json(output_root / "run.json", run | {"status": "failed"})
        raise RuntimeError(f"No episodes converted: {errors[:3]}")
    manifest = aggregate(output_root, results, run)
    atomic_json(output_root / "run.json", run | {"status": manifest["status"]})
    print(_json({"status": manifest["status"], "counts": manifest["counts"], "output": str(output_root)}), flush=True)
    return 0 if manifest["all_selected_sources_represented"] else 2


def process(args):
    """Own the output lock for the entire conversion and final publication."""
    if args.workers < 1 or (args.limit_tasks is not None and args.limit_tasks < 1):
        raise ValueError("workers and limit-tasks must be positive")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.output) + ".lock", timeout=0):
            return run_processing(args)
    except Timeout as error:
        raise ValueError("Another conversion owns this output directory") from error
