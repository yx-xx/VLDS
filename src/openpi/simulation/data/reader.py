"""Read recorded demonstrations without loading models or converting whole datasets."""

# ruff: noqa: RUF001 -- Chinese user-facing messages.

import dataclasses
import io
import json
import math
import pathlib
import re

import av
import cv2
import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from openpi.simulation.catalog import operation_type
from openpi.simulation.data.replay import ReplaySource
from openpi.simulation.data.replay import read_replay_manifest
from openpi.simulation.data.replay import resolve_replay_source


@dataclasses.dataclass
class Episode:
    path: pathlib.Path
    name: str
    task: str
    frames: int
    fps: float
    cameras: tuple[str, ...]
    format: str
    group: str = ""
    videos: dict[str, pathlib.Path] = dataclasses.field(default_factory=dict)
    replay: bool = False
    fields: tuple[str, ...] = ()
    replay_source: ReplaySource | None = None
    replay_error: str = ""
    processed_root: pathlib.Path | None = None
    processed_index: int | None = None

    @property
    def kind(self):
        return operation_type(self.task.lower())


@dataclasses.dataclass
class TaskSummary:
    name: str
    local_episodes: list[Episode] = dataclasses.field(default_factory=list)
    total_episodes: int | None = None
    example_episode: int | None = None

    @property
    def kind(self):
        return operation_type(self.name.lower())


@dataclasses.dataclass
class DatasetIndex:
    root: pathlib.Path
    episodes: list[Episode]
    declared_episodes: int | None = None
    notes: list[str] = dataclasses.field(default_factory=list)
    tasks: list[TaskSummary] = dataclasses.field(default_factory=list)
    cameras: tuple[str, ...] = ()
    robot: str = ""
    fields: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.tasks:
            groups = {}
            for episode in self.episodes:
                groups.setdefault(episode.task, TaskSummary(episode.task)).local_episodes.append(episode)
            self.tasks = list(groups.values())
        if not self.fields:
            self.fields = tuple(dict.fromkeys(key for episode in self.episodes for key in episode.fields))
        if not self.cameras:
            self.cameras = tuple(dict.fromkeys(camera for episode in self.episodes for camera in episode.cameras))


def read_jsonlines(path):
    if not path.is_file():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list | tuple):
        return " ".join(text(v) for v in value)
    return str(value)


def _fps(value, default):
    result = float(default if value is None else value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("数据集帧率须为有限的正数。")
    return result


def scan_dataset(path: str | pathlib.Path) -> DatasetIndex:
    root = pathlib.Path(path).expanduser().resolve()
    if not root.exists():
        raise ValueError("数据路径不存在。请选择 LeRobot 数据集根目录或 LIBERO / ALOHA HDF5 文件。")
    if root.is_dir() and (root / "manifest.json").is_file():
        manifest = json.loads((root / "manifest.json").read_text())
        if str(manifest.get("schema", "")).startswith("vlds.libero.trajectories."):
            return _scan_processed_libero(root)
    if root.is_dir() and (root / "meta/info.json").is_file():
        return _scan_lerobot(root)
    files = [root] if root.is_file() else sorted([*root.rglob("*.hdf5"), *root.rglob("*.h5")])
    episodes, notes = [], []
    for file in files:
        if file.suffix.lower() not in (".h5", ".hdf5"):
            continue
        try:
            episodes.extend(_scan_hdf5(file))
        except (OSError, ValueError, KeyError) as error:
            notes.append(f"{file.name}: {error}")
    if not episodes:
        detail = "；".join(notes[:3])
        raise ValueError("没有找到可浏览的演示。支持 LeRobot v2.x、LIBERO HDF5、ALOHA HDF5。" + detail)
    return DatasetIndex(root, episodes, notes=notes)


def _scan_processed_libero(root):
    from openpi.datasets.libero.reader import ProcessedLiberoDataset

    dataset = ProcessedLiberoDataset(root)
    cameras = ("images/agentview_rgb", "images/eye_in_hand_rgb")
    episodes = [
        Episode(
            path=root / entry["shard"],
            name=f"{entry['suite']} / {entry['task']} / {entry['episode']}",
            task=entry["language"],
            frames=entry["observations"],
            fps=entry["actions"] / entry["duration_seconds"],
            cameras=cameras,
            format="VLDS LIBERO HDF5",
            group=entry["group"],
            replay=True,
            fields=(*cameras, "actions", "robot_state/position", "gripper_command", "command_intervals"),
            processed_root=root,
            processed_index=index,
        )
        for index, entry in enumerate(dataset.episodes)
    ]
    return DatasetIndex(
        root,
        episodes,
        declared_episodes=len(episodes),
        robot="Franka Panda",
        notes=[
            "已处理示教；命令区间保留完整动作，动力系统结果在浏览时另行载入。",
            "初始录制图像缺失；终端观测无对应的完整仿真状态。",
        ],
    )


def _scan_hdf5(path):
    episodes = []
    with h5py.File(path, "r") as file:
        if "data" in file:
            data = file["data"]
            info = json.loads(text(data.attrs.get("problem_info", "{}")))
            task = text(info.get("language_instruction", path.stem.replace("_demo", "").replace("_", " ")))
            args = json.loads(text(data.attrs.get("env_args", "{}")))
            fps = _fps(args.get("env_kwargs", {}).get("control_freq", 20), 20)
            for name in sorted(data, key=lambda k: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", k)]):
                group = data[name]
                if not isinstance(group, h5py.Group):
                    continue
                cameras = tuple(
                    f"obs/{k}"
                    for k, v in group.get("obs", {}).items()
                    if isinstance(v, h5py.Dataset) and ("rgb" in k or "image" in k) and "depth" not in k
                )
                lengths = [len(group[k]) for k in (*cameras, "actions", "states") if k in group]
                if not lengths or min(lengths) <= 0:
                    continue
                replay = "states" in group and bool(group.attrs.get("model_file", ""))
                if cameras or replay:
                    episodes.append(
                        Episode(
                            path,
                            f"{path.stem} / {name}",
                            text(group.attrs.get("task", task)),
                            min(lengths),
                            _fps(group.attrs.get("fps", fps), fps),
                            cameras,
                            "LIBERO HDF5",
                            f"data/{name}",
                            replay=replay,
                            fields=tuple(
                                key
                                for key in (
                                    *cameras,
                                    "actions",
                                    "states",
                                    "obs/joint_states",
                                    "obs/ee_states",
                                    "obs/gripper_states",
                                )
                                if key in group
                            ),
                        )
                    )
        elif "observations/images" in file:
            cameras = tuple(f"observations/images/{key}" for key in file["observations/images"] if "depth" not in key)
            lengths = [len(file[k]) for k in (*cameras, "action", "observations/qpos") if k in file]
            if cameras and lengths and min(lengths) > 0:
                task = text(file.attrs.get("task", path.parent.name.replace("_", " ")))
                episodes.append(
                    Episode(
                        path,
                        path.stem,
                        task,
                        min(lengths),
                        _fps(file.attrs.get("fps", 50), 50),
                        cameras,
                        "ALOHA HDF5",
                        fields=tuple(key for key in (*cameras, "action", "observations/qpos") if key in file),
                    )
                )
    return episodes


def _scan_lerobot(root):
    info = json.loads((root / "meta/info.json").read_text())
    if not str(info.get("codebase_version", "v2.0")).startswith("v2"):
        raise ValueError("此浏览器支持 LeRobot v2.x；当前目录使用其他布局。请加载 v2 数据或原始 HDF5。")
    fps = _fps(info.get("fps", 30), 30)
    features = info.get("features", {})
    cameras = tuple(k for k, value in features.items() if value.get("dtype") in ("video", "image"))
    tasks = {int(t["task_index"]): t["task"] for t in read_jsonlines(root / "meta/tasks.jsonl")}
    records = read_jsonlines(root / "meta/episodes.jsonl")
    if not records:
        records = [
            {"episode_index": int(p.stem.split("_")[-1])} for p in sorted(root.glob("data/**/episode_*.parquet"))
        ]
    episodes = []
    replay_links, notes = read_replay_manifest(root)
    missing_videos = 0
    for record in records:
        index = int(record["episode_index"])
        values = {"episode_index": index, "episode_chunk": index // int(info.get("chunks_size", 1000))}
        path = root / info["data_path"].format(**values)
        if not path.is_file():
            continue  # Partial, intentionally downloaded datasets are supported.
        videos = {}
        available = []
        for camera in cameras:
            if features[camera]["dtype"] == "video":
                video = root / info["video_path"].format(video_key=camera, **values)
                if not video.is_file():
                    missing_videos += 1
                    continue
                videos[camera] = video
            available.append(camera)
        link = replay_links.get(str(index))
        replay_source, replay_error = None, ""
        if not available and link is None:
            continue
        frames = pq.read_metadata(path).num_rows
        if not frames:
            continue
        if link is not None:
            try:
                replay_source = resolve_replay_source(root, link, frames)
            except ValueError as error:
                replay_error = str(error)
                notes.append(f"回合 {index} 三维参考回放不可用：{error}")
        if not available and replay_source is None:
            continue
        instruction = text(record.get("tasks", []))
        if not instruction:
            task_id = pq.read_table(path, columns=["task_index"])["task_index"][0].as_py() if tasks else None
            instruction = str(tasks.get(task_id, "未标注任务"))
        episodes.append(
            Episode(
                path,
                f"episode_{index:06d}",
                instruction,
                frames,
                fps,
                tuple(available),
                "LeRobot " + str(info.get("codebase_version", "v2.x")),
                videos=videos,
                replay=replay_source is not None,
                replay_source=replay_source,
                replay_error=replay_error,
            )
        )
    if missing_videos:
        notes.append(f"有 {missing_videos} 个相机文件尚未下载。")
    if not episodes:
        notes.append("本地没有可播放的回合。可以先浏览任务目录，再按需下载演示。")
    return DatasetIndex(
        root,
        episodes,
        info.get("total_episodes"),
        notes,
        tasks=_task_summaries(tasks, records, episodes),
        cameras=cameras,
        robot=str(info.get("robot_type", "")),
        fields=tuple(features),
    )


def _task_summaries(tasks, records, episodes):
    """Include undownloaded tasks, without reading remote media or all frame data."""
    groups = {name: TaskSummary(name) for name in tasks.values()}
    local = {episode.name: episode for episode in episodes}
    memberships = {}
    counts = {}
    complete = bool(records)
    for record in records:
        name = f"episode_{int(record['episode_index']):06d}"
        names = record.get("tasks", [])
        if isinstance(names, str):
            names = [names]
        if not names and name in local:
            names = [local[name].task]
        if not names and len(tasks) == 1:
            names = list(tasks.values())
        complete = complete and bool(names)
        memberships[name] = names
        for task in set(names):
            summary = groups.setdefault(task, TaskSummary(task))
            counts[task] = counts.get(task, 0) + 1
            if summary.example_episode is None:
                summary.example_episode = int(record["episode_index"])
    for episode in episodes:
        for task in memberships.get(episode.name) or [episode.task]:
            groups.setdefault(task, TaskSummary(task)).local_episodes.append(episode)
    for name, summary in groups.items():
        summary.total_episodes = counts.get(name, 0) if complete else counts.get(name)
    return list(groups.values())


def rgb_image(value, root=None):
    """Decode one image only; arrays are RGB, encoded bytes use their native codec."""
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path"):
            path = pathlib.Path(value["path"])
            with Image.open(path if path.is_absolute() else root / path) as image:
                return np.asarray(image.convert("RGB"))
        else:
            raise ValueError("此图像没有 bytes 或 path。")
    if isinstance(value, bytes | bytearray | memoryview):
        with Image.open(io.BytesIO(value)) as image:
            return np.asarray(image.convert("RGB"))
    array = np.asarray(value)
    if array.ndim == 1:
        decoded = cv2.imdecode(array.astype(np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("无法解码此帧的压缩图像。")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)
    if array.dtype.kind == "f" and array.size and array.max() <= 1:
        array = array * 255
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"不支持的图像尺寸：{array.shape}")
    if array.shape[-1] == 1:
        return np.repeat(array, 3, axis=-1)
    return array[..., :3]


class EpisodeReader:
    """Single-thread owner; keeps at most one Parquet row group and one active video decoder."""

    def __init__(self, episode: Episode):
        self.episode = episode
        self.segmentation = None
        self.processed_group = None
        if episode.processed_root is not None:
            from openpi.datasets.libero.reader import ProcessedLiberoDataset
            from openpi.datasets.libero.reader import read_record

            dataset = ProcessedLiberoDataset(episode.processed_root)
            self.file, self.processed_group = dataset._open(episode.processed_index)  # noqa: SLF001 -- validated source/VDS access.
            group = self.processed_group
            try:
                if "command_intervals" in group:
                    intervals = read_record(group, "command_intervals")
                    candidates = read_record(group, "candidates")
                    self.segmentation = {
                        "intervals": intervals,
                        "time": group["time"][:].tolist(),
                        "candidate_counts": [
                            sum(c["command_interval_id"] == i for c in candidates) for i in range(len(intervals))
                        ],
                    }
                source = group["source"]
                episode.replay = "states" in source and bool(source.attrs.get("model_file", ""))
                if episode.replay:
                    source_path = (episode.path.parent / group.attrs["source_file"]).resolve()
                    indices = group["simulation_state_index"][:]
                    valid = group["simulation_state_valid"][:]
                    if len(indices) != episode.frames or len(valid) != episode.frames:
                        raise ValueError("处理后仿真帧映射长度错误。")
                    if np.any(indices[valid] < 0) or np.any(indices[valid] >= len(source["states"])):
                        raise ValueError("处理后仿真帧映射越界。")
                    episode.replay_source = ReplaySource(
                        source_path,
                        text(group.attrs["source_group"]),
                        tuple(
                            int(index) if available else None for index, available in zip(indices, valid, strict=True)
                        ),
                    )
                else:
                    episode.replay_error = "来源缺少完整场景 XML 或仿真状态。"
            except Exception:
                self.file.close()
                raise
        else:
            self.file = h5py.File(episode.path, "r") if episode.format.endswith("HDF5") else None
        self.parquet = pq.ParquetFile(episode.path) if self.file is None else None
        self.table = None
        self.table_start = -1
        self.table_group = -1
        self.video = None
        self.video_camera = None
        self.video_next = -1
        self.video_iterator = None
        self.video_stream = None
        self.root = next((p for p in episode.path.parents if (p / "meta/info.json").is_file()), episode.path.parent)
        from openpi.datasets.dynamics.playback import available_profiles
        from openpi.datasets.dynamics.playback import load_for_episode
        from openpi.datasets.dynamics.playback import segmentation

        preferred = "coherent" if "coherent" in available_profiles(episode) else "replay"
        bundle = load_for_episode(episode, profile=preferred)
        if bundle is not None:
            self.segmentation = segmentation(bundle)

    def _row(self, frame, camera):
        columns = [
            key
            for key in self.parquet.schema_arrow.names
            if key
            in (
                "action",
                "actions",
                "state",
                "observation.state",
                "simulation.state",
                "timestamp",
                "task_index",
                "next.done",
                "next.reward",
                "next.success",
            )
            or (key == camera and camera not in self.episode.videos)
        ]
        if (
            self.table is not None
            and self.table.column_names == columns
            and self.table_start <= frame < self.table_start + self.table.num_rows
        ):
            return frame - self.table_start
        start = 0
        for group in range(self.parquet.num_row_groups):
            length = self.parquet.metadata.row_group(group).num_rows
            if frame < start + length:
                self.table = self.parquet.read_row_group(group, columns=columns)
                self.table_start, self.table_group = start, group
                return frame - start
            start += length
        raise IndexError(frame)

    def read(self, frame: int, camera: str | None):
        episode = self.episode
        if not 0 <= frame < episode.frames:
            raise ValueError("帧编号超出此回合范围。")
        values = {}
        image = None
        if self.processed_group is not None:
            group = self.processed_group
            available = bool(group["image_valid"][frame])
            if camera and available:
                image = rgb_image(group[camera][frame])
            values = {
                "time": float(group["time"][frame]),
                "image_valid": available,
                "simulation_state_valid": bool(group["simulation_state_valid"][frame]),
                **{
                    f"robot_state/{key}": np.asarray(value[frame]).tolist()
                    for key, value in group["robot_state"].items()
                },
            }
            if frame < len(group["actions"]):
                values["actions"] = group["actions"][frame].tolist()
                values["gripper_command"] = int(group["gripper_command"][frame])
            else:
                values["note"] = "终端观测：没有下一动作，也没有对应的完整仿真状态。"
        elif self.file is not None:
            group = self.file[episode.group] if episode.group else self.file
            if camera:
                image = rgb_image(group[camera][frame])
            for key in (
                "action",
                "actions",
                "observations/qpos",
                "states",
                "timestamp",
                "timestamps",
                "rewards",
                "dones",
                "obs/joint_states",
                "obs/ee_states",
                "obs/gripper_states",
            ):
                if key in group and group[key].ndim and len(group[key]) > frame:
                    values[key] = np.asarray(group[key][frame]).tolist()
        else:
            row = self._row(frame, camera)
            for key in (
                "action",
                "actions",
                "state",
                "observation.state",
                "simulation.state",
                "timestamp",
                "task_index",
                "next.done",
                "next.reward",
                "next.success",
            ):
                if key in self.table.column_names:
                    values[key] = self.table[key][row].as_py()
            if camera in episode.videos:
                image = self._video_frame(camera, frame, values.get("timestamp", frame / episode.fps))
            elif camera:
                image = rgb_image(self.table[camera][row].as_py(), self.root)
        return image, values

    def _video_frame(self, camera, frame, target_time):
        if self.video_camera != camera:
            if self.video is not None:
                self.video.close()
            self.video = av.open(str(self.episode.videos[camera]))
            self.video_stream = self.video.streams.video[0]
            self.video_stream.codec_context.thread_count = 2
            self.video_iterator = iter(self.video.decode(self.video_stream))
            self.video_camera, self.video_next = camera, 0
        if frame != self.video_next:
            self.video.seek(int(target_time / self.video_stream.time_base), stream=self.video_stream, backward=True)
            self.video_iterator = iter(self.video.decode(self.video_stream))
        for decoded in self.video_iterator:
            timestamp = float(decoded.pts * self.video_stream.time_base) if decoded.pts is not None else target_time
            if timestamp >= target_time - 0.5 / self.episode.fps:
                self.video_next = frame + 1
                return decoded.to_ndarray(format="rgb24")
        raise ValueError(f"视频缺少第 {frame} 帧，无法与此回合数据对齐。")

    def close(self):
        if self.video is not None:
            self.video.close()
        if self.file is not None:
            self.file.close()
        if self.parquet is not None:
            self.parquet.close()
        self.table = None
