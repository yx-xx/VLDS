"""Explicit links from processed episodes to recorded MuJoCo trajectories."""

# ruff: noqa: RUF001 -- Chinese validation messages.
import dataclasses
import json
from pathlib import Path

import h5py


@dataclasses.dataclass(frozen=True)
class ReplaySource:
    path: Path
    group: str
    frame_indices: tuple[int, ...]

    def frame(self, index: int) -> int:
        return self.frame_indices[index]


def read_replay_manifest(root: Path) -> tuple[dict, list[str]]:
    path = root / "meta/replay.json"
    if not path.is_file():
        return {}, []
    try:
        manifest = json.loads(path.read_text())
        if manifest.get("version") != 1 or not isinstance(manifest.get("episodes"), dict):
            raise ValueError("需要 version=1 和 episodes 对象")
        return manifest["episodes"], []
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {}, [f"三维参考回放配置无效：{error}"]


def resolve_replay_source(root: Path, record: dict, frames: int) -> ReplaySource:
    """Require an explicit, exact frame alignment; never infer scene state from robot qpos."""
    try:
        path = Path(record["path"]).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        group_name = record["group"]
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("来源回合 group 不能为空")
        with h5py.File(path, "r") as file:
            group = file[group_name]
            states = group["states"]
            if states.ndim != 2 or not len(states) or not group.attrs.get("model_file", ""):
                raise ValueError("来源需要二维 states 和 model_file XML")
            indices = record.get("frame_indices")
            if indices is None:
                if len(states) != frames:
                    raise ValueError("来源与处理后帧数不同，必须提供 frame_indices")
                indices = list(range(frames))
            if not isinstance(indices, list) or len(indices) != frames:
                raise ValueError("frame_indices 长度须等于处理后回合帧数")
            if any(type(index) is not int or not 0 <= index < len(states) for index in indices):
                raise ValueError("frame_indices 必须是来源范围内的整数帧编号")
        return ReplaySource(path, group_name, tuple(indices))
    except (KeyError, TypeError, OSError) as error:
        raise ValueError(f"无法读取三维回放来源：{error}") from error
