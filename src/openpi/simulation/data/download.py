"""Download one explicitly selected demonstration, with its metadata and camera files."""

# ruff: noqa: RUF001
import json
import pathlib
from urllib.parse import quote

import requests

from openpi.simulation.paths import ROOT

SOURCES = {
    "libero": ("LIBERO · Franka 操作演示", "physical-intelligence/libero", "v2.0"),
    "aloha": ("ALOHA · 人工方块交接演示", "lerobot/aloha_sim_transfer_cube_human", "v2.1"),
    "pusht": ("PushT · 平面推物演示", "lerobot/pusht", "v2.1"),
    "xarm": ("xArm · 物体抬升演示", "lerobot/xarm_lift_medium", "v2.1"),
}
PREVIEW_ROOT = ROOT / "data/dataset_previews"
ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")


def preview_directory(source):
    return PREVIEW_ROOT / SOURCES[source][1].replace("/", "--")


def _relative_path(name):
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("远程元数据包含无效文件路径。")
    return pathlib.Path(path)


def download_preview(
    source, episode_index=0, endpoint=ENDPOINTS[0], progress=lambda _message: None, *, cancelled=lambda: False
):
    if source not in SOURCES or episode_index < 0:
        raise ValueError("请选择有效的数据源与非负回合编号。")
    if endpoint not in ENDPOINTS:
        raise ValueError("请选择官方源或镜像源。")
    _, repo, revision = SOURCES[source]
    root = preview_directory(source)
    root.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        if cancelled():
            raise InterruptedError("下载已停止。")
        relative = _relative_path(name)
        target = root / relative
        if target.is_file():
            return target
        progress(f"正在下载 {name}")
        url = f"{endpoint}/datasets/{repo}/resolve/{revision}/{quote(name, safe='/')}"
        temporary = target.with_suffix(target.suffix + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with requests.get(url, stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if cancelled():
                            raise InterruptedError("下载已停止。")
                        stream.write(chunk)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    info = json.loads(fetch("meta/info.json").read_text())
    fetch("meta/tasks.jsonl")
    episodes_path = fetch("meta/episodes.jsonl")
    available = {
        int(json.loads(line)["episode_index"]) for line in episodes_path.read_text().splitlines() if line.strip()
    }
    if episode_index not in available:
        raise ValueError(f"数据源不包含回合 {episode_index}；请查看有效回合编号。")
    values = {"episode_index": episode_index, "episode_chunk": episode_index // int(info.get("chunks_size", 1000))}
    fetch(info["data_path"].format(**values))
    for camera, feature in info["features"].items():
        if feature["dtype"] == "video":
            fetch(info["video_path"].format(video_key=camera, **values))
    (root / "preview_source.json").write_text(
        json.dumps({"repo_id": repo, "revision": revision, "endpoint": endpoint, "mode": "selected_episodes"}, indent=2)
    )
    progress(f"回合 {episode_index} 已保存；只下载了所选回合及必要元数据。")
    return root
