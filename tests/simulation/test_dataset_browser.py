"""Check recorded image/action alignment, partial datasets, seeking, and state replay contracts."""

# ruff: noqa: SLF001 -- Exercise format and process boundaries.
import io
import json
import os
import subprocess
import sys
import textwrap
from unittest import mock

import av
import h5py
import mujoco
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.simulation.data import download as dataset_download
from openpi.simulation.data.reader import EpisodeReader
from openpi.simulation.data.reader import rgb_image
from openpi.simulation.data.reader import scan_dataset
from openpi.simulation.workers import replay as dataset_replay_worker


def png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def lerobot_fixture(root, *, video=False):
    (root / "meta").mkdir()
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "chunks_size": 1000,
        "total_episodes": 2,
        "features": {"image": {"dtype": "video" if video else "image"}},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "push the block"}) + "\n")
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i}) for i in [0, 1]))
    table = pa.table(
        {
            "image": [{"bytes": png((i * 40, 0, 0)), "path": None} for i in range(4)],
            "state": [[float(i), 1.0] for i in range(4)],
            "actions": [[float(i), -1.0] for i in range(4)],
            "task_index": [0] * 4,
            "timestamp": [i / 10 for i in range(4)],
        }
    )
    path = root / "data/chunk-000/episode_000000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(table, path, row_group_size=2)
    return info


def test_libero_episode_order_images_actions_and_3d_availability(tmp_path):
    path = tmp_path / "task_demo.hdf5"
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "open the drawer"})
        for number in [10, 2]:
            demo = data.create_group(f"demo_{number}")
            demo.create_dataset("actions", data=np.arange(21).reshape(3, 7))
            demo.create_dataset("states", data=np.zeros((3, 3)))
            demo.create_dataset("obs/agentview_rgb", data=np.full((3, 4, 4, 3), number, dtype=np.uint8))
            demo.create_dataset("obs/eye_in_hand_rgb", data=np.full((3, 4, 4, 3), number + 1, dtype=np.uint8))
            if number == 2:
                demo.attrs["model_file"] = "<mujoco/>"
    index = scan_dataset(path)
    assert [ep.group for ep in index.episodes] == ["data/demo_2", "data/demo_10"]
    assert [ep.replay for ep in index.episodes] == [True, False]
    assert index.episodes[0].kind == "开合操作"
    reader = EpisodeReader(index.episodes[0])
    try:
        image, values = reader.read(2, "obs/eye_in_hand_rgb")
        np.testing.assert_array_equal(image, 3)
        assert values["actions"] == list(range(14, 21))
        with pytest.raises(ValueError, match="范围"):
            reader.read(3, "obs/agentview_rgb")
    finally:
        reader.close()


@pytest.mark.parametrize("compressed", [False, True])
def test_aloha_raw_and_compressed_images_remain_rgb(tmp_path, compressed):
    path = tmp_path / "episode_0.hdf5"
    with h5py.File(path, "w") as file:
        file.attrs["fps"] = 50
        file.create_dataset("action", data=np.arange(28).reshape(2, 14))
        file.create_dataset("observations/qpos", data=np.ones((2, 14)))
        if compressed:
            images = file.create_dataset("observations/images/top", (2,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for i in range(2):
                images[i] = np.frombuffer(png((255, 0, 0)), dtype=np.uint8)
        else:
            file.create_dataset("observations/images/top", data=np.tile([255, 0, 0], (3, 4, 4, 1)).astype(np.uint8))
    ep = scan_dataset(path).episodes[0]
    assert ep.frames == 2
    reader = EpisodeReader(ep)
    try:
        image, values = reader.read(1, ep.cameras[0])
        np.testing.assert_array_equal(image[0, 0], [255, 0, 0])
        assert values["action"] == list(range(14, 28))
        assert values["observations/qpos"] == [1.0] * 14
    finally:
        reader.close()


def test_partial_lerobot_with_embedded_images_and_row_group_seeking(tmp_path):
    lerobot_fixture(tmp_path)
    index = scan_dataset(tmp_path)
    assert index.declared_episodes == 2
    assert len(index.episodes) == 1
    ep = index.episodes[0]
    assert ep.task == "push the block"
    reader = EpisodeReader(ep)
    try:
        for frame in [0, 3, 2, 1, 0]:
            image, values = reader.read(frame, "image")
            np.testing.assert_array_equal(image[0, 0], [frame * 40, 0, 0])
            assert values["state"][0] == frame
            assert values["actions"][0] == frame
            assert reader.table.num_rows == 2
    finally:
        reader.close()


def test_lerobot_video_seeks_in_both_directions(tmp_path):
    lerobot_fixture(tmp_path, video=True)
    video = tmp_path / "videos/chunk-000/image/episode_000000.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for i in range(4):
            pixels = np.full((16, 16, 3), i * 60, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    ep = scan_dataset(tmp_path).episodes[0]
    reader = EpisodeReader(ep)
    try:
        for frame in [0, 1, 3, 2, 0]:
            image, values = reader.read(frame, "image")
            assert abs(float(image.mean()) - frame * 60) < 4
            assert values["actions"][0] == frame
    finally:
        reader.close()


def test_missing_videos_and_unsupported_layout_have_explanations(tmp_path):
    lerobot_fixture(tmp_path, video=True)
    index = scan_dataset(tmp_path)
    assert not index.episodes
    assert len(index.tasks) == 1
    assert not index.tasks[0].local_episodes
    assert index.tasks[0].total_episodes == 2
    assert any("没有可播放" in note for note in index.notes)
    info = json.loads((tmp_path / "meta/info.json").read_text())
    info["codebase_version"] = "v3.0"
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="v2"):
        scan_dataset(tmp_path)


def test_3d_only_demonstration_and_invalid_fps(tmp_path):
    path = tmp_path / "states.hdf5"
    with h5py.File(path, "w") as file:
        demo = file.create_group("data/demo_0")
        demo.attrs["model_file"] = "<mujoco/>"
        demo.create_dataset("states", data=np.zeros((3, 5)))
    ep = scan_dataset(path).episodes[0]
    assert ep.replay
    assert ep.cameras == ()
    assert ep.fields == ("states",)
    reader = EpisodeReader(ep)
    try:
        image, values = reader.read(0, None)
        assert image is None
        assert values["states"] == [0.0] * 5
    finally:
        reader.close()
    with h5py.File(path, "a") as file:
        file["data"].attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": -1}})
    with pytest.raises(ValueError, match="帧率"):
        scan_dataset(path)


def test_rgb_image_supports_lerobot_path_and_chw(tmp_path):
    (tmp_path / "image.png").write_bytes(png((0, 0, 255)))
    np.testing.assert_array_equal(rgb_image({"path": "image.png"}, tmp_path)[0, 0], [0, 0, 255])
    assert rgb_image(np.ones((3, 6, 8), dtype=np.float32)).shape == (6, 8, 3)


def test_download_only_selected_episode_and_required_files(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_download, "PREVIEW_ROOT", tmp_path)
    info = {
        "chunks_size": 1000,
        "data_path": "data/{episode_index}.parquet",
        "video_path": "videos/{video_key}/{episode_index}.mp4",
        "features": {"top": {"dtype": "video"}, "state": {"dtype": "float32"}},
    }
    payloads = {
        "meta/info.json": json.dumps(info).encode(),
        "meta/tasks.jsonl": b"{}\n",
        "meta/episodes.jsonl": b'{"episode_index": 0}\n{"episode_index": 1}\n',
        "data/1.parquet": b"parquet",
        "videos/top/1.mp4": b"video",
    }
    calls = []

    def get(url, **_kwargs):
        name = url.split("/resolve/v2.1/")[1]
        calls.append(name)
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [payloads[name]]
        return response

    monkeypatch.setattr(dataset_download.requests, "get", get)
    path = dataset_download.download_preview("aloha", 1)
    assert set(calls) == set(payloads)
    assert not (path / "data/0.parquet").exists()
    assert not list(path.rglob("*.part"))
    calls.clear()
    dataset_download.download_preview("aloha", 1)
    assert calls == []
    with pytest.raises(ValueError, match="不包含"):
        dataset_download.download_preview("aloha", 20)


@pytest.mark.parametrize("path", ["/outside", "../outside", "meta/../../outside"])
def test_remote_metadata_paths_stay_inside_cache(path):
    with pytest.raises(ValueError, match="路径"):
        dataset_download._relative_path(path)


@pytest.mark.parametrize("checkout", ["/author/LIBERO/libero/libero", "/author/libero-dev/chiliocosm"])
def test_replay_loads_assets_from_original_dataset_checkout(tmp_path, monkeypatch, checkout):
    assets = tmp_path / "installed/assets"
    (assets / "textures").mkdir(parents=True)
    texture = assets / "textures/checker.png"
    texture.write_bytes(png((255, 0, 0)))
    libero = mock.MagicMock()
    libero.get_libero_path.return_value = str(assets)
    monkeypatch.setitem(sys.modules, "libero", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "libero.libero", libero)
    xml = (
        '<mujoco><asset><texture name="checker" type="2d" '
        f'file="{checkout}/assets/textures/checker.png"/></asset>'
        '<worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
    )
    relocated = dataset_replay_worker.relocate_assets(xml, tmp_path / "demo.hdf5")
    assert str(texture) in relocated
    # Loading the model catches unresolved resources beyond string replacement.
    model = mujoco.MjModel.from_xml_string(relocated)
    assert model.ntex == 1


def test_replay_sets_positions_and_velocities_without_simulating():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="slide"/><geom size=".1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    dataset_replay_worker.set_recorded_state(model, data, [2.5, 0.2, 0.4])
    assert data.time == 2.5
    np.testing.assert_allclose(data.qpos, [0.2])
    np.testing.assert_allclose(data.qvel, [0.4])
    for state in [[0, 0], [0, np.nan, 0]]:
        with pytest.raises(ValueError, match="状态"):
            dataset_replay_worker.set_recorded_state(model, data, state)


def test_task_directory_includes_undownloaded_and_multitask_episodes(tmp_path):
    lerobot_fixture(tmp_path)
    tasks = ["push the block", "open the drawer", "turn on the stove"]
    (tmp_path / "meta/tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": number, "task": task}) for number, task in enumerate(tasks))
    )
    records = [
        {"episode_index": 0, "tasks": tasks[:2]},
        {"episode_index": 1, "tasks": tasks[1:]},
        {"episode_index": 2, "tasks": [tasks[2]]},
    ]
    (tmp_path / "meta/episodes.jsonl").write_text("\n".join(json.dumps(record) for record in records))
    index = scan_dataset(tmp_path)
    assert [task.name for task in index.tasks] == tasks
    assert [len(task.local_episodes) for task in index.tasks] == [1, 1, 0]
    assert [task.total_episodes for task in index.tasks] == [1, 2, 2]
    assert [task.example_episode for task in index.tasks] == [0, 0, 1]
    assert index.tasks[0].local_episodes[0] is index.tasks[1].local_episodes[0]
    assert index.cameras == ("image",)


def test_unknown_task_episode_counts_are_not_reported_as_zero(tmp_path):
    lerobot_fixture(tmp_path)
    (tmp_path / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "push the block"})
        + "\n"
        + json.dumps({"task_index": 1, "task": "open the drawer"})
    )
    index = scan_dataset(tmp_path)
    assert index.tasks[1].total_episodes is None
    assert index.tasks[1].example_episode is None
    assert not index.tasks[1].local_episodes


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires a desktop display")
def test_visual_browser_task_selection_and_navigation(tmp_path):
    lerobot_fixture(tmp_path)
    tasks = ["push the block", "turn on the stove"]
    (tmp_path / "meta/tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": number, "task": task}) for number, task in enumerate(tasks))
    )
    (tmp_path / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps({"episode_index": number, "tasks": [task]}) for number, task in enumerate(tasks))
    )
    (tmp_path / "preview_source.json").write_text(json.dumps({"repo_id": "physical-intelligence/libero"}))
    code = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path
        from unittest import mock

        from openpi.simulation.ui.inspector import InspectorWindow
        from openpi.simulation.runtime.controller import InspectorController
        from openpi.simulation.ui.tk_runtime import tk

        root = tk.Tk()
        root.withdraw()
        control = InspectorController()
        window = InspectorWindow(root, control, dataset_path=sys.argv[1])
        browser = window.dataset_browser

        def wait_for(predicate):
            deadline = time.monotonic() + 15
            while not predicate():
                root.update()
                if time.monotonic() > deadline:
                    raise AssertionError(browser.status.cget("text"))
                time.sleep(0.01)
            root.update()

        try:
            wait_for(lambda: browser.photo is not None)
            assert len(browser.task_rows) == 2
            for selected in (0, 1, 0, 1):
                window.tabs.select(selected)
                root.update()
                sizes = [(button.winfo_reqwidth(), button.winfo_reqheight()) for button in window.mode_switches]
                assert sizes[selected][0] > sizes[1-selected][0], sizes
                assert sizes[selected][1] > sizes[1-selected][1], sizes
                assert window.active_page.get() == selected

            browser.search.set("炉灶")
            wait_for(lambda: browser.episode is None)
            assert len(browser.task_rows) == 1
            assert browser.photo is None
            assert browser.task_download.winfo_manager() == "pack"
            assert browser.play_button.instate(["disabled"])
            with mock.patch.object(browser, "download") as download:
                browser.download_selected_task()
                download.assert_called_once_with()
                assert browser.remote_episode.get() == "1"
                assert browser.online_box.current() == 0

            browser.only_local.set(True)
            browser._filter_tasks()
            assert not browser.task_rows
            browser.search.set("")
            wait_for(lambda: browser.photo is not None)
            assert len(browser.task_rows) == 1
            browser.only_local.set(False)
            browser._filter_tasks()
            assert len(browser.task_rows) == 2
            browser.toggle_play()
            window.tabs.select(0)
            root.update()
            assert not browser.playing

            browser.search.set("炉灶")
            browser.load(Path(sys.argv[1]) / "missing")
            wait_for(lambda: "不存在" in browser.status.cget("text"))
            browser.load(sys.argv[1])
            wait_for(lambda: browser.photo is not None)
            assert len(browser.task_rows) == 2
            assert browser.episode.name == "episode_000000"
            print("visual browser checks passed")
        finally:
            browser.close()
            control.close()
            root.destroy()
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "visual browser checks passed" in process.stdout
