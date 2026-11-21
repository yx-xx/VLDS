"""Check processed-to-source alignment and atomic live updates using real data files."""

# ruff: noqa: SLF001 -- Process and reader boundary tests.
import contextlib
import io
import json
import os
import subprocess
import sys
import textwrap
from unittest import mock

import h5py
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.simulation.data.reader import EpisodeReader
from openpi.simulation.data.reader import scan_dataset
from openpi.simulation.data.watch import StableFileWatch
from openpi.simulation.data.watch import file_signature
from openpi.simulation.runtime.replay import DatasetReplay
from openpi.simulation.workers.replay import RecordedTrajectory
from openpi.simulation.workers.replay import replace_trajectory

MODEL = '<mujoco><worldbody><body><joint type="slide"/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'


def recording(path, *, offset=0, invalid=False):
    with h5py.File(path, "w") as file:
        demo = file.create_group("data/demo_0")
        demo.attrs["model_file"] = MODEL
        states = np.array([[i / 20, offset + i / 10, 0.0] for i in range(8)])
        if invalid:
            states[2, 1] = np.nan
        demo.create_dataset("states", data=states)
    return path


def processed(root, source, *, cameras=True, indices=None):
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    image = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(image, format="PNG")
    columns = {"actions": [[42.0 + i] for i in range(4)], "state": [[float(i)] for i in range(4)]}
    if cameras:
        columns["image"] = [{"bytes": image.getvalue(), "path": None}] * 4
    pq.write_table(pa.table(columns), root / "data/episode_000000.parquet")
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "features": {"image": {"dtype": "image"}} if cameras else {},
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "open the drawer"}))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["open the drawer"]}))
    manifest = {
        "version": 1,
        "episodes": {
            "0": {
                "path": os.path.relpath(source, root),
                "group": "data/demo_0",
                "frame_indices": indices if indices is not None else [7, 4, 1, 0],
            }
        },
    }
    (root / "meta/replay.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("cameras", [False, True])
def test_processed_episode_preserves_numeric_data_and_explicit_source_alignment(tmp_path, cameras):
    source = recording(tmp_path / "source.hdf5")
    root = tmp_path / "processed"
    processed(root, source, cameras=cameras)
    index = scan_dataset(root)
    assert not index.notes
    episode = index.episodes[0]
    assert episode.replay
    assert episode.replay_source.path == source
    assert episode.replay_source.frame_indices == (7, 4, 1, 0)
    reader = EpisodeReader(episode)
    try:
        image, values = reader.read(2, "image" if cameras else None)
        assert values["actions"] == [44.0]
        assert values["state"] == [2.0]
        assert (image is not None) == cameras
    finally:
        reader.close()
    control = DatasetReplay()
    control.process = mock.Mock()
    control.process.poll.return_value = None
    control.process.stdin = io.StringIO()
    control.source = episode.replay_source
    control.send("frame", frame=2)
    assert json.loads(control.process.stdin.getvalue()) == {"command": "frame", "frame": 1, "timeline_frame": 2}


@pytest.mark.parametrize("indices", [[0, 1], [0, 1, 2, 8], [0, 1, 2, -1], [0, 1, 2, 1.5], [0, 1, 2, True]])
def test_invalid_source_mapping_disables_only_3d_and_explains_problem(tmp_path, indices):
    source = recording(tmp_path / "source.hdf5")
    root = tmp_path / "processed"
    processed(root, source, indices=indices)
    index = scan_dataset(root)
    assert len(index.episodes) == 1
    assert not index.episodes[0].replay
    assert "frame_indices" in index.episodes[0].replay_error
    assert index.notes
    reader = EpisodeReader(index.episodes[0])
    try:
        assert reader.read(0, "image")[0] is not None
    finally:
        reader.close()


def test_source_length_requires_explicit_alignment(tmp_path):
    root = tmp_path / "processed"
    manifest = processed(root, recording(tmp_path / "source.hdf5"))
    del manifest["episodes"]["0"]["frame_indices"]
    (root / "meta/replay.json").write_text(json.dumps(manifest))
    assert "帧数不同" in scan_dataset(root).episodes[0].replay_error


def test_atomic_reload_keeps_native_data_object_and_replaces_recorded_states(tmp_path):
    source = recording(tmp_path / "source.hdf5")
    current = RecordedTrajectory(source, "data/demo_0", 2)
    original_data = current.data
    replacement_path = recording(tmp_path / "replacement.hdf5", offset=1)
    replacement_path.replace(source)
    replacement = RecordedTrajectory(source, "data/demo_0", 2, previous=current)
    assert replacement.model is current.model
    viewer = mock.Mock()
    viewer.handle.lock.side_effect = contextlib.nullcontext
    replacement = replace_trajectory(viewer, current, replacement)
    try:
        assert not current.file.id.valid
        assert replacement.data is original_data
        np.testing.assert_allclose(replacement.data.qpos, [1.2])
        replacement.seek(7)
        np.testing.assert_allclose(replacement.data.qpos, [1.7])
        viewer.attach_native.assert_not_called()
        viewer.detach.assert_not_called()
    finally:
        replacement.close()


def test_bad_replacement_does_not_destroy_current_scene(tmp_path):
    source = recording(tmp_path / "source.hdf5")
    current = RecordedTrajectory(source, "data/demo_0", 2)
    recording(tmp_path / "bad.hdf5", invalid=True).replace(source)
    try:
        with pytest.raises(ValueError, match="无效"):
            RecordedTrajectory(source, "data/demo_0", 2, previous=current)
        assert current.file.id.valid
        np.testing.assert_allclose(current.data.qpos, [0.2])
    finally:
        current.close()


def test_watch_accepts_atomic_replacement_and_debounces_in_place_writes(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("old")
    watch = StableFileWatch()
    initial = file_signature([path])
    assert not watch.changed(initial)
    replacement = tmp_path / "replacement.json"
    replacement.write_text("new")
    replacement.replace(path)
    updated = file_signature([path])
    assert updated != initial
    assert watch.changed(updated)
    assert not watch.changed(updated)
    path.write_text("still being written")
    in_place = file_signature([path])
    assert not watch.changed(in_place)
    assert watch.changed(in_place)
    assert not watch.changed(in_place)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires a desktop display")
def test_browser_refreshes_atomic_processed_output_and_preserves_frame(tmp_path):
    root = tmp_path / "processed"
    processed(root, recording(tmp_path / "source.hdf5"))
    code = textwrap.dedent("""
        import time
        import sys
        from pathlib import Path
        from unittest import mock
        import pyarrow as pa
        import pyarrow.parquet as pq
        from openpi.simulation.ui.inspector import InspectorWindow
        from openpi.simulation.ui.dataset_browser import DatasetBrowser
        from openpi.simulation.runtime.controller import InspectorController
        from openpi.simulation.ui.tk_runtime import tk
        root = tk.Tk()
        root.withdraw()
        control = InspectorController()
        with mock.patch.object(DatasetBrowser, "refresh_library"):
            window = InspectorWindow(root, control, dataset_path=sys.argv[1])
        browser = window.dataset_browser
        def wait_for(predicate):
            deadline = time.monotonic() + 12
            while not predicate():
                root.update()
                if time.monotonic() > deadline:
                    raise AssertionError(browser.status.cget("text"))
                time.sleep(0.01)
        try:
            wait_for(lambda: browser.photo is not None)
            browser.step(2)
            wait_for(lambda: "44." in browser.values.get("1.0", "end"))
            browser.auto_refresh.set(True)
            wait_for(lambda: browser.watch.current is not None)
            path = Path(sys.argv[1]) / "data/episode_000000.parquet"
            original = pq.read_table(path)
            columns = original.to_pydict()
            columns["actions"] = [[100.0+i] for i in range(4)]
            temporary = path.with_suffix(".tmp")
            pq.write_table(pa.table(columns), temporary)
            temporary.replace(path)
            wait_for(lambda: "102." in browser.values.get("1.0", "end"))
            assert browser.frame == 2
            assert browser.episode.name == "episode_000000"
            assert browser.episode.replay_source.frame(2) == 1
            browser.auto_refresh.set(False)
            # Shorter processed output clamps the old cursor and rejects the obsolete mapping.
            pq.write_table(pa.table({k:v[:2] for k,v in columns.items()}), temporary)
            temporary.replace(path)
            browser.reload_current()
            wait_for(lambda: not browser.reloading and browser.frame == 1)
            wait_for(lambda: "101." in browser.values.get("1.0", "end"))
            assert not browser.episode.replay
            assert "frame_indices" in browser.replay_note.cget("text")
            print("live processed dataset refresh passed")
        finally:
            browser.close()
            control.close()
            root.destroy()
    """)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=35,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "refresh passed" in process.stdout


def test_export_sample_preserves_source_frames_timing_and_existing_outputs(tmp_path):
    from scripts.export_simulation_episode import export_episode

    source = recording(tmp_path / "source.hdf5")
    output = tmp_path / "sampled"
    export_episode(source, "data/demo_0", output, start=1, stop=8, stride=3)
    episode = scan_dataset(output).episodes[0]
    assert episode.frames == 3
    assert episode.fps == pytest.approx(20 / 3)
    assert episode.replay_source.frame_indices == (1, 4, 7)
    reader = EpisodeReader(episode)
    try:
        image, values = reader.read(1, None)
        assert image is None
        np.testing.assert_allclose(values["simulation.state"], [0.2, 0.4, 0])
        assert values["timestamp"] == pytest.approx(0.15)
    finally:
        reader.close()
    manifest = (output / "meta/replay.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        export_episode(source, "data/demo_0", output)
    assert (output / "meta/replay.json").read_bytes() == manifest


def test_reload_does_not_reuse_a_worker_invalidated_by_pending_stop(tmp_path):
    root = tmp_path / "processed"
    processed(root, recording(tmp_path / "source.hdf5"))
    episode = scan_dataset(root).episodes[0]
    control = DatasetReplay()
    control.process = mock.Mock()
    control.process.poll.return_value = None
    control.process_generation = 3
    control.generation = 4  # stop() has invalidated the old reader, but cleanup is still pending.
    with mock.patch.object(control, "open") as open_worker:
        control.reload(episode, frame=2)
    open_worker.assert_called_once_with(episode, frame=2)
    control.process.stdin.write.assert_not_called()
