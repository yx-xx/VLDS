"""Validate segmented observations, recorded-state alignment and timeline navigation."""

# ruff: noqa: SLF001, RUF001 -- UI/read boundary checks and actual interface text.
import io
import json
import os
import queue
import subprocess
import sys
import textwrap
from unittest import mock

import h5py
import mujoco
import numpy as np
import pytest

from openpi.datasets.libero import SEGMENT_COLORS
from openpi.simulation import paths
from openpi.simulation.data.discovery import discover_datasets
from openpi.simulation.data.reader import EpisodeReader
from openpi.simulation.data.reader import scan_dataset
from openpi.simulation.runtime.replay import DatasetReplay
from openpi.simulation.ui.dataset_browser import DatasetBrowser
from openpi.simulation.workers.replay import RecordedTrajectory
from openpi.simulation.workers.trajectory_overlay import TrajectoryOverlay


def segmented_dataset(tmp_path):
    source = tmp_path / "source.hdf5"
    count = 6
    with h5py.File(source, "w") as file:
        group = file.create_group("data/demo_0")
        group.attrs["model_file"] = (
            '<mujoco><worldbody><body><joint type="slide"/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        group["states"] = np.c_[np.arange(count) * 0.05, np.arange(count) * 0.01, np.zeros(count)]
        group["actions"] = np.c_[np.zeros((count, 6)), [-1, -1, 1, 1, -1, -1]]
        group["obs/agentview_rgb"] = np.broadcast_to(np.arange(count)[:, None, None, None], (count, 4, 4, 3))
        group["obs/eye_in_hand_rgb"] = np.broadcast_to(np.arange(count)[:, None, None, None] + 10, (count, 4, 4, 3))
    root = tmp_path / "processed/segmented"
    shard = root / "libero_spatial/task.hdf5"
    shard.parent.mkdir(parents=True)
    intervals = [
        {"command_interval_id": i, "start": i * 2, "end": (i + 1) * 2, "command": command, "command_label": label}
        for i, (command, label) in enumerate([(-1, "OPEN"), (1, "CLOSE"), (-1, "OPEN")])
    ]
    with h5py.File(shard, "w") as file, h5py.File(source) as raw:
        file["provenance"] = json.dumps(
            {"source_size": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns}
        )
        group = file.create_group("episodes/demo_0")
        group.attrs.update(source_file="../../../source.hdf5", source_group="/data/demo_0")
        group["source"] = h5py.ExternalLink("../../../source.hdf5", "/data/demo_0")
        group["actions"] = raw["data/demo_0/actions"][:]
        group["time"] = np.arange(count + 1) * 0.05
        group["gripper_command"] = [-1, -1, 1, 1, -1, -1]
        group["image_valid"] = [False, *([True] * count)]
        group["simulation_state_valid"] = [*([True] * count), False]
        group["simulation_state_index"] = [*range(count), -1]
        group["robot_state/position"] = np.c_[np.arange(count + 1) * 0.01, np.zeros((count + 1, 2))]
        group["robot_state/quaternion_xyzw"] = np.tile([0, 0, 0, 1], (count + 1, 1))
        group["robot_state/gripper_position"] = np.tile([0.04, -0.04], (count + 1, 1))
        group["command_intervals"] = json.dumps(intervals)
        group["candidates"] = json.dumps(
            [{"command_interval_id": i, "start": i * 2, "end": (i + 1) * 2} for i in range(3)]
        )
        for camera in ["agentview_rgb", "eye_in_hand_rgb"]:
            layout = h5py.VirtualLayout(shape=(count + 1, 4, 4, 3), dtype=np.int64)
            layout[1:] = h5py.VirtualSource(
                "../../../source.hdf5", f"/data/demo_0/obs/{camera}", shape=(count, 4, 4, 3)
            )
            group.create_virtual_dataset(f"images/{camera}", layout, fillvalue=0)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "vlds.libero.trajectories.v2",
                "episodes": [
                    {
                        "shard": "libero_spatial/task.hdf5",
                        "group": "/episodes/demo_0",
                        "suite": "libero_spatial",
                        "task": "pick_up_the_bowl_demo",
                        "episode": "demo_0",
                        "language": "pick up the bowl",
                        "actions": count,
                        "observations": count + 1,
                        "duration_seconds": count * 0.05,
                    }
                ],
            }
        )
    )
    return root


def test_segmented_reader_keeps_missing_frames_and_source_alignment(tmp_path):
    root = segmented_dataset(tmp_path)
    index = scan_dataset(root)
    episode = index.episodes[0]
    assert episode.frames == 7
    assert episode.fps == pytest.approx(20)
    reader = EpisodeReader(episode)
    try:
        assert reader.segmentation["candidate_counts"] == [1, 1, 1]
        assert [i["start"] for i in reader.segmentation["intervals"]] == [0, 2, 4]
        image, values = reader.read(0, episode.cameras[0])
        assert image is None
        assert not values["image_valid"]
        for frame in [1, 5, 2]:
            image, values = reader.read(frame, episode.cameras[0])
            np.testing.assert_array_equal(image, frame - 1)
            assert episode.replay_source.frame(frame) == frame
            assert values["simulation_state_valid"]
        image, values = reader.read(6, episode.cameras[0])
        np.testing.assert_array_equal(image, 5)
        assert not values["simulation_state_valid"]
        assert episode.replay_source.frame(6) is None
        assert "actions" not in values
    finally:
        reader.close()
    assert root in discover_datasets([root.parent])


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires a desktop display")
def test_segmented_browser_navigation_and_bounded_playback(tmp_path):
    root = segmented_dataset(tmp_path)
    code = textwrap.dedent("""
        import sys, time
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
                    raise AssertionError(browser.status.cget('text'))
                time.sleep(0.01)
            root.update()
        try:
            wait_for(lambda: browser.segments.data is not None and not browser.render_pending)
            assert browser.segments.winfo_manager() == 'pack'
            assert browser.flip.get()
            browser.segments.overlay_current_only.set(True)
            browser.segments._overlay_changed()
            assert browser.replay.overlay_options['current_only']
            assert browser.image is None
            assert '缺失' in browser.preview_message
            browser.segments._move(1)
            wait_for(lambda: browser.frame == 2 and browser.photo is not None and not browser.render_pending)
            assert browser.segments.selected == 1
            assert 'CLOSE' in browser.segments.selection.get()
            browser.speed.set('2×')
            browser.segments._play()
            wait_for(lambda: not browser.playing and browser.frame == 4)
            assert browser.segments.selected == 2
            browser.seek_frame(6)
            wait_for(lambda: not browser.render_pending)
            assert '终端观测' in browser.replay_note.cget('text')
            with mock.patch.object(browser.replay, 'reload') as reload:
                browser.open_replay()
                reload.assert_not_called()
            browser.search.set('missing task')
            wait_for(lambda: browser.episode is None)
            assert not browser.segments.winfo_manager()
            print('segmented browser navigation passed')
        finally:
            browser.close()
            control.close()
            root.destroy()
    """)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "segmented browser navigation passed" in result.stdout


def test_native_overlay_uses_world_positions_colors_and_shared_boundaries(tmp_path):
    episode = scan_dataset(segmented_dataset(tmp_path)).episodes[0]
    reader = EpisodeReader(episode)
    reader.close()
    control = DatasetReplay()
    spec = control._overlay(episode)
    source = episode.replay_source
    recording = RecordedTrajectory(source.path, source.group, 3, overlay=spec, timeline_frame=3)
    try:
        overlay = recording.overlay
        scene = mujoco.MjvScene(recording.model, maxgeom=50)
        original_state = recording.data.qpos.copy()
        overlay.draw(scene)
        assert overlay.selected == 1
        assert scene.ngeom == 13
        assert not overlay.sampled
        np.testing.assert_allclose(
            scene.geoms[2].rgba[:3], [int(SEGMENT_COLORS[1][i : i + 2], 16) / 255 for i in (1, 3, 5)]
        )
        assert scene.geoms[2].size[0] > scene.geoms[0].size[0]
        cursor = scene.geoms[scene.ngeom - 3]
        assert scene.geoms[scene.ngeom - 1].label == "NOW 2/3 CLOSE"
        np.testing.assert_allclose(cursor.pos, [0.03, 0, 0])
        np.testing.assert_array_equal(recording.data.qpos, original_state)
        overlay.configure(current_only=True)
        overlay.draw(scene)
        assert scene.ngeom == 7
        assert scene.geoms[2].label == "2"
        np.testing.assert_allclose(scene.geoms[2].pos, overlay.positions[2])
        assert scene.geoms[3].label == "END"
        np.testing.assert_allclose(scene.geoms[3].pos, overlay.positions[4])
        recording.seek(4, 4)
        overlay.draw(scene)
        assert overlay.selected == 2
        assert scene.geoms[scene.ngeom - 1].label == "NOW 3/3 OPEN"
        with pytest.raises(ValueError, match="cursor"):
            overlay.set_frame(6, 5)
        assert overlay.frame == 4
        overlay.configure(visible=False)
        overlay.draw(scene)
        assert scene.ngeom == 0
        overlay.configure(visible=True, current_only=False, markers=False)
        overlay.draw(scene)
        assert scene.ngeom == 9
        small_scene = mujoco.MjvScene(recording.model, maxgeom=10)
        overlay.configure(markers=True)
        overlay.draw(small_scene)
        assert overlay.sampled
        assert small_scene.ngeom <= small_scene.maxgeom
        assert [small_scene.geoms[i].label for i in range(3, 7)] == ["1", "2", "3", "END"]
    finally:
        recording.close()


def test_overlay_rejects_wrong_recording_and_incomplete_segments(tmp_path):
    episode = scan_dataset(segmented_dataset(tmp_path)).episodes[0]
    reader = EpisodeReader(episode)
    reader.close()
    source = episode.replay_source
    spec = DatasetReplay()._overlay(episode)
    with pytest.raises(ValueError, match="recording"):
        TrajectoryOverlay(spec, source.path, "data/another_demo")
    with h5py.File(episode.path, "r+") as file:
        group = file[episode.group]
        intervals = json.loads(group["command_intervals"][()])
        del group["command_intervals"]
        group["command_intervals"] = json.dumps(intervals[:-1])
    with pytest.raises(ValueError, match="cover"):
        TrajectoryOverlay(spec, source.path, source.group)


def test_replay_reload_carries_overlay_source_and_retains_display_options(tmp_path):
    episode = scan_dataset(segmented_dataset(tmp_path)).episodes[0]
    reader = EpisodeReader(episode)
    reader.close()
    control = DatasetReplay()
    control.configure_overlay(visible=True, current_only=True, markers=False)
    control.process = mock.Mock()
    control.process.poll.return_value = None
    control.process.stdin = io.StringIO()
    control.process_generation = control.generation
    control.reload(episode, frame=2)
    command = json.loads(control.process.stdin.getvalue())
    assert command["frame"] == command["timeline_frame"] == 2
    assert command["overlay"]["path"] == str(episode.path)
    assert command["overlay"]["group"] == episode.group
    assert command["overlay"]["current_only"]
    assert not command["overlay"]["markers"]


def test_failed_recorded_state_preserves_overlay_cursor_and_robot_state(tmp_path):
    episode = scan_dataset(segmented_dataset(tmp_path)).episodes[0]
    reader = EpisodeReader(episode)
    reader.close()
    source = episode.replay_source
    with h5py.File(source.path, "r+") as file:
        file[source.group]["states"][4, 1] = np.nan
    recording = RecordedTrajectory(source.path, source.group, 3, overlay=DatasetReplay()._overlay(episode))
    try:
        original_state = recording.data.qpos.copy()
        with pytest.raises(ValueError, match="无效"):
            recording.seek(4, 4)
        assert recording.overlay.frame == recording.frame == 3
        assert recording.overlay.selected == 1
        np.testing.assert_array_equal(recording.data.qpos, original_state)
        with pytest.raises(ValueError, match="cursor"):
            recording.seek(2, 1)
        np.testing.assert_array_equal(recording.data.qpos, original_state)
    finally:
        recording.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires a desktop display")
def test_segmented_replay_renders_inside_browser_and_clears_missing_terminal_state(tmp_path):
    root = segmented_dataset(tmp_path)
    code = textwrap.dedent("""
        import sys, time
        import numpy as np
        from openpi.simulation.ui.inspector import InspectorWindow
        from openpi.simulation.runtime.controller import InspectorController
        from openpi.simulation.ui.tk_runtime import tk

        root = tk.Tk()
        root.withdraw()
        control = InspectorController()
        window = InspectorWindow(root, control, dataset_path=sys.argv[1])
        browser = window.dataset_browser
        def wait_for(predicate):
            deadline = time.monotonic() + 20
            while not predicate():
                root.update()
                if time.monotonic() > deadline:
                    raise AssertionError(browser.status.cget('text') + browser.replay_note.cget('text'))
                time.sleep(0.01)
            root.update()
        try:
            wait_for(lambda: browser.segments.data is not None and not browser.render_pending)
            browser.open_replay()
            wait_for(lambda: browser.replay_ready and browser.scene_view.photo is not None
                     and np.array(browser.scene_view.image).shape == (120, 160, 3))
            assert '--render-buffer' in browser.replay.process.args
            before = np.array(browser.scene_view.image)
            browser.replay.send('view', action='rotate', dx=0.1, dy=0.1)
            wait_for(lambda: np.any(np.array(browser.scene_view.image) != before))
            browser.segments._move(1)
            wait_for(lambda: browser.frame == 2 and browser.replay_pending is None)
            assert browser.segments.selected == 1
            browser.replay.configure_overlay(current_only=True)
            browser.segments._play()
            wait_for(lambda: not browser.playing and browser.frame == 4)
            browser.seek_frame(6)
            wait_for(lambda: not browser.render_pending)
            assert not browser.replay_ready
            assert browser.scene_view.image is None
            assert '终端观测' in browser.scene_view.message
            browser.seek_frame(1)
            wait_for(lambda: not browser.render_pending)
            browser.open_replay()
            wait_for(lambda: browser.replay_ready and browser.scene_view.photo is not None)
            assert not any(isinstance(child, tk.Toplevel) for child in root.winfo_children())
            root.deiconify()
            root.geometry('1280x850')
            wait_for(lambda: browser.scene_view.canvas.winfo_width() > 500)
            assert browser.scene_view.canvas.winfo_width() > browser.views.winfo_width() * 0.72, (browser.scene_view.canvas.winfo_width(), browser.views.winfo_width())
            assert not browser.values.frame.winfo_manager()
            assert not browser.data_options.winfo_manager()
            def inside_window(widget):
                return widget.winfo_rooty() + widget.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            assert inside_window(browser.play_button)
            assert inside_window(browser.scale)
            browser._toggle_values()
            root.update()
            assert browser.values.frame.winfo_manager() == 'pack'
            assert inside_window(browser.values)
            assert inside_window(browser.play_button)
            browser._toggle_values()
            browser._toggle_data_options()
            root.update()
            assert browser.data_options.winfo_manager() == 'pack'
        finally:
            browser.close()
            control.close()
            root.destroy()
    """)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        cwd=paths.ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires a desktop display")
def test_processed_browser_automatically_switches_replays_and_recovers_after_terminal(tmp_path):
    root = segmented_dataset(tmp_path)
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "r+") as file:
        file.copy("data/demo_0", "data/demo_1")
        states = file["data/demo_1/states"][:]
        states[:, 1] += 0.2
        file["data/demo_1/states"][:] = states
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["episodes"][0]
    with h5py.File(root / entry["shard"], "r+") as file:
        file["provenance"][()] = json.dumps(
            {"source_size": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns}
        )
        file.copy("episodes/demo_0", "episodes/demo_1")
        group = file["episodes/demo_1"]
        group.attrs["source_group"] = "/data/demo_1"
        del group["source"]
        group["source"] = h5py.ExternalLink("../../../source.hdf5", "/data/demo_1")
        positions = group["robot_state/position"][:]
        positions[:, 0] += 0.2
        group["robot_state/position"][:] = positions
    other = root / "libero_spatial/other_task.hdf5"
    other.write_bytes((root / entry["shard"]).read_bytes())
    manifest["episodes"].extend(
        [
            {**entry, "group": "/episodes/demo_1", "episode": "demo_1"},
            {
                **entry,
                "shard": "libero_spatial/other_task.hdf5",
                "task": "close_drawer",
                "language": "close the drawer",
            },
        ]
    )
    manifest_path.write_text(json.dumps(manifest))
    code = textwrap.dedent("""
        import sys, time
        import numpy as np
        from openpi.simulation.ui.inspector import InspectorWindow
        from openpi.simulation.runtime.controller import InspectorController
        from openpi.simulation.ui.tk_runtime import tk

        root = tk.Tk()
        root.withdraw()
        control = InspectorController()
        window = InspectorWindow(root, control, dataset_path=sys.argv[1])
        browser = window.dataset_browser
        errors = []
        root.report_callback_exception = lambda *args: errors.append(str(args[1]))
        def wait_for(predicate):
            deadline = time.monotonic() + 20
            while not predicate():
                root.update()
                if time.monotonic() > deadline:
                    raise AssertionError(browser.status.cget('text') + browser.replay_note.cget('text') + str(errors))
                time.sleep(0.01)
            root.update()
        def ready():
            return (browser.replay_ready and not browser.render_pending
                    and browser.replay_pending is None and browser.scene_view.image is not None)
        try:
            wait_for(ready)
            assert browser.auto_replay.get()
            # Reselecting the current episode must republish its cached read metadata.
            browser.episode_box.event_generate('<<ComboboxSelected>>')
            wait_for(ready)
            assert browser.segments.data is not None
            before = np.array(browser.scene_view.image)
            browser.episode_box.current(1)
            browser.episode_box.event_generate('<<ComboboxSelected>>')
            wait_for(lambda: ready() and browser.replay.source.group == '/data/demo_1'
                     and np.any(np.array(browser.scene_view.image) != before))
            row = next(key for key, task in browser.task_rows.items() if task.name == 'close the drawer')
            browser.task_tree.selection_set(row)
            browser.task_tree.event_generate('<<TreeviewSelect>>')
            wait_for(lambda: ready() and browser.episode.task == 'close the drawer')
            browser.seek_frame(6)
            wait_for(lambda: not browser.render_pending and browser.scene_view.image is None)
            assert not browser.replay_ready
            assert '终端观测' in browser.scene_view.message
            browser.seek_frame(1)
            wait_for(ready)
            browser.auto_replay.set(False)
            browser._auto_replay_changed()
            browser.task.set('pick up the bowl')
            browser._filter_episodes()
            wait_for(lambda: not browser.render_pending and browser.replay.process is None)
            assert not browser.replay_ready
            assert '加载三维回放' in browser.scene_view.message
            browser.reload_current()
            wait_for(lambda: not browser.reloading and not browser.render_pending)
            assert not browser.auto_replay.get()
            browser.auto_replay.set(True)
            browser._auto_replay_changed()
            wait_for(ready)
            root.deiconify()
            root.geometry('1280x850')
            wait_for(lambda: browser.episode_box.winfo_width() > 600)
            assert browser.episode_box.winfo_width() > browser.views.winfo_width() * 0.85
            assert browser.episode_box.winfo_rooty() < browser.scene_view.winfo_rooty()
            assert not errors, errors
        finally:
            browser.close()
            control.close()
            root.destroy()
    """)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        cwd=paths.ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 0, process.stderr + process.stdout


def test_replay_frame_sync_keeps_only_latest_frame_when_renderer_is_busy():
    from types import SimpleNamespace

    replay = mock.Mock()
    browser = SimpleNamespace(
        episode=SimpleNamespace(replay=True, replay_source=None),
        replay_ready=True,
        replay_loading=False,
        replay_after_read=False,
        replay_pending=10,
        replay_deferred=None,
        replay_sent_at=0.0,
        replay=replay,
    )

    DatasetBrowser._sync_replay_frame(browser, 11)
    assert browser.replay_pending == 10
    assert browser.replay_deferred == 11
    replay.send.assert_not_called()

    DatasetBrowser._sync_replay_frame(browser, 12)
    assert browser.replay_pending == 10
    assert browser.replay_deferred == 12
    replay.send.assert_not_called()


def test_field_overlay_matches_episode_and_time_without_changing_robot_state(tmp_path):
    from openpi.fields import primitive
    from openpi.fields import seds
    from openpi.fields import visualization

    root = segmented_dataset(tmp_path)
    episode = scan_dataset(root).episodes[0]
    EpisodeReader(episode).close()
    source = episode.replay_source
    recording = RecordedTrajectory(source.path, source.group, 1, overlay=DatasetReplay()._overlay(episode))
    try:
        time = np.arange(3) * 0.05
        position = recording.overlay.positions[:3]
        field = primitive.Primitive(seds.PositionField(seds.linear_factors(2), position[-1]), np.eye(3), -1)
        arrays, metadata = visualization.sample_field(field, position, np.tile(np.eye(3), (3, 1, 1)), time, density=3)
        metadata.update(
            processed_path=str(episode.path), processed_group=episode.group, interval={"start": 0, "end": 2}
        )
        path = tmp_path / "preview.npz"
        visualization.save_snapshot(path, arrays, metadata)
        recording.overlay.configure(field_preview=str(path), visible=False)
        original = recording.data.qpos.copy()
        scene = mujoco.MjvScene(recording.model, maxgeom=1000)
        recording.overlay.draw(scene)
        assert recording.overlay.status()["field_geoms"] > 0
        labels = [scene.geoms[i].label for i in range(scene.ngeom)]
        assert "FIELD GOAL" in labels
        assert "NOMINAL" in labels
        cursor = scene.geoms[labels.index("NOMINAL")]
        np.testing.assert_allclose(cursor.pos, arrays["nominal"][1])
        np.testing.assert_array_equal(recording.data.qpos, original)
        # Saved full-episode traces must take priority over local, reset-start traces.
        arrays["sequence_nominal"] = arrays["nominal"][: len(position)] + [0.002, 0, 0]
        visualization.save_snapshot(path, arrays, metadata)
        recording.overlay.configure(field_preview=str(path))
        np.testing.assert_array_equal(recording.overlay.field.arrays["nominal"], arrays["sequence_nominal"])
        # Advancing outside this field's source interval must hide it.
        recording.seek(3, 3)
        recording.overlay.draw(scene)
        assert scene.ngeom == 0
        # A bad replacement preserves the previously loaded field.
        metadata["processed_group"] = "/episodes/other"
        visualization.save_snapshot(path, arrays, metadata)
        previous = recording.overlay.field
        with pytest.raises(ValueError, match="match"):
            recording.overlay.configure(field_preview=str(path))
        assert recording.overlay.field is previous
        recording.seek(1, 1)
        recording.overlay.configure(visible=True)
        small = mujoco.MjvScene(recording.model, maxgeom=40)
        recording.overlay.draw(small)
        assert small.ngeom <= small.maxgeom
        recording.overlay.configure(field_preview=None)
        recording.overlay.draw(scene)
        assert recording.overlay.status()["field_geoms"] == 0
    finally:
        recording.close()


def test_switching_profiles_prepares_snapshots_off_thread_and_reuses_each_profile_cache(tmp_path):
    from collections import OrderedDict
    from types import SimpleNamespace

    from openpi.simulation.ui.field_preview import FieldPreviewPanel

    class DeferredExecutor:
        def __init__(self):
            self.jobs = []

        def submit(self, callback):
            self.jobs.append(callback)

    record = {
        "start": 0,
        "end": 2,
        "accepted": True,
        "report": {},
        "maximum_speed_scaled_per_second": 1.0,
    }
    result = {
        "records": [record],
        "limits": {},
        "tolerance": [0.01],
        "summary": {"actions_covered": 2, "maximum_scaled_error": 0.0, "position_rms_native": 0.0},
        "coordinates": {"space": "world Cartesian", "units": "m"},
    }
    executor = DeferredExecutor()
    panel = SimpleNamespace(
        episode=SimpleNamespace(processed_root=tmp_path),
        offline_selected=None,
        frame=0,
        generation=1,
        native=mock.Mock(),
        choice=mock.Mock(),
        detail=mock.Mock(),
        play_button=mock.Mock(),
        temporary=SimpleNamespace(name=str(tmp_path)),
        auto_cache=OrderedDict(),
        offline_pending=set(),
        executor=executor,
        events=queue.Queue(),
        configure_overlay=mock.Mock(),
        replay=mock.Mock(),
        _buttons=mock.Mock(),
        after=mock.Mock(),
        closed=False,
    )
    panel.detail.cget.return_value = ""
    panel._offline_key = FieldPreviewPanel._offline_key.__get__(panel)
    panel._activate_offline = FieldPreviewPanel._activate_offline.__get__(panel)
    panel._queue_offline_snapshot = FieldPreviewPanel._queue_offline_snapshot.__get__(panel)
    panel._ensure_offline = FieldPreviewPanel._ensure_offline.__get__(panel)
    panel.poll = FieldPreviewPanel.poll.__get__(panel)
    snapshots = [({"goal": np.array([value, 0.0, 0.0])}, {}) for value in (0.1, 0.2, 0.3)]
    with mock.patch("openpi.datasets.dynamics.playback.preview", side_effect=snapshots) as sample:
        paths = []
        for profile, value in [
            ("replay", 0.1),
            ("production_libero", 0.2),
            ("coherent", 0.3),
            ("replay", 0.1),
            ("coherent", 0.3),
        ]:
            panel.profile_code = profile
            panel.offline = {"result": result, "frame_offset": 0, "path": str(tmp_path / profile)}
            panel.offline_selected = None
            panel.offline_pending.clear()
            before = sample.call_count
            panel._ensure_offline()
            # Crossing a boundary only schedules work; it never samples on Tk.
            assert sample.call_count == before
            if panel.offline_pending:
                assert len(executor.jobs) == 1
                executor.jobs.pop(0)()
                FieldPreviewPanel.poll(panel)
                assert sample.call_count == before + 1
            else:
                assert sample.call_count == before
            path = panel.replay.configure_overlay.call_args.kwargs["field_preview"]
            paths.append(path)
            with np.load(path, allow_pickle=False) as snapshot:
                np.testing.assert_array_equal(snapshot["goal"], [value, 0.0, 0.0])
        assert sample.call_count == 3
        assert paths[0] == paths[3]
        assert paths[2] == paths[4]
        assert len(set(paths)) == 3


def test_manual_pilot_updates_both_timeline_and_field_controls():
    from types import SimpleNamespace

    from openpi.simulation.ui.field_preview import FieldPreviewPanel

    bundle = {
        "profile": "coherent_flow",
        "manual": True,
        "frame_offset": 0,
        "arrays": {"time": np.array([0.0, 0.05])},
        "result": {"records": [{"start": 0, "end": 1}], "limits": {"dt": 0.05}},
    }
    events = queue.Queue()
    events.put((2, 3, "file", ("dynamics", bundle), None))
    panel = SimpleNamespace(
        generation=2,
        job=3,
        closed=False,
        busy=True,
        events=events,
        automatic=mock.Mock(),
        native=mock.Mock(),
        replay=mock.Mock(),
        offline_pending=set(),
        snapshot="previous",
        on_segmentation=mock.Mock(),
        _set_profile_choices=mock.Mock(),
        _ensure_offline=mock.Mock(),
        _prefetch_offline=mock.Mock(),
        _buttons=mock.Mock(),
        after=mock.Mock(),
    )
    panel.poll = FieldPreviewPanel.poll.__get__(panel)
    panel.poll()
    assert panel.offline is bundle
    assert panel.snapshot is None
    assert panel.profile_code == "coherent_flow"
    assert panel.segmentation["kind"] == "dynamics"
    assert panel.automatic.set.call_count == 1
    assert panel.automatic.set.call_args.args == (True,)
    panel.on_segmentation.assert_called_once_with(panel.segmentation)
    panel.replay.configure_overlay.assert_called_once_with(
        intervals=panel.segmentation["intervals"], field_preview=None
    )
    panel._ensure_offline.assert_called_once()


def test_same_profile_pilots_keep_distinct_snapshot_files(tmp_path):
    from types import SimpleNamespace

    from openpi.simulation.ui.field_preview import FieldPreviewPanel

    class InlineExecutor:
        def submit(self, callback):
            callback()

    panel = SimpleNamespace(
        offline=None,
        episode=SimpleNamespace(processed_root=tmp_path),
        generation=1,
        profile_code="coherent_flow",
        auto_cache={},
        offline_pending=set(),
        executor=InlineExecutor(),
        temporary=SimpleNamespace(name=str(tmp_path)),
        events=queue.Queue(),
    )
    panel._offline_key = FieldPreviewPanel._offline_key.__get__(panel)
    with mock.patch(
        "openpi.datasets.dynamics.playback.preview",
        side_effect=[({"goal": np.ones(3)}, {}), ({"goal": np.zeros(3)}, {})],
    ):
        paths = []
        for name in ("pilot-a", "pilot-b"):
            panel.offline = {"path": str(tmp_path / name / "episode_00000.json")}
            FieldPreviewPanel._queue_offline_snapshot(panel, 0)
            paths.append(panel.events.get_nowait()[3][3])
        assert paths[0] != paths[1]
        with np.load(paths[0], allow_pickle=False) as snapshot:
            np.testing.assert_array_equal(snapshot["goal"], np.ones(3))
        with np.load(paths[1], allow_pickle=False) as snapshot:
            np.testing.assert_array_equal(snapshot["goal"], np.zeros(3))


def test_native_pilot_keeps_load_and_interval_play_enabled_without_a_world_snapshot():
    from types import SimpleNamespace

    from openpi.simulation.ui.field_preview import FieldPreviewPanel

    panel = SimpleNamespace(
        busy=False,
        episode=SimpleNamespace(processed_root=None),
        snapshot=None,
        offline={"result": {}},
        fit_button=mock.Mock(),
        load_button=mock.Mock(),
        choice=mock.Mock(),
        profile_choice=mock.Mock(),
        export_button=mock.Mock(),
        focus_button=mock.Mock(),
        play_button=mock.Mock(),
    )
    FieldPreviewPanel._buttons(panel)
    panel.fit_button.configure.assert_called_once_with(state="disabled")
    panel.load_button.configure.assert_called_once_with(state="normal")
    panel.play_button.configure.assert_called_once_with(state="normal")
    panel.focus_button.configure.assert_called_once_with(state="disabled")
