"""Validate segmented observations, recorded-state alignment and timeline navigation."""

# ruff: noqa: SLF001, RUF001 -- UI/read boundary checks and actual interface text.
import io
import json
import os
import subprocess
import sys
import textwrap
from unittest import mock

import h5py
import mujoco
import numpy as np
import pytest

from openpi.datasets.libero import SEGMENT_COLORS
from openpi.simulation.data.discovery import discover_datasets
from openpi.simulation.data.reader import EpisodeReader
from openpi.simulation.data.reader import scan_dataset
from openpi.simulation.runtime.replay import DatasetReplay
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
