"""Replay recorded MuJoCo states in the inspector viewport; no policy inference."""

# ruff: noqa: RUF001 -- Chinese interface messages.
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import queue
import sys
import threading
import time
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from openpi_client.embedded_viewer import EmbeddedViewer
from openpi_client.simulation_viewer import InteractiveViewer

if __package__:
    from openpi.simulation.workers.trajectory_overlay import TrajectoryOverlay
else:
    from trajectory_overlay import TrajectoryOverlay


def emit(**event):
    print("@dataset " + json.dumps(event, ensure_ascii=False), flush=True)


def relocate_assets(xml, source):
    tree = ET.fromstring(xml)
    assets_root = None
    for element in tree.iter():
        filename = element.get("file")
        if not filename:
            continue
        parts = pathlib.PurePosixPath(filename).parts
        if "robosuite" in parts:
            index = max(i for i, part in enumerate(parts) if part == "robosuite")
            spec = importlib.util.find_spec("robosuite")
            if spec:
                element.set("file", str(pathlib.Path(spec.origin).parent.joinpath(*parts[index + 1 :])))
        elif "assets" in parts and ("libero" in parts or "chiliocosm" in parts):
            # Original LIBERO demos also name the author's older chiliocosm checkout.
            from libero.libero import get_libero_path

            index = parts.index("assets")
            if assets_root is None:
                assets_root = pathlib.Path(get_libero_path("assets"))
            element.set("file", str(assets_root.joinpath(*parts[index + 1 :])))
        elif not pathlib.Path(filename).is_absolute():
            candidate = source.parent / filename
            if candidate.is_file():
                element.set("file", str(candidate))
    return ET.tostring(tree, encoding="unicode")


def set_recorded_state(model, data, state):
    state = np.asarray(state)
    expected = 1 + model.nq + model.nv + model.na
    if state.shape != (expected,) or not np.isfinite(state).all():
        raise ValueError(f"记录状态长度或数值无效，预期 {expected}，实际 {state.shape}。")
    data.time = state[0]
    data.qpos[:] = state[1 : 1 + model.nq]
    data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
    if model.na:
        data.act[:] = state[-model.na :]
    mujoco.mj_forward(model, data)


class RecordedTrajectory:
    """Own an HDF5 snapshot, validating replacement data before touching the displayed scene."""

    def __init__(self, path, group, frame=0, previous=None, *, overlay=None, timeline_frame=None):
        self.path = pathlib.Path(path)
        self.group = group
        self.file = h5py.File(self.path, "r")
        try:
            recording = self.file[group]
            xml = recording.attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode()
            self.xml = relocate_assets(xml, self.path)
            self.states = recording["states"]
            if self.states.ndim != 2 or not len(self.states):
                raise ValueError("三维回放需要非空的二维 states。")
            self.model = (
                previous.model
                if previous is not None and previous.xml == self.xml
                else mujoco.MjModel.from_xml_string(self.xml)
            )
            self.data = mujoco.MjData(self.model)
            self.overlay = TrajectoryOverlay(overlay, self.path, self.group) if overlay else None
            self.seek(frame, timeline_frame)
        except Exception:
            self.file.close()
            raise

    def seek(self, frame, timeline_frame=None):
        if not 0 <= frame < len(self.states):
            raise ValueError("三维回放的帧编号超出范围。")
        state = self.states[frame]
        if self.overlay:
            self.overlay.validate_frame(frame if timeline_frame is None else timeline_frame, frame)
        set_recorded_state(self.model, self.data, state)
        if self.overlay:
            self.overlay.set_frame(frame if timeline_frame is None else timeline_frame, frame)
        self.frame = frame

    def draw_overlay(self, viewer):
        if isinstance(viewer, EmbeddedViewer):
            viewer.set_overlay(self.overlay)
            return
        with viewer.handle.lock():
            if self.overlay:
                self.overlay.draw(viewer.handle.user_scn)
            else:
                viewer.handle.user_scn.ngeom = 0

    def overlay_status(self):
        return self.overlay.status() if self.overlay else {}

    def close(self):
        self.file.close()


def replace_trajectory(viewer, current, replacement):
    if current.model is replacement.model:
        # Keep the native window, its camera and the MjData object held by the viewer.
        with viewer.handle.lock():
            set_recorded_state(current.model, current.data, replacement.states[replacement.frame])
        replacement.data = current.data
    else:
        with viewer.handle.lock():
            camera = viewer.handle.cam
            position = (camera.lookat.copy(), camera.distance, camera.azimuth, camera.elevation)
        viewer.detach()
        viewer.attach_native(replacement.model, replacement.data, hide_collision_shells=True, show_ui=False)
        with viewer.handle.lock():
            camera = viewer.handle.cam
            camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = position
    current.close()
    return replacement


def run(args):
    commands = queue.Queue()

    def read_commands():
        for line in sys.stdin:
            try:
                command = json.loads(line)
                if isinstance(command, dict):
                    commands.put(command)
            except json.JSONDecodeError:
                continue
        commands.put({"command": "stop"})

    threading.Thread(target=read_commands, daemon=True).start()
    viewer = EmbeddedViewer(args.render_buffer) if args.render_buffer else InteractiveViewer()
    trajectory = None
    try:
        trajectory = RecordedTrajectory(
            args.path,
            args.group,
            args.frame,
            overlay=args.overlay,
            timeline_frame=args.timeline_frame,
        )
        camera = {}
        if trajectory.overlay:
            positions = trajectory.overlay.positions
            camera = {
                "lookat": (positions.min(axis=0) + positions.max(axis=0)) / 2,
                "distance": max(1.2, float(np.ptp(positions, axis=0).max()) * 2.5),
            }
        viewer.attach_native(trajectory.model, trajectory.data, hide_collision_shells=True, show_ui=False, **camera)
        viewer.paused = True
        expected_paused = True
        trajectory.draw_overlay(viewer)
        viewer.sync()
        emit(ready=True, source_frame=args.frame, **trajectory.overlay_status())
        while viewer.enabled:
            target = None
            while not commands.empty():
                command = commands.get_nowait()
                action = command.get("command")
                if args.render_buffer and viewer.command(command):
                    continue
                if action == "stop":
                    return
                if action == "frame":
                    target = (int(command["frame"]), command.get("timeline_frame", command["frame"]))
                elif action == "playing":
                    viewer.paused = expected_paused = not command["playing"]
                elif action == "overlay":
                    if trajectory.overlay:
                        trajectory.overlay.configure(**command)
                        trajectory.draw_overlay(viewer)
                        emit(overlay_updated=True, **trajectory.overlay_status())
                elif action == "reload":
                    replacement = None
                    try:
                        replacement = RecordedTrajectory(
                            command["path"],
                            command["group"],
                            int(command["frame"]),
                            previous=trajectory,
                            overlay=command.get("overlay"),
                            timeline_frame=command.get("timeline_frame"),
                        )
                        trajectory = replace_trajectory(viewer, trajectory, replacement)
                        trajectory.draw_overlay(viewer)
                        target = None
                        emit(ready=True, reloaded=True, source_frame=trajectory.frame, **trajectory.overlay_status())
                    except Exception as error:
                        if replacement is not None:
                            replacement.close()
                        viewer.paused = expected_paused = True
                        emit(error="更新三维数据失败，保留上一版场景：" + str(error))
            if target is not None:
                source_frame, timeline_frame = target
                try:
                    with viewer.handle.lock():
                        trajectory.seek(source_frame, timeline_frame)
                    trajectory.draw_overlay(viewer)
                    viewer.sync()
                    emit(
                        frame=timeline_frame,
                        source_frame=source_frame,
                        simulation_time=float(trajectory.data.time),
                        **trajectory.overlay_status(),
                    )
                except (ValueError, IndexError, OSError) as error:
                    viewer.paused = expected_paused = True
                    emit(error="当前帧无法回放，保留上一帧：" + str(error))
            viewer.sync()
            if viewer.paused != expected_paused:
                expected_paused = viewer.paused
                emit(playing=not viewer.paused)
            if viewer.single_step:
                viewer.single_step = False
                emit(step=True)
            time.sleep(0.005)
    finally:
        viewer.close()
        if trajectory is not None:
            trajectory.close()
        emit(stopped=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--timeline-frame", type=int)
    parser.add_argument("--overlay", type=json.loads)
    parser.add_argument("--render-buffer", help="Render into the inspector instead of a separate window.")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        emit(error=str(error))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
