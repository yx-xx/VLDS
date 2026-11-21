"""Replay recorded MuJoCo states in a native 3D window; no policy inference."""

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
from openpi_client.simulation_viewer import InteractiveViewer


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

    def __init__(self, path, group, frame=0, previous=None):
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
            self.seek(frame)
        except Exception:
            self.file.close()
            raise

    def seek(self, frame):
        if not 0 <= frame < len(self.states):
            raise ValueError("三维回放的帧编号超出范围。")
        set_recorded_state(self.model, self.data, self.states[frame])
        self.frame = frame

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
        viewer.attach_native(replacement.model, replacement.data, hide_collision_shells=True)
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
    viewer = InteractiveViewer()
    trajectory = None
    try:
        trajectory = RecordedTrajectory(args.path, args.group, args.frame)
        viewer.attach_native(trajectory.model, trajectory.data, hide_collision_shells=True)
        viewer.paused = True
        expected_paused = True
        emit(ready=True, source_frame=args.frame)
        while viewer.enabled:
            target = None
            while not commands.empty():
                command = commands.get_nowait()
                action = command.get("command")
                if action == "stop":
                    return
                if action == "frame":
                    target = (int(command["frame"]), command.get("timeline_frame", command["frame"]))
                elif action == "playing":
                    viewer.paused = expected_paused = not command["playing"]
                elif action == "reload":
                    replacement = None
                    try:
                        replacement = RecordedTrajectory(
                            command["path"], command["group"], int(command["frame"]), previous=trajectory
                        )
                        trajectory = replace_trajectory(viewer, trajectory, replacement)
                        target = None
                        emit(ready=True, reloaded=True, source_frame=trajectory.frame)
                    except Exception as error:
                        if replacement is not None:
                            replacement.close()
                        viewer.paused = expected_paused = True
                        emit(error="更新三维数据失败，保留上一版场景：" + str(error))
            if target is not None:
                source_frame, timeline_frame = target
                try:
                    with viewer.handle.lock():
                        trajectory.seek(source_frame)
                    viewer.sync()
                    emit(frame=timeline_frame, source_frame=source_frame, simulation_time=float(trajectory.data.time))
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
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        emit(error=str(error))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
