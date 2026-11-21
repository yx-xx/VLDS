"""Draw recorded world-frame end-effector paths in MuJoCo's auxiliary scene.

This module also runs in the separate Python 3.8 LIBERO environment.
"""

# ruff: noqa: B905, RUF007 -- zip strict and itertools.pairwise need Python 3.10.
from __future__ import annotations

import json
import pathlib

import h5py
import mujoco
import numpy as np


class TrajectoryOverlay:
    """Small per-episode display snapshot, independent of simulation physics."""

    def __init__(self, specification, source_path, source_group):
        path = pathlib.Path(specification["path"])
        with h5py.File(path, "r") as file:
            group = file[specification["group"]]
            recorded_path = (path.parent / group.attrs["source_file"]).resolve()
            if recorded_path != pathlib.Path(source_path).resolve() or group.attrs["source_group"] != source_group:
                raise ValueError("Trajectory overlay does not refer to the replayed recording.")
            self.positions = group["robot_state/position"][:]
            self.times = group["time"][:]
            self.indices = group["simulation_state_index"][:]
            self.valid = group["simulation_state_valid"][:]
            self.intervals = json.loads(group["command_intervals"][()])
        count = len(self.positions)
        if (
            self.positions.shape != (count, 3)
            or count < 2
            or not np.isfinite(self.positions).all()
            or self.times.shape != (count,)
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
            or self.indices.shape != (count,)
            or self.valid.shape != (count,)
        ):
            raise ValueError("Invalid trajectory overlay coordinates, time or source mapping.")
        end = 0
        for item in self.intervals:
            start, stop = item["start"], item["end"]
            if start != end or not start < stop < count or item["command_label"] not in ("OPEN", "CLOSE"):
                raise ValueError("Trajectory intervals must partition all recorded actions.")
            end = stop
        if not self.intervals or end != count - 1:
            raise ValueError("Trajectory intervals do not cover the recorded trajectory.")
        self.colors = np.array(
            [[int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)] for color in specification["colors"]]
        )
        if not len(self.colors):
            raise ValueError("Trajectory overlay needs at least one segment color.")
        self.visible = specification.get("visible", True)
        self.current_only = specification.get("current_only", False)
        self.markers = specification.get("markers", True)
        self.frame = 0
        self.selected = 0
        self.drawn_geoms = 0
        self.sampled = False

    def validate_frame(self, frame, source_frame):
        if not 0 <= frame < len(self.positions) or not self.valid[frame] or self.indices[frame] != source_frame:
            raise ValueError("Trajectory cursor does not match the replayed source frame.")

    def set_frame(self, frame, source_frame):
        self.validate_frame(frame, source_frame)
        self.frame = frame
        self.selected = next(
            (i for i, item in enumerate(self.intervals) if item["start"] <= frame < item["end"]),
            len(self.intervals) - 1,
        )

    def configure(self, **options):
        for key in ("visible", "current_only", "markers"):
            if key in options:
                setattr(self, key, bool(options[key]))

    def draw(self, scene):
        """Replace display-only geoms; caller holds the native viewer lock."""
        scene.ngeom = 0
        self.drawn_geoms = 0
        self.sampled = False
        if not self.visible:
            return
        shown = [self.selected] if self.current_only else list(range(len(self.intervals)))
        # Reserve room for the cursor and boundaries before allocating path connectors.
        reserve = 3 + (len(shown) + 1 if self.markers else 0)
        capacity = max(0, scene.maxgeom - reserve)
        if capacity < len(shown):
            raise ValueError("MuJoCo auxiliary scene has too few slots for the command intervals.")
        lengths = np.array([self.intervals[i]["end"] - self.intervals[i]["start"] for i in shown])
        total = int(lengths.sum())
        self.sampled = total > capacity
        quotas = 1 + ((capacity - len(shown)) * lengths / total).astype(int) if self.sampled else lengths
        for index, quota in zip(shown, quotas):
            item = self.intervals[index]
            nodes = np.unique(np.linspace(item["start"], item["end"], int(quota) + 1, dtype=int))
            active = index == self.selected
            rgba = [*self.colors[index % len(self.colors)], 1.0 if active else 0.65]
            radius = 0.004 if active else 0.002
            for left, right in zip(nodes[:-1], nodes[1:]):
                if np.linalg.norm(self.positions[right] - self.positions[left]) < 1e-8:
                    continue
                geom = self._geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, self.positions[left], rgba)
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, self.positions[left], self.positions[right]
                )
        if self.markers:
            for index in shown:
                item = self.intervals[index]
                rgba = [*self.colors[index % len(self.colors)], 1.0]
                self._geom(
                    scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    0.006,
                    self.positions[item["start"]],
                    rgba,
                    str(index + 1),
                )
            last = self.intervals[shown[-1]]["end"]
            self._geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, 0.006, self.positions[last], [0.9, 0.9, 0.9, 1], "END")
        label = f"NOW {self.selected + 1}/{len(self.intervals)} {self.intervals[self.selected]['command_label']}"
        cursor = self.positions[self.frame]
        label_position = cursor + np.array([0, 0, 0.055])
        self._geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, 0.008, cursor, [1, 1, 1, 1])
        leader = self._geom(scene, mujoco.mjtGeom.mjGEOM_LINE, 1, cursor, [1, 1, 1, 1])
        mujoco.mjv_connector(leader, mujoco.mjtGeom.mjGEOM_LINE, 1, cursor, label_position)
        self._geom(scene, mujoco.mjtGeom.mjGEOM_LABEL, 0, label_position, [1, 1, 1, 1], label)
        self.drawn_geoms = scene.ngeom

    @staticmethod
    def _geom(scene, kind, radius, position, rgba, label=""):
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("MuJoCo auxiliary scene geometry limit reached.")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, kind, np.full(3, radius), position, np.eye(3).ravel(), np.array(rgba, dtype=np.float32)
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.label = label
        scene.ngeom += 1
        return geom

    def status(self):
        return {"overlay_geoms": self.drawn_geoms, "segment": self.selected + 1, "overlay_sampled": self.sampled}
