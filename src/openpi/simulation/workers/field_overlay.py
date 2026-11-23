"""Render a precomputed field preview; compatible with the isolated Python 3.8 worker."""

# ruff: noqa: B905, RUF007 -- This module runs on Python 3.8.
from __future__ import annotations

import json
import pathlib

import mujoco
import numpy as np


class FieldOverlay:
    def __init__(self, path, processed_path, processed_group, frame_count):
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata"]))
            self.arrays = {key: archive[key] for key in ("grid", "arrows", "velocity", "demo", "nominal", "goal")}
            if "sequence_nominal" in archive:
                self.arrays["nominal"] = archive["sequence_nominal"]
        info = self.metadata
        if (
            info.get("schema") != "vlds.field_preview.v1"
            or info.get("frame") != "world"
            or info.get("control_point") != "gripper0_grip_site"
            or pathlib.Path(info["processed_path"]).resolve() != pathlib.Path(processed_path).resolve()
            or info["processed_group"] != processed_group
        ):
            raise ValueError("Field preview does not match the replayed episode or coordinate frame.")
        self.continuous = "continuous_report" in info
        self.start, self.end = info["interval"]["start"], info["interval"]["end"]
        if not 0 <= self.start < self.end < frame_count:
            raise ValueError("Field preview interval is outside the replay timeline.")
        arrays = self.arrays
        for key, value in arrays.items():
            shape_valid = value.shape == (3,) if key == "goal" else value.ndim == 2 and value.shape[1] == 3
            if not shape_valid or not np.isfinite(value).all():
                raise ValueError("Field preview has invalid geometry: " + key)
        if (
            not 1 <= len(arrays["grid"]) <= 1331
            or arrays["grid"].shape != arrays["arrows"].shape
            or arrays["grid"].shape != arrays["velocity"].shape
            or len(arrays["demo"]) != self.end - self.start + 1
            or len(arrays["nominal"]) < len(arrays["demo"])
        ):
            raise ValueError("Field preview has inconsistent sampling or rollout lengths.")
        self.drawn_geoms = 0

    def draw(self, scene, frame, make_geom, *, vectors=True, rollout=True):
        self.drawn_geoms = 0
        if not self.start <= frame <= self.end:
            return
        initial = scene.ngeom
        arrays = self.arrays
        # Bound geometry use even for very long demonstrations or a small native user scene.
        budget = max(0, min(1800, (scene.maxgeom - scene.ngeom) // 2 - 8))
        if budget < 12:
            return
        if vectors:
            spacing = max(float(np.ptp(arrays["grid"], axis=0).max()), 0.05)
            count = min(len(arrays["grid"]), budget // 2)
            indices = np.unique(np.linspace(0, len(arrays["grid"]) - 1, count, dtype=int))
            maximum = max(float(np.linalg.norm(arrays["velocity"], axis=1).max()), 1e-12)
            for i in indices:
                if np.linalg.norm(arrays["arrows"][i]) < 1e-7:
                    continue
                ratio = float(np.linalg.norm(arrays["velocity"][i]) / maximum)
                color = [0.12 + 0.75 * ratio, 0.72 - 0.18 * ratio, 0.85 - 0.65 * ratio, 0.85]
                start = arrays["grid"][i]
                geom = make_geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, spacing * 0.004, start, color)
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_ARROW, spacing * 0.004, start, start + arrays["arrows"][i]
                )
        paths = [(arrays["demo"], [0.95, 0.55, 0.12, 1.0])]
        if rollout:
            paths.append((arrays["nominal"], [0.85, 0.15, 0.65, 1.0]))
        quota = max(2, (budget - (scene.ngeom - initial)) // len(paths))
        for points, color in paths:
            nodes = np.unique(np.linspace(0, len(points) - 1, min(len(points), quota + 1), dtype=int))
            for a, b in zip(nodes[:-1], nodes[1:]):
                if np.linalg.norm(points[b] - points[a]) < 1e-8:
                    continue
                geom = make_geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025, points[a], color)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025, points[a], points[b])
        make_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            0.008,
            arrays["goal"],
            [0.1, 0.85, 0.3, 1],
            "GOAL" if self.continuous else "FIELD GOAL",
        )
        if rollout:
            cursor = arrays["nominal"][frame - self.start]
            make_geom(
                scene,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                0.006,
                cursor,
                [0.85, 0.15, 0.65, 1],
                "" if self.continuous else "NOMINAL",
            )
            actual = arrays["demo"][frame - self.start]
            if np.linalg.norm(actual - cursor) > 1e-8:
                geom = make_geom(scene, mujoco.mjtGeom.mjGEOM_LINE, 1, actual, [1, 1, 1, 0.8])
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1, actual, cursor)
        self.drawn_geoms = scene.ngeom - initial
