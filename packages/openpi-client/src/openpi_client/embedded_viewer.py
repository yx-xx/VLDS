"""Render interactive MuJoCo views into the desktop's frame buffer without a native window."""

from __future__ import annotations

import contextlib

import mujoco
import numpy as np

from openpi_client.simulation_frames import MAX_HEIGHT
from openpi_client.simulation_frames import MAX_WIDTH
from openpi_client.simulation_frames import FrameBuffer


class _Renderer:
    """Explicit GL lifetime works with both MuJoCo 2.3 (ALOHA) and 3.2 (LIBERO)."""

    def __init__(self, model, *, height, width):
        self.model = model
        self.gl = mujoco.GLContext(width, height)
        self.context = None
        try:
            self.gl.make_current()
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            self.rect = mujoco.MjrRect(0, 0, width, height)
            self.pixels = np.empty((height, width, 3), dtype=np.uint8)
        except Exception:
            self.close()
            raise

    def update_scene(self, data, *, camera, scene_option):
        self.gl.make_current()
        mujoco.mjv_updateScene(self.model, data, scene_option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)

    def render(self):
        self.gl.make_current()
        mujoco.mjr_render(self.rect, self.scene, self.context)
        mujoco.mjr_readPixels(self.pixels, None, self.rect, self.context)
        return self.pixels[::-1]

    def close(self):
        if self.gl is not None:
            self.gl.make_current()
            if self.context is not None:
                self.context.free()
                self.context = None
            self.gl.free()
            self.gl = None


class EmbeddedViewer:
    """Same simulator attachment contract as InteractiveViewer, with camera commands from Tk."""

    def __init__(self, path):
        self.frames = FrameBuffer(path)
        self.handle = None
        self.enabled = True
        self.paused = False
        self.single_step = False
        self.renderer = None
        self.model = self.data = None
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.width, self.height = 800, 600
        self.overlay = None
        self.dirty = True
        self.last_time = None
        self.home = None

    @staticmethod
    def lock():
        return contextlib.nullcontext()

    def attach(self, env):
        self.attach_native(env.sim.model._model, env.sim.data._data, hide_collision_shells=True)  # noqa: SLF001

    def attach_native(
        self,
        model,
        data,
        *,
        hide_collision_shells=False,
        lookat=(0.0, 0.0, 0.8),
        distance=2.2,
        show_ui=True,
    ):
        self.detach()
        self.model, self.data = model, data
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        if hide_collision_shells:
            self.opt.geomgroup[0] = 0
            self.opt.geomgroup[1] = 1
        self.cam.lookat[:] = lookat
        self.cam.distance = distance
        self.cam.azimuth, self.cam.elevation = 135, -25
        self.home = (np.array(lookat), distance, 135, -25)
        self.handle = self
        self._resize()
        self.sync()

    def _resize(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.model is None:
            return
        # This changes the render target capacity only, not the physics or model cameras.
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.height)
        self.renderer = _Renderer(self.model, height=self.height, width=self.width)
        self.dirty = True

    def set_overlay(self, overlay):
        self.overlay = overlay
        self.dirty = True

    def command(self, message):
        if message.get("command") != "view":
            return False
        action = message.get("action")
        if action == "resize":
            width = max(160, min(MAX_WIDTH, int(message["width"])))
            height = max(120, min(MAX_HEIGHT, int(message["height"])))
            if (width, height) != (self.width, self.height):
                self.width, self.height = width, height
                self._resize()
        elif self.renderer is not None:
            if action == "reset":
                lookat, distance, azimuth, elevation = self.home
                self.cam.lookat[:] = lookat
                self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
            elif action in ("rotate", "pan", "zoom"):
                movement = {
                    "rotate": mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                    "pan": mujoco.mjtMouse.mjMOUSE_MOVE_V,
                    "zoom": mujoco.mjtMouse.mjMOUSE_ZOOM,
                }[action]
                dx, dy = float(message.get("dx", 0)), float(message.get("dy", 0))
                if np.isfinite([dx, dy]).all():
                    mujoco.mjv_moveCamera(self.model, movement, dx, dy, self.renderer.scene, self.cam)
            self.dirty = True
        return True

    def sync(self):
        if self.renderer is None or (not self.dirty and self.last_time == float(self.data.time)):
            return
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        if self.overlay is not None:
            self.overlay.draw(self.renderer.scene, clear=False)
        self.frames.write(self.renderer.render())
        self.last_time = float(self.data.time)
        self.dirty = False

    def detach(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.handle = self.model = self.data = None
        self.overlay = None
        self.last_time = None

    def close(self):
        self.enabled = False
        self.detach()
        self.frames.close()
