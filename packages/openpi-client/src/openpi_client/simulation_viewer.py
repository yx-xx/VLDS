"""Native MuJoCo interface shared by the supported simulation environments."""

import os
import sys
import time


class InteractiveViewer:
    """MuJoCo's native 3D interface, sharing the simulator controlled by the policy."""

    def __init__(self):
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise ValueError("The interactive viewer requires a desktop DISPLAY.")
        self.handle = None
        self.enabled = True
        self.paused = False
        self.single_step = False

    def on_key(self, key: int) -> None:
        if key == 32:  # Space
            self.paused = not self.paused
        elif key == 262:  # GLFW right arrow
            self.paused = True
            self.single_step = True

    def attach(self, env) -> None:
        # robosuite 1.4 exposes the native objects only through these wrapper attributes.
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
        if not self.enabled:
            return
        from mujoco import viewer as mujoco_viewer

        options = {} if show_ui else {"show_left_ui": False, "show_right_ui": False}
        self.handle = mujoco_viewer.launch_passive(
            model,
            data,
            key_callback=self.on_key,
            **options,
        )
        with self.handle.lock():
            # Match robosuite's camera rendering: hide collision shells and show visual meshes.
            if hide_collision_shells:
                self.handle.opt.geomgroup[0] = 0
                self.handle.opt.geomgroup[1] = 1
            self.handle.cam.lookat[:] = lookat
            self.handle.cam.distance = distance
            self.handle.cam.azimuth = 135
            self.handle.cam.elevation = -25
        self.sync()

    def sync(self) -> None:
        if self.handle is not None:
            if self.handle.is_running():
                self.handle.sync()
            else:
                self.close()

    def wait_for_step(self) -> None:
        self.sync()
        while self.enabled and self.paused and not self.single_step:
            time.sleep(0.02)
            self.sync()
        self.single_step = False

    def detach(self) -> None:
        if self.handle is not None:
            if not self.handle.is_running():
                self.enabled = False
            self.handle.close()
            # close() signals the render thread asynchronously. Wait for it to
            # release the model before resetting the environment or exiting GLFW.
            deadline = time.monotonic() + 5
            # MuJoCo 2.3 (ALOHA) has no Handle.m; its is_running() waits for exit completion.
            while time.monotonic() < deadline:
                active = self.handle.m is not None if hasattr(self.handle, "m") else self.handle.is_running()
                if not active:
                    break
                time.sleep(0.01)
            self.handle = None

    def close(self) -> None:
        self.enabled = False
        self.detach()
