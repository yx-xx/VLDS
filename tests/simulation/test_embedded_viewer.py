"""Exercise real cross-environment RGB transport, camera control and GL resource lifetimes."""

import os
import subprocess
import sys
import textwrap

import numpy as np
from openpi_client.simulation_frames import FrameBuffer
import pytest

from openpi.simulation import paths


def test_frame_buffer_keeps_latest_complete_frame_and_cleans_up():
    frames = FrameBuffer()
    path = frames.path
    writer = FrameBuffer(path)
    try:
        assert frames.read() is None
        writer.write(np.full((32, 48, 3), 12, dtype=np.uint8))
        np.testing.assert_array_equal(frames.read(), 12)
        assert frames.read() is None
        writer.write(np.full((64, 80, 3), 24, dtype=np.uint8))
        writer.write(np.full((64, 80, 3), 36, dtype=np.uint8))
        image = frames.read()
        assert image.shape == (64, 80, 3)
        np.testing.assert_array_equal(image, 36)
        with pytest.raises(ValueError, match="uint8 RGB"):
            writer.write(np.zeros((2, 2)))
    finally:
        writer.close()
        frames.close()
    assert not path.exists()
    assert frames.read() is None
    frames.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires a desktop GL display")
@pytest.mark.parametrize("interpreter", [sys.executable, str(paths.ROOT / "examples/libero/.venv/bin/python")])
def test_embedded_renderer_in_both_environments_without_visible_glfw_window(interpreter):
    if not os.path.isfile(interpreter):
        pytest.skip("LIBERO environment is not installed")
    frames = FrameBuffer()
    code = textwrap.dedent('''
        import sys
        import glfw
        import mujoco
        import numpy as np
        from openpi_client.embedded_viewer import EmbeddedViewer
        from openpi_client.simulation_frames import FrameBuffer

        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <light pos="0 0 3"/>
          <geom type="plane" size="2 2 0.1" rgba="0.3 0.4 0.5 1"/>
          <body pos="0 0 0.2"><joint type="slide" axis="1 0 0"/>
            <geom type="box" size="0.12 0.12 0.12" rgba="1 0 0 1"/>
          </body></worldbody></mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        viewer = EmbeddedViewer(sys.argv[1])
        reader = FrameBuffer(sys.argv[1])
        try:
            viewer.attach_native(model, data, lookat=(0, 0, 0.2), distance=1.5)
            assert not glfw.get_window_attrib(glfw.get_current_context(), glfw.VISIBLE)
            before = reader.read()
            assert before.shape == (600, 800, 3)
            assert before.std() > 5
            original = data.qpos.copy()
            viewer.command(dict(command='view', action='rotate', dx=0.2, dy=0.1))
            viewer.sync()
            rotated = reader.read()
            assert np.any(before != rotated)
            np.testing.assert_array_equal(data.qpos, original)
            viewer.command(dict(command='view', action='pan', dx=0.1, dy=0.1))
            viewer.command(dict(command='view', action='zoom', dy=0.2))
            viewer.sync()
            assert np.any(reader.read() != rotated)
            viewer.command(dict(command='view', action='focus', lookat=[0.1, 0, 0.3], distance=0.7))
            np.testing.assert_array_equal(viewer.cam.lookat, [0.1, 0, 0.3])
            assert viewer.cam.distance == 0.7
            viewer.command(dict(command='view', action='focus', lookat=[float('nan'), 0, 0], distance=-1))
            np.testing.assert_array_equal(viewer.cam.lookat, [0.1, 0, 0.3])
            np.testing.assert_array_equal(data.qpos, original)
            viewer.command(dict(command='view', action='reset'))
            np.testing.assert_array_equal(viewer.cam.lookat, [0, 0, 0.2])
            assert viewer.cam.distance == 1.5
            viewer.command(dict(command='view', action='resize', width=320, height=200))
            viewer.sync()
            small = reader.read()
            assert small.shape == (200, 320, 3)
            data.qpos[0], data.time = 0.4, 0.05
            mujoco.mj_forward(model, data)
            viewer.sync()
            assert np.any(reader.read() != small)
            viewer.sync()
            assert reader.read() is None
            # Paused configuration changes must publish a fresh RGB frame at the same physics time.
            class TintOverlay:
                def __init__(self, color):
                    self.color = color
                def draw(self, scene, clear=False):
                    scene.geoms[0].rgba[:3] = self.color
            viewer.set_overlay(TintOverlay([1, 0, 0]))
            viewer.sync()
            red = reader.read()
            assert red is not None
            viewer.set_overlay(TintOverlay([0, 1, 0]))
            viewer.sync()
            green = reader.read()
            assert green is not None and np.any(red != green)
            assert data.time == 0.05
            viewer.set_overlay(None)
            # Switching scenes releases the old GL context, then creates another hidden one.
            viewer.detach()
            viewer.attach_native(model, data)
            assert not glfw.get_window_attrib(glfw.get_current_context(), glfw.VISIBLE)
            viewer.sync()
            assert reader.read() is not None
        finally:
            viewer.close()
            reader.close()
    ''')
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYOPENGL_PLATFORM", None)
    environment.update(MUJOCO_GL="glfw", LIBGL_ALWAYS_SOFTWARE="1", __GLX_VENDOR_LIBRARY_NAME="mesa")
    try:
        subprocess.run(
            [interpreter, "-c", code, str(frames.path)],
            cwd=paths.ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert frames.read().shape == (200, 320, 3)
    finally:
        frames.close()
