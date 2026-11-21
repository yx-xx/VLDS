import contextlib
import importlib.util
import json
import pathlib
import sys
import types
from unittest import mock

import cv2
import numpy as np
from openpi_client import simulation_viewer
from openpi_client import websocket_client_policy
import pytest

from openpi.simulation.paths import ROOT


@pytest.fixture
def evaluator():
    libero_package = types.ModuleType("libero.libero")
    libero_package.benchmark = mock.Mock()
    libero_package.get_libero_path = mock.Mock(return_value="/tmp/bddl_files")
    environments = types.ModuleType("libero.libero.envs")
    environments.OffScreenRenderEnv = mock.Mock()
    module_path = ROOT / "examples/libero/main.py"
    spec = importlib.util.spec_from_file_location("libero_evaluator_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "libero": types.ModuleType("libero"),
            "libero.libero": libero_package,
            "libero.libero.envs": environments,
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
        yield module


@pytest.mark.parametrize(
    ("action_chunk", "expected_error"),
    [
        (np.zeros((5, 7)), False),
        (np.full((5, 7), np.nan), True),
        (np.zeros((5, 8)), True),
        (np.zeros((2, 7)), True),
        (np.zeros(7), True),
    ],
)
@pytest.mark.parametrize("viewer_enabled", [False, True])
def test_rollout_validates_actions_and_writes_results(
    evaluator, tmp_path: pathlib.Path, action_chunk, expected_error, viewer_enabled
):
    observation = {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    task_suite = mock.Mock(n_tasks=2)
    task_suite.get_task.return_value = mock.Mock(language="move block", problem_folder="suite", bddl_file="task.bddl")
    task_suite.get_task_init_states.return_value = [np.zeros(1)]
    evaluator.benchmark.get_benchmark_dict.return_value = {"libero_spatial": lambda: task_suite}
    environment = mock.Mock()
    environment.set_init_state.return_value = observation
    environment.step.return_value = (observation, 0, False, {})
    evaluator.OffScreenRenderEnv.return_value = environment
    args = evaluator.Args(
        task_ids=[1],
        max_steps=2,
        num_steps_wait=1,
        num_trials_per_task=1,
        video_out_path=str(tmp_path),
        viewer=viewer_enabled,
        camera_preview=viewer_enabled,
    )
    with (
        mock.patch.object(websocket_client_policy, "WebsocketClientPolicy") as policy_factory,
        mock.patch.object(evaluator.imageio, "mimwrite") as write_video,
        mock.patch.object(evaluator, "InteractiveViewer") as viewer_factory,
        mock.patch.object(evaluator, "CameraPreview") as preview_factory,
    ):
        policy_factory.return_value.infer.return_value = {"actions": action_chunk}
        expectation = pytest.raises(RuntimeError, match="rollout") if expected_error else contextlib.nullcontext()
        with expectation:
            evaluator.eval_libero(args)
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["robot"] == "Panda"
    assert results["viewer"] == viewer_enabled
    assert results["max_steps"] == 2
    assert results["total_episodes"] == 1
    assert results["episodes"][0]["task_id"] == 1
    assert len(results["errors"]) == int(expected_error)
    assert environment.step.call_count == (1 if expected_error else 3)
    assert evaluator.OffScreenRenderEnv.call_args.kwargs["robots"] == ["Panda"]
    environment.seed.assert_called_once_with(7)
    environment.close.assert_called_once()
    write_video.assert_called_once()
    if viewer_enabled:
        viewer_factory.return_value.attach.assert_called_once_with(environment)
        viewer_factory.return_value.sync.assert_called()
        viewer_factory.return_value.wait_for_step.assert_called()
        preview_factory.return_value.show.assert_called()
        preview_factory.return_value.close.assert_called_once()
        viewer_factory.return_value.close.assert_called_once()
    else:
        viewer_factory.assert_not_called()
        preview_factory.assert_not_called()


@pytest.mark.parametrize("task_ids", [[], [-1], [2], [0, 0]])
def test_invalid_task_selection(evaluator, task_ids):
    task_suite = mock.Mock(n_tasks=2)
    evaluator.benchmark.get_benchmark_dict.return_value = {"libero_spatial": lambda: task_suite}
    with pytest.raises(ValueError, match="task_ids"):
        evaluator.eval_libero(evaluator.Args(task_ids=task_ids))


@pytest.mark.parametrize("key", [-1, ord("q"), ord("Q"), 27])
def test_camera_preview_preserves_observations(evaluator, monkeypatch, key):
    monkeypatch.setenv("DISPLAY", ":1")
    observation = {
        "agentview_image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "robot0_eye_in_hand_image": np.full((2, 3, 3), 100, dtype=np.uint8),
    }
    originals = {name: image.copy() for name, image in observation.items()}
    with mock.patch.multiple(
        cv2,
        namedWindow=mock.DEFAULT,
        resizeWindow=mock.DEFAULT,
        getWindowProperty=mock.DEFAULT,
        imshow=mock.DEFAULT,
        waitKey=mock.DEFAULT,
        destroyWindow=mock.DEFAULT,
    ) as gui:
        gui["getWindowProperty"].return_value = 1
        gui["waitKey"].return_value = key
        viewer = evaluator.CameraPreview()
        viewer.show(observation, task_id=0, episode_idx=0, step=1, status="Running")
        frame = gui["imshow"].call_args.args[1]
        np.testing.assert_array_equal(frame[44:, :3], originals["agentview_image"][::-1, ::-1, ::-1])
        np.testing.assert_array_equal(frame[44:, 3:], originals["robot0_eye_in_hand_image"][::-1, ::-1, ::-1])
        assert viewer.enabled == (key == -1)
        viewer.close()
        viewer.show(observation, task_id=0, episode_idx=0, step=2, status="Running")
        gui["imshow"].assert_called_once()
        gui["destroyWindow"].assert_called_once()
    for name, image in observation.items():
        np.testing.assert_array_equal(image, originals[name])


def test_camera_preview_requires_display(evaluator, monkeypatch):
    monkeypatch.setattr(evaluator.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(ValueError, match="DISPLAY"):
        evaluator.CameraPreview()


def test_camera_preview_handles_closed_window(evaluator, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":1")
    with mock.patch.multiple(
        cv2,
        namedWindow=mock.DEFAULT,
        resizeWindow=mock.DEFAULT,
        getWindowProperty=mock.DEFAULT,
        imshow=mock.DEFAULT,
        destroyWindow=mock.DEFAULT,
    ) as gui:
        gui["getWindowProperty"].return_value = 0
        viewer = evaluator.CameraPreview()
        viewer.show({}, task_id=0, episode_idx=0, step=0, status="Running")
        assert not viewer.enabled
        gui["imshow"].assert_not_called()
        gui["destroyWindow"].assert_called_once()


def test_viewer_closes_on_evaluation_failure(evaluator):
    with (
        mock.patch.object(evaluator, "InteractiveViewer") as viewer_factory,
        mock.patch.object(evaluator, "_eval_libero", side_effect=RuntimeError("evaluation failed")),
        pytest.raises(RuntimeError, match="evaluation failed"),
    ):
        evaluator.eval_libero(evaluator.Args(viewer=True))
    viewer_factory.return_value.close.assert_called_once()


def test_interactive_viewer_attaches_current_simulator_after_reset(evaluator, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":1")
    native_model, native_data = object(), object()
    env = mock.Mock()
    env.sim.model = mock.Mock(_model=native_model)
    env.sim.data = mock.Mock(_data=native_data)
    first = mock.MagicMock(m=None)
    second = mock.MagicMock(m=None)
    first.cam.lookat = np.zeros(3)
    second.cam.lookat = np.zeros(3)
    with mock.patch("mujoco.viewer.launch_passive", side_effect=[first, second]) as launch:
        viewer = evaluator.InteractiveViewer()
        viewer.attach(env)
        assert launch.call_args.args == (native_model, native_data)
        assert launch.call_args.kwargs["key_callback"] == viewer.on_key
        native_model, native_data = object(), object()
        env.sim.model = mock.Mock(_model=native_model)
        env.sim.data = mock.Mock(_data=native_data)
        viewer.attach(env)
        first.close.assert_called_once()
        assert launch.call_args.args == (native_model, native_data)
        second.sync.assert_called_once()
        viewer.close()
        second.close.assert_called_once()


def test_interactive_viewer_pause_and_single_step(evaluator, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":1")
    viewer = evaluator.InteractiveViewer()
    viewer.handle = mock.Mock(m=None)
    viewer.on_key(32)
    assert viewer.paused
    # Release exactly one control step from a paused loop.
    with mock.patch.object(simulation_viewer.time, "sleep", side_effect=lambda _: viewer.on_key(262)) as sleep:
        viewer.wait_for_step()
    sleep.assert_called_once()
    assert viewer.paused
    assert not viewer.single_step
    viewer.on_key(32)
    assert not viewer.paused
    viewer.close()


def test_interactive_viewer_stays_hidden_after_window_closed(evaluator, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":1")
    viewer = evaluator.InteractiveViewer()
    handle = mock.Mock(m=None)
    handle.is_running.return_value = False
    viewer.handle = handle
    viewer.paused = True
    viewer.wait_for_step()
    assert not viewer.enabled
    handle.close.assert_called_once()
    with mock.patch("mujoco.viewer.launch_passive") as launch:
        viewer.attach(mock.Mock())
    launch.assert_not_called()


def test_interactive_viewer_requires_display(evaluator, monkeypatch):
    monkeypatch.setattr(evaluator.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(ValueError, match="DISPLAY"):
        evaluator.InteractiveViewer()
