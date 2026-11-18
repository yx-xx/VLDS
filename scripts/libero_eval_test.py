import contextlib
import importlib.util
import json
import pathlib
import sys
import types
from unittest import mock

import cv2
import numpy as np
from openpi_client import websocket_client_policy
import pytest


@pytest.fixture
def evaluator():
    libero_package = types.ModuleType("libero.libero")
    libero_package.benchmark = mock.Mock()
    libero_package.get_libero_path = mock.Mock(return_value="/tmp/bddl_files")
    environments = types.ModuleType("libero.libero.envs")
    environments.OffScreenRenderEnv = mock.Mock()
    module_path = pathlib.Path(__file__).resolve().parents[1] / "examples/libero/main.py"
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
    )
    with (
        mock.patch.object(websocket_client_policy, "WebsocketClientPolicy") as policy_factory,
        mock.patch.object(evaluator.imageio, "mimwrite") as write_video,
        mock.patch.object(evaluator, "LiveViewer") as viewer_factory,
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
        viewer_factory.return_value.show.assert_called()
        viewer_factory.return_value.close.assert_called_once()
    else:
        viewer_factory.assert_not_called()


@pytest.mark.parametrize("task_ids", [[], [-1], [2], [0, 0]])
def test_invalid_task_selection(evaluator, task_ids):
    task_suite = mock.Mock(n_tasks=2)
    evaluator.benchmark.get_benchmark_dict.return_value = {"libero_spatial": lambda: task_suite}
    with pytest.raises(ValueError, match="task_ids"):
        evaluator.eval_libero(evaluator.Args(task_ids=task_ids))


@pytest.mark.parametrize("key", [-1, ord("q"), ord("Q"), 27])
def test_live_viewer_preserves_observations(evaluator, monkeypatch, key):
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
        viewer = evaluator.LiveViewer()
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


def test_live_viewer_requires_display(evaluator, monkeypatch):
    monkeypatch.setattr(evaluator.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(ValueError, match="DISPLAY"):
        evaluator.LiveViewer()


def test_live_viewer_handles_closed_window(evaluator, monkeypatch):
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
        viewer = evaluator.LiveViewer()
        viewer.show({}, task_id=0, episode_idx=0, step=0, status="Running")
        assert not viewer.enabled
        gui["imshow"].assert_not_called()
        gui["destroyWindow"].assert_called_once()


def test_viewer_closes_on_evaluation_failure(evaluator):
    with (
        mock.patch.object(evaluator, "LiveViewer") as viewer_factory,
        mock.patch.object(evaluator, "_eval_libero", side_effect=RuntimeError("evaluation failed")),
        pytest.raises(RuntimeError, match="evaluation failed"),
    ):
        evaluator.eval_libero(evaluator.Args(viewer=True))
    viewer_factory.return_value.close.assert_called_once()
