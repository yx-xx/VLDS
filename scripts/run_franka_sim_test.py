import json
import pathlib
from unittest import mock

import pytest

from scripts import run_franka_sim


def test_default_franka_run():
    args = run_franka_sim.parse_args([])
    assert args.models == ["pi05"]
    assert args.task_ids == [0]
    assert args.trials == 1
    assert not args.viewer
    assert args.mode == "single"
    assert args.suites == ["libero_spatial"]
    assert args.output_dir == run_franka_sim.ROOT / "data/franka_sim"


def test_full_evaluation_mode():
    args = run_franka_sim.parse_args(["--mode", "full"])
    assert args.suites == list(run_franka_sim.TASK_SUITES)
    assert args.all_tasks
    assert args.trials == 50
    assert not args.viewer
    assert args.output_dir.name.startswith("full_")


def test_multi_scene_view_mode(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":1")
    args = run_franka_sim.parse_args(["--mode", "view"])
    assert args.suites == ["libero_spatial", "libero_object", "libero_goal"]
    assert args.task_ids == [0, 1, 2]
    assert args.trials == 1
    assert args.viewer
    assert not args.all_tasks
    assert args.output_dir.name.startswith("view_")


def test_mode_overrides(tmp_path: pathlib.Path):
    args = run_franka_sim.parse_args(
        ["--mode", "full", "--suite", "libero_goal", "--trials", "2", "--output-dir", str(tmp_path)]
    )
    assert args.suites == ["libero_goal"]
    assert args.trials == 2
    assert args.output_dir == tmp_path


def test_all_three_models():
    args = run_franka_sim.parse_args(["--models", "pi0", "pi0_fast", "pi05"])
    assert [run_franka_sim.MODELS[model] for model in args.models] == [
        "pi0_libero",
        "pi0_fast_libero",
        "pi05_libero",
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--trials", "0"],
        ["--max-steps", "0"],
        ["--startup-timeout", "0"],
        ["--task-ids", "-1"],
        ["--task-ids", "0", "0"],
        ["--models", "pi0", "pi0"],
        ["--models", "pi0", "pi05", "--checkpoint", "/tmp/checkpoint"],
        ["--port", "65536"],
    ],
)
def test_reject_invalid_arguments(arguments):
    with pytest.raises(SystemExit):
        run_franka_sim.parse_args(arguments)


def test_small_client_run(tmp_path: pathlib.Path):
    args = run_franka_sim.parse_args(["--task-ids", "1", "3", "--max-steps", "5"])
    command = run_franka_sim.client_command(args, tmp_path)
    assert command[command.index("--args.host") + 1] == "127.0.0.1"
    assert command[command.index("--args.num-trials-per-task") + 1] == "1"
    assert command[command.index("--args.task-ids") + 1 : command.index("--args.max-steps")] == ["1", "3"]
    assert command[command.index("--args.max-steps") + 1] == "5"


def test_full_suite_client_run(tmp_path: pathlib.Path):
    args = run_franka_sim.parse_args(["--all-tasks"])
    assert "--args.task-ids" not in run_franka_sim.client_command(args, tmp_path)


def test_viewer_client_run(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":1")
    args = run_franka_sim.parse_args(["--viewer"])
    assert "--args.viewer" in run_franka_sim.client_command(args, tmp_path)


@pytest.mark.parametrize("arguments", [["--viewer"], ["--mode", "view"]])
def test_viewer_requires_display(monkeypatch, arguments):
    monkeypatch.setattr(run_franka_sim.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(SystemExit):
        run_franka_sim.parse_args(arguments)


def test_all_models_and_suites_share_summary(tmp_path: pathlib.Path):
    sim_python = tmp_path / "python"
    sim_python.touch()
    completed = []

    def run_model(args, model, environment):
        result = {"model": model, "task_suite": args.suite}
        completed.append(result)
        return result

    with (
        mock.patch.object(run_franka_sim, "simulation_env", return_value={}),
        mock.patch.object(run_franka_sim, "run_model", side_effect=run_model),
    ):
        run_franka_sim.main(
            [
                "--mode",
                "full",
                "--models",
                "pi0",
                "pi05",
                "--trials",
                "1",
                "--output-dir",
                str(tmp_path),
                "--sim-python",
                str(sim_python),
            ]
        )
    assert len(completed) == 10
    assert {(result["model"], result["task_suite"]) for result in completed} == {
        (model, suite) for model in ("pi0", "pi05") for suite in run_franka_sim.TASK_SUITES
    }
    assert json.loads((tmp_path / "summary.json").read_text()) == completed


def test_failed_server_does_not_start_simulation():
    server = mock.Mock(returncode=1)
    server.poll.return_value = 1
    with pytest.raises(RuntimeError, match="server exited"):
        run_franka_sim.wait_for_server(server, port=8000, timeout=1)


def test_server_stops_if_client_fails(tmp_path: pathlib.Path):
    args = run_franka_sim.parse_args(["--output-dir", str(tmp_path)])
    server = mock.Mock()
    server.poll.return_value = None
    with (
        mock.patch.object(run_franka_sim.socket, "socket"),
        mock.patch.object(run_franka_sim, "wait_for_server"),
        mock.patch.object(run_franka_sim.subprocess, "Popen", return_value=server) as start_server,
        mock.patch.object(run_franka_sim.subprocess, "run", side_effect=RuntimeError("client failed")),
        pytest.raises(RuntimeError, match="client failed"),
    ):
        run_franka_sim.run_model(args, "pi05", {})
    server.terminate.assert_called_once()
    server.wait.assert_called_once_with(timeout=30)
    command = start_server.call_args.args[0]
    assert command[command.index("--host") + 1] == "127.0.0.1"
