"""Validate scene selection, adapter contracts and policy reuse across manual scene switches."""

# ruff: noqa: SLF001 -- These tests exercise the process lifecycle boundaries.

import json
import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap
from unittest import mock

import numpy as np
import pytest

from openpi.simulation import catalog
from openpi.simulation import paths
from openpi.simulation.runtime import controller
from openpi.simulation.workers import scene as simulation_worker


@pytest.fixture
def small_catalog():
    return {
        "libero": {"available": True, "suites": [{"id": "libero_spatial", "tasks": [{"id": 0, "name": "move bowl"}]}]},
        "aloha": {
            "available": True,
            "suites": [
                {"id": "transfer", "tasks": [{"id": 0, "name": "Transfer cube"}]},
                {"id": "insertion", "tasks": [{"id": 0, "name": "Insert peg"}]},
            ],
        },
    }


def test_catalog_keeps_benchmark_task_order(tmp_path, monkeypatch):
    libero = tmp_path / "libero"
    (libero / "benchmark").mkdir(parents=True)
    task_map = {name: [f"task_{i}" for i in range(10)] for name in catalog.LIBERO_SUITES}
    (libero / "benchmark/libero_suite_task_map.py").write_text("libero_task_map = " + repr(task_map))
    (libero / "benchmark/__init__.py").write_text("task_orders = " + repr([list(reversed(range(10)))]))
    monkeypatch.setattr(paths, "LIBERO_ROOT", libero)
    result = catalog.build_catalog()["libero"]["suites"]
    assert result[0]["tasks"][0] == {"id": 0, "name": "task 9", "kind": "抓取放置"}
    assert result[-1]["tasks"][0]["name"] == "task 0"


@pytest.mark.parametrize(
    "overrides", [{"suite": "missing"}, {"task_id": -1}, {"seed": 2**32}, {"model": "wrong"}, {"initial_state": -1}]
)
def test_reject_invalid_selection(small_catalog, overrides):
    with pytest.raises(ValueError, match="任务|种子|模型"):
        catalog.validate_selection({"dataset": "libero", "suite": "libero_spatial", **overrides}, small_catalog)


def test_insertion_requires_matching_checkpoint(small_catalog, tmp_path):
    selected = catalog.validate_selection({"dataset": "aloha", "suite": "insertion"}, small_catalog)
    with pytest.raises(ValueError, match="ALOHA"):
        catalog.policy_spec(selected)
    selected["checkpoint"] = str(tmp_path)
    assert catalog.policy_spec(selected) == ("pi0_aloha_sim", str(tmp_path))


@pytest.mark.parametrize("dimensions", [7, 14])
def test_action_contracts(dimensions):
    np.testing.assert_array_equal(simulation_worker.validate_actions(np.zeros((5, dimensions)), dimensions), 0)
    for invalid in (
        np.zeros((5, 21 - dimensions)),
        np.full((5, dimensions), np.nan),
        np.zeros((0, dimensions)),
        np.zeros(dimensions),
    ):
        with pytest.raises(ValueError, match="horizon"):
            simulation_worker.validate_actions(invalid, dimensions)


def test_software_backend_matches_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "LIBERO_ROOT", tmp_path / "libero")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("PYTHONPATH", "/external/ros")
    for dataset, backend in (("libero", "glx"), ("aloha", "glfw")):
        environment = controller.simulation_environment({"dataset": dataset, "software_rendering": True})
        assert environment["MUJOCO_GL"] == backend
        assert "PYOPENGL_PLATFORM" not in environment
        assert "PYTHONPATH" not in environment


def test_same_model_is_reused_after_scene_switch(small_catalog):
    with mock.patch.object(catalog, "build_catalog", return_value=small_catalog):
        control = controller.InspectorController()
    control.selection = catalog.validate_selection({"dataset": "libero", "suite": "libero_spatial"}, small_catalog)
    control.worker = mock.Mock()
    control.worker.poll.return_value = None
    control.policy = mock.Mock()
    control.policy.poll.return_value = None
    control.policy_key = catalog.policy_spec(control.selection)
    control.policy_ready = True
    control.policy_port = 8123
    with mock.patch.object(controller.subprocess, "Popen") as popen, mock.patch.object(control, "_stop_policy") as stop:
        control._load_policy()
    popen.assert_not_called()
    stop.assert_not_called()
    message = json.loads(control.worker.stdin.write.call_args.args[0])
    assert message == {"command": "policy_ready", "port": 8123}


def test_stale_policy_monitor_cannot_connect_new_scene(small_catalog):
    with mock.patch.object(catalog, "build_catalog", return_value=small_catalog):
        control = controller.InspectorController()
    old_policy = mock.Mock()
    control.policy = mock.Mock()
    with mock.patch.object(control, "send") as send:
        control._wait_for_policy(old_policy, 8123, ("config", "checkpoint"), "server.log")
    send.assert_not_called()
    assert control.events.empty()


@pytest.mark.skipif(sys.platform != "linux" or not os.environ.get("DISPLAY"), reason="Requires a Linux desktop display")
def test_official_launcher_uses_desktop_fonts():
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    native = pathlib.Path("/usr/lib") / version / "lib-dynload" / f"_tkinter{sysconfig.get_config_var('EXT_SUFFIX')}"
    if not native.is_file():
        pytest.skip("No matching system Tk extension installed")
    # Start a fresh interpreter so earlier imports cannot hide a bootstrap-order bug.
    # Import the real launcher and inspector in their normal order, then run its main.
    code = textwrap.dedent(
        """
        import json
        import sys
        from unittest import mock

        from scripts import view_simulation
        from openpi.simulation.ui import inspector
        from openpi.simulation.ui.tk_runtime import font, tk, ttk
        import _tkinter

        roots = []
        result = {}

        def inspect_window(root):
            roots.append(root)
            root.withdraw()
            root.update_idletasks()
            style = ttk.Style(root)
            result.update(
                backend=_tkinter.__file__,
                families=list(font.families(root)),
                body=font.nametofont("TkDefaultFont", root=root).actual(),
                numbers=font.nametofont("TkFixedFont", root=root).actual(),
                labels=font.Font(root=root, font=style.lookup("TLabel", "font")).actual(),
                heading=font.Font(root=root, font=style.lookup("Heading.TLabel", "font")).actual(),
            )

        try:
            with mock.patch.object(tk.Tk, "mainloop", inspect_window), mock.patch.object(
                sys, "argv", ["scripts/view_simulation.py"]
            ):
                view_simulation.main()
            print(json.dumps(result))
        finally:
            for root in roots:
                root.destroy()
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=paths.ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    result = json.loads(process.stdout)
    assert pathlib.Path(result["backend"]) == native
    assert result["labels"]["family"] == result["body"]["family"]
    assert result["heading"]["family"] == result["body"]["family"]
    assert result["heading"]["weight"] == "bold"
    if "Noto Sans CJK SC" in result["families"]:
        assert result["body"]["family"] == "Noto Sans CJK SC"
    if "DejaVu Sans Mono" in result["families"]:
        assert result["numbers"]["family"] == "DejaVu Sans Mono"
