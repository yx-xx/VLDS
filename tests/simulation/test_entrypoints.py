"""Exercise the launch paths across working directories and Python environments."""

import json
import os
import subprocess
import sys

import pytest

from openpi.simulation import paths


def headless_environment():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("DISPLAY", None)
    return environment


@pytest.mark.parametrize("entrypoint", ["script", "module"])
def test_catalog_launchers_work_outside_repository_without_gui(tmp_path, entrypoint):
    # Run the real launcher in a fresh process to detect eager GUI or model imports.
    code = """
import runpy
import sys

entrypoint, launcher = sys.argv[1:]
sys.argv = [launcher, "--catalog"]
if entrypoint == "script":
    runpy.run_path(launcher, run_name="__main__")
else:
    runpy.run_module("openpi.simulation", run_name="__main__")
assert not {"tkinter", "mujoco", "jax", "torch"}.intersection(sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-c", code, entrypoint, str(paths.ROOT / "scripts/view_simulation.py")],
        cwd=tmp_path,
        env=headless_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    catalog = json.loads(process.stdout)
    assert set(catalog) == {"libero", "aloha"}
    assert all("available" in environment and "suites" in environment for environment in catalog.values())


def test_franka_script_remains_a_usable_entrypoint(tmp_path):
    process = subprocess.run(
        [sys.executable, str(paths.ROOT / "scripts/run_franka_sim.py"), "--help"],
        cwd=tmp_path,
        env=headless_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "--mode" in process.stdout
    assert "--viewer" in process.stdout


@pytest.mark.parametrize("worker", ["scene.py", "replay.py"])
@pytest.mark.parametrize("interpreter", [sys.executable, str(paths.ROOT / "examples/libero/.venv/bin/python")])
def test_workers_run_directly_in_each_environment(tmp_path, worker, interpreter):
    if not os.path.isfile(interpreter):
        pytest.skip("Optional LIBERO environment is not installed")
    process = subprocess.run(
        [interpreter, str(paths.WORKERS_ROOT / worker), "--help"],
        cwd=tmp_path,
        env=headless_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "usage:" in process.stdout
