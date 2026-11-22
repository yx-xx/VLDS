import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.fields import primitive
from openpi.fields import seds
from openpi.simulation.runtime.field_control import OSCFieldAdapter
from openpi.simulation.runtime.field_execution import run_worker


def test_osc_adapter_refreshes_zero_rotation_and_reverses_live_action_scaling():
    controller = SimpleNamespace(
        impedance_mode="fixed",
        use_delta=True,
        use_ori=True,
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-np.array([0.05] * 3 + [0.5] * 3),
        output_max=np.array([0.05] * 3 + [0.5] * 3),
    )
    calls = []

    def set_goal(action, set_pos=None, set_ori=None):
        calls.append((action, set_ori))
        controller.goal_pos = set_pos if set_pos is not None else np.array([9, 9, 9])
        controller.goal_ori = set_ori

    controller.set_goal = set_goal
    adapter = OSCFieldAdapter(controller)
    target = primitive.Primitive(seds.PositionField(seds.linear_factors(1), np.zeros(3)), np.eye(3), -1)
    action, expected = adapter.action(target, np.array([0.02, 0, 0]), np.eye(3))
    controller.set_goal(action[:6])
    assert action[-1] == -1
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][1], np.eye(3))
    np.testing.assert_allclose(action[:3], [-0.02, 0, 0], atol=1e-14)
    assert adapter.verify_goal(expected) == (0, 0)
    target.rotation = Rotation.from_rotvec([0, 0, 0.3]).as_matrix()
    action, expected = adapter.action(target, np.array([0.02, 0, 0]), np.eye(3))
    assert action[5] == pytest.approx(0.05)
    controller.set_goal(action[:6])
    adapter.verify_goal(expected)


def test_worker_timeout_kills_process_and_records_failure_terminal_state(tmp_path):
    pid_file = tmp_path / "pid"
    script = "import os, pathlib, time; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(3)"
    result = run_worker(
        [sys.executable, "-c", script], cwd=tmp_path, env=os.environ.copy(), output=tmp_path, timeout=0.2
    )
    assert result["status"] == "timeout"
    assert json.loads((tmp_path / "worker.json").read_text())["status"] == "timeout"
    assert json.loads((tmp_path / "result.json").read_text())["infrastructure_failure"]
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    assert not Path(f"/proc/{pid}").exists()


def test_worker_nonzero_exit_kept_as_failure(tmp_path):
    result = run_worker(
        [sys.executable, "-c", "raise SystemExit(7)"], cwd=tmp_path, env=os.environ.copy(), output=tmp_path, timeout=3
    )
    assert result["status"] == "failed"
    assert result["returncode"] == 7


def test_pose_reads_refresh_derived_mujoco_transforms():
    from openpi.simulation.workers.fields import actual_pose

    class Sim:
        def __init__(self):
            self.x = np.array([9.0, 9, 9])
            self.data = SimpleNamespace(
                get_site_xpos=lambda _: self.x,
                get_site_xmat=lambda _: np.eye(3),
                get_site_xvelp=lambda _: np.zeros(3),
                get_site_xvelr=lambda _: np.zeros(3),
            )

        def forward(self):
            self.x = np.array([1.0, 2, 3])

    p, _, _, _ = actual_pose(SimpleNamespace(sim=Sim()))
    np.testing.assert_array_equal(p, [1, 2, 3])
