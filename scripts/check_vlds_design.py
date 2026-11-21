"""Check VLDS design equations and installed controller helpers without a simulator."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, help="Optional JSON report path; always prints the report.")
args = parser.parse_args()
spec = importlib.util.find_spec("robosuite")
if spec is None or spec.origin is None:
    raise SystemExit("Run with the project's LIBERO environment, which includes robosuite.")
pkg = Path(spec.origin).parent
rng = np.random.default_rng(20261121)


def functions_from(path, names, ns):
    tree = ast.parse(path.read_text())
    selected = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), "exec"), ns)


ns = {"np": np, "math": math, "EPS": np.finfo(float).eps * 4.0}
functions_from(pkg / "utils/transform_utils.py", ["axisangle2quat", "quat2mat"], ns)
ns["trans"] = SimpleNamespace(**{name: ns[name] for name in ("axisangle2quat", "quat2mat")})
functions_from(pkg / "utils/control_utils.py", ["set_goal_orientation", "set_goal_position"], ns)
functions_from(pkg / "controllers/base_controller.py", ["scale_action"], ns)
functions_from(pkg / "controllers/osc.py", ["set_goal"], ns)

minimum_covariance = 1.0
max_sym_eigenvalue = -100.0
max_radial_ratio = -100.0
max_euler_change = -100.0
max_float_error = 0.0
for _ in range(100):
    raw = rng.normal(size=(4, 19))

    def lower(x):
        a = np.zeros((3, 3))
        a[np.tril_indices(3)] = x
        a[np.diag_indices(3)] = np.logaddexp(0, np.diag(a))
        return a

    matrices, covs = [], []
    for component in raw:
        L, B = lower(component[4:10]), lower(component[13:19])
        a, b, c = component[10:13]
        skew = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
        S = L @ L.T + 1e-6 * np.eye(3)
        A = skew - B @ B.T - 0.01 * np.eye(3)
        matrices.append(A)
        covs.append(S)
        minimum_covariance = min(minimum_covariance, np.linalg.eigvalsh(S).min())
        max_sym_eigenvalue = max(max_sym_eigenvalue, np.linalg.eigvalsh((A + A.T) / 2).max())
    As, Ss = np.array(matrices), np.array(covs)

    def field(e, dtype=np.float64, matrices=As, covariances=Ss, factors=raw):
        y = np.asarray(e / 0.1, dtype=dtype)
        a = matrices.astype(dtype)
        s = covariances.astype(dtype)
        rr = factors.astype(dtype)
        diff = y - rr[:, 1:4]
        loggate = rr[:, 0] - 0.5 * (np.linalg.slogdet(s)[1] + np.einsum("ki,kij,kj->k", diff, np.linalg.inv(s), diff))
        h = np.exp(loggate - logsumexp(loggate))
        return 0.1 * np.einsum("k,kij,j->i", h, a, y)

    assert np.array_equal(field(np.zeros(3)), np.zeros(3))
    for _ in range(100):
        e = rng.normal(size=3) * 0.2
        f = field(e)
        ratio = e @ f / (e @ e)
        max_radial_ratio = max(max_radial_ratio, ratio)
        assert ratio <= -0.01 + 1e-10
        alpha = min(1.0, 0.1 / np.linalg.norm(f), -(e @ f) / (0.05 * (f @ f)))
        delta_v = 0.5 * (np.sum((e + 0.05 * alpha * f) ** 2) - np.sum(e**2))
        max_euler_change = max(max_euler_change, delta_v)
        assert delta_v <= 1e-12
        f32 = field(e, np.float32)
        assert np.isfinite(f32).all()
        max_float_error = max(max_float_error, np.linalg.norm(f - f32))
assert minimum_covariance > 0
assert max_sym_eigenvalue <= -0.01 + 1e-10

max_rotation_error, max_frame_error, max_scaling_error = 0.0, 0.0, 0.0
for _ in range(100):
    Rh, C = Rotation.random(2, random_state=rng).as_matrix()
    Rs = Rh @ C
    max_frame_error = max(max_frame_error, np.max(np.abs(Rh @ (Rh.T @ Rs) - Rs)))
    error_axis = rng.normal(size=3)
    error_axis /= np.linalg.norm(error_axis)
    angle = rng.uniform(0.001, np.pi - 0.1)
    Rg = Rotation.from_rotvec(angle * error_axis).as_matrix() @ Rs
    er = Rotation.from_matrix(Rg @ Rs.T).as_rotvec()
    omega = 2 * er
    omega *= min(1.0, 0.5 / np.linalg.norm(omega))
    delta = 0.05 * omega
    expected = Rotation.from_rotvec(delta).as_matrix() @ Rs
    observed = ns["set_goal_orientation"](delta, Rs)
    max_rotation_error = max(max_rotation_error, np.max(np.abs(expected - observed)))
    after = Rotation.from_matrix(Rg @ observed.T).magnitude()
    assert after < angle + 1e-7
    ctrl = SimpleNamespace(
        action_scale=None,
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-np.array([0.05] * 3 + [0.5] * 3),
        output_max=np.array([0.05] * 3 + [0.5] * 3),
    )
    wanted = np.r_[rng.uniform(-0.002, 0.002, 3), delta]
    action = wanted / ctrl.output_max
    got = ns["scale_action"](ctrl, action)
    max_scaling_error = max(max_scaling_error, np.max(np.abs(wanted - got)))
assert max_rotation_error < 2e-7
assert max_frame_error < 1e-12
assert max_scaling_error < 1e-12

ctrl.update = lambda: None
ctrl.scale_action = lambda a: ns["scale_action"](ctrl, a)
ctrl.impedance_mode = "fixed"
ctrl.use_delta = True
ctrl.use_ori = True
ctrl.ee_pos = np.array([0.3, 0.0, 0.5])
ctrl.ee_ori_mat = np.eye(3)
ctrl.goal_ori = Rotation.from_rotvec([0.2, 0.1, 0.0]).as_matrix()
old = ctrl.goal_ori.copy()
ctrl.position_limits = ctrl.orientation_limits = None
ctrl.interpolator_pos = ctrl.interpolator_ori = None
ns["set_goal"](ctrl, np.zeros(6))
assert np.array_equal(ctrl.goal_ori, old)
ns["set_goal"](ctrl, np.zeros(6), set_ori=np.eye(3))
assert np.array_equal(ctrl.goal_ori, np.eye(3))

# HOLD accumulator exercises the specification rule only.
inside = [True, True, False, False, True, True]
dwell = 0.0
events = []
for i, valid in enumerate(inside):
    dwell += 0.05 if valid else 0.0
    if dwell >= 0.2 - 1e-12:
        events.append(i)
assert events == [5]
report = {
    "python_version": sys.version.split()[0],
    "robosuite_version": importlib.metadata.version("robosuite"),
    "numpy_version": np.__version__,
    "scipy_version": importlib.metadata.version("scipy"),
    "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "seed": 20261121,
    "fields": 100,
    "nonzero_state_probes": 10000,
    "minimum_covariance_eigenvalue": float(minimum_covariance),
    "largest_symmetric_A_eigenvalue": float(max_sym_eigenvalue),
    "largest_radial_ratio": float(max_radial_ratio),
    "largest_nominal_Euler_delta_V": float(max_euler_change),
    "max_float32_field_absolute_difference": float(max_float_error),
    "rotation_frame_scaling_trials": 100,
    "max_rotation_element_error": float(max_rotation_error),
    "max_frame_composition_error": float(max_frame_error),
    "max_scaling_error": float(max_scaling_error),
    "zero_rotation_retains_old_target": True,
    "explicit_set_ori_refreshes_target": True,
    "hold_pause_rule": "passed",
    "source_sha256": {
        str(p.relative_to(pkg)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            pkg / "utils/transform_utils.py",
            pkg / "utils/control_utils.py",
            pkg / "controllers/base_controller.py",
            pkg / "controllers/osc.py",
        ]
    },
    "scope": "formula and extracted installed helper bodies only; no simulator, fitter, learned model or real robot",
}
if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
