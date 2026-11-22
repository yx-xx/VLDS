"""Float64, four-component position SEDS adapter for VLDS design-v1.

The 4 x 19 factors contain logit, mean(3), lower L(6), skew xyz(3),
lower B(6). Lower triangles are row-major; diagonal entries use softplus.
Fitting minimizes measured, time-weighted velocity MSE in this stable family.
It does not reproduce the original SEDS SQP solver or certify robot dynamics.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import logsumexp

VERSION = "seds_mse_pos3_k4_v1"
TRIL = np.tril_indices(3)
DIAGONAL = np.array([0, 2, 5])


@dataclasses.dataclass(frozen=True)
class SEDSConfig:
    length_scale: float = 0.1
    covariance_floor: float = 1e-6
    stability_margin: float = 0.01
    max_condition: float = 1e6
    max_matrix_norm: float = 20.0
    restarts: int = 3
    max_iterations: int = 300
    seed: int = 20261122

    def __post_init__(self):
        for key, value in dataclasses.asdict(self).items():
            if not np.isfinite(value) or value < 0 or (key != "seed" and value == 0):
                raise ValueError(f"Invalid SEDS configuration: {key}")
        if not 0 < self.covariance_floor < 1 or self.max_condition < 1:
            raise ValueError("Invalid covariance configuration")
        if self.restarts != 3 or self.max_iterations > 300:
            raise ValueError("design-v1 uses three initializations and at most 300 iterations each")


def finite_array(value, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite array of shape {shape}; got {array.shape}")
    return array


DEFAULT_CONFIG = SEDSConfig()


def lower(raw):
    raw = finite_array(raw)
    matrix = np.zeros((*raw.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., TRIL[0], TRIL[1]] = raw
    matrix[..., np.arange(3), np.arange(3)] = np.logaddexp(0.0, raw[..., DIAGONAL])
    return matrix


def _inverse_softplus(value):
    value = np.asarray(value, dtype=np.float64)
    if np.any(value <= 0):
        raise ValueError("Positive diagonal required")
    return value + np.log(-np.expm1(-value))


def _encode_lower(matrix):
    result = matrix[..., TRIL[0], TRIL[1]].copy()
    result[..., DIAGONAL] = _inverse_softplus(result[..., DIAGONAL])
    return result


def decode(raw, config=DEFAULT_CONFIG):
    raw = finite_array(raw, (4, 19))
    factor_l, factor_b = lower(raw[:, 4:10]), lower(raw[:, 13:19])
    covariance = factor_l @ factor_l.transpose(0, 2, 1) + config.covariance_floor * np.eye(3)
    skew = np.zeros((4, 3, 3), dtype=np.float64)
    x, y, z = raw[:, 10:13].T
    skew[:, 0, 1], skew[:, 1, 0] = -z, z
    skew[:, 0, 2], skew[:, 2, 0] = y, -y
    skew[:, 1, 2], skew[:, 2, 1] = -x, x
    matrices = skew - factor_b @ factor_b.transpose(0, 2, 1) - config.stability_margin * np.eye(3)
    return {
        "weights": np.exp(raw[:, 0] - logsumexp(raw[:, 0])),
        "means": raw[:, 1:4].copy(),
        "covariances": covariance,
        "matrices": matrices,
    }


class PositionField:
    """One fixed target and fixed factors, evaluated at the actual control point."""

    def __init__(self, raw, goal, config=DEFAULT_CONFIG):
        self.raw = finite_array(raw, (4, 19)).copy()
        self.goal = finite_array(goal, (3,)).copy()
        self.config = config
        self.physical = decode(self.raw, config)
        if any(not np.isfinite(value).all() for value in self.physical.values()):
            raise ValueError("Nonfinite decoded field")
        covariance = self.physical["covariances"]
        matrices = self.physical["matrices"]
        condition = np.linalg.cond(covariance)
        norms = np.linalg.norm(matrices, ord=2, axis=(1, 2))
        margin = -np.linalg.eigvalsh((matrices + matrices.transpose(0, 2, 1)) / 2)[:, -1]
        self.diagnostics = {
            "max_covariance_condition": float(condition.max()),
            "max_matrix_norm_per_second": float(norms.max()),
            "minimum_stability_margin_per_second": float(margin.min()),
        }
        if not all(np.isfinite(x) for x in self.diagnostics.values()):
            raise ValueError("Nonfinite decoded field")
        if condition.max() > config.max_condition or norms.max() > config.max_matrix_norm + 1e-10:
            raise ValueError(f"Field exceeds covariance condition or matrix norm limit: {self.diagnostics}")
        if margin.min() < config.stability_margin - 1e-10:
            raise ValueError("Field violates stability margin")
        self.cholesky = np.linalg.cholesky(covariance)
        self.log_weights = self.raw[:, 0] - logsumexp(self.raw[:, 0])
        self.log_determinant = 2 * np.log(np.diagonal(self.cholesky, axis1=1, axis2=2)).sum(axis=1)

    def velocity(self, position):
        position = finite_array(position)
        if position.shape[-1:] != (3,):
            raise ValueError("Position must end in three coordinates")
        shape = position.shape
        y = (position.reshape(-1, 3) - self.goal) / self.config.length_scale
        log_gates = []
        for k in range(4):
            mean = self.physical["means"][k]
            chol, log_weight, log_det = self.cholesky[k], self.log_weights[k], self.log_determinant[k]
            standardized = solve_triangular(chol, (y - mean).T, lower=True, check_finite=False)
            log_gates.append(log_weight - 0.5 * (log_det + (standardized**2).sum(axis=0)))
        log_gates = np.stack(log_gates, axis=1)
        gates = np.exp(log_gates - logsumexp(log_gates, axis=1, keepdims=True))
        result = self.config.length_scale * np.einsum("nk,kij,nj->ni", gates, self.physical["matrices"], y)
        return finite_array(result).reshape(shape)

    def record(self):
        return {
            "version": VERSION,
            "config": dataclasses.asdict(self.config),
            "goal_position": self.goal.tolist(),
            "field_raw": self.raw.tolist(),
            "field_physical": {key: value.tolist() for key, value in self.physical.items()},
            "constraints": self.diagnostics,
        }

    @classmethod
    def from_record(cls, record):
        if record["version"] != VERSION:
            raise ValueError("Unsupported field version")
        field = cls(record["field_raw"], record["goal_position"], SEDSConfig(**record["config"]))
        if "field_physical" not in record:
            raise ValueError("Archive is missing decoded physical parameters")
        for key, value in field.physical.items():
            saved = finite_array(record["field_physical"].get(key), value.shape)
            if not np.allclose(value, saved, atol=1e-12, rtol=1e-10):
                raise ValueError("Archive factors disagree with physical parameters")
        return field


def linear_factors(rate, config=DEFAULT_CONFIG):
    if not np.isfinite(rate) or rate <= config.stability_margin:
        raise ValueError("Linear contraction rate must exceed the stability margin")
    raw = np.zeros((4, 19), dtype=np.float64)
    raw[:, 4:10] = _encode_lower(np.eye(3) * np.sqrt(1 - config.covariance_floor))
    raw[:, 13:19] = _encode_lower(np.eye(3) * np.sqrt(rate - config.stability_margin))
    return raw


def time_weights(time):
    time = finite_array(time)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("At least two strictly increasing sample times required")
    delta = np.diff(time)
    return np.r_[delta[0] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1] / 2]


def _initialization(y, velocity, weights, config, rng):
    # Deterministic k-means++ followed by Lloyd updates; no optional sklearn dependency.
    centers = [y[int(rng.integers(len(y)))]]
    for _ in range(3):
        distances = np.min(np.sum((y[:, None] - np.array(centers)) ** 2, axis=-1), axis=1)
        probabilities = distances * weights
        index = rng.choice(len(y), p=probabilities / probabilities.sum()) if probabilities.sum() > 1e-14 else 0
        centers.append(y[index])
    centers = np.array(centers)
    for _ in range(15):
        labels = np.argmin(np.sum((y[:, None] - centers) ** 2, axis=-1), axis=1)
        for k in range(4):
            if np.any(labels == k):
                centers[k] = np.average(y[labels == k], axis=0, weights=weights[labels == k])
    raw = linear_factors(1.0, config)
    raw[:, 1:4] = centers
    for k in range(4):
        mask = labels == k
        mass = float(weights[mask].sum())
        raw[k, 0] = np.log(max(mass / weights.sum(), 1e-4))
        centered = y[mask] - centers[k]
        covariance = (centered.T * weights[mask]) @ centered / max(mass, 1e-12) + np.eye(3) * 0.005
        raw[k, 4:10] = _encode_lower(np.linalg.cholesky(covariance))
    # Project a weighted linear regression onto the strict Euclidean Lyapunov cone.
    a = np.linalg.solve((y.T * weights) @ y + np.eye(3) * 1e-5, (y.T * weights) @ velocity).T
    eigenvalues, eigenvectors = np.linalg.eigh((a + a.T) / 2)
    symmetric = (eigenvectors * np.minimum(eigenvalues, -config.stability_margin - 0.02)) @ eigenvectors.T
    b = np.linalg.cholesky(-symmetric - config.stability_margin * np.eye(3))
    skew = (a - a.T) / 2
    raw[:, 10:13] = [skew[2, 1], skew[0, 2], skew[1, 0]]
    raw[:, 13:19] = _encode_lower(b)
    return raw


def velocity_objective(position, velocity, weights, config):
    """Torch autograd supplies a CPU float64 Jacobian to SciPy L-BFGS."""
    import torch
    import torch.nn.functional as nnf

    y = torch.tensor(position, dtype=torch.float64)
    v = torch.tensor(velocity, dtype=torch.float64)
    w = torch.tensor(weights / weights.sum(), dtype=torch.float64)
    indices = torch.tril_indices(3, 3)
    eye = torch.eye(3, dtype=torch.float64)
    # Normalize the objective only, preserving its minimizer and physical reported MSE.
    scale = max(float(np.sum(weights[:, None] * velocity**2) / weights.sum()), 1e-6)

    def triangular(values):
        diagonal = torch.zeros_like(values)
        diagonal[:, DIAGONAL] = nnf.softplus(values[:, DIAGONAL]) - values[:, DIAGONAL]
        result = torch.zeros((4, 3, 3), dtype=torch.float64)
        result[:, indices[0], indices[1]] = values + diagonal
        return result

    def evaluate(flat):
        raw = torch.tensor(flat.reshape(4, 19), dtype=torch.float64, requires_grad=True)
        factor_l, factor_b = triangular(raw[:, 4:10]), triangular(raw[:, 13:19])
        covariance = factor_l @ factor_l.transpose(1, 2) + config.covariance_floor * eye
        chol = torch.linalg.cholesky(covariance)
        delta = (y[:, None, :] - raw[None, :, 1:4]).permute(1, 2, 0)
        standardized = torch.linalg.solve_triangular(chol, delta, upper=False)
        mahalanobis = torch.sum(standardized**2, dim=1).T
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(dim=1)
        gates = torch.softmax(raw[None, :, 0] - 0.5 * (mahalanobis + logdet), dim=1)
        x, yy, z = raw[:, 10:13].unbind(dim=1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, yy, z, zero, -x, -yy, x, zero), dim=1).reshape(4, 3, 3)
        matrices = skew - factor_b @ factor_b.transpose(1, 2) - config.stability_margin * eye
        prediction = torch.einsum("nk,kij,nj->ni", gates, matrices, y)
        loss = torch.sum(w[:, None] * (prediction - v) ** 2) / scale
        loss.backward()
        return float(loss.detach()), raw.grad.numpy().ravel().copy()

    return evaluate


def fit_linear(position, velocity, time, config=DEFAULT_CONFIG):
    position = finite_array(position)
    velocity = finite_array(velocity, position.shape)
    weights = time_weights(time)
    if position.shape != (len(weights), 3):
        raise ValueError("Position, velocity and time must have matching sample counts")
    goal = position[-1].copy()
    centered = position - goal
    denominator = float(np.sum(weights[:, None] * centered**2))
    rate = float(-np.sum(weights[:, None] * centered * velocity) / max(denominator, 1e-30))
    rate = float(np.clip(rate, max(0.02, config.stability_margin + 0.01), 20.0))
    simple = PositionField(linear_factors(rate, config), goal, config)
    simple_mse = float(np.average(np.sum((simple.velocity(position) - velocity) ** 2, axis=1), weights=weights))
    radial = np.sum(centered * velocity, axis=1)
    report = {
        "version": VERSION,
        "samples": len(position),
        "objective": "trapezoidal-time-weighted physical velocity MSE",
        "linear_baseline_rate_per_second": rate,
        "linear_baseline_velocity_mse": simple_mse,
        "radial_conflict_fraction": float(np.average(radial > 1e-6, weights=weights)),
        "radial_conflict_tolerance_m2_per_second": 1e-6,
        "radial_conflict_indices": np.flatnonzero(radial > 1e-6).tolist(),
        "restarts": [],
    }
    report.update(
        kind="linear_baseline", optimizer_converged=None, velocity_mse=simple_mse, parameter_supervision=False
    )
    return simple, report


def fit(position, velocity, time, config=DEFAULT_CONFIG):
    position = finite_array(position)
    velocity = finite_array(velocity, position.shape)
    weights = time_weights(time)
    simple, report = fit_linear(position, velocity, time, config)
    goal, simple_mse = simple.goal, report["linear_baseline_velocity_mse"]
    centered = position - goal
    if len(position) < 8:
        report.update(
            kind="short_linear", optimizer_converged=True, velocity_mse=simple_mse, parameter_supervision=False
        )
        return simple, report
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    candidates = []
    try:
        y, v = centered / config.length_scale, velocity / config.length_scale
        objective = velocity_objective(y, v, weights, config)
        # A fixed box bounds ||B||_F^2 <= 6*1.6^2 and ||Omega||_2 <= sqrt(3)*2.5,
        # hence ||A||_2 < 20. The deployed decoder itself remains unchanged.
        b_maximum = min(
            1.6, np.sqrt(max(0.01, config.max_matrix_norm - config.stability_margin - np.sqrt(3) * 2.5) / 6) * 0.999
        )
        bounds = [(None, None)] * 76
        for component in range(4):
            for j in range(10, 13):
                bounds[component * 19 + j] = (-2.5, 2.5)
            for j in range(6):
                bounds[component * 19 + 13 + j] = (
                    (-12.0, float(_inverse_softplus(b_maximum))) if j in DIAGONAL else (-b_maximum, b_maximum)
                )
        report["optimizer_search_box"] = {
            "skew_absolute_max": 2.5,
            "B_element_absolute_max": b_maximum,
            "B_diagonal_raw_min": -12.0,
        }
        for restart in range(config.restarts):
            rng = np.random.default_rng(config.seed + restart)
            initial = _initialization(y, v, weights, config, rng)
            result = minimize(
                objective,
                initial.ravel(),
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": config.max_iterations, "maxls": 30, "ftol": 1e-10, "gtol": 1e-6},
            )
            record = {
                "seed": config.seed + restart,
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "converged": bool(result.success),
                "message": str(result.message),
            }
            try:
                raw = result.x.reshape(4, 19).copy()
                raw[:, 0] -= raw[:, 0].mean()
                raw = raw[np.lexsort(raw[:, 1:4].T[::-1])]
                field = PositionField(raw, goal, config)
                mse = float(np.average(np.sum((field.velocity(position) - velocity) ** 2, axis=1), weights=weights))
                record["velocity_mse"] = mse
                record["constraints_passed"] = True
                candidates.append((mse, field, restart, bool(result.success)))
            except ValueError as error:
                record.update(constraints_passed=False, reason=str(error))
            report["restarts"].append(record)
    finally:
        torch.set_num_threads(previous_threads)
    if not candidates:
        report.update(kind="seds", optimizer_converged=False, velocity_mse=None, parameter_supervision=False)
        return None, report
    mse, field, restart, converged = min(candidates, key=lambda item: item[0])
    report.update(
        kind="seds",
        optimizer_converged=converged,
        velocity_mse=mse,
        selected_restart=restart,
        parameter_supervision=True,
    )
    return field, report
