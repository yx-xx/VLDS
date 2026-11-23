"""Gaussian-gated maps with one common quadratic Lyapunov certificate.

This is a discrete LPV estimator, not the original SEDS optimizer. Every local
map contracts in the saved metric. A positive scalar speed cap preserves that
certificate. Fits are checked by closed-loop rollouts, not derivative MSE alone.
"""

from __future__ import annotations

import contextlib
import dataclasses

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import softmax

VERSION = "gaussian_contractive_map_v1"


@dataclasses.dataclass(frozen=True)
class FitConfig:
    components: int = 4
    regression_iterations: int = 100
    rollout_iterations: int = 160
    contraction: float = 0.9999
    maximum_metric_condition: float = 1000.0
    fit_margin: float = 0.9


DEFAULT_CONFIG = FitConfig()


def _project(matrices, radius):
    left, singular, right = np.linalg.svd(matrices)
    return (left * np.minimum(singular, radius)[..., None, :]) @ right


def learned_metric(reference, maximum_condition, goal=None):
    """Use a stable regression's Lyapunov metric, bounded away from singularity."""
    reference = np.asarray(reference)
    goal = reference[-1] if goal is None else np.asarray(goal)
    x, y = reference[:-1] - goal, reference[1:] - goal
    dimension = x.shape[1]
    ridge = 1e-3
    transition = np.linalg.solve(x.T @ x + ridge * np.eye(dimension), x.T @ y + ridge * 0.99 * np.eye(dimension)).T
    eigenvalues, vectors = np.linalg.eig(transition)
    if np.linalg.cond(vectors) > 1e8:
        return np.eye(dimension)
    stable = eigenvalues * np.minimum(1.0, 0.98 / np.maximum(abs(eigenvalues), 1e-15))
    transition = (vectors @ np.diag(stable) @ np.linalg.inv(vectors)).real
    metric = solve_discrete_lyapunov(transition.T, np.eye(dimension))
    values, vectors = np.linalg.eigh((metric + metric.T) / 2)
    if values.min() <= 0 or not np.isfinite(values).all():
        return np.eye(dimension)
    values = np.maximum(values, values.max() / maximum_condition)
    metric = (vectors * values) @ vectors.T
    metric *= dimension / np.trace(metric)
    return np.linalg.cholesky(metric).T


class ContractiveField:
    def __init__(self, record):
        if record.get("version") != VERSION:
            raise ValueError("Unsupported contractive field version")
        self.goal = np.asarray(record["goal"], dtype=float)
        self.tolerance = np.asarray(record["tolerance"], dtype=float)
        self.transform = np.asarray(record["transform"], dtype=float)
        self.means = np.asarray(record["means"], dtype=float)
        self.variance = np.asarray(record["variance"], dtype=float)
        self.matrices = np.asarray(record["matrices"], dtype=float)
        self.dt = float(record["dt"])
        self.speed = float(record["maximum_speed_scaled_per_second"])
        d, k = len(self.goal), len(self.means)
        expected = ((d,), (d,), (d, d), (k, d), (d,), (k, d, d))
        values = (self.goal, self.tolerance, self.transform, self.means, self.variance, self.matrices)
        if any(a.shape != s or not np.isfinite(a).all() for a, s in zip(values, expected, strict=True)):
            raise ValueError("Invalid contractive field dimensions or values")
        if (
            not k
            or np.any(self.tolerance <= 0)
            or np.any(self.variance <= 0)
            or not np.isfinite([self.dt, self.speed]).all()
            or min(self.dt, self.speed) <= 0
            or np.linalg.cond(self.transform) ** 2 > 1000.0 + 1e-5
        ):
            raise ValueError("Invalid contractive metric or limits")
        self.inverse = np.linalg.inv(self.transform)
        self.radius = float(np.linalg.svd(self.matrices, compute_uv=False).max())
        if self.radius >= 1.0:
            raise ValueError("Local maps lack a strict common Lyapunov certificate")

    def record(self):
        return {
            "version": VERSION,
            **{
                name: getattr(self, name).tolist()
                for name in ("goal", "tolerance", "transform", "means", "variance", "matrices")
            },
            "dt": self.dt,
            "maximum_speed_scaled_per_second": self.speed,
            "maximum_contraction": self.radius,
            "metric_condition": float(np.linalg.cond(self.transform) ** 2),
        }

    def step(self, position):
        position = np.asarray(position, dtype=float)
        if position.shape[-1:] != self.goal.shape or not np.isfinite(position).all():
            raise ValueError("Invalid state for field evaluation")
        z = ((position - self.goal) / self.tolerance) @ self.transform.T
        gates = softmax(-0.5 * ((z[..., None, :] - self.means) ** 2 / self.variance).sum(axis=-1), axis=-1)
        next_z = np.einsum("...k,kij,...j->...i", gates, self.matrices, z)
        delta = (next_z - z) @ self.inverse.T
        norm = np.linalg.norm(delta, axis=-1, keepdims=True)
        alpha = np.minimum(1.0, self.speed * self.dt / np.maximum(norm, 1e-30))
        return position + alpha * delta * self.tolerance

    def velocity(self, position):
        return (self.step(position) - position) / self.dt

    def rollout(self, initial, steps):
        position = np.asarray(initial).copy()
        values = [position.copy()]
        for _ in range(steps):
            position = self.step(position)
            values.append(position.copy())
        return np.asarray(values)


def _regression(reference, tolerance, dt, speed, transform, config, goal=None):
    scaled = reference / tolerance
    goal = reference[-1] if goal is None else np.asarray(goal)
    z = (scaled - goal / tolerance) @ transform.T
    x, y = z[:-1], z[1:]
    k = min(config.components, max(1, len(x) // 6))
    means = x[np.linspace(0, len(x) - 1, k).astype(int)]
    variance = np.maximum(np.var(x, axis=0) * 0.4, 0.25)
    gates = softmax(-0.5 * ((x[:, None, :] - means) ** 2 / variance).sum(-1), axis=-1)
    design = (gates[:, :, None] * x[:, None, :]).reshape(len(x), -1)
    d = x.shape[1]
    gram = design.T @ design + 1e-4 * np.eye(k * d)
    target = design.T @ y
    matrices = np.linalg.solve(gram, target).reshape(k, d, d).transpose(0, 2, 1)
    matrices = _project(matrices, config.contraction)
    step = 1.0 / max(np.linalg.eigvalsh(gram)[-1], 1e-12)
    for _ in range(config.regression_iterations):
        flat = matrices.transpose(0, 2, 1).reshape(k * d, d)
        gradient = (gram @ flat - target).reshape(k, d, d).transpose(0, 2, 1)
        matrices = _project(matrices - step * gradient, config.contraction)
    return ContractiveField(
        {
            "version": VERSION,
            "goal": np.asarray(goal),
            "tolerance": tolerance,
            "transform": transform,
            "means": means,
            "variance": variance,
            "matrices": matrices,
            "dt": dt,
            "maximum_speed_scaled_per_second": speed,
        }
    )


def maximum_error(field, reference, carried):
    starts = np.stack([reference[0], carried])
    trace = field.rollout(starts, len(reference) - 1)
    error = np.linalg.norm((trace - reference[:, None]) / field.tolerance, axis=-1)
    return float(error.max()), trace


def refine_field(field, reference, carried, config):
    import torch

    torch.set_num_threads(1)
    dtype = torch.float64
    means = torch.tensor(field.means, dtype=dtype)
    variance = torch.tensor(field.variance, dtype=dtype)
    inverse = torch.tensor(field.inverse, dtype=dtype)
    target = torch.tensor((reference - field.goal) / field.tolerance, dtype=dtype)
    start = torch.tensor((np.stack([reference[0], carried]) - field.goal) / field.tolerance, dtype=dtype)
    transform = torch.tensor(field.transform, dtype=dtype)
    matrices = torch.nn.Parameter(torch.tensor(field.matrices, dtype=dtype))
    optimizer = torch.optim.Adam([matrices], lr=0.003)
    best, _ = maximum_error(field, reference, carried)
    saved = field.matrices.copy()
    for _ in range(config.rollout_iterations):
        x = start @ transform.T
        values = [start]
        for _ in range(len(reference) - 1):
            h = torch.softmax(-0.5 * ((x[:, None] - means) ** 2 / variance).sum(-1), dim=-1)
            next_x = torch.einsum("bk,kij,bj->bi", h, matrices, x)
            delta = (next_x - x) @ inverse.T
            norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            alpha = torch.clamp(field.speed * field.dt / torch.clamp(norm, min=1e-30), max=1.0)
            x = x + alpha * (next_x - x)
            values.append(x @ inverse.T)
        error = torch.linalg.vector_norm(torch.stack(values) - target[:, None], dim=-1)
        maximum = float(error.max().detach())
        if maximum < best:
            best, saved = maximum, matrices.detach().numpy().copy()
        if maximum <= config.fit_margin:
            break
        loss = (error**2).mean() + 0.2 * (error**4).mean().sqrt()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([matrices], 100.0)
        optimizer.step()
        with torch.no_grad():
            projected = _project(matrices.detach().numpy(), config.contraction)
            matrices.copy_(torch.tensor(projected, dtype=dtype))
    record = field.record()
    record["matrices"] = saved
    return ContractiveField(record)


def fit(reference, tolerance, dt, speed, *, carried=None, goal=None, refine=True, config=DEFAULT_CONFIG):
    reference, tolerance = np.asarray(reference), np.asarray(tolerance)
    if reference.ndim != 2 or len(reference) < 2 or tolerance.shape != (reference.shape[1],):
        raise ValueError("Expected aligned multidimensional observations and tolerance")
    if (
        not np.isfinite(reference).all()
        or not np.isfinite(tolerance).all()
        or np.any(tolerance <= 0)
        or not np.isfinite([dt, speed]).all()
        or min(dt, speed) <= 0
        or config.components < 1
        or not 0 < config.contraction < 1
        or config.regression_iterations < 0
        or config.rollout_iterations < 0
        or not 1 <= config.maximum_metric_condition <= 1000
    ):
        raise ValueError("Invalid reference or tolerance")
    carried = reference[0] if carried is None else np.asarray(carried)
    goal = reference[-1] if goal is None else np.asarray(goal)
    if carried.shape != reference.shape[1:] or not np.isfinite(carried).all():
        raise ValueError("Invalid carried state")
    if goal.shape != reference.shape[1:] or not np.isfinite(goal).all():
        raise ValueError("Invalid goal state")
    transforms = [np.eye(reference.shape[1])]
    with contextlib.suppress(ValueError, np.linalg.LinAlgError):
        transforms.append(learned_metric(reference / tolerance, config.maximum_metric_condition, goal=goal / tolerance))
    candidates = [
        _regression(reference, tolerance, dt, speed, transform, config, goal=goal) for transform in transforms
    ]
    field = min(candidates, key=lambda candidate: maximum_error(candidate, reference, carried)[0])
    error, _ = maximum_error(field, reference, carried)
    if refine and config.fit_margin < error < 4.0 and config.rollout_iterations:
        field = refine_field(field, reference, carried, config)
    error, trace = maximum_error(field, reference, carried)
    return field, trace, error
