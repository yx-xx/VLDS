"""A directly trainable structured VLDS prediction model.

This module implements the first executable model described by the VLDS design:
H=3 ordered goals, a current four-component stable position field, and losses
that compare field behaviour at sampled states.  The input encoders consume
already-embedded language and visual features so they can be replaced by the
openpi vision-language backbone later without changing the output contract.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
from torch.nn import functional as nnf

from openpi.models.vlds_config import VLDSConfig


@dataclasses.dataclass
class VLDSPrediction:
    """Structured output of :class:`StructuredVLDSModel`."""

    goal_position: torch.Tensor
    goal_rotation_6d: torch.Tensor
    gripper_logits: torch.Tensor
    primitive_logits: torch.Tensor
    hold_log_seconds: torch.Tensor
    valid_logits: torch.Tensor
    handoff_logits: torch.Tensor
    field_raw: torch.Tensor
    field_valid_logit: torch.Tensor
    field_scale: torch.Tensor
    features: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dataclasses.asdict(self)


_TRIL = torch.tril_indices(3, 3)
_DIAGONAL = torch.tensor([0, 2, 5])


def _lower(raw: torch.Tensor) -> torch.Tensor:
    """Decode six lower-triangular factors with positive diagonal."""

    result = torch.zeros(*raw.shape[:-1], 3, 3, dtype=raw.dtype, device=raw.device)
    indices = _TRIL.to(raw.device)
    result[..., indices[0], indices[1]] = raw
    diagonal = _DIAGONAL.to(raw.device)
    result[..., torch.arange(3, device=raw.device), torch.arange(3, device=raw.device)] = nnf.softplus(
        raw[..., diagonal]
    )
    return result


def decode_seds_field(raw: torch.Tensor, config: VLDSConfig) -> dict[str, torch.Tensor]:
    """Decode ``[B, 4, 19]`` factors into a differentiable stable field.

    The symmetric part of every matrix is ``-B B^T - margin I``.  Therefore
    the decoded field has a strict common attractor at the supplied goal; this
    constraint is architectural rather than supplied by a soft loss.
    """

    if raw.ndim != 3 or raw.shape[1:] != (config.field_components, config.field_factor_dim):
        raise ValueError(f"expected raw field shape [B, {config.field_components}, {config.field_factor_dim}]")
    factor_l = _lower(raw[..., 4:10])
    factor_b = _lower(raw[..., 13:19])
    eye = torch.eye(3, dtype=raw.dtype, device=raw.device)
    covariance = factor_l @ factor_l.transpose(-1, -2) + config.covariance_floor * eye

    x, y, z = raw[..., 10], raw[..., 11], raw[..., 12]
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(*x.shape, 3, 3)
    matrices = skew - factor_b @ factor_b.transpose(-1, -2) - config.stability_margin * eye
    return {
        "weights_logits": raw[..., 0],
        "means": raw[..., 1:4],
        "covariances": covariance,
        "matrices": matrices,
    }


def evaluate_seds_field(
    raw: torch.Tensor,
    goal: torch.Tensor,
    positions: torch.Tensor,
    config: VLDSConfig,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a batch of position fields at one or more positions.

    Args:
        raw: Field factors with shape ``[B, 4, 19]``.
        goal: Shared attractor with shape ``[B, 3]``.
        positions: Query positions with shape ``[B, N, 3]`` or ``[B, 3]``.
        scale: Optional positive velocity scale with shape ``[B, 1]``.

    Returns:
        Velocity vectors with shape ``[B, N, 3]``.
    """

    if positions.ndim == 2:
        positions = positions[:, None, :]
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B, N, 3] or [B, 3]")
    if goal.ndim != 2 or goal.shape[-1] != 3:
        raise ValueError("goal must have shape [B, 3]")
    physical = decode_seds_field(raw, config)
    y = (positions - goal[:, None, :]) / config.length_scale
    means = physical["means"]
    covariance = physical["covariances"]
    chol = torch.linalg.cholesky(covariance)

    delta = y[:, None, :, :] - means[:, :, None, :]
    delta = delta.transpose(-1, -2)
    standardized = torch.linalg.solve_triangular(chol, delta, upper=False).transpose(-1, -2)
    mahalanobis = (standardized**2).sum(dim=-1)
    logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    log_prob = physical["weights_logits"][:, :, None] - 0.5 * (mahalanobis + logdet[:, :, None])
    weights = torch.softmax(log_prob, dim=1)
    velocity = torch.einsum("bkn,bkij,bnj->bni", weights, physical["matrices"], y)
    velocity = config.length_scale * velocity
    if scale is not None:
        velocity = velocity * scale[:, None, :]
    return velocity


class StructuredVLDSModel(nn.Module):
    """Horizon-three structured policy head with a replaceable input encoder."""

    def __init__(self, config: VLDSConfig | None = None) -> None:
        super().__init__()
        self.config = VLDSConfig() if config is None else config
        config = self.config
        hidden = config.hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(nn.Linear(config.context_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.language_encoder = nn.Sequential(nn.Linear(config.language_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        plan_dim = config.plan_dim
        self.plan_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, config.horizon * plan_dim)
        )
        self.field_head = nn.Sequential(
            nn.Linear(hidden + plan_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.field_raw_dim + 2),
        )

    def forward(
        self,
        state_history: torch.Tensor,
        execution_context: torch.Tensor,
        language: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> VLDSPrediction:
        config = self.config
        if state_history.ndim != 3 or state_history.shape[-1] != config.state_dim:
            raise ValueError(f"state_history must have shape [B, T, {config.state_dim}]")
        if execution_context.ndim != 2 or execution_context.shape[-1] != config.context_dim:
            raise ValueError(f"execution_context must have shape [B, {config.context_dim}]")
        if language is None:
            language = torch.zeros(
                state_history.shape[0], config.language_dim, dtype=state_history.dtype, device=state_history.device
            )
        if language.ndim != 2 or language.shape[-1] != config.language_dim:
            raise ValueError(f"language must have shape [B, {config.language_dim}]")
        state_features = self.state_encoder(state_history)
        if history_mask is None:
            history_mask = torch.ones(state_history.shape[:2], dtype=state_history.dtype, device=state_history.device)
        weights = history_mask.to(state_features.dtype)[..., None]
        history_mean = (state_features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        history_last = state_features[:, -1]
        context_features = self.context_encoder(execution_context)
        language_features = self.language_encoder(language)
        features = self.fusion(torch.cat((history_mean, history_last, context_features, language_features), dim=-1))

        plan_dim = config.plan_dim
        plan = self.plan_head(features).reshape(-1, config.horizon, plan_dim)
        plan_position = plan[..., :3]
        plan_rotation_6d = plan[..., 3:9]
        gripper_logits = plan[..., 9:10]
        primitive_logits = plan[..., 10:12]
        hold_log_seconds = plan[..., 12:13]
        valid_logits = plan[..., 13:14]
        handoff_logits = plan[..., 14 : 14 + config.handoff_role_dim]

        current_condition = torch.cat((features, plan[:, 0]), dim=-1)
        field_output = self.field_head(current_condition)
        field_raw = field_output[..., : config.field_raw_dim].reshape(
            -1, config.field_components, config.field_factor_dim
        )
        field_valid_logit = field_output[..., config.field_raw_dim : config.field_raw_dim + 1]
        field_scale = nnf.softplus(field_output[..., config.field_raw_dim + 1 :]) + 0.05
        return VLDSPrediction(
            goal_position=plan_position,
            goal_rotation_6d=plan_rotation_6d,
            gripper_logits=gripper_logits,
            primitive_logits=primitive_logits,
            hold_log_seconds=hold_log_seconds,
            valid_logits=valid_logits,
            handoff_logits=handoff_logits,
            field_raw=field_raw,
            field_valid_logit=field_valid_logit,
            field_scale=field_scale,
            features=features,
        )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def vlds_loss(
    prediction: VLDSPrediction,
    batch: dict[str, torch.Tensor],
    config: VLDSConfig,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute masked structured losses and return scalar diagnostics."""

    weights = {
        "goal": 1.0,
        "rotation": 0.25,
        "command": 0.2,
        "primitive": 0.2,
        "hold": 0.1,
        "field_behavior": 1.0,
        "field_parameter": 0.1,
        "field_valid": 0.05,
        "field_scale": 0.1,
        "handoff": 0.1,
        "structure": 0.05,
        **(weights or {}),
    }
    valid = batch["valid_mask"].bool()
    current_valid = valid[:, 0]
    goal_error = nnf.smooth_l1_loss(prediction.goal_position, batch["goal_position"], reduction="none")
    rotation_error = nnf.mse_loss(prediction.goal_rotation_6d, batch["goal_rotation_6d"], reduction="none")
    command_error = nnf.binary_cross_entropy_with_logits(
        prediction.gripper_logits, batch["gripper_command"].to(prediction.gripper_logits.dtype), reduction="none"
    )
    primitive_error = nnf.cross_entropy(
        prediction.primitive_logits.reshape(-1, 2), batch["primitive_kind"].long().reshape(-1), reduction="none"
    ).reshape_as(valid)
    hold_error = nnf.smooth_l1_loss(prediction.hold_log_seconds, batch["hold_log_seconds"], reduction="none")
    hold_mask = valid & batch["primitive_kind"].bool()
    handoff_error = nnf.cross_entropy(
        prediction.handoff_logits.reshape(-1, config.handoff_role_dim),
        batch["handoff_role"].long().reshape(-1),
        reduction="none",
    ).reshape_as(valid)

    field_mask = batch["field_supervision_mask"].bool() & current_valid
    predicted_velocity = evaluate_seds_field(
        prediction.field_raw,
        prediction.goal_position[:, 0],
        batch["field_positions"],
        config,
        prediction.field_scale,
    )
    behavior_mask = field_mask[:, None] & batch["field_sample_mask"].bool()
    behavior_error = ((predicted_velocity - batch["field_velocity"]) ** 2).sum(dim=-1)
    field_parameter_error = nnf.smooth_l1_loss(
        prediction.field_raw, batch["field_raw"].to(prediction.field_raw.dtype), reduction="none"
    )
    field_valid_error = nnf.binary_cross_entropy_with_logits(
        prediction.field_valid_logit.squeeze(-1),
        batch["field_supervision_mask"].to(prediction.field_valid_logit.dtype),
        reduction="none",
    )
    field_scale_error = nnf.smooth_l1_loss(
        prediction.field_scale, batch["field_scale"].to(prediction.field_scale.dtype), reduction="none"
    )
    speed_penalty = nnf.relu(predicted_velocity.norm(dim=-1) - config.max_speed) ** 2

    components = {
        "goal": _masked_mean(goal_error, valid),
        "rotation": _masked_mean(rotation_error, valid),
        "command": _masked_mean(command_error, valid),
        "primitive": _masked_mean(primitive_error, valid),
        "hold": _masked_mean(hold_error, hold_mask),
        "handoff": _masked_mean(handoff_error, valid),
        "field_behavior": _masked_mean(behavior_error, behavior_mask),
        "field_parameter": _masked_mean(field_parameter_error, field_mask),
        "field_valid": field_valid_error.mean(),
        "field_scale": _masked_mean(field_scale_error, field_mask),
        "structure": _masked_mean(speed_penalty, behavior_mask),
    }
    total = sum(weights[name] * components[name] for name in components)
    metrics = {name: float(value.detach()) for name, value in components.items()}
    metrics["total"] = float(total.detach())
    metrics["field_valid_fraction"] = float(field_mask.float().mean().detach())
    return total, metrics


__all__ = [
    "StructuredVLDSModel",
    "VLDSPrediction",
    "decode_seds_field",
    "evaluate_seds_field",
    "vlds_loss",
]
