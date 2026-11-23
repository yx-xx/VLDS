"""Structured VLDS batches and a deterministic synthetic smoke dataset."""

from __future__ import annotations

import dataclasses

import torch
from torch.utils.data import Dataset

from openpi.models.vlds import evaluate_seds_field
from openpi.models.vlds_config import VLDSConfig


@dataclasses.dataclass(frozen=True)
class VLDSSchema:
    """Names and shapes used by the first ``vlds.fields.v1`` batch."""

    version: str = "vlds.fields.v1"
    horizon: int = 3
    history_length: int = 4
    field_shape: tuple[int, int] = (4, 19)


def validate_vlds_batch(batch: dict[str, torch.Tensor], config: VLDSConfig) -> None:
    """Validate the tensor contract before a production batch reaches the loss."""

    required = {
        "state_history",
        "execution_context",
        "language",
        "goal_position",
        "goal_rotation_6d",
        "gripper_command",
        "primitive_kind",
        "hold_log_seconds",
        "valid_mask",
        "handoff_role",
        "field_raw",
        "field_supervision_mask",
        "field_scale",
        "field_positions",
        "field_velocity",
        "field_sample_mask",
    }
    missing = sorted(required - batch.keys())
    if missing:
        raise ValueError(f"VLDS batch is missing fields: {missing}")
    size = batch["state_history"].shape[0]
    expected = {
        "state_history": (size, config.history_length, config.state_dim),
        "execution_context": (size, config.context_dim),
        "language": (size, config.language_dim),
        "goal_position": (size, config.horizon, 3),
        "goal_rotation_6d": (size, config.horizon, 6),
        "gripper_command": (size, config.horizon, 1),
        "primitive_kind": (size, config.horizon),
        "hold_log_seconds": (size, config.horizon, 1),
        "valid_mask": (size, config.horizon),
        "handoff_role": (size, config.horizon),
        "field_raw": (size, config.field_components, config.field_factor_dim),
        "field_supervision_mask": (size,),
        "field_scale": (size, 1),
        "field_positions": (size, batch["field_positions"].shape[1], 3),
        "field_velocity": (size, batch["field_positions"].shape[1], 3),
        "field_sample_mask": (size, batch["field_positions"].shape[1]),
    }
    for key, shape in expected.items():
        if tuple(batch[key].shape) != shape:
            raise ValueError(f"{key} has shape {tuple(batch[key].shape)}, expected {shape}")
    for key, value in batch.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{key} contains nonfinite values")
    if batch["handoff_role"].min() < 0 or batch["handoff_role"].max() >= config.handoff_role_dim:
        raise ValueError("handoff_role contains an unknown class")


class SyntheticVLDSDataset(Dataset[dict[str, torch.Tensor]]):
    """Generate causal structured queries with analytically valid stable fields.

    This is intentionally a smoke dataset, not a replacement for the audited
    LIBERO training set.  Its purpose is to verify tensor shapes, masking,
    differentiable field evaluation, and the ability of the new head to
    overfit a small fixed set.
    """

    def __init__(self, size: int = 128, config: VLDSConfig | None = None, seed: int = 20261123) -> None:
        super().__init__()
        config = VLDSConfig() if config is None else config
        self.config = config
        self.schema = VLDSSchema(horizon=config.horizon, history_length=config.history_length)
        generator = torch.Generator().manual_seed(seed)
        feature_dim = config.state_dim + config.context_dim + config.language_dim
        state_history = torch.randn(size, config.history_length, config.state_dim, generator=generator)
        execution_context = torch.randn(size, config.context_dim, generator=generator)
        language = torch.randn(size, config.language_dim, generator=generator)
        latent = torch.cat((state_history.mean(dim=1), execution_context, language), dim=-1)

        def teacher_matrix(output_dim: int, scale: float = 0.25) -> torch.Tensor:
            return torch.randn(feature_dim, output_dim, generator=generator) * scale

        position = torch.tanh(latent @ teacher_matrix(config.horizon * 3)).reshape(size, config.horizon, 3) * 0.15
        rotation = torch.tanh(latent @ teacher_matrix(config.horizon * 6)).reshape(size, config.horizon, 6)
        command_score = latent @ teacher_matrix(config.horizon)
        gripper_command = (command_score > 0).to(torch.float32).unsqueeze(-1)
        primitive_score = latent @ teacher_matrix(config.horizon)
        primitive_kind = (primitive_score > 0.6).to(torch.long)
        hold_log_seconds = torch.log(torch.full((size, config.horizon, 1), 0.2 / 0.05))
        handoff_role = torch.remainder(
            torch.round(torch.abs(latent @ teacher_matrix(config.horizon))), config.handoff_role_dim
        ).to(torch.long)
        valid_score = latent @ teacher_matrix(config.horizon)
        valid_mask = torch.sigmoid(valid_score) > 0.25
        valid_mask[:, 0] = True

        field_raw = latent @ teacher_matrix(config.field_raw_dim, scale=0.08)
        field_raw = field_raw.reshape(size, config.field_components, config.field_factor_dim)
        field_raw[..., 0] += torch.tensor([1.5, 0.5, -0.5, -1.5])
        field_raw[..., 10:13] *= 0.2
        field_raw[..., 13:19] *= 0.25
        field_raw[..., 4:10] *= 0.25

        sample_count = 12
        field_positions = position[:, 0, None, :] + torch.randn(size, sample_count, 3, generator=generator) * 0.04
        field_scale = torch.nn.functional.softplus(latent @ teacher_matrix(1, scale=0.1)) + 0.5
        with torch.no_grad():
            field_velocity = evaluate_seds_field(
                field_raw,
                position[:, 0],
                field_positions,
                config,
                field_scale,
            )

        self.data = {
            "state_history": state_history,
            "execution_context": execution_context,
            "language": language,
            "goal_position": position,
            "goal_rotation_6d": rotation,
            "gripper_command": gripper_command,
            "primitive_kind": primitive_kind,
            "hold_log_seconds": hold_log_seconds,
            "valid_mask": valid_mask,
            "handoff_role": handoff_role,
            "field_raw": field_raw,
            "field_supervision_mask": torch.ones(size, dtype=torch.bool),
            "field_scale": field_scale,
            "field_positions": field_positions,
            "field_velocity": field_velocity,
            "field_sample_mask": torch.ones(size, sample_count, dtype=torch.bool),
        }

    def __len__(self) -> int:
        return len(self.data["state_history"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}
