"""Configuration for the first directly trainable VLDS model."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class VLDSConfig:
    """Small, explicit configuration shared by the model and smoke dataset.

    The initial implementation intentionally uses vector inputs for language and
    images.  A production encoder can replace those vectors without changing the
    structured prediction heads or the field decoder.
    """

    horizon: int = 3
    field_components: int = 4
    field_factor_dim: int = 19
    history_length: int = 4
    state_dim: int = 12
    context_dim: int = 12
    language_dim: int = 32
    hidden_dim: int = 128
    handoff_role_dim: int = 4
    length_scale: float = 0.1
    covariance_floor: float = 1e-6
    stability_margin: float = 0.01
    max_speed: float = 2.0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.field_components != 4 or self.field_factor_dim != 19:
            raise ValueError("design-v1 uses exactly four 19-dimensional field factors")
        if self.history_length < 1 or self.state_dim < 1 or self.context_dim < 1 or self.language_dim < 1:
            raise ValueError("input dimensions must be positive")
        if self.hidden_dim < 8 or self.handoff_role_dim < 2:
            raise ValueError("hidden_dim or handoff_role_dim is too small")
        if self.length_scale <= 0 or self.covariance_floor <= 0 or self.stability_margin <= 0 or self.max_speed <= 0:
            raise ValueError("field scales and limits must be positive")

    @property
    def field_raw_dim(self) -> int:
        return self.field_components * self.field_factor_dim

    @property
    def plan_dim(self) -> int:
        return 3 + 6 + 1 + 2 + 1 + 1 + self.handoff_role_dim
