"""Explicit pose references for the installed fixed-impedance OSC controller."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.fields import primitive


class OSCFieldAdapter:
    """Refresh orientation even when zero rotation would retain an old OSC goal."""

    def __init__(self, controller, config=primitive.DEFAULT_CONTROL):
        if controller.impedance_mode != "fixed" or not controller.use_delta or not controller.use_ori:
            raise ValueError("Field execution requires fixed-impedance delta OSC_POSE")
        self.controller, self.config = controller, config
        self.original_set_goal = controller.set_goal
        self.reference_rotation = None
        self.reference_position = None
        self.input_min = np.broadcast_to(np.asarray(controller.input_min, dtype=float), (6,)).copy()
        self.input_max = np.broadcast_to(np.asarray(controller.input_max, dtype=float), (6,)).copy()
        self.output_min = np.broadcast_to(np.asarray(controller.output_min, dtype=float), (6,)).copy()
        self.output_max = np.broadcast_to(np.asarray(controller.output_max, dtype=float), (6,)).copy()
        if (
            not np.isfinite(np.r_[self.input_min, self.input_max, self.output_min, self.output_max]).all()
            or np.any(self.input_max <= self.input_min)
            or np.any(self.output_max <= self.output_min)
        ):
            raise ValueError("Invalid live controller action scaling")
        controller.set_goal = self.set_goal

    def set_goal(self, action, set_pos=None, set_ori=None):
        if self.reference_rotation is None:
            raise ValueError("No actual-state pose reference prepared for this control step")
        self.original_set_goal(action, set_pos=self.reference_position, set_ori=self.reference_rotation)

    def action(self, field, position, rotation):
        velocity, omega, alpha = primitive.feedback(field, position, rotation, self.config)
        delta = self.config.reference_seconds * np.r_[velocity, omega]
        action = self.input_min + (delta - self.output_min) * (self.input_max - self.input_min) / (
            self.output_max - self.output_min
        )
        if np.any(action < self.input_min - 1e-10) or np.any(action > self.input_max + 1e-10):
            raise ValueError("Reference exceeds live controller action range")
        self.reference_position = np.asarray(position) + delta[:3]
        self.reference_rotation = Rotation.from_rotvec(delta[3:]).as_matrix() @ rotation
        return np.r_[action, field.command], {
            "expected_position": self.reference_position.copy(),
            "expected_rotation": self.reference_rotation.copy(),
            "alpha": alpha,
        }

    def verify_goal(self, expected):
        position_error = float(np.linalg.norm(self.controller.goal_pos - expected["expected_position"]))
        rotation_error = float(np.max(np.abs(self.controller.goal_ori - expected["expected_rotation"])))
        if position_error > 1e-7 or rotation_error > 1e-6:
            raise ValueError(
                f"Installed OSC goal mismatch: position={position_error:.6g} m; rotation={rotation_error:.6g}"
            )
        return position_error, rotation_error
