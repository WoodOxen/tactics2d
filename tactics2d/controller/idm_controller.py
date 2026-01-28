# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""IDM controller implementation."""


from typing import Optional, Tuple

import numpy as np

from tactics2d.participant.trajectory.state import State

from .controller_base import ControllerBase


class IDMController(ControllerBase):
    """Intelligent Driver Model (IDM) controller for car-following behavior.

    The IDM is a car-following model that computes acceleration based on:
    - Current speed and desired speed
    - Distance to leading vehicle (if present)
    - Relative speed to leading vehicle (if present)

    Attributes:
        desired_speed (float): Desired speed in free traffic (m/s).
        time_headway (float): Desired time headway to leading vehicle (s).
        min_spacing (float): Minimum desired net distance to leading vehicle (m).
        max_acceleration (float): Maximum acceleration capability (m/s²).
        comfortable_deceleration (float): Comfortable deceleration (m/s²).
        delta (float): Acceleration exponent (usually 4.0).
    """

    def __init__(
        self,
        desired_speed: float = 10.0,
        time_headway: float = 1.5,
        min_spacing: float = 2.0,
        max_acceleration: float = 1.0,
        comfortable_deceleration: float = 3.0,
        delta: float = 4.0,
    ):
        """Initialize IDM controller with parameters.

        Args:
            desired_speed: Desired speed in free traffic (m/s). Default 10.0.
            time_headway: Desired time headway to leading vehicle (s). Default 1.5.
            min_spacing: Minimum desired net distance to leading vehicle (m). Default 2.0.
            max_acceleration: Maximum acceleration capability (m/s²). Default 1.0.
            comfortable_deceleration: Comfortable deceleration (m/s²). Default 3.0.
            delta: Acceleration exponent (usually 4.0). Default 4.0.
        """
        self.desired_speed = desired_speed
        self.time_headway = time_headway
        self.min_spacing = min_spacing
        self.max_acceleration = max_acceleration
        self.comfortable_deceleration = comfortable_deceleration
        self.delta = delta

    def step(self, ego_state: State, leading_state: State = None, **kwargs) -> Tuple[float, float]:
        """Compute control commands using IDM.

        Args:
            ego_state: Current state of the ego vehicle.
            leading_state: State of the leading vehicle (optional). If None,
                free-flow acceleration is used.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tuple[float, float]: (steering, acceleration) commands.
                steering is always 0.0 (pure longitudinal control).
                acceleration is computed by IDM (m/s²).
        """
        if leading_state is None:
            # Free-flow acceleration: a_max * [1 - (v/v_des)^δ]
            speed = ego_state.speed if ego_state.speed is not None else 0.0
            if self.desired_speed > 0:
                acceleration = self.max_acceleration * (
                    1.0 - (speed / self.desired_speed) ** self.delta
                )
            else:
                # If desired speed is zero, accelerate to stop
                acceleration = -self.comfortable_deceleration if speed > 0 else 0.0
        else:
            # Car-following acceleration
            acceleration = self._idm_acceleration(ego_state, leading_state)

        # Clip acceleration to physical limits
        acceleration = np.clip(acceleration, -self.comfortable_deceleration, self.max_acceleration)

        # IDM is purely longitudinal, no steering control
        return 0.0, acceleration

    def _idm_acceleration(self, ego_state: State, leading_state: State) -> float:
        """Compute IDM acceleration given ego and leading vehicle states.

        Args:
            ego_state: State of the ego vehicle.
            leading_state: State of the leading vehicle.

        Returns:
            Acceleration command (m/s²).
        """
        # Current speeds
        v = ego_state.speed if ego_state.speed is not None else 0.0
        v_lead = leading_state.speed if leading_state.speed is not None else 0.0

        # Distance between vehicles
        dx = leading_state.x - ego_state.x
        dy = leading_state.y - ego_state.y
        distance = np.hypot(dx, dy)

        # Relative speed (positive if leading vehicle is faster)
        dv = v_lead - v

        # Desired dynamic distance: s* = s0 + v*T + (v*dv)/(2*sqrt(a_max*b))
        # Ensure s_star is at least min_spacing
        s_star = (
            self.min_spacing
            + v * self.time_headway
            + (v * dv) / (2 * np.sqrt(self.max_acceleration * self.comfortable_deceleration))
        )
        s_star = max(s_star, self.min_spacing)

        # IDM acceleration formula:
        # a = a_max * [1 - (v/v_des)^δ - (s*/s)^2]
        if distance > 0:
            # Handle zero desired_speed case
            if self.desired_speed > 0:
                speed_ratio_term = (v / self.desired_speed) ** self.delta
            else:
                # If desired_speed is zero, treat as wanting to stop
                speed_ratio_term = 1.0 if v > 0 else 0.0

            acceleration = self.max_acceleration * (
                1.0 - speed_ratio_term - (s_star / distance) ** 2
            )
        else:
            # Avoid division by zero - apply maximum deceleration
            acceleration = -self.comfortable_deceleration

        return acceleration

    def configure(self, **kwargs) -> None:
        """Configure IDM parameters.

        Args:
            **kwargs: Parameter names and values to update.
                Supported parameters: desired_speed, time_headway, min_spacing,
                max_acceleration, comfortable_deceleration, delta.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"IDMController has no parameter '{key}'")
