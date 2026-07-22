# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unicycle dynamics for BITS-style trajectory decoding."""

from typing import Optional

import torch

from .config import BitsConfig


def integrate_unicycle_controls(
    controls: torch.Tensor, current_states: torch.Tensor, config: Optional[BitsConfig] = None
) -> tuple:
    """Integrate acceleration/yaw-rate controls with TBSIM-style unicycle dynamics.

    Args:
        controls: Control tensor ``(*, T, 2)`` with ``[acceleration, yaw_rate]``.
        current_states: Current state tensor ``(*, 4)`` with
            ``[x, y, speed, yaw]``.
        config: Dynamics configuration. Defaults to ``BitsConfig`` with
            ``future_steps`` inferred from ``controls``.

    Returns:
        A tuple ``(positions, yaws)`` where each has shape ``(*, T, 2)`` and
        ``(*, T, 1)`` respectively.
    """

    resolved_config = config or BitsConfig(future_steps=controls.shape[-2])
    states = ensure_unicycle_state(current_states, controls)
    positions = []
    yaws = []
    for step in range(controls.shape[-2]):
        states = unicycle_step(states, controls[..., step, :], resolved_config)
        positions.append(states[..., 0:2])
        yaws.append(states[..., 3:4])
    return torch.stack(positions, dim=-2), torch.stack(yaws, dim=-2)


def unicycle_step(states: torch.Tensor, controls: torch.Tensor, config: BitsConfig) -> torch.Tensor:
    # TBSIM Unicycle state is [x, y, speed, yaw], with controls
    # [acceleration, yaw_rate]. Clamp controls from current speed, then
    # integrate position with a half-step acceleration approximation.
    acceleration, yaw_rate = clip_unicycle_controls(states, controls, config)
    dt = float(config.dt)
    speed = states[..., 2:3]
    yaw = states[..., 3:4]
    next_speed_for_position = speed + acceleration * dt * 0.5
    dx = torch.cos(yaw) * next_speed_for_position * dt
    dy = torch.sin(yaw) * next_speed_for_position * dt
    next_speed = speed + acceleration * dt
    next_yaw = yaw + yaw_rate * dt
    return torch.cat([states[..., 0:1] + dx, states[..., 1:2] + dy, next_speed, next_yaw], dim=-1)


def clip_unicycle_controls(
    states: torch.Tensor, controls: torch.Tensor, config: BitsConfig
) -> tuple:
    speed = states[..., 2:3]
    speed_for_yaw = torch.clamp(torch.abs(speed), min=0.1)
    yaw_bound = torch.minimum(
        torch.as_tensor(config.dynamics_max_steer, dtype=speed.dtype, device=speed.device)
        * speed_for_yaw,
        torch.as_tensor(config.dynamics_max_yawvel, dtype=speed.dtype, device=speed.device)
        / speed_for_yaw,
    )
    yaw_bound = torch.clamp(yaw_bound, min=0.1)

    acceleration = controls[..., 0:1]
    yaw_rate = controls[..., 1:2]
    acceleration_lower = torch.clamp(
        torch.as_tensor(config.dynamics_speed_min, dtype=speed.dtype, device=speed.device) - speed,
        max=float(config.dynamics_acceleration_max),
    )
    acceleration_lower = torch.clamp(
        acceleration_lower, min=float(config.dynamics_acceleration_min)
    )
    acceleration_upper = torch.clamp(
        torch.as_tensor(config.dynamics_speed_max, dtype=speed.dtype, device=speed.device) - speed,
        min=float(config.dynamics_acceleration_min),
    )
    acceleration_upper = torch.clamp(
        acceleration_upper, max=float(config.dynamics_acceleration_max)
    )
    return (
        torch.clamp(acceleration, acceleration_lower, acceleration_upper),
        torch.clamp(yaw_rate, -yaw_bound, yaw_bound),
    )


def ensure_unicycle_state(current_states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    states = current_states.to(device=controls.device, dtype=controls.dtype)
    while states.ndim < controls.ndim - 1:
        states = states.unsqueeze(1)
    return states.expand(*controls.shape[:-2], states.shape[-1])
