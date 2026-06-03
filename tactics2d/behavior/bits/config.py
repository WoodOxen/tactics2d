# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Configuration for BITS-style behavior imitation."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class BitsConfig:
    """Parameters shared by BITS data preparation and models."""

    history_steps: int = 10
    future_steps: int = 20
    dt: float = 0.1
    max_agents: int = 20
    max_agents_distance: float = 30.0
    raster_size: int = 224
    pixel_size: float = 0.5
    default_vehicle_length: float = 4.8
    default_vehicle_width: float = 1.9
    include_non_vehicle_neighbors: bool = False
    likelihood_weight: float = 1.0
    progress_weight: float = 1.0
    lane_weight: float = 100.0
    collision_weight: float = 100.0
    drivable_distance_clip: float = 10.0
    prediction_loss_weight: float = 1.0
    goal_loss_weight: float = 0.0
    pixel_bce_loss_weight: float = 0.0
    pixel_ce_loss_weight: float = 1.0
    pixel_res_loss_weight: float = 1.0
    pixel_yaw_loss_weight: float = 1.0
    dynamics_max_steer: float = 0.5
    dynamics_max_yawvel: float = 8.0
    dynamics_acceleration_min: float = -6.0
    dynamics_acceleration_max: float = 4.0
    dynamics_speed_min: float = -10.0
    dynamics_speed_max: float = 30.0

    @property
    def step_ms(self) -> int:
        """Return the configured sampling interval in milliseconds."""

        return int(round(self.dt * 1000))

    @property
    def torch_loss_weights(self) -> Dict[str, float]:
        """Return TBSIM-style loss weights for the torch reproduction path."""

        return {
            "prediction_loss": self.prediction_loss_weight,
            "goal_loss": self.goal_loss_weight,
            "pixel_bce_loss": self.pixel_bce_loss_weight,
            "pixel_ce_loss": self.pixel_ce_loss_weight,
            "pixel_res_loss": self.pixel_res_loss_weight,
            "pixel_yaw_loss": self.pixel_yaw_loss_weight,
        }

    @property
    def cost_weights(self) -> Dict[str, float]:
        """Return TBSIM-style closed-loop planning cost weights."""

        return {
            "likelihood_weight": self.likelihood_weight,
            "progress_weight": self.progress_weight,
            "lane_weight": self.lane_weight,
            "collision_weight": self.collision_weight,
        }
