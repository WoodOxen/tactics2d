# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Configuration for the InterSim-style interactive behavior model."""

from dataclasses import dataclass
from typing import Optional

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.


@dataclass(frozen=True)
class InterSimConfig:
    """Parameters for directed relation detection and conflict resolution.

    ``horizon_steps`` is the InterSim-native alias of ``planning_steps``. All
    learning-based predictor options must stay disabled.
    """

    dt: float = 0.1
    horizon_steps: int = 80
    interaction_distance: float = 40.0
    lane_match_radius: float = 5.0
    lane_heading_tolerance_deg: float = 20.0
    stop_margin: float = 2.0
    cruise_speed: float = 70.0
    cruise_accel: float = 2.7
    max_target_speed: float = 70.0
    yield_speed_ratio: float = 0.8
    # Braking limit override; None falls back to the participant's max_decel.
    vehicle_decel_limit: Optional[float] = 15.0
    # Opt-in red-light braking.
    respect_traffic_light: bool = False
    # Add geometry-implied successors to broken lane links.
    augment_lane_graph: bool = False
    max_resolution_iters: int = 6
    default_vehicle_length: float = 4.8
    default_vehicle_width: float = 1.9
    use_relation_model: bool = False
    relation_model_path: Optional[str] = None
    use_marginal_model: bool = False
    marginal_model_path: Optional[str] = None
    # Closed-loop cadence in steps (WOMD 10 Hz).
    planning_warmup_steps: int = 11
    planning_interval: int = 10
    scenario_steps: int = 91
    # "directed" brakes the later reactor only; "yield_all" brakes both sides.
    relation_mode: str = "directed"

    @property
    def step_ms(self) -> int:
        """Return the configured sampling interval in milliseconds."""

        return int(round(self.dt * 1000))

    @property
    def planning_steps(self) -> int:
        """Return the number of future planning states."""

        return self.horizon_steps
