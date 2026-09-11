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

    ``horizon_steps`` is the InterSim-native name for the number of future
    states returned by ``InterSimBehaviorModel.predict``. Use
    ``planning_steps`` when code needs the same semantic field across behavior
    models. All learning-based options (relation/marginal predictors) are
    disabled by default; enabling one raises ``NotImplementedError``.
    """

    dt: float = 0.1
    horizon_steps: int = 80
    interaction_distance: float = 40.0
    lane_match_radius: float = 5.0
    lane_heading_tolerance_deg: float = 20.0
    stop_margin: float = 2.0
    # Upstream env cruises toward 70/frame_rate (70 m/s at 10 Hz) with a
    # straight-road acceleration of ~2.7 m/s^2; 25 m/s here was an artificial
    # ceiling absent upstream, so it is raised to ~no effective cap.
    cruise_speed: float = 70.0
    cruise_accel: float = 2.7
    max_target_speed: float = 70.0
    yield_speed_ratio: float = 0.8
    # Per-vehicle braking limit override. Upstream's emergency braking is
    # A_SLOWDOWN_DESIRE/frame_rate per frame ~= 15 m/s^2, so it defaults to that
    # rather than the tactics2d Vehicle default (10). ``None`` would fall back to
    # the participant's own ``max_decel``.
    vehicle_decel_limit: Optional[float] = 15.0
    # Red-light stopping is opt-in: the upstream InterSim closed loop (§11) does
    # not brake for traffic signals, so it stays off for protocol comparability.
    respect_traffic_light: bool = False
    # Map-fidelity toggle: add geometry-implied successors to broken lane links
    # so lane-following chains keep moving past proto connectivity gaps.
    augment_lane_graph: bool = False
    max_resolution_iters: int = 6
    default_vehicle_length: float = 4.8
    default_vehicle_width: float = 1.9
    use_relation_model: bool = False
    relation_model_path: Optional[str] = None
    use_marginal_model: bool = False
    marginal_model_path: Optional[str] = None
    # Closed-loop cadence (WOMD 10 Hz defaults mirrored from upstream InterSim).
    planning_warmup_steps: int = 11
    planning_interval: int = 10
    scenario_steps: int = 91
    # Relation resolution: "directed" brakes the later reactor only; "yield_all"
    # brakes every agent involved in an imminent conflict (all-yield reference).
    relation_mode: str = "directed"

    @property
    def step_ms(self) -> int:
        """Return the configured sampling interval in milliseconds."""

        return int(round(self.dt * 1000))

    @property
    def planning_steps(self) -> int:
        """Return the number of future planning states."""

        return self.horizon_steps
