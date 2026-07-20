# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Configuration for the LimSim-style interactive behavior model."""

from dataclasses import dataclass, field
from typing import Tuple

from .action import LimSimAction


@dataclass(frozen=True)
class LimSimConfig:
    """Parameters for interaction grouping, rollout, and MCTS scoring.

    ``horizon_steps`` is the LimSim planner-native name for the number of
    future states returned by ``LimSimBehaviorModel.predict``. Use
    ``planning_steps`` when code needs the same semantic field across behavior
    models.
    """

    horizon_steps: int = 50
    dt: float = 0.1
    mcts_iterations: int = 200
    exploration_weight: float = 0.707
    interaction_distance: float = 30.0
    conflict_distance: float = 3.0
    lane_match_radius: float = 4.0
    lane_heading_match_weight: float = 2.0
    max_lateral_offset_for_lane_rollout: float = 2.5
    max_group_size: int = 3
    max_routes_per_agent: int = 3
    min_speed: float = 0.0
    max_speed: float = 35.0
    acceleration: float = 0.7
    deceleration: float = -0.7
    decision_resolution: float = 1.5
    default_vehicle_length: float = 4.8
    default_vehicle_width: float = 1.9
    default_lane_width: float = 3.6
    lateral_speed: float = 1.17
    frenet_target_speed_offsets: Tuple[float, ...] = (-1.0, 0.0, 1.0)
    frenet_lateral_offsets: Tuple[float, ...] = (-0.3, 0.0, 0.3)
    frenet_collision_penalty: float = 5000.0
    frenet_obstacle_buffer: float = 2.0
    frenet_proximity_weight: float = 30.0
    frenet_speed_weight: float = 1.0
    frenet_lateral_weight: float = 2.0
    frenet_accel_weight: float = 0.3
    frenet_jerk_weight: float = 0.1
    frenet_stop_line_penalty: float = 3000.0
    frenet_stop_distance_buffer: float = 1.0
    frenet_stop_speed_threshold: float = 0.5
    frenet_stop_deceleration: float = 4.0
    frenet_junction_conflict_penalty: float = 800.0
    frenet_junction_conflict_time_window: float = 1.5
    frenet_junction_conflict_distance: float = 6.0
    traffic_light_stop_states: Tuple[str, ...] = (
        "LANE_STATE_STOP",
        "LANE_STATE_CAUTION",
        "STOP",
        "CAUTION",
        "RED",
        "YELLOW",
    )
    same_lane_time_headway: float = 3.5
    junction_interaction_distance: float = 20.0
    terminal_depth: int = 4
    use_frenet_refinement: bool = False
    candidate_actions: Tuple[LimSimAction, ...] = field(
        default_factory=lambda: (
            LimSimAction.KS,
            LimSimAction.AC,
            LimSimAction.DC,
            LimSimAction.LCL,
            LimSimAction.LCR,
        )
    )

    @property
    def step_ms(self) -> int:
        """Return the configured sampling interval in milliseconds."""

        return int(round(self.dt * 1000))

    @property
    def planning_steps(self) -> int:
        """Return the number of future planning states."""

        return self.horizon_steps
