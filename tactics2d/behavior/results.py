# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared results returned by behavior-model rollouts."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tactics2d.participant.trajectory import Trajectory


@dataclass
class ClosedLoopMetrics:
    """Metrics shared by joint closed-loop behavior benchmarks."""

    front_collisions: int = 0
    side_collisions: int = 0
    rear_collisions: int = 0
    progress: float = 0.0
    total_agents_controlled: int = 0
    collided: bool = False


@dataclass
class BehaviorRolloutResult:
    """Common result returned by every behavior-model rollout."""

    ego_id: object = None
    frames: List[int] = field(default_factory=list)
    controlled_ids: List[object] = field(default_factory=list)
    trajectories: Dict[object, Trajectory] = field(default_factory=dict)
    cycles: int = 0
    metrics: Optional[ClosedLoopMetrics] = None


@dataclass
class TrajectoryRolloutResult(BehaviorRolloutResult):
    """Output of a receding-horizon replay on the recorded frame grid."""

    trajectory: Optional[Trajectory] = None
    plans: Dict[int, List[Tuple[int, float, float]]] = field(default_factory=dict)
