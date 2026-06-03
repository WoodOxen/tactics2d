# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reward model for LimSim-style MCTS."""

from typing import Dict, Sequence

import numpy as np

from .action import LimSimAction
from .config import LimSimConfig
from .interaction import first_collision_info, minimum_pair_distance
from .schema import AgentDecisionState


class LimSimReward:
    """Score joint action rollouts with safety, progress, and comfort terms."""

    def __init__(self, config: LimSimConfig):
        self.config = config

    def evaluate(
        self,
        initial_agents: Sequence[AgentDecisionState],
        trajectories: Dict[object, Sequence[AgentDecisionState]],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
    ) -> float:
        """Evaluate a joint rollout."""

        ordered = [trajectories[agent.agent_id] for agent in initial_agents]
        obstacle_ordered = [list(trajectory) for trajectory in obstacle_trajectories if trajectory]
        collision_ordered = ordered + obstacle_ordered
        reward = 0.0
        collision_info = first_collision_info(collision_ordered)
        if collision_info is not None:
            reward -= self.config.collision_penalty
            step, source, target = collision_info
            reward -= self.config.collision_time_penalty * (len(collision_ordered[0]) - step)
            reward -= self.config.collision_speed_penalty * (source.speed + target.speed)

        min_distance = minimum_pair_distance(collision_ordered)
        if min_distance < self.config.conflict_distance:
            reward -= self.config.proximity_penalty * (self.config.conflict_distance - min_distance)

        reward -= self._closing_speed_penalty(collision_ordered)

        for agent in initial_agents:
            trajectory = trajectories[agent.agent_id]
            if not trajectory:
                continue
            last_state = trajectory[-1]
            progress = max(last_state.route_progress - agent.route_progress, 0.0)
            reward += self.config.progress_weight * progress
            reward += self.config.speed_weight * np.mean([state.speed for state in trajectory])
            reward -= self.config.comfort_weight * abs(last_state.action.acceleration)
            if abs(last_state.lateral_offset) < 0.5:
                reward += 0.2
            if last_state.action in {LimSimAction.AC, LimSimAction.KS}:
                reward += 0.2
            if last_state.lane_id in last_state.route_lane_ids:
                reward += 0.2
            if last_state.action.is_lane_change:
                reward -= self.config.lane_change_penalty
        return float(reward)

    def _closing_speed_penalty(self, trajectories: Sequence[Sequence[AgentDecisionState]]) -> float:
        penalty = 0.0
        if len(trajectories) < 2:
            return penalty
        steps = min(len(trajectory) for trajectory in trajectories)
        for step in range(steps):
            for i, source in enumerate(trajectories):
                for target in trajectories[i + 1 :]:
                    source_state = source[step]
                    target_state = target[step]
                    if source_state.lane_id != target_state.lane_id:
                        continue
                    rear, front = source_state, target_state
                    if rear.route_progress > front.route_progress:
                        rear, front = front, rear
                    gap = front.route_progress - rear.route_progress
                    closing_speed = rear.speed - front.speed
                    safe_gap = rear.length + max(rear.speed, 0.0)
                    if closing_speed > 0.0 and gap < safe_gap:
                        penalty += (
                            self.config.closing_speed_penalty * closing_speed * (safe_gap - gap)
                        )
        return penalty
