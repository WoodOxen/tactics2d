# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reward model for LimSim-style MCTS.

Per-step bonuses (lane-center, speed, route-lane, continuity) are summed over
the trajectory and normalized to [0, 1]; terminal bonus up to 0.8, collision 0.0.
"""

from dataclasses import dataclass
from typing import Dict, Sequence

from .action import LimSimAction
from .config import LimSimConfig
from .decision_state import AgentDecisionState
from .interaction import cross_trajectory_safety_summary, trajectory_safety_summary


@dataclass(frozen=True)
class PreparedObstacleSafety:
    """Fixed obstacle trajectories and their prefix safety statistics."""

    trajectories: tuple
    prefix_summaries: Dict[int, tuple]

    def summary(self, steps: int):
        """Return obstacle-only safety statistics for a rollout prefix."""

        return self.prefix_summaries.get(steps, (None, float("inf"), 0.0))


class LimSimReward:
    """Score joint action rollouts with safety, progress, and comfort terms."""

    def __init__(self, config: LimSimConfig):
        self.config = config

    @staticmethod
    def prepare_obstacles(
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
    ) -> PreparedObstacleSafety:
        """Precompute safety terms that do not change during one MCTS search."""

        trajectories = tuple(
            tuple(trajectory) for trajectory in obstacle_trajectories if trajectory
        )
        if not trajectories:
            return PreparedObstacleSafety(trajectories=(), prefix_summaries={})
        max_steps = min(len(trajectory) for trajectory in trajectories)
        summaries = {
            steps: trajectory_safety_summary([trajectory[:steps] for trajectory in trajectories])
            for steps in range(1, max_steps + 1)
        }
        return PreparedObstacleSafety(trajectories=trajectories, prefix_summaries=summaries)

    def evaluate(
        self,
        initial_agents: Sequence[AgentDecisionState],
        trajectories: Dict[object, Sequence[AgentDecisionState]],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        obstacle_safety: PreparedObstacleSafety = None,
    ) -> float:
        """Evaluate a joint rollout, returning a value in **[0, 1]**.

        The range keeps the MCTS early-termination threshold (reward > 0.8)
        meaningful.
        """

        ordered = [trajectories[agent.agent_id] for agent in initial_agents]
        if obstacle_safety is None:
            obstacle_ordered = [
                list(trajectory) for trajectory in obstacle_trajectories if trajectory
            ]
        else:
            obstacle_ordered = list(obstacle_safety.trajectories)
        collision_ordered = ordered + obstacle_ordered

        # --- collision → 0.0 (matches original paper's terminal-on-collision) ---
        if obstacle_safety is None or not obstacle_ordered:
            collision, min_distance, closing_factor = trajectory_safety_summary(collision_ordered)
        else:
            steps = min(len(trajectory) for trajectory in collision_ordered if trajectory)
            candidate_safety = trajectory_safety_summary(
                [trajectory[:steps] for trajectory in ordered]
            )
            obstacle_only_safety = obstacle_safety.summary(steps)
            cross_safety = cross_trajectory_safety_summary(ordered, obstacle_ordered, steps=steps)
            collision = candidate_safety[0] or obstacle_only_safety[0] or cross_safety[0]
            min_distance = min(candidate_safety[1], obstacle_only_safety[1], cross_safety[1])
            closing_factor = candidate_safety[2] + obstacle_only_safety[2] + cross_safety[2]
        if collision is not None:
            return 0.0

        rewards = []
        for agent in initial_agents:
            trajectory = trajectories[agent.agent_id]
            if not trajectory:
                rewards.append(0.0)
                continue

            reward = 0.0
            n_steps = len(trajectory)
            last_state = trajectory[-1]

            # --- per-step bonuses (0.2 / n_steps per term, mapped to
            #     trajectory-step granularity) ---
            for state in trajectory:
                if abs(state.lateral_offset) < 0.5:
                    reward += 0.2 / n_steps
                if state.action in {LimSimAction.AC, LimSimAction.KS}:
                    reward += 0.2 / n_steps
                if state.lane_id is not None and state.lane_id in state.route_lane_ids:
                    reward += 0.2 / n_steps

            # action continuity: same action in consecutive decisions
            for i in range(1, len(trajectory)):
                if trajectory[i].action == trajectory[i - 1].action:
                    reward += 0.2 / n_steps

            # --- terminal-state bonus (original: up to 0.8) ---
            if last_state.lane_id is not None and last_state.lane_id in last_state.route_lane_ids:
                if abs(last_state.lateral_offset) < 0.5:
                    reward += 0.8
                else:
                    reward += 0.2

            # --- auxiliary signals (Tactics2D extensions, kept at small weight) ---
            progress = max(last_state.route_progress - agent.route_progress, 0.0)
            reward += 0.05 * min(abs(progress) / 20.0, 1.0)

            if last_state.action.is_lane_change:
                reward -= 0.15

            rewards.append(max(0.0, min(1.0, reward)))

        # --- proximity / closing-speed adjustments (shared across agents) ---
        avg_reward = sum(rewards) / len(rewards)

        if min_distance < self.config.conflict_distance:
            avg_reward -= 0.05 * (self.config.conflict_distance - min_distance)

        # Normalized per rollout step and capped: only a collision may drive the
        # reward to 0.0, not a sustained approach.
        lengths = [len(trajectory) for trajectory in collision_ordered if trajectory]
        steps = min(lengths) if lengths else 1
        avg_reward -= min(self.config.reward_closing_penalty_cap, 0.02 * closing_factor / steps)

        return float(max(0.0, min(1.0, avg_reward)))
