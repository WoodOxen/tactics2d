# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reward model for LimSim-style MCTS.

Per-step bonuses (lane-centre, speed, route-lane, continuity) are summed over
the trajectory and normalised to [0, 1]; terminal bonus up to 0.8, collision 0.0.
"""

from typing import Dict, Sequence

from .action import LimSimAction
from .config import LimSimConfig
from .decision_state import AgentDecisionState
from .interaction import trajectory_safety_summary


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
        """Evaluate a joint rollout, returning a value in **[0, 1]**.

        The range keeps the MCTS early-termination threshold (reward > 0.8)
        meaningful.
        """

        ordered = [trajectories[agent.agent_id] for agent in initial_agents]
        obstacle_ordered = [list(trajectory) for trajectory in obstacle_trajectories if trajectory]
        collision_ordered = ordered + obstacle_ordered

        # --- collision → 0.0 (matches original paper's terminal-on-collision) ---
        collision, min_distance, closing_factor = trajectory_safety_summary(collision_ordered)
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

        # Normalised per rollout step and capped: only a collision may drive the
        # reward to 0.0, not a sustained approach.
        lengths = [len(trajectory) for trajectory in collision_ordered if trajectory]
        steps = min(lengths) if lengths else 1
        avg_reward -= min(self.config.reward_closing_penalty_cap, 0.02 * closing_factor / steps)

        return float(max(0.0, min(1.0, avg_reward)))
