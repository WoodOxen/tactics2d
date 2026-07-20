# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reward model for LimSim-style MCTS.

The reward structure follows the original LimSim paper: per-step bonuses
(lane-centre, speed, route-lane, continuity) summed over the trajectory and
normalised to [0, 1], with a terminal-state bonus of up to 0.8.  A collision
returns 0.0.
"""

from typing import Dict, Sequence

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
        """Evaluate a joint rollout, returning a value in **[0, 1]**.

        The range matches the original LimSim paper so that the MCTS
        early-termination threshold (reward > 0.8) and the decaying-budget
        chain search behave as intended.
        """

        ordered = [trajectories[agent.agent_id] for agent in initial_agents]
        obstacle_ordered = [list(trajectory) for trajectory in obstacle_trajectories if trajectory]
        collision_ordered = ordered + obstacle_ordered

        # --- collision → 0.0 (matches original paper's terminal-on-collision) ---
        if first_collision_info(collision_ordered) is not None:
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

            # --- per-step bonuses (original LimSim §3.3, 0.2 / max_decision_num each) ---
            # mapped to trajectory-step granularity: 0.2 / n_steps per term
            for state in trajectory:
                if abs(state.lateral_offset) < 0.5:
                    reward += 0.2 / n_steps
                if state.action in {LimSimAction.AC, LimSimAction.KS}:
                    reward += 0.2 / n_steps
                if state.lane_id is not None and state.lane_id in state.route_lane_ids:
                    reward += 0.2 / n_steps

            # action continuity (original: same action in consecutive decisions)
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

        min_distance = minimum_pair_distance(collision_ordered)
        if min_distance < self.config.conflict_distance:
            avg_reward -= 0.05 * (self.config.conflict_distance - min_distance)

        closing_factor = self._closing_speed_factor(collision_ordered)
        avg_reward -= 0.02 * closing_factor

        return float(max(0.0, min(1.0, avg_reward)))

    def _closing_speed_factor(self, trajectories: Sequence[Sequence[AgentDecisionState]]) -> float:
        factor = 0.0
        if len(trajectories) < 2:
            return factor
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
                        factor += closing_speed * (safe_gap - gap)
        return factor
