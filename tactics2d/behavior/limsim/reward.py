# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reward and stage-collision model for LimSim-style MCTS."""

import math
from typing import Dict, Optional, Sequence

from tactics2d.geometry import spatial

from .action import LimSimAction
from .config import LimSimConfig
from .schema import AgentDecisionState, DecisionStep


class LimSimReward:
    """Score FlowState histories with the original LimSim reward terms."""

    def __init__(self, config: LimSimConfig):
        self.config = config

    def evaluate(
        self,
        initial_agents: Sequence[AgentDecisionState],
        trajectories: Dict[object, Sequence[AgentDecisionState]],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        max_decision_num: Optional[int] = None,
        obstacle_initial_states: Sequence[AgentDecisionState] = (),
        completed_decisions: Sequence[Sequence[DecisionStep]] = (),
    ) -> float:
        """Evaluate one joint rollout and return an official [0, 1] score.

        ``max_decision_num`` is supplied by MCTS from the actual FlowState
        depth. In that mode collision was already evaluated while constructing
        the current layer and is not scanned a second time.
        """

        if not initial_agents:
            return 0.0

        flowstate_depth_supplied = max_decision_num is not None
        steps_per_decision = max(1, int(round(self.config.decision_resolution / self.config.dt)))
        if flowstate_depth_supplied:
            # MCTS stores exactly one official FlowState endpoint per layer.
            stage_trajectories = {
                agent.agent_id: [agent] + list(trajectories[agent.agent_id])
                for agent in initial_agents
            }
        else:
            # Standalone callers retain the existing dt-sampled trajectory
            # contract and are reduced to decision-resolution endpoints here.
            stage_trajectories = {
                agent.agent_id: [agent]
                + list(trajectories[agent.agent_id][steps_per_decision - 1 :: steps_per_decision])
                for agent in initial_agents
            }
        if max_decision_num is None:
            max_decision_num = 1 + max(
                (
                    (len(trajectories[agent.agent_id]) + steps_per_decision - 1)
                    // steps_per_decision
                    for agent in initial_agents
                ),
                default=0,
            )
        max_decision_num = max(1, int(max_decision_num))

        if not flowstate_depth_supplied and self._has_collision_in_stages(
            stage_trajectories,
            obstacle_trajectories,
            max_decision_num,
            obstacle_initial_states,
            completed_decisions,
        ):
            return 0.0

        rewards = []
        for agent in initial_agents:
            trajectory = trajectories[agent.agent_id]
            if not trajectory and not flowstate_depth_supplied:
                rewards.append(0.0)
                continue

            reward = 0.0
            decision_states = stage_trajectories[agent.agent_id][:max_decision_num]
            last_state = decision_states[-1]
            for index, state in enumerate(decision_states):
                if abs(state.lateral_offset) < 0.5:
                    reward += 0.2 / max_decision_num
                if index > 0 and state.action in {LimSimAction.AC, LimSimAction.KS}:
                    reward += 0.2 / max_decision_num
                if index > 1 and state.action == decision_states[index - 1].action:
                    reward += 0.2 / max_decision_num
                if state.lane_id is not None and state.lane_id in state.available_lane_ids:
                    reward += 0.2 / max_decision_num

            if (
                last_state.lane_id is not None
                and last_state.lane_id in last_state.available_lane_ids
            ):
                reward += 0.8 if abs(last_state.lateral_offset) < 0.5 else 0.2

            rewards.append(max(0.0, min(1.0, reward)))

        return float(sum(rewards) / len(rewards))

    def has_collision_at_stage(
        self,
        decision_states: Sequence[AgentDecisionState],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        stage_index: int = 0,
        obstacle_initial_states: Sequence[AgentDecisionState] = (),
        completed_decisions: Sequence[Sequence[DecisionStep]] = (),
    ) -> bool:
        """Check one official FlowState layer, excluding external-external pairs."""

        decision_states = tuple(decision_states)
        for index, decision_state in enumerate(decision_states):
            for other_state in decision_states[:index]:
                if self._decision_vehicle_collides(decision_state, other_state):
                    return True

        external_states = []
        if stage_index == 0:
            external_states.extend(obstacle_initial_states)
        else:
            steps_per_decision = max(
                1, int(round(self.config.decision_resolution / self.config.dt))
            )
            obstacle_index = stage_index * steps_per_decision - 1
            external_states.extend(
                trajectory[obstacle_index]
                for trajectory in obstacle_trajectories
                if obstacle_index < len(trajectory)
            )
        external_states.extend(
            decisions[stage_index].expected_state
            for decisions in completed_decisions
            if stage_index < len(decisions)
        )

        for decision_state in decision_states:
            for external_state in external_states:
                if self._decision_vehicle_collides(decision_state, external_state):
                    return True
        return False

    def _has_collision_in_stages(
        self,
        stage_trajectories: Dict[object, Sequence[AgentDecisionState]],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
        max_decision_num: int,
        obstacle_initial_states: Sequence[AgentDecisionState],
        completed_decisions: Sequence[Sequence[DecisionStep]],
    ) -> bool:
        collision_stage_count = min(max_decision_num, self.config.max_flowstate_depth)
        for stage_index in range(collision_stage_count):
            decision_states = [
                trajectory[stage_index]
                for trajectory in stage_trajectories.values()
                if stage_index < len(trajectory)
            ]
            if self.has_collision_at_stage(
                decision_states,
                obstacle_trajectories,
                stage_index,
                obstacle_initial_states,
                completed_decisions,
            ):
                return True
        return False

    @staticmethod
    def _decision_vehicle_collides(
        decision_state: AgentDecisionState, other_state: AgentDecisionState
    ) -> bool:
        if decision_state.agent_id == other_state.agent_id:
            return False
        distance = math.hypot(decision_state.x - other_state.x, decision_state.y - other_state.y)
        distance_threshold = math.hypot(
            decision_state.length + other_state.length,
            decision_state.width + other_state.width,
        )
        if distance > distance_threshold:
            return False
        decision_shape = spatial.oriented_box(
            decision_state.x,
            decision_state.y,
            decision_state.heading,
            decision_state.length * 2.0,
            decision_state.width * 1.5,
        )
        other_shape = spatial.oriented_box(
            other_state.x, other_state.y, other_state.heading, other_state.length, other_state.width
        )
        return bool(decision_shape.intersects(other_shape))
