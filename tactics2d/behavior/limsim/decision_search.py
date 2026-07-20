# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Decision-search adapter for LimSim-style joint behavior decisions."""

import itertools
import random
from typing import Dict, List, Optional, Sequence, Tuple

from tactics2d.map.element import Map
from tactics2d.search import MCTS

from .action import LimSimAction
from .config import LimSimConfig
from .lane_follower import LaneFollower, is_action_valid
from .reward import LimSimReward
from .schema import AgentDecisionState, JointDecisionState


class LimSimDecisionSearch:
    """Joint high-level behavior search using the existing Tactics2D MCTS."""

    def __init__(self, config: LimSimConfig):
        self.config = config
        self.follower = LaneFollower(config)
        self.reward = LimSimReward(config)
        self._expand_cache = {}

    def plan(
        self,
        agents: Sequence[AgentDecisionState],
        map_: Optional[Map],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
    ) -> Tuple[Dict[object, LimSimAction], Dict[object, List[AgentDecisionState]], object]:
        """Plan one joint action for a group of interacting agents.

        Implements the original LimSim-style chained MCTS: the search runs
        sequentially per decision step, with a decaying iteration budget
        ``budget = base_budget / (depth/2 + 1)`` so that the immediate next
        action receives the most computation.  Each step inherits the best
        child of the previous step as its root, matching the paper's
        receding-horizon-within-MCTS design.
        """

        self._expand_cache.clear()

        start_state = JointDecisionState(
            agents=tuple(agents), depth=0, trajectories=tuple(tuple() for _ in agents)
        )

        def terminal_fn(state: JointDecisionState) -> bool:
            return state.depth >= self.config.terminal_depth

        def expand_fn(state: JointDecisionState):
            return self._expand(state, map_)

        def reward_fn(state: JointDecisionState) -> float:
            trajectories = state.trajectory_dict()
            if not any(trajectories.values()):
                trajectories = {
                    agent.agent_id: self.follower.rollout(agent, agent.action, map_)
                    for agent in state.agents
                }
            return self.reward.evaluate(agents, trajectories, obstacle_trajectories)

        def simulate_fn(state: JointDecisionState) -> JointDecisionState:
            """Rollout to terminal with lightweight random-step generation.

            Uses :meth:`_simulate_step` instead of :meth:`_expand` to avoid
            computing the full Cartesian product of all agent actions on every
            simulation step.
            """
            current = state
            while not terminal_fn(current):
                next_state = self._simulate_step(current, map_)
                if next_state is None:
                    break
                current = next_state
            return current

        # --- chained MCTS: one search per decision step with decaying budget ---
        base_budget = max(1, self.config.mcts_iterations)
        current_state = start_state
        best_intermediate = start_state
        last_root = None

        for depth in range(self.config.terminal_depth):
            budget = max(2, int(base_budget / (depth / 2.0 + 1.0)))
            mcts = MCTS(
                terminal_fn=terminal_fn,
                expand_fn=expand_fn,
                reward_fn=reward_fn,
                simulate_fn=simulate_fn,
                exploration_weight=self.config.exploration_weight,
            )
            _, last_root = mcts.plan(start=current_state, max_try=budget)

            selected = self._best_state_from_root(last_root)
            if selected is None:
                break
            best_intermediate = selected

            # early termination: good-enough terminal state (original paper §3.3)
            if terminal_fn(selected) and reward_fn(selected) > 0.8:
                break

            # follow best child as the root for the next decision step
            current_state = selected

        # --- one-step fallback: did MCTS beat the greedy immediate choice? ---
        selected = best_intermediate
        root_fallback = self._best_one_step_state(start_state, map_, agents, obstacle_trajectories)
        if root_fallback is not None and reward_fn(root_fallback) > reward_fn(selected):
            selected = root_fallback

        actions = {}
        for agent, trajectory in zip(selected.agents, selected.trajectories):
            if trajectory:
                actions[agent.agent_id] = trajectory[0].action
            else:
                actions[agent.agent_id] = agent.action
        trajectories = selected.trajectory_dict()
        for index, agent in enumerate(selected.agents):
            if len(trajectories.get(agent.agent_id, [])) < self.config.horizon_steps:
                remaining = self.follower.rollout(
                    agent,
                    agent.action,
                    map_,
                    steps=self.config.horizon_steps - len(trajectories.get(agent.agent_id, [])),
                )
                trajectories[agent.agent_id] = trajectories.get(agent.agent_id, []) + remaining
        return actions, trajectories, last_root

    def _expand(self, state: JointDecisionState, map_: Optional[Map]) -> List[JointDecisionState]:
        cache_key = id(state)
        if cache_key in self._expand_cache:
            return self._expand_cache[cache_key]

        action_sets = []
        for agent in state.agents:
            actions = [
                action
                for action in self.config.candidate_actions
                if is_action_valid(agent, action, map_)
            ]
            actions = self._prune_actions(agent, actions)
            action_sets.append(actions or [LimSimAction.KS])

        steps_per_decision = max(1, int(round(self.config.decision_resolution / self.config.dt)))

        # --- pre-compute individual agent rollouts (key optimization) ---
        # Without this, each expand call does 5^N × N rollouts.
        # With caching, we do N × len(actions) rollouts and then assemble
        # joint states via cheap list operations.
        agent_rollouts = {}
        for idx, agent in enumerate(state.agents):
            per_action = {}
            for action in action_sets[idx]:
                segment = self.follower.rollout(agent, action, map_, steps=steps_per_decision)
                next_agent = segment[-1] if segment else agent.with_updates(action=action)
                per_action[action] = (segment, next_agent)
            agent_rollouts[idx] = per_action

        expanded = []
        for joint_actions in itertools.product(*action_sets):
            next_agents = []
            next_trajectories = []
            for index, (agent, action) in enumerate(zip(state.agents, joint_actions)):
                segment, next_agent = agent_rollouts[index][action]
                next_agents.append(next_agent)
                history = list(state.trajectories[index]) if state.trajectories else []
                history.extend(segment)
                next_trajectories.append(tuple(history[: self.config.horizon_steps]))
            expanded.append(
                JointDecisionState(
                    agents=tuple(next_agents),
                    depth=state.depth + 1,
                    trajectories=tuple(next_trajectories),
                )
            )

        self._expand_cache[cache_key] = expanded
        return expanded

    def _simulate_step(
        self, state: JointDecisionState, map_: Optional[Map]
    ) -> Optional[JointDecisionState]:
        """Generate a single random child without the full Cartesian product.

        This is the lightweight counterpart of :meth:`_expand`, used exclusively
        during simulation rollouts.  It picks one random action per agent,
        rolls out only those agent-action pairs, and returns one child state
        in O(N) instead of O(|A|^N).
        """

        steps_per_decision = max(1, int(round(self.config.decision_resolution / self.config.dt)))
        next_agents = []
        next_trajectories = []

        for index, agent in enumerate(state.agents):
            actions = [
                action
                for action in self.config.candidate_actions
                if is_action_valid(agent, action, map_)
            ]
            actions = self._prune_actions(agent, actions)
            action = random.choice(actions or [LimSimAction.KS])
            segment = self.follower.rollout(agent, action, map_, steps=steps_per_decision)
            next_agent = segment[-1] if segment else agent.with_updates(action=action)
            next_agents.append(next_agent)
            history = list(state.trajectories[index]) if state.trajectories else []
            history.extend(segment)
            next_trajectories.append(tuple(history[: self.config.horizon_steps]))

        return JointDecisionState(
            agents=tuple(next_agents), depth=state.depth + 1, trajectories=tuple(next_trajectories)
        )

    def _prune_actions(
        self, agent: AgentDecisionState, actions: List[LimSimAction]
    ) -> List[LimSimAction]:
        """Drop context-irrelevant actions to reduce the MCTS branching factor.

        Heuristics:
        - Drop AC at near-max speed (acceleration has no effect).
        - Drop DC at near-min speed (already stopped or crawling).
        - Always keep at least KS as a safe fallback.
        """
        if len(actions) <= 2:
            return actions  # already minimal

        keep = set(actions)
        speed_ratio = agent.speed / max(self.config.max_speed, 1.0)

        if speed_ratio >= 0.95:
            keep.discard(LimSimAction.AC)
        if speed_ratio <= 0.02:
            keep.discard(LimSimAction.DC)

        result = [a for a in actions if a in keep]
        # ensure KS is always available as the safe option
        if LimSimAction.KS not in result:
            result.append(LimSimAction.KS)
        return result

    def _best_state_from_root(self, root) -> Optional[JointDecisionState]:
        if root is None or not root.children:
            return None
        node = root
        while node.children:
            visited = [child for child in node.children if child.visits > 0]
            candidates = visited or node.children
            node = max(candidates, key=lambda child: child.total_reward / max(child.visits, 1))
        return node.state

    def _best_one_step_state(
        self,
        state: JointDecisionState,
        map_: Optional[Map],
        initial_agents: Sequence[AgentDecisionState],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
    ) -> Optional[JointDecisionState]:
        candidates = self._expand(state, map_)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: self.reward.evaluate(
                initial_agents, candidate.trajectory_dict(), obstacle_trajectories
            ),
        )
