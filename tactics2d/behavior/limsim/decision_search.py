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
from .schema import AgentDecisionState, DecisionStep, JointDecisionState


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
        """Plan one joint action while preserving the existing return contract."""

        actions, trajectories, _, root = self._plan_with_decisions(
            agents, map_, obstacle_trajectories=obstacle_trajectories, start_frame=0
        )
        return actions, trajectories, root

    def _plan_with_decisions(
        self,
        agents: Sequence[AgentDecisionState],
        map_: Optional[Map],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        start_frame: int = 0,
        obstacle_initial_states: Sequence[AgentDecisionState] = (),
        completed_decisions: Sequence[Sequence[DecisionStep]] = (),
    ) -> Tuple[
        Dict[object, LimSimAction],
        Dict[object, List[AgentDecisionState]],
        Dict[object, List[DecisionStep]],
        object,
    ]:
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
            agents=tuple(agents),
            depth=0,
            trajectories=tuple(tuple() for _ in agents),
            active_agent_ids=frozenset(agent.agent_id for agent in agents),
            has_collision=self.reward.has_collision_at_stage(
                agents,
                obstacle_trajectories,
                stage_index=0,
                obstacle_initial_states=obstacle_initial_states,
                completed_decisions=completed_decisions,
            ),
        )

        def terminal_fn(state: JointDecisionState) -> bool:
            return self._is_terminal(state)

        def expand_fn(state: JointDecisionState):
            return self._expand(
                state, map_, obstacle_trajectories, obstacle_initial_states, completed_decisions
            )

        def reward_fn(state: JointDecisionState) -> float:
            if state.has_collision:
                return 0.0
            trajectories = state.trajectory_dict()
            return self.reward.evaluate(
                agents,
                trajectories,
                max_decision_num=state.depth + 1,
            )

        def simulate_fn(state: JointDecisionState) -> JointDecisionState:
            """Rollout to terminal with lightweight random-step generation.

            Uses :meth:`_simulate_step` instead of :meth:`_expand` to avoid
            computing the full Cartesian product of all agent actions on every
            simulation step.
            """
            current = state
            while not terminal_fn(current):
                next_state = self._simulate_step(
                    current,
                    map_,
                    obstacle_trajectories,
                    obstacle_initial_states,
                    completed_decisions,
                )
                if next_state is None:
                    break
                current = next_state
            return current

        # --- chained MCTS: one search per decision step with decaying budget ---
        base_budget = max(1, self.config.mcts_iterations)
        mcts = MCTS(
            terminal_fn=terminal_fn,
            expand_fn=expand_fn,
            reward_fn=reward_fn,
            simulate_fn=simulate_fn,
            exploration_weight=self.config.exploration_weight,
        )
        search_root = MCTS.Node(start_state)
        current_node = search_root
        decision_step_ms = int(round(self.config.decision_resolution * 1000))

        for depth in range(self.config.terminal_depth):
            budget = max(2, int(base_budget / (depth / 2.0 + 1.0)))
            mcts.search(current_node, max_try=budget)

            selected_node = self._best_child_from_node(current_node)
            if selected_node is None:
                break
            current_node = selected_node

            # LimSim evaluates the best existing leaf, not merely the immediate
            # child selected for the next chained search stage.
            best_descendant = self._best_descendant_from_node(current_node)
            if terminal_fn(best_descendant.state) and reward_fn(best_descendant.state) > 0.8:
                current_node = best_descendant
                break

        selected_node = self._best_descendant_from_node(current_node)
        selected = selected_node.state
        selected_reward = reward_fn(selected)
        path_nodes = self._path_from_root(search_root, selected_node)
        decision_sequences = {agent.agent_id: [] for agent in agents}

        if path_nodes and selected_reward >= 0.5:
            for node in path_nodes:
                expected_frame = start_frame + node.state.depth * decision_step_ms
                for selected_agent in node.state.active_agents:
                    decision_sequences[selected_agent.agent_id].append(
                        DecisionStep(
                            action=selected_agent.action,
                            expected_state=selected_agent,
                            expected_frame=expected_frame,
                        )
                    )

        for agent in agents:
            decisions = decision_sequences[agent.agent_id]
            if (
                decisions
                and agent.available_lane_ids
                and decisions[-1].expected_state.lane_id not in agent.available_lane_ids
            ):
                decision_sequences[agent.agent_id] = []

        actions = {}
        trajectories = {}
        for agent in agents:
            decisions = decision_sequences.get(agent.agent_id, ())
            actions[agent.agent_id] = (
                decisions[0].action if decisions else self._fallback_action(agent, map_)
            )
            trajectories[agent.agent_id] = self._dense_rough_trajectory(
                agent,
                decisions,
                start_frame,
                map_,
                fallback_action=actions[agent.agent_id],
            )
        return actions, trajectories, decision_sequences, search_root

    def _expand(
        self,
        state: JointDecisionState,
        map_: Optional[Map],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        obstacle_initial_states: Sequence[AgentDecisionState] = (),
        completed_decisions: Sequence[Sequence[DecisionStep]] = (),
    ) -> List[JointDecisionState]:
        cache_key = id(state)
        if cache_key in self._expand_cache:
            return self._expand_cache[cache_key]
        if self._is_terminal(state):
            self._expand_cache[cache_key] = []
            return []

        active_indexes = [
            index
            for index, agent in enumerate(state.agents)
            if agent.agent_id in state.resolved_active_agent_ids
        ]
        action_sets = []
        for index in active_indexes:
            agent = state.agents[index]
            actions = self._candidate_actions(agent, map_)
            action_sets.append(actions or [LimSimAction.KS])

        # FlowState expands one direct DECISION_RESOLUTION endpoint per action.
        # The dt-sampled rough trajectory is generated only after MCTS selects
        # a DecisionStep sequence.
        agent_endpoints = {}
        for index, actions in zip(active_indexes, action_sets):
            agent = state.agents[index]
            per_action = {}
            for action in actions:
                per_action[action] = self.follower.flowstate_transition(agent, action, map_)
            agent_endpoints[index] = per_action

        expanded = []
        for joint_actions in itertools.product(*action_sets):
            endpoints = {
                index: agent_endpoints[index][action]
                for index, action in zip(active_indexes, joint_actions)
            }
            expanded.append(
                self._next_state_from_endpoints(
                    state,
                    endpoints,
                    obstacle_trajectories,
                    obstacle_initial_states,
                    completed_decisions,
                )
            )

        self._expand_cache[cache_key] = expanded
        return expanded

    def _simulate_step(
        self,
        state: JointDecisionState,
        map_: Optional[Map],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        obstacle_initial_states: Sequence[AgentDecisionState] = (),
        completed_decisions: Sequence[Sequence[DecisionStep]] = (),
    ) -> Optional[JointDecisionState]:
        """Generate a single random child without the full Cartesian product.

        This is the lightweight counterpart of :meth:`_expand`, used exclusively
        during simulation rollouts.  It picks one random action per agent,
        rolls out only those agent-action pairs, and returns one child state
        in O(N) instead of O(|A|^N).
        """

        if self._is_terminal(state):
            return None

        endpoints = {}
        for index, agent in enumerate(state.agents):
            if agent.agent_id not in state.resolved_active_agent_ids:
                continue
            actions = self._candidate_actions(agent, map_)
            action = random.choice(actions or [LimSimAction.KS])
            endpoints[index] = self.follower.flowstate_transition(agent, action, map_)

        return self._next_state_from_endpoints(
            state, endpoints, obstacle_trajectories, obstacle_initial_states, completed_decisions
        )

    def _next_state_from_endpoints(
        self,
        state: JointDecisionState,
        endpoints: Dict[int, Optional[AgentDecisionState]],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
        obstacle_initial_states: Sequence[AgentDecisionState],
        completed_decisions: Sequence[Sequence[DecisionStep]],
    ) -> JointDecisionState:
        """Build one FlowState layer, dropping vehicles beyond their lane corridor."""

        next_agents = list(state.agents)
        next_trajectories = (
            [list(trajectory) for trajectory in state.trajectories]
            if state.trajectories
            else [[] for _ in state.agents]
        )
        next_active_ids = set()
        for index, next_agent in endpoints.items():
            if next_agent is None:
                continue
            next_agents[index] = next_agent
            next_trajectories[index].append(next_agent)
            next_active_ids.add(next_agent.agent_id)

        next_depth = state.depth + 1
        has_collision = False
        if next_active_ids and next_depth < self.config.max_flowstate_depth:
            active_states = [agent for agent in next_agents if agent.agent_id in next_active_ids]
            has_collision = self.reward.has_collision_at_stage(
                active_states,
                obstacle_trajectories,
                stage_index=next_depth,
                obstacle_initial_states=obstacle_initial_states,
                completed_decisions=completed_decisions,
            )

        return JointDecisionState(
            agents=tuple(next_agents),
            depth=next_depth,
            trajectories=tuple(tuple(trajectory) for trajectory in next_trajectories),
            active_agent_ids=frozenset(next_active_ids),
            has_collision=has_collision,
        )

    def _dense_rough_trajectory(
        self,
        agent: AgentDecisionState,
        decisions: Sequence[DecisionStep],
        start_frame: int,
        map_: Optional[Map],
        fallback_action: LimSimAction,
    ) -> List[AgentDecisionState]:
        """Generate the dt-sampled output after MCTS selects endpoint decisions."""

        states = []
        current = agent
        current_frame = start_frame
        horizon_frame = start_frame + self.config.horizon_steps * self.config.step_ms
        for decision in decisions:
            segment_end = min(decision.expected_frame, horizon_frame)
            steps = int(round((segment_end - current_frame) / self.config.step_ms))
            if steps <= 0:
                continue
            segment = self.follower.rollout(current, decision.action, map_, steps=steps)
            states.extend(segment)
            if len(segment) < steps:
                return states
            current = segment[-1]
            current_frame += len(segment) * self.config.step_ms
            if current_frame >= horizon_frame:
                return states[: self.config.horizon_steps]

        remaining_steps = self.config.horizon_steps - len(states)
        if remaining_steps > 0:
            continuation_action = decisions[-1].action if decisions else fallback_action
            states.extend(
                self.follower.rollout(current, continuation_action, map_, steps=remaining_steps)
            )
        return states[: self.config.horizon_steps]

    def _is_terminal(self, state: JointDecisionState) -> bool:
        """Return whether an official FlowState layer stops expanding."""

        return (
            state.depth >= self.config.max_flowstate_depth
            or not state.active_agents
            or state.has_collision
        )

    def _candidate_actions(
        self, agent: AgentDecisionState, map_: Optional[Map]
    ) -> List[LimSimAction]:
        """Return the route-compatible action set used by every MCTS stage."""

        longitudinal = {LimSimAction.KS, LimSimAction.AC, LimSimAction.DC}
        base_actions = [
            action for action in self.config.candidate_actions if action in longitudinal
        ]
        if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
            return base_actions
        if not agent.available_lane_ids:
            return base_actions

        lane = map_.lanes[agent.lane_id]
        lane_width = self.follower._lane_width(lane)
        needs_lane_change = (
            agent.lane_id not in agent.available_lane_ids
            or abs(agent.lateral_offset) >= lane_width / 4.0
        )
        if not needs_lane_change:
            return base_actions

        centerline = lane.centerline()
        lane_length = float(centerline.length) if centerline is not None else 0.0
        remaining_distance = max(lane_length - agent.route_progress, 0.0)
        if remaining_distance <= agent.speed * self.config.decision_resolution:
            return (
                [LimSimAction.DC]
                if LimSimAction.DC in self.config.candidate_actions
                else base_actions[:1]
            )

        lane_change_actions = set()
        if agent.behaviour in {LimSimAction.LCL, LimSimAction.LCR}:
            lane_change_actions.add(agent.behaviour)
        elif agent.lane_id not in agent.available_lane_ids:
            if self._neighbor_chain_reaches_available(agent.lane_id, "left_neighbors", map_, agent):
                lane_change_actions.add(LimSimAction.LCL)
            elif self._neighbor_chain_reaches_available(
                agent.lane_id, "right_neighbors", map_, agent
            ):
                lane_change_actions.add(LimSimAction.LCR)

        return [
            action
            for action in self.config.candidate_actions
            if action in longitudinal
            or (action in lane_change_actions and is_action_valid(agent, action, map_))
        ]

    def _fallback_action(self, agent: AgentDecisionState, map_: Optional[Map]) -> LimSimAction:
        """Preserve route-required behaviour when no MCTS sequence is usable."""

        actions = self._candidate_actions(agent, map_)
        if agent.behaviour.is_lane_change and agent.behaviour in actions:
            return agent.behaviour
        if agent.available_lane_ids and agent.lane_id not in agent.available_lane_ids:
            for action in (LimSimAction.LCL, LimSimAction.LCR):
                if action in actions:
                    return action
        if LimSimAction.KS in actions:
            return LimSimAction.KS
        return actions[0] if actions else LimSimAction.KS

    def _neighbor_chain_reaches_available(
        self, lane_id, neighbor_attribute: str, map_: Map, agent: AgentDecisionState
    ) -> bool:
        pending = [lane_id]
        visited = {lane_id}
        while pending:
            current = map_.lanes.get(pending.pop())
            if current is None:
                continue
            for neighbor_id in getattr(current, neighbor_attribute):
                if neighbor_id in agent.available_lane_ids:
                    return True
                if neighbor_id in map_.lanes and neighbor_id not in visited:
                    visited.add(neighbor_id)
                    pending.append(neighbor_id)
        return False

    def _best_child_from_node(self, node):
        if node is None or not node.children:
            return None
        visited = [child for child in node.children if child.visits > 0]
        candidates = visited or node.children
        best_reward = max(child.total_reward / max(child.visits, 1) for child in candidates)
        best_children = [
            child
            for child in candidates
            if child.total_reward / max(child.visits, 1) == best_reward
        ]
        return random.choice(best_children)

    def _best_descendant_from_node(self, node):
        while node.children and node.state.depth < self.config.max_flowstate_depth:
            child = self._best_child_from_node(node)
            if child is None:
                break
            node = child
        return node

    def _path_from_root(self, root, node):
        path = []
        while node is not None and node is not root:
            path.append(node)
            node = node.parent
        return list(reversed(path)) if node is root else []
