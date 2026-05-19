# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public LimSim-style behavior model entry point."""

from typing import Dict, Iterable, Optional, Sequence

from tactics2d.map.element import Map

from .action import LimSimAction
from .config import LimSimConfig
from .decision_search import LimSimDecisionSearch
from .frenet_planner import FrenetTrajectoryPlanner
from .interaction import InteractionGraph
from .planner import LaneFollower
from .roi import RoISelector
from .scene import SceneBuilder
from .schema import PlanningResult, states_to_trajectory


class LimSimBehaviorModel:
    """Reproduce LimSim's MCT-based interactive behavior layer on Tactics2D data.

    This implementation focuses on the non-LLM LimSim pipeline: local interaction
    grouping, discrete joint behavior decisions, and trajectory rollout.
    """

    def __init__(self, config: Optional[LimSimConfig] = None):
        self.config = config or LimSimConfig()
        self.scene_builder = SceneBuilder(self.config)
        self.interaction_graph = InteractionGraph(self.config)
        self.decision_search = LimSimDecisionSearch(self.config)
        self.follower = LaneFollower(self.config)
        self.trajectory_planner = FrenetTrajectoryPlanner(self.config)

    def plan(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
        roi_center: Optional[Sequence[float]] = None,
        roi_radius: Optional[float] = None,
        roi_outer_radius: Optional[float] = None,
        ego_id: Optional[object] = None,
    ) -> PlanningResult:
        """Plan future trajectories for active agents at one frame."""

        selected_ids = list(agent_ids) if agent_ids is not None else None
        background_ids = []
        if selected_ids is None and roi_radius is not None:
            if ego_id is not None:
                selection = RoISelector.select_around_agent(
                    participants,
                    frame,
                    ego_id=ego_id,
                    radius=roi_radius,
                    outer_radius=roi_outer_radius,
                )
            elif roi_center is not None:
                selection = RoISelector.select_by_radius(
                    participants,
                    frame,
                    center=roi_center,
                    radius=roi_radius,
                    outer_radius=roi_outer_radius,
                )
            else:
                raise ValueError("roi_center or ego_id must be provided when roi_radius is set.")
            selected_ids = selection.agent_ids
            background_ids = selection.background_agent_ids

        scene_states = self.scene_builder.build(participants, map_, frame, selected_ids)
        background_states = self.scene_builder.build(participants, map_, frame, background_ids)
        background_trajectories = [
            self.follower.rollout(background_state, LimSimAction.KS, map_)
            for background_state in background_states.values()
        ]
        groups = self.interaction_graph.build_groups(scene_states, map_)
        result = PlanningResult(
            groups=groups,
            roi_agent_ids=list(scene_states.keys()),
            background_agent_ids=background_ids,
        )
        rough_trajectories = {}

        for group in groups:
            agents = [scene_states[agent_id] for agent_id in group]
            if len(agents) <= 1:
                agent = agents[0]
                action = self._choose_single_agent_action(
                    agent, map_, background_trajectories, time_ms=frame
                )
                result.actions[agent.agent_id] = action
                rough_trajectories[agent.agent_id] = self.follower.rollout(agent, action, map_)
                continue

            actions, trajectories, root = self.decision_search.plan(
                agents, map_, obstacle_trajectories=background_trajectories
            )
            result.root_nodes[tuple(group)] = root
            for agent in agents:
                result.actions[agent.agent_id] = actions[agent.agent_id]
                rough_trajectories[agent.agent_id] = trajectories[agent.agent_id]

        final_state_trajectories = {}
        for agent in scene_states.values():
            planning_obstacles = list(background_trajectories)
            planning_obstacles.extend(
                trajectory
                for other_id, trajectory in final_state_trajectories.items()
                if other_id != agent.agent_id
            )
            planning_obstacles.extend(
                trajectory
                for other_id, trajectory in rough_trajectories.items()
                if other_id != agent.agent_id and other_id not in final_state_trajectories
            )
            planned_states = self.trajectory_planner.plan(
                agent,
                result.actions[agent.agent_id],
                map_,
                planning_obstacles,
                time_ms=frame,
            )
            final_state_trajectories[agent.agent_id] = planned_states
            result.trajectories[agent.agent_id] = states_to_trajectory(
                agent.agent_id, planned_states, frame, self.config.dt
            )

        return result

    def _choose_single_agent_action(
        self,
        agent,
        map_: Optional[Map],
        background_trajectories,
        time_ms: Optional[int] = None,
    ) -> LimSimAction:
        candidates = [
            action
            for action in self.config.candidate_actions
            if action in {LimSimAction.KS, LimSimAction.AC, LimSimAction.DC}
        ]
        if not background_trajectories:
            return LimSimAction.KS

        best_action = LimSimAction.KS
        best_reward = float("-inf")
        for action in candidates:
            trajectory = self.trajectory_planner.plan(
                agent, action, map_, background_trajectories, time_ms=time_ms
            )
            reward = self.decision_search.reward.evaluate(
                [agent],
                {agent.agent_id: trajectory},
                background_trajectories,
            )
            if reward > best_reward:
                best_reward = reward
                best_action = action
        return best_action
