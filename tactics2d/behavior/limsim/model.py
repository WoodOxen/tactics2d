# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public LimSim-style behavior model entry point."""

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.geometry import normalize_angle
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import Trajectory

from .action import LimSimAction
from .config import LimSimConfig
from .decision_search import LimSimDecisionSearch
from .frenet_planner import FrenetTrajectoryPlanner
from .interaction import InteractionGraph
from .planner import LaneFollower
from .prediction import LimSimPredictor
from .roi import RoISelector
from .scene import SceneBuilder
from .schema import AgentDecisionState, PlanningResult, states_to_trajectory


class LimSimBehaviorModel(BehaviorModelBase):
    """Reproduce LimSim's MCT-based interactive behavior layer on Tactics2D data.

    This implementation focuses on the non-LLM LimSim pipeline: local interaction
    grouping, discrete joint behavior decisions, and trajectory rollout.
    """

    def __init__(self, config: Optional[LimSimConfig] = None, parallel_workers: int = 0):
        self.config = config or LimSimConfig()
        self.parallel_workers = parallel_workers
        self.scene_builder = SceneBuilder(self.config)
        self.interaction_graph = InteractionGraph(self.config)
        self.decision_search = LimSimDecisionSearch(self.config)
        self.follower = LaneFollower(self.config)
        self.trajectory_planner = FrenetTrajectoryPlanner(self.config)
        self.predictor = LimSimPredictor(self.config)

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
        last_planned_trajectories: Optional[Dict[object, Trajectory]] = None,
    ) -> PlanningResult:
        """Plan future trajectories for active agents at one frame.

        Args:
            participants: Tactics2D traffic participants keyed by agent id.
            map_: Semantic map used for lane matching, route rollout, and
                conflict checks. If ``None``, map-dependent planning is skipped.
            frame: Current frame timestamp in milliseconds.
            agent_ids: Optional explicit ids to control. If omitted, all active
                vehicles are considered unless an RoI is requested.
            roi_center: Center point used with ``roi_radius`` when no ``ego_id``
                is provided.
            roi_radius: Inner RoI radius. Vehicles inside this region are
                controlled by LimSim.
            roi_outer_radius: Optional outer RoI radius. Vehicles between the
                inner and outer radii are treated as background obstacles.
            ego_id: Optional participant id whose current position defines the
                RoI center.
            last_planned_trajectories: Previously planned trajectories used by
                the predictor to reuse committed lane-change plans.

        Returns:
            PlanningResult containing planned trajectories plus LimSim-specific
            diagnostics such as actions, interaction groups, and RoI ids.
        """

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

        selected_ids = self._filter_controlled_vehicle_ids(participants, selected_ids)
        background_ids = self._filter_controlled_vehicle_ids(participants, background_ids)

        scene_states = self._filter_lane_matched_states(
            self.scene_builder.build(participants, map_, frame, selected_ids), map_
        )
        background_states = self._filter_lane_matched_states(
            self.scene_builder.build(participants, map_, frame, background_ids), map_
        )
        background_ids = [agent_id for agent_id in background_ids if agent_id in background_states]
        background_trajectories = self._predict_obstacle_trajectories(
            participants,
            map_,
            frame,
            background_states,
            last_planned_trajectories=last_planned_trajectories,
        )
        scene_predictions = self._predict_obstacle_trajectories(
            participants,
            map_,
            frame,
            scene_states,
            last_planned_trajectories=last_planned_trajectories,
        )
        groups = self.interaction_graph.build_groups(scene_states, map_)
        result = PlanningResult(
            groups=groups,
            roi_agent_ids=list(scene_states.keys()),
            background_agent_ids=background_ids,
        )
        rough_trajectories = {}

        for group in groups:
            agents = [scene_states[agent_id] for agent_id in group]
            group_obstacles = list(background_trajectories.values())
            group_obstacles.extend(
                scene_predictions[other_id]
                for other_id in scene_states
                if other_id not in group and other_id in scene_predictions
            )
            if len(agents) <= 1:
                agent = agents[0]
                action = self._choose_single_agent_action(
                    agent, map_, group_obstacles, time_ms=frame
                )
                result.actions[agent.agent_id] = action
                rough_trajectories[agent.agent_id] = self.follower.rollout(agent, action, map_)
                continue

            actions, trajectories, root = self.decision_search.plan(
                agents, map_, obstacle_trajectories=group_obstacles
            )
            result.root_nodes[tuple(group)] = root
            for agent in agents:
                result.actions[agent.agent_id] = actions[agent.agent_id]
                rough_trajectories[agent.agent_id] = trajectories[agent.agent_id]

        final_state_trajectories = {}
        for agent in scene_states.values():
            planning_obstacles = list(background_trajectories.values())
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
                agent, result.actions[agent.agent_id], map_, planning_obstacles, time_ms=frame
            )
            final_state_trajectories[agent.agent_id] = planned_states
            result.trajectories[agent.agent_id] = states_to_trajectory(
                agent.agent_id, planned_states, frame, self.config.dt
            )

        return result

    def predict(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> Dict[object, Trajectory]:
        """Plan future trajectories for selected agents.

        This method provides the shared behavior-model interface. Use
        :meth:`plan` when LimSim-specific diagnostics such as actions, groups,
        and MCTS root nodes are needed.
        """

        return self.plan(participants, map_, frame, agent_ids=agent_ids).trajectories

    def _filter_controlled_vehicle_ids(
        self, participants: Dict[object, object], agent_ids: Optional[Iterable[object]]
    ) -> list:
        """Keep LimSim-controlled agents aligned with vehicle-only semantics."""

        if agent_ids is None:
            agent_ids = participants.keys()
        return [
            agent_id for agent_id in agent_ids if isinstance(participants.get(agent_id), Vehicle)
        ]

    def _filter_lane_matched_states(
        self, states: Dict[object, AgentDecisionState], map_: Optional[Map]
    ) -> Dict[object, AgentDecisionState]:
        """Keep map-based planning on lane-matched vehicles only."""

        if map_ is None:
            return states
        return {agent_id: state for agent_id, state in states.items() if state.lane_id is not None}

    def _predict_obstacle_trajectories(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        background_states: Dict[object, AgentDecisionState],
        last_planned_trajectories: Optional[Dict[object, Trajectory]] = None,
    ):
        predictions = self.predictor.predict(
            participants,
            map_,
            frame,
            agent_ids=background_states.keys(),
            last_planned_trajectories=last_planned_trajectories,
        )
        trajectories = {}
        for agent_id, background_state in background_states.items():
            predicted_states = self._trajectory_to_decision_states(
                predictions.get(agent_id), background_state
            )
            if predicted_states:
                trajectories[agent_id] = predicted_states
            else:
                trajectories[agent_id] = self.follower.rollout(
                    background_state, LimSimAction.KS, map_
                )
        return trajectories

    def _trajectory_to_decision_states(
        self, trajectory: Optional[Trajectory], reference_state: AgentDecisionState
    ):
        if trajectory is None:
            return []
        states = []
        previous_progress = reference_state.route_progress
        for frame in trajectory.frames[: self.config.horizon_steps]:
            raw_state = trajectory.get_state(frame)
            dx = raw_state.x - reference_state.x
            dy = raw_state.y - reference_state.y
            projected_progress = (
                reference_state.route_progress
                + dx * np.cos(reference_state.heading)
                + dy * np.sin(reference_state.heading)
            )
            previous_progress = max(previous_progress, projected_progress)
            states.append(
                reference_state.with_updates(
                    x=raw_state.x,
                    y=raw_state.y,
                    heading=normalize_angle(raw_state.heading),
                    speed=max(raw_state.speed or 0.0, 0.0),
                    route_progress=previous_progress,
                )
            )
        return states

    def _choose_single_agent_action(
        self, agent, map_: Optional[Map], background_trajectories, time_ms: Optional[int] = None
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
                [agent], {agent.agent_id: trajectory}, background_trajectories
            )
            if reward > best_reward:
                best_reward = reward
                best_action = action
        return best_action
