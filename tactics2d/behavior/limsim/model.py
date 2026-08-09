# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public LimSim-style behavior model entry point."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import Trajectory

from .action import LimSimAction
from .config import LimSimConfig
from .decision_search import LimSimDecisionSearch
from .frenet_planner import FrenetTrajectoryPlanner
from .interaction import InteractionGraph
from .lane_follower import LaneFollower
from .roi import RoISelector
from .scene import SceneBuilder
from .schema import (
    AgentDecisionState,
    DecisionStep,
    PlanningResult,
    decision_sequence_covers_horizon,
    states_to_trajectory,
)


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
        try:
            decision_interval_ms = int(round(float(self.config.decision_interval) * 1000))
        except (TypeError, ValueError) as error:
            raise ValueError("decision_interval must be a positive number of seconds.") from error
        if decision_interval_ms <= 0:
            raise ValueError("decision_interval must be positive.")
        self._decision_interval_ms = decision_interval_ms
        self._reset_decision_cache()

    def plan(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        route_map: Dict[object, Tuple[str, ...]],
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
            route_map: ``{agent_id: (lane_id_0, lane_id_1, ...)}`` mapping
                each vehicle to its ordered lane sequence.  Pass an empty
                dict to fall back to pure lane-topology routing (successors
                are chosen arbitrarily; not recommended).
            agent_ids: Optional explicit ids to control. If omitted, all active
                vehicles are considered unless an RoI is requested.
            roi_center: Center point used with ``roi_radius`` when no ``ego_id``
                is provided.
            roi_radius: Inner RoI radius. Vehicles inside this region are
                planning candidates for LimSim. A replay controller may apply
                an additional interaction-based takeover policy.
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
            self.scene_builder.build(participants, map_, frame, selected_ids, route_map=route_map),
            map_,
        )
        background_states = self._filter_lane_matched_states(
            self.scene_builder.build(
                participants, map_, frame, background_ids, route_map=route_map
            ),
            map_,
        )
        background_ids = [agent_id for agent_id in background_ids if agent_id in background_states]
        background_trajectories = self._predict_obstacle_trajectories(
            participants,
            map_,
            frame,
            background_states,
            last_planned_trajectories=last_planned_trajectories,
        )
        groups = self.interaction_graph.build_groups(scene_states, map_)
        result = PlanningResult(
            groups=groups,
            roi_agent_ids=list(scene_states.keys()),
            background_agent_ids=background_ids,
        )
        rough_trajectories = {}
        if self._decision_refresh_due(frame):
            refreshed_decisions = {}
            for group in groups:
                agents = [scene_states[agent_id] for agent_id in group]
                _, trajectories, decisions, root = self.decision_search._plan_with_decisions(
                    agents,
                    map_,
                    obstacle_trajectories=tuple(background_trajectories.values()),
                    start_frame=frame,
                    obstacle_initial_states=tuple(background_states.values()),
                    completed_decisions=tuple(
                        sequence for sequence in refreshed_decisions.values() if sequence
                    ),
                )
                result.root_nodes[tuple(group)] = root
                for agent in agents:
                    agent_decisions = decisions.get(agent.agent_id, [])
                    refreshed_decisions[agent.agent_id] = agent_decisions
                    if agent_decisions:
                        rough_trajectories[agent.agent_id] = trajectories[agent.agent_id]

            self._decision_sequences = refreshed_decisions
            self._last_decision_frame = frame

        active_decisions = {}
        for agent in scene_states.values():
            cached_decisions = self._remaining_decisions(agent.agent_id, frame)
            decisions = (
                cached_decisions
                if decision_sequence_covers_horizon(
                    cached_decisions, frame, self.config.horizon_steps, self.config.step_ms
                )
                else []
            )
            active_decisions[agent.agent_id] = decisions
            action = (
                decisions[0].action
                if decisions
                else self.decision_search._fallback_action(agent, map_)
            )
            result.actions[agent.agent_id] = action
            if not decisions:
                rough_trajectories[agent.agent_id] = self.follower.rollout(agent, action, map_)
            elif agent.agent_id not in rough_trajectories:
                rough_trajectories[agent.agent_id] = self.decision_search._dense_rough_trajectory(
                    agent,
                    decisions,
                    frame,
                    map_,
                    fallback_action=action,
                )

        final_state_trajectories = {}
        for agent in scene_states.values():
            action = result.actions[agent.agent_id]
            decisions = active_decisions[agent.agent_id]
            rough = rough_trajectories.get(agent.agent_id, [])

            if not self.config.use_frenet_refinement:
                # --- pure LimSim mode: use MCTS rough trajectories directly ---
                # This matches the original LimSim paper's trajectory generation
                # (kinematic lane-following from MCTS-selected actions).
                final_state_trajectories[agent.agent_id] = rough
                result.trajectories[agent.agent_id] = states_to_trajectory(
                    agent.agent_id, rough, frame, self.config.dt
                )
                continue

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
                agent, action, map_, planning_obstacles, time_ms=frame, decision_sequence=decisions
            )
            final_state_trajectories[agent.agent_id] = planned_states
            result.trajectories[agent.agent_id] = states_to_trajectory(
                agent.agent_id, planned_states, frame, self.config.dt
            )

        return result

    def _reset_decision_cache(self):
        """Clear high-level decisions at the boundary of one simulation run."""

        self._last_decision_frame = None
        self._decision_sequences = {}

    def _decision_refresh_due(self, frame: int) -> bool:
        if self._last_decision_frame is None:
            return True
        if frame < self._last_decision_frame:
            self._reset_decision_cache()
            return True
        return frame - self._last_decision_frame >= self._decision_interval_ms

    def _remaining_decisions(self, agent_id, frame: int) -> List[DecisionStep]:
        return [
            decision
            for decision in self._decision_sequences.get(agent_id, ())
            if decision.expected_frame > frame
        ]

    def predict(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
        route_map: Optional[Dict[object, Tuple[str, ...]]] = None,
    ) -> Dict[object, Trajectory]:
        """Plan future trajectories for selected agents.

        This method provides the shared behavior-model interface. Use
        :meth:`plan` when LimSim-specific diagnostics such as actions, groups,
        and MCTS root nodes are needed.

        .. note::

            **route_map should always be provided.**  Without it LimSim relies
            on pure lane-topology inference (blindly following the first
            successor lane), which produces incorrect routing at intersections
            and dead-ends.  Call :func:`~tactics2d.dataset_parser.route_extractor.extract_all_lane_sequences`
            or provide your own mapping.
        """

        return self.plan(
            participants,
            map_,
            frame,
            route_map=route_map if route_map is not None else {},
            agent_ids=agent_ids,
        ).trajectories

    def predict_batch(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frames: List[int],
        agent_ids: Optional[Iterable[object]] = None,
        max_workers: Optional[int] = None,
        route_map: Optional[Dict[object, Tuple[str, ...]]] = None,
    ) -> Dict[int, Dict[object, Trajectory]]:
        """Predict an ordered frame batch in an isolated LimSim session.

        LimSim's high-level decisions, route cursor, and MCTS expansion cache
        are temporal state. A batch therefore runs chronologically on one fresh
        model instead of sharing this instance across worker threads.
        ``max_workers`` is retained for interface compatibility but ignored.
        """

        del max_workers
        batch_model = type(self)(config=self.config, parallel_workers=0)
        selected_ids = None if agent_ids is None else tuple(agent_ids)
        kwargs = {"agent_ids": selected_ids}
        if route_map is not None:
            kwargs["route_map"] = route_map

        result = {}
        for frame in sorted(set(frames)):
            try:
                result[frame] = batch_model.predict(participants, map_, frame, **kwargs)
            except Exception:
                result[frame] = {}
        return result

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
        """Predict obstacle trajectories reusing pre-built states.

        Avoids redundant scene building by using *background_states* directly
        and only consulting the predictor for trajectory-reuse lookups.
        """
        last_planned = last_planned_trajectories or {}
        trajectories = {}
        for agent_id, background_state in background_states.items():
            # try to reuse a previously planned trajectory segment
            remaining = None
            cached_traj = last_planned.get(agent_id)
            if cached_traj is not None:
                remaining_frames = [f for f in cached_traj.frames if f > frame]
                if remaining_frames:
                    remaining = Trajectory(
                        id_=agent_id, fps=cached_traj.fps, stable_freq=cached_traj.stable_freq
                    )
                    for sf in remaining_frames:
                        remaining.add_state(cached_traj.get_state(sf))

            if remaining is not None:
                predicted_states = self._trajectory_to_decision_states(remaining, background_state)
                if predicted_states:
                    trajectories[agent_id] = predicted_states
                    continue

            # fallback: lane-following rollout from pre-built state
            trajectories[agent_id] = self.follower.rollout(background_state, LimSimAction.KS, map_)
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
                    heading=spatial.normalize_angle(raw_state.heading),
                    speed=max(raw_state.speed or 0.0, 0.0),
                    route_progress=previous_progress,
                )
            )
        return states
