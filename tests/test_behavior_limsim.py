# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for LimSim behavior planning and integration."""

import numpy as np
import pytest
from shapely.geometry import LineString

from tactics2d.behavior.limsim import LimSimBehaviorModel, LimSimConfig
from tactics2d.behavior.limsim.action import LimSimAction
from tactics2d.behavior.limsim.decision_search import LimSimDecisionSearch
from tactics2d.behavior.limsim.frenet_planner import FrenetTrajectoryPlanner
from tactics2d.behavior.limsim.schema import AgentDecisionState, DecisionStep, JointDecisionState
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.search import MCTS


def _lane(lane_id, x_left, x_right, y_start, y_end):
    left_side = LineString([(x_left, y_start), (x_left, y_end)])
    right_side = LineString([(x_right, y_start), (x_right, y_end)])
    return Lane(
        id_=lane_id,
        left_side=left_side,
        right_side=right_side,
        custom_tags={
            "centerline": np.array(
                [[(x_left + x_right) / 2.0, y_start], [(x_left + x_right) / 2.0, y_end]]
            )
        },
    )


def _parallel_map():
    map_ = Map(name="parallel")
    a = _lane("A", 0.0, 2.0, 0.0, 80.0)
    b = _lane("B", 2.0, 4.0, 0.0, 80.0)
    map_.add_lane(a)
    map_.add_lane(b)
    return map_


def _vehicle(agent_id, frame, x, y, heading=np.pi / 2, speed=5.0):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    trajectory.add_state(
        State(
            frame=frame,
            x=x,
            y=y,
            heading=heading,
            vx=speed * np.cos(heading),
            vy=speed * np.sin(heading),
        )
    )
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=4.5, width=1.8)


def test_limsim_plans_for_all_agents():
    """plan() produces actions and trajectories for every controlled agent."""
    config = LimSimConfig(horizon_steps=6, dt=0.2, mcts_iterations=20, interaction_distance=15.0)
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=5.0), 2: _vehicle(2, 0, 1.0, 10.0, speed=2.0)}

    result = LimSimBehaviorModel(config).plan(participants, _parallel_map(), route_map={}, frame=0)

    assert set(result.actions) == {1, 2}
    assert set(result.trajectories) == {1, 2}
    assert len(result.trajectories[1].frames) == config.horizon_steps


def test_agent_decision_state_preserves_existing_positional_field_order():
    state = AgentDecisionState(
        "ego", 1.0, 2.0, 0.3, 4.0, "A", 0.2, ("A", "B"), 7.0, 4.5, 1.8, LimSimAction.AC
    )

    assert state.route_progress == 7.0
    assert state.action == LimSimAction.AC
    assert state.available_lane_ids == frozenset()


def test_limsim_decelerates_for_obstacle():
    """A faster agent behind a slower one decelerates (DC) to avoid collision."""
    config = LimSimConfig(horizon_steps=50, mcts_iterations=200, interaction_distance=20.0)
    participants = {1: _vehicle(1, 0, 1.0, 0.0, speed=8.0), 2: _vehicle(2, 0, 1.0, 25.0, speed=3.0)}

    result = LimSimBehaviorModel(config).plan(
        participants, _parallel_map(), route_map={1: ("A",), 2: ("A",)}, frame=0
    )

    assert result.actions[1] == LimSimAction.DC


def test_limsim_chained_mcts_advances_one_decision_stage_at_a_time(monkeypatch):
    """Each decaying search budget continues from one selected child node."""

    calls = []
    stage_actions = [LimSimAction.AC, LimSimAction.AC, LimSimAction.DC, LimSimAction.KS]

    def fake_search(self, root, max_try):
        calls.append((root.state.depth, max_try))
        action = stage_actions[root.state.depth]
        child_state = next(
            state for state in self.expand_fn(root.state) if state.agents[0].action == action
        )
        child = self.Node(child_state, parent=root)
        child.visits = 1
        child.total_reward = 1.0
        root.children = [child]
        root.visits += 1
        return root

    monkeypatch.setattr(MCTS, "search", fake_search)
    config = LimSimConfig(horizon_steps=50, terminal_depth=4, mcts_iterations=200)
    agent = AgentDecisionState(
        "ego", 0.0, 0.0, 0.0, 5.0, lane_id="A", available_lane_ids=frozenset({"A"})
    )

    actions, trajectories, decisions, root = LimSimDecisionSearch(config)._plan_with_decisions(
        [agent], map_=None, start_frame=1000
    )

    assert calls == [(0, 200), (1, 133), (2, 100), (3, 80)]
    assert root.state.depth == 0
    assert actions["ego"] == LimSimAction.AC
    assert len(trajectories["ego"]) == config.horizon_steps
    assert [step.action for step in decisions["ego"]] == stage_actions
    assert [step.expected_frame for step in decisions["ego"]] == [2500, 4000, 5500, 7000]

    selected_nodes = []
    node = root
    while node.children:
        node = node.children[0]
        selected_nodes.append(node)
    assert [step.expected_state for step in decisions["ego"]] == [
        node.state.agents[0] for node in selected_nodes
    ]
    assert len(selected_nodes[-1].state.trajectories[0]) == len(stage_actions)


def test_limsim_chained_mcts_stops_on_good_terminal_descendant(monkeypatch):
    calls = []

    def fake_search(self, root, max_try):
        calls.append((root.state.depth, max_try))
        node = root
        while node.state.depth < config.max_flowstate_depth:
            child_state = next(
                state
                for state in self.expand_fn(node.state)
                if state.agents[0].action == LimSimAction.KS
            )
            child = self.Node(child_state, parent=node)
            child.visits = 1
            child.total_reward = 1.0
            node.children = [child]
            node.visits = 1
            node = child
        return root

    monkeypatch.setattr(MCTS, "search", fake_search)
    config = LimSimConfig(horizon_steps=50, terminal_depth=4, mcts_iterations=200)
    agent = AgentDecisionState(
        "ego",
        1.0,
        0.0,
        np.pi / 2.0,
        5.0,
        lane_id="A",
        route_lane_ids=("A",),
        available_lane_ids=frozenset({"A"}),
    )

    _, _, decisions, _ = LimSimDecisionSearch(config)._plan_with_decisions(
        [agent], _parallel_map(), start_frame=0
    )

    assert calls == [(0, 200)]
    assert len(decisions["ego"]) == config.max_flowstate_depth


def test_limsim_rejects_low_reward_mcts_decision(monkeypatch):
    config = LimSimConfig(horizon_steps=10, terminal_depth=1, mcts_iterations=2)
    search = LimSimDecisionSearch(config)
    agent = AgentDecisionState("ego", 0.0, 0.0, 0.0, 5.0)
    monkeypatch.setattr(search.reward, "evaluate", lambda *args, **kwargs: 0.0)

    actions, trajectories, decisions, _ = search._plan_with_decisions([agent], map_=None)

    assert decisions["ego"] == []
    assert actions["ego"] == LimSimAction.KS
    assert len(trajectories["ego"]) == config.horizon_steps
    assert trajectories["ego"][0].x == pytest.approx(0.5)
    assert all(state.action == LimSimAction.KS for state in trajectories["ego"])


def test_limsim_rejects_decision_ending_outside_available_lanes(monkeypatch):
    config = LimSimConfig(horizon_steps=1, terminal_depth=1, mcts_iterations=2)
    search = LimSimDecisionSearch(config)
    agent = AgentDecisionState(
        "ego",
        1.0,
        0.0,
        np.pi / 2.0,
        5.0,
        lane_id="A",
        route_lane_ids=("A", "B"),
        available_lane_ids=frozenset({"A"}),
    )

    def fake_search(self, root, max_try):
        endpoint = agent.with_updates(lane_id="B", action=LimSimAction.AC)
        child_state = JointDecisionState(agents=(endpoint,), depth=1, trajectories=((endpoint,),))
        child = self.Node(child_state, parent=root)
        child.visits = 1
        child.total_reward = 1.0
        root.children = [child]
        return root

    monkeypatch.setattr(MCTS, "search", fake_search)
    monkeypatch.setattr(search.reward, "evaluate", lambda *args, **kwargs: 1.0)

    actions, _, decisions, _ = search._plan_with_decisions([agent], _parallel_map())

    assert decisions["ego"] == []
    assert actions["ego"] == LimSimAction.KS


def test_limsim_model_caches_decisions_for_three_seconds(monkeypatch):
    config = LimSimConfig(horizon_steps=15, terminal_depth=2, use_frenet_refinement=True)
    model = LimSimBehaviorModel(config)
    participants = {agent_id: _vehicle(agent_id, 0, float(agent_id), 0.0) for agent_id in (1, 2, 3)}

    def build_scene(participants, map_, frame, agent_ids=None, route_map=None):
        selected = list(participants) if agent_ids is None else list(agent_ids)
        active = {1, 2} if frame == 0 else {1, 2, 3}
        return {
            agent_id: AgentDecisionState(
                agent_id, x=float(agent_id), y=frame / 1000.0, heading=0.0, speed=5.0
            )
            for agent_id in selected
            if agent_id in active
        }

    monkeypatch.setattr(model.scene_builder, "build", build_scene)
    monkeypatch.setattr(
        model.interaction_graph,
        "build_groups",
        lambda states, map_: [list(states)] if states else [],
    )
    monkeypatch.setattr(model, "_predict_obstacle_trajectories", lambda *args, **kwargs: {})

    decision_frames = []

    def plan_with_decisions(
        agents,
        map_,
        obstacle_trajectories=(),
        start_frame=0,
        obstacle_initial_states=(),
        completed_decisions=(),
    ):
        decision_frames.append(start_frame)
        trajectories = {}
        decisions = {}
        actions = {}
        for agent in agents:
            current = agent
            rough = []
            decisions[agent.agent_id] = []
            for depth, action in enumerate((LimSimAction.AC, LimSimAction.DC), start=1):
                segment = model.follower.rollout(current, action, map_, steps=15)
                rough.extend(segment)
                current = segment[-1]
                decisions[agent.agent_id].append(
                    DecisionStep(
                        action=action,
                        expected_state=current,
                        expected_frame=start_frame + depth * 1500,
                    )
                )
            actions[agent.agent_id] = decisions[agent.agent_id][0].action
            trajectories[agent.agent_id] = rough[: config.horizon_steps]
        return actions, trajectories, decisions, object()

    monkeypatch.setattr(model.decision_search, "_plan_with_decisions", plan_with_decisions)

    low_level_calls = []

    def low_level_plan(
        agent, action, map_, obstacle_trajectories=(), time_ms=None, decision_sequence=()
    ):
        low_level_calls.append(
            (agent.agent_id, time_ms, action, tuple(step.action for step in decision_sequence))
        )
        return model.follower.rollout(agent, action, map_, steps=config.horizon_steps)

    monkeypatch.setattr(model.trajectory_planner, "plan", low_level_plan)

    frames = [*range(0, 3000, 500), 2999, 3000]
    results = {
        frame: model.plan(participants, map_=None, route_map={}, frame=frame) for frame in frames
    }

    assert config.decision_interval == 3.0
    assert decision_frames == [0, 3000]
    assert [call[1] for call in low_level_calls if call[0] == 1] == frames
    assert next(call for call in low_level_calls if call[:2] == (1, 0))[3] == (
        LimSimAction.AC,
        LimSimAction.DC,
    )
    assert next(call for call in low_level_calls if call[:2] == (1, 1500))[3] == (LimSimAction.DC,)
    assert next(call for call in low_level_calls if call[:2] == (1, 2500))[2:] == (
        LimSimAction.KS,
        (),
    )
    assert next(call for call in low_level_calls if call[:2] == (3, 500))[2:] == (
        LimSimAction.KS,
        (),
    )
    assert results[2500].root_nodes == {}
    assert results[3000].root_nodes

    model._reset_decision_cache()
    model.plan(participants, map_=None, route_map={}, frame=3500)
    assert decision_frames == [0, 3000, 3500]


def test_limsim_cached_decision_gap_preserves_route_required_lane_change(monkeypatch):
    config = LimSimConfig(horizon_steps=15, terminal_depth=2)
    model = LimSimBehaviorModel(config)
    map_ = _parallel_map()
    map_.lanes["A"].right_neighbors.add("B")
    map_.lanes["B"].left_neighbors.add("A")
    participant = _vehicle("ego", 1500, 3.0, 0.0)
    wrong_lane_state = AgentDecisionState(
        "ego",
        3.0,
        0.0,
        np.pi / 2.0,
        5.0,
        lane_id="B",
        route_lane_ids=("A",),
        available_lane_ids=frozenset({"A"}),
    )

    def build_scene(participants, map_, frame, agent_ids=None, route_map=None):
        selected = list(participants) if agent_ids is None else list(agent_ids)
        return {"ego": wrong_lane_state} if "ego" in selected else {}

    monkeypatch.setattr(model.scene_builder, "build", build_scene)
    monkeypatch.setattr(
        model.interaction_graph,
        "build_groups",
        lambda states, map_: [list(states)] if states else [],
    )
    monkeypatch.setattr(model, "_predict_obstacle_trajectories", lambda *args, **kwargs: {})

    model._last_decision_frame = 0
    model._decision_sequences = {
        "ego": [DecisionStep(LimSimAction.KS, wrong_lane_state, expected_frame=1600)]
    }

    result = model.plan({"ego": participant}, map_, frame=1500, route_map={"ego": ("A",)})

    assert result.actions["ego"] == LimSimAction.LCL
    assert result.trajectories["ego"].frames


def test_limsim_predict_batch_uses_one_isolated_chronological_session(monkeypatch):
    model = LimSimBehaviorModel()
    route_map = {"ego": ("A",)}
    calls = []
    decision_frames = []

    def fake_predict(self, participants, map_, frame, agent_ids=None, route_map=None):
        calls.append((id(self), frame, agent_ids, route_map))
        if self._decision_refresh_due(frame):
            decision_frames.append(frame)
            self._last_decision_frame = frame
        return {}

    monkeypatch.setattr(LimSimBehaviorModel, "predict", fake_predict)

    result = model.predict_batch(
        {},
        None,
        frames=[3000, 2500, 0, 500],
        agent_ids=(agent_id for agent_id in ["ego"]),
        max_workers=4,
        route_map=route_map,
    )

    assert list(result) == [0, 500, 2500, 3000]
    assert [call[1] for call in calls] == [0, 500, 2500, 3000]
    assert decision_frames == [0, 3000]
    assert len({call[0] for call in calls}) == 1
    assert calls[0][0] != id(model)
    assert all(call[2] == ("ego",) and call[3] is route_map for call in calls)
    assert model._last_decision_frame is None


def test_frenet_planner_consumes_remaining_decision_stages():
    config = LimSimConfig(horizon_steps=30, terminal_depth=2)
    planner = FrenetTrajectoryPlanner(config)
    map_ = _parallel_map()
    agent = AgentDecisionState(
        "ego", 1.0, 0.0, np.pi / 2.0, 5.0, lane_id="A", route_lane_ids=("A",)
    )
    first_segment = planner.fallback.rollout(agent, LimSimAction.AC, map_, steps=15)
    second_segment = planner.fallback.rollout(first_segment[-1], LimSimAction.DC, map_, steps=15)
    decisions = [
        DecisionStep(action=LimSimAction.AC, expected_state=first_segment[-1], expected_frame=1500),
        DecisionStep(
            action=LimSimAction.DC, expected_state=second_segment[-1], expected_frame=3000
        ),
    ]

    states = planner.plan(agent, LimSimAction.AC, map_=map_, time_ms=0, decision_sequence=decisions)
    after_first_boundary = planner.plan(
        states[14], LimSimAction.DC, map_=map_, time_ms=1500, decision_sequence=decisions
    )

    assert len(states) == config.horizon_steps
    assert states[14].action == LimSimAction.AC
    assert states[15].action == LimSimAction.DC
    assert states[15].speed < states[14].speed
    assert all(
        later.route_progress >= earlier.route_progress for earlier, later in zip(states, states[1:])
    )
    assert after_first_boundary[0].action == LimSimAction.DC
    assert len(after_first_boundary) == config.horizon_steps


def test_frenet_planner_offsets_obstacles_for_later_decision_stages(monkeypatch):
    config = LimSimConfig(horizon_steps=30, terminal_depth=2)
    planner = FrenetTrajectoryPlanner(config)
    agent = AgentDecisionState("ego", 0.0, 0.0, 0.0, 5.0)
    decisions = [
        DecisionStep(
            action=action,
            expected_state=agent.with_updates(action=action),
            expected_frame=expected_frame,
        )
        for action, expected_frame in ((LimSimAction.AC, 1500), (LimSimAction.DC, 3000))
    ]
    obstacle = [
        AgentDecisionState("obstacle", float(step), 2.0, 0.0, 5.0)
        for step in range(config.horizon_steps)
    ]
    calls = []

    def plan_action(
        current, action, map_, obstacle_trajectories, time_ms, steps, target_state=None
    ):
        calls.append((action, time_ms, steps, obstacle_trajectories[0][0].x))
        return planner.fallback.rollout(current, action, map_, steps=steps)

    monkeypatch.setattr(planner, "_plan_action", plan_action)

    planner.plan(
        agent,
        LimSimAction.AC,
        map_=None,
        obstacle_trajectories=[obstacle],
        time_ms=0,
        decision_sequence=decisions,
    )

    assert calls == [(LimSimAction.AC, 0, 15, 0.0), (LimSimAction.DC, 1500, 15, 15.0)]
