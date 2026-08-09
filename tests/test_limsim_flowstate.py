# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the official LimSim FlowState semantics."""

import numpy as np
import pytest
from shapely.geometry import LineString

from tactics2d.behavior.limsim import LimSimBehaviorModel, LimSimConfig
from tactics2d.behavior.limsim.action import LimSimAction
from tactics2d.behavior.limsim.decision_search import LimSimDecisionSearch
from tactics2d.behavior.limsim.schema import (
    AgentDecisionState,
    DecisionStep,
    JointDecisionState,
    states_to_trajectory,
)
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.search import MCTS


def _agent(agent_id, x=0.0, y=0.0, heading=0.0, speed=5.0, route_progress=0.0):
    return AgentDecisionState(
        agent_id=agent_id,
        x=x,
        y=y,
        heading=heading,
        speed=speed,
        lane_id="A",
        route_lane_ids=("A",),
        route_progress=route_progress,
        available_lane_ids=frozenset({"A"}),
    )


def _joint_state(agents, depth=0, trajectories=None):
    return JointDecisionState(
        agents=tuple(agents),
        depth=depth,
        trajectories=(
            tuple(tuple() for _ in agents) if trajectories is None else tuple(trajectories)
        ),
        active_agent_ids=frozenset(agent.agent_id for agent in agents),
    )


def _vehicle(agent_id, x, y):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    trajectory.add_state(State(frame=0, x=x, y=y, heading=0.0, vx=5.0, vy=0.0))
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=4.5, width=1.8)


def test_official_flowstate_depth_is_separate_from_chained_search_count():
    config = LimSimConfig()

    assert config.terminal_depth == 4
    assert config.max_flowstate_depth == 5

    with pytest.raises(ValueError, match="max_decision_time"):
        LimSimConfig(max_decision_time=0.0).max_flowstate_depth


def test_new_config_fields_preserve_existing_positional_arguments():
    config = LimSimConfig(50, 0.1, 10)
    state = AgentDecisionState("ego", 0.0, 0.0, 0.0, 5.0)

    assert config.mcts_iterations == 10
    assert config.planning_interval == 0.5
    assert config.decision_interval == 3.0
    assert config.default_target_speed == pytest.approx(30.0 / 3.6)
    assert state.behaviour == LimSimAction.KS
    assert state.target_speed == pytest.approx(30.0 / 3.6)


@pytest.mark.parametrize(
    ("action", "expected_x", "expected_speed", "expected_offset"),
    (
        (LimSimAction.KS, 7.5, 5.0, 0.0),
        (LimSimAction.AC, 9.8625, 6.05, 0.0),
        (LimSimAction.DC, 5.1375, 3.95, 0.0),
        (LimSimAction.LCL, 7.5, 5.0, 1.755),
        (LimSimAction.LCR, 7.5, 5.0, -1.755),
    ),
)
def test_flowstate_transition_uses_official_direct_endpoint_formula(
    action, expected_x, expected_speed, expected_offset
):
    follower = LimSimDecisionSearch(LimSimConfig()).follower

    endpoint = follower.flowstate_transition(_agent("ego"), action, map_=None)

    assert endpoint is not None
    assert endpoint.x == pytest.approx(expected_x)
    assert endpoint.speed == pytest.approx(expected_speed)
    assert endpoint.lateral_offset == pytest.approx(expected_offset)
    assert endpoint.action == action


def test_ks_recenters_only_inside_official_lane_width_quarter():
    map_ = Map(name="flowstate_centering")
    map_.add_lane(
        Lane(
            id_="A",
            left_side=LineString([(0.0, 2.0), (20.0, 2.0)]),
            right_side=LineString([(0.0, -2.0), (20.0, -2.0)]),
            custom_tags={"centerline": np.asarray([(0.0, 0.0), (20.0, 0.0)])},
        )
    )
    follower = LimSimDecisionSearch(LimSimConfig()).follower

    centered = follower.flowstate_transition(
        _agent("center", speed=1.0).with_updates(lateral_offset=0.5),
        LimSimAction.KS,
        map_,
    )
    unchanged = follower.flowstate_transition(
        _agent("offset", speed=1.0).with_updates(lateral_offset=1.0),
        LimSimAction.KS,
        map_,
    )

    assert centered is not None and centered.lateral_offset == 0.0
    assert unchanged is not None and unchanged.lateral_offset == pytest.approx(1.0)


def test_candidate_lane_change_uses_fixed_behaviour_not_previous_sampled_action():
    map_ = Map(name="fixed_behaviour")
    for lane_id, center_x in (("A", 0.0), ("B", -2.0)):
        map_.add_lane(
            Lane(
                id_=lane_id,
                left_side=LineString([(center_x - 1.0, 0.0), (center_x - 1.0, 20.0)]),
                right_side=LineString([(center_x + 1.0, 0.0), (center_x + 1.0, 20.0)]),
                custom_tags={"centerline": np.asarray([(center_x, 0.0), (center_x, 20.0)])},
            )
        )
    map_.lanes["A"].left_neighbors.add("B")
    map_.lanes["B"].right_neighbors.add("A")
    search = LimSimDecisionSearch(LimSimConfig())
    agent = _agent("ego", heading=np.pi / 2.0).with_updates(
        lateral_offset=0.6,
        action=LimSimAction.LCR,
        behaviour=LimSimAction.LCL,
    )

    actions = search._candidate_actions(agent, map_)
    endpoint = search.follower.flowstate_transition(agent, LimSimAction.KS, map_)

    assert LimSimAction.LCL in actions
    assert LimSimAction.LCR not in actions
    assert endpoint is not None
    assert endpoint.behaviour == LimSimAction.LCL
    assert endpoint.action == LimSimAction.KS


def test_dense_output_is_generated_from_timed_decision_steps_after_search():
    config = LimSimConfig(horizon_steps=30, dt=0.1)
    search = LimSimDecisionSearch(config)
    agent = _agent("ego")
    decisions = (
        DecisionStep(
            action=LimSimAction.AC,
            expected_state=agent.with_updates(action=LimSimAction.AC),
            expected_frame=1500,
        ),
        DecisionStep(
            action=LimSimAction.DC,
            expected_state=agent.with_updates(action=LimSimAction.DC),
            expected_frame=3000,
        ),
    )

    states = search._dense_rough_trajectory(
        agent, decisions, start_frame=0, map_=None, fallback_action=LimSimAction.KS
    )
    trajectory = states_to_trajectory(agent.agent_id, states, start_frame=0, dt=config.dt)

    assert len(states) == 30
    assert all(state.action == LimSimAction.AC for state in states[:15])
    assert all(state.action == LimSimAction.DC for state in states[15:])
    assert trajectory.frames[0] == 100
    assert trajectory.frames[-1] == 3000


def test_incomplete_stage_removes_only_that_vehicle_and_preserves_history_order(monkeypatch):
    config = LimSimConfig(horizon_steps=75)
    search = LimSimDecisionSearch(config)
    agents = (_agent("left", x=-30.0), _agent("exit"), _agent("right", x=30.0))
    calls = []

    def transition(agent, action, map_):
        calls.append(agent.agent_id)
        if agent.agent_id == "exit":
            return None
        return agent.with_updates(x=agent.x + 1.0, action=action)

    monkeypatch.setattr(search.follower, "flowstate_transition", transition)

    first_children = search._expand(_joint_state(agents), map_=None)
    first = first_children[0]

    assert len(first_children) == 27
    assert first.agent_ids == ("left", "exit", "right")
    assert first.resolved_active_agent_ids == frozenset({"left", "right"})
    assert [len(history) for history in first.trajectories] == [1, 0, 1]
    assert first.agents[1] is agents[1]

    calls.clear()
    second_children = search._expand(first, map_=None)
    second = second_children[0]

    assert len(second_children) == 9
    assert calls.count("left") == 3
    assert calls.count("exit") == 0
    assert calls.count("right") == 3
    assert second.agent_ids == first.agent_ids
    assert second.resolved_active_agent_ids == first.resolved_active_agent_ids
    assert [len(history) for history in second.trajectories] == [2, 0, 2]


def test_empty_active_layer_is_terminal_and_removed_vehicle_never_reenters(monkeypatch):
    config = LimSimConfig(candidate_actions=(LimSimAction.KS,))
    search = LimSimDecisionSearch(config)
    agent = _agent("exit")
    calls = []

    def transition(agent, action, map_):
        calls.append(agent.agent_id)
        return None

    monkeypatch.setattr(search.follower, "flowstate_transition", transition)

    empty_layer = search._expand(_joint_state((agent,)), map_=None)[0]

    assert empty_layer.depth == 1
    assert empty_layer.active_agents == ()
    assert empty_layer.trajectories == ((),)
    assert search._expand(empty_layer, map_=None) == []
    assert search._simulate_step(empty_layer, map_=None) is None
    assert calls == ["exit"]


def test_vehicle_at_exact_lane_end_remains_until_it_crosses_the_end():
    map_ = Map(name="flowstate_lane_end")
    map_.add_lane(
        Lane(
            id_="A",
            left_side=LineString([(-1.0, 0.0), (-1.0, 4.0)]),
            right_side=LineString([(1.0, 0.0), (1.0, 4.0)]),
            custom_tags={"centerline": np.asarray([(0.0, 0.0), (0.0, 4.0)])},
        )
    )
    config = LimSimConfig(
        dt=0.5,
        decision_resolution=1.5,
        candidate_actions=(LimSimAction.KS,),
    )
    search = LimSimDecisionSearch(config)
    agent = _agent("ego", y=1.0, heading=np.pi / 2.0, speed=2.0, route_progress=1.0)
    root = _joint_state((agent,))

    endpoint = search.follower.flowstate_transition(agent, LimSimAction.KS, map_)
    at_end = search._next_state_from_endpoints(root, {0: endpoint}, (), (), ())
    beyond = search.follower.flowstate_transition(at_end.agents[0], LimSimAction.KS, map_)
    after_exit = search._next_state_from_endpoints(
        at_end,
        {0: beyond},
        (),
        (),
        (),
    )

    assert endpoint is not None
    assert endpoint.route_progress == pytest.approx(4.0)
    assert at_end.resolved_active_agent_ids == frozenset({"ego"})
    assert beyond is None
    assert after_exit.active_agents == ()


def test_vehicle_at_exact_lane_end_does_not_enter_successor_early():
    map_ = Map(name="flowstate_successor_boundary")
    for lane_id, start, end in (("A", 0.0, 4.0), ("B", 4.0, 8.0)):
        map_.add_lane(
            Lane(
                id_=lane_id,
                left_side=LineString([(-1.0, start), (-1.0, end)]),
                right_side=LineString([(1.0, start), (1.0, end)]),
                custom_tags={"centerline": np.asarray([(0.0, start), (0.0, end)])},
            )
        )
    map_.lanes["A"].successors.add("B")
    follower = LimSimDecisionSearch(LimSimConfig()).follower
    agent = _agent("ego", y=1.0, heading=np.pi / 2.0, speed=2.0, route_progress=1.0).with_updates(
        route_lane_ids=("A", "B"), available_lane_ids=frozenset({"A", "B"})
    )

    at_end = follower.flowstate_transition(agent, LimSimAction.KS, map_)
    in_successor = follower.flowstate_transition(at_end, LimSimAction.KS, map_)

    assert at_end is not None
    assert at_end.lane_id == "A"
    assert at_end.route_progress == pytest.approx(4.0)
    assert in_successor is not None
    assert in_successor.lane_id == "B"
    assert in_successor.route_progress == pytest.approx(3.0)


def test_removed_vehicle_has_no_exit_action_but_keeps_official_reward(monkeypatch):
    config = LimSimConfig(
        horizon_steps=30,
        terminal_depth=1,
        candidate_actions=(LimSimAction.KS,),
    )
    search = LimSimDecisionSearch(config)
    exiting = _agent("exit", x=-30.0)
    staying = _agent("stay", x=30.0)

    def transition(agent, action, map_):
        if agent.agent_id == "exit":
            return None
        return agent.with_updates(x=agent.x + 1.0, action=action)

    def fake_search(self, root, max_try):
        node = root
        while not self.terminal_fn(node.state):
            child = self.Node(self.expand_fn(node.state)[0], parent=node)
            child.visits = 1
            child.total_reward = 1.0
            node.children = [child]
            node.visits += 1
            node = child
        return root

    monkeypatch.setattr(search.follower, "flowstate_transition", transition)
    monkeypatch.setattr(MCTS, "search", fake_search)

    _, _, decisions, root = search._plan_with_decisions([exiting, staying], map_=None)
    removed_reward = search.reward.evaluate(
        [exiting], {exiting.agent_id: []}, max_decision_num=config.max_flowstate_depth + 1
    )

    assert decisions["exit"] == []
    assert len(decisions["stay"]) == config.max_flowstate_depth
    assert root.children[0].state.resolved_active_agent_ids == frozenset({"stay"})
    assert removed_reward == pytest.approx(0.8 + 0.4 / 6.0)


def test_collision_checks_only_flowstate_stages_and_not_background_pairs():
    config = LimSimConfig()
    reward = LimSimDecisionSearch(config).reward
    controlled = _agent("ego", speed=0.0)
    controlled_trajectory = [controlled.with_updates(action=LimSimAction.KS)] * int(
        round(config.decision_resolution / config.dt)
    )
    far = _agent("background", x=100.0, speed=0.0)
    middle_only = [far] * len(controlled_trajectory)
    middle_only[len(middle_only) // 2] = far.with_updates(x=0.0)

    middle_reward = reward.evaluate(
        [controlled],
        {controlled.agent_id: controlled_trajectory},
        (middle_only,),
        obstacle_initial_states=(far,),
    )
    endpoint_collision = list(middle_only)
    endpoint_collision[-1] = far.with_updates(x=0.0)
    endpoint_reward = reward.evaluate(
        [controlled],
        {controlled.agent_id: controlled_trajectory},
        (endpoint_collision,),
        obstacle_initial_states=(far,),
    )
    overlapping_background = [_agent("other_background", x=100.0, speed=0.0)] * len(
        controlled_trajectory
    )
    background_pair_reward = reward.evaluate(
        [controlled],
        {controlled.agent_id: controlled_trajectory},
        (middle_only, overlapping_background),
        obstacle_initial_states=(far, overlapping_background[0]),
    )

    assert middle_reward == 1.0
    assert endpoint_reward == 0.0
    assert background_pair_reward == 1.0


def test_collision_categories_use_official_root_and_completed_decision_indexes():
    reward = LimSimDecisionSearch(LimSimConfig()).reward
    controlled = _agent("ego", speed=0.0)
    close = _agent("other", speed=0.0)
    far = close.with_updates(x=100.0)
    completed = (
        DecisionStep(LimSimAction.KS, close, expected_frame=1500),
        DecisionStep(LimSimAction.KS, far, expected_frame=3000),
    )
    enlarged_only = close.with_updates(x=6.0)

    assert reward.has_collision_at_stage(
        (controlled,), stage_index=0, obstacle_initial_states=(close,)
    )
    assert reward.has_collision_at_stage(
        (controlled,), stage_index=0, completed_decisions=(completed,)
    )
    assert not reward.has_collision_at_stage(
        (controlled,), stage_index=1, completed_decisions=(completed,)
    )
    assert not controlled.footprint.intersects(enlarged_only.footprint)
    assert reward._decision_vehicle_collides(controlled, enlarged_only)


def test_terminal_time_layer_skips_collision_check():
    config = LimSimConfig()
    search = LimSimDecisionSearch(config)
    agent = _agent("ego", speed=0.0)
    endpoint = agent.with_updates(action=LimSimAction.KS)
    far = _agent("background", x=100.0, speed=0.0)
    preterminal_state = _joint_state(
        (agent,),
        depth=config.max_flowstate_depth - 2,
        trajectories=((agent,) * (config.max_flowstate_depth - 2),),
    )
    preterminal_obstacle = [far] * 60
    preterminal_obstacle[-1] = far.with_updates(x=0.0)
    colliding = search._next_state_from_endpoints(
        preterminal_state,
        {0: endpoint},
        (preterminal_obstacle,),
        (far,),
        (),
    )

    state = _joint_state(
        (agent,),
        depth=config.max_flowstate_depth - 1,
        trajectories=((agent,) * (config.max_flowstate_depth - 1),),
    )
    obstacle = [far] * 75
    obstacle[-1] = far.with_updates(x=0.0)

    assert colliding.depth == config.max_flowstate_depth - 1
    assert colliding.has_collision
    assert search.reward.has_collision_at_stage(
        (endpoint,), (obstacle,), stage_index=config.max_flowstate_depth
    )

    terminal = search._next_state_from_endpoints(
        state,
        {0: endpoint},
        (obstacle,),
        (far,),
        (),
    )

    assert terminal.depth == config.max_flowstate_depth
    assert terminal.active_agents
    assert not terminal.has_collision


def test_later_groups_see_only_background_and_accepted_prior_decisions(monkeypatch):
    config = LimSimConfig(horizon_steps=5, use_frenet_refinement=False)
    model = LimSimBehaviorModel(config)
    participants = {
        1: _vehicle(1, 0.0, 0.0),
        2: _vehicle(2, 1.0, 0.0),
        3: _vehicle(3, 2.0, 0.0),
        9: _vehicle(9, 50.0, 0.0),
    }
    scene_states = {
        agent_id: _agent(agent_id, x=participants[agent_id].get_state(0).x)
        for agent_id in participants
    }
    prediction_calls = []
    planner_calls = []

    def build_scene(participants, map_, frame, agent_ids=None, route_map=None):
        return {agent_id: scene_states[agent_id] for agent_id in (agent_ids or ())}

    def predict_background(participants, map_, frame, background_states, **kwargs):
        prediction_calls.append(tuple(background_states))
        return {9: [background_states[9]] * config.horizon_steps}

    def plan_with_decisions(
        agents,
        map_,
        obstacle_trajectories=(),
        start_frame=0,
        obstacle_initial_states=(),
        completed_decisions=(),
    ):
        agent = agents[0]
        planner_calls.append(
            (
                tuple(trajectory[0].agent_id for trajectory in obstacle_trajectories),
                tuple(state.agent_id for state in obstacle_initial_states),
                tuple(decisions[0].expected_state.agent_id for decisions in completed_decisions),
            )
        )
        trajectory = [
            agent.with_updates(x=agent.x + 0.1 * (index + 1), action=LimSimAction.KS)
            for index in range(config.horizon_steps)
        ]
        decisions = []
        if agent.agent_id != 2:
            decisions = [
                DecisionStep(
                    LimSimAction.KS,
                    trajectory[-1],
                    expected_frame=start_frame + int(config.decision_resolution * 1000),
                )
            ]
        return (
            {agent.agent_id: LimSimAction.KS},
            {agent.agent_id: trajectory},
            {agent.agent_id: decisions},
            object(),
        )

    monkeypatch.setattr(model.scene_builder, "build", build_scene)
    monkeypatch.setattr(model, "_predict_obstacle_trajectories", predict_background)
    monkeypatch.setattr(
        model.interaction_graph, "build_groups", lambda states, map_: [[1], [2], [3]]
    )
    monkeypatch.setattr(model.decision_search, "_plan_with_decisions", plan_with_decisions)

    model.plan(
        participants,
        map_=None,
        frame=0,
        route_map={},
        roi_center=(0.0, 0.0),
        roi_radius=10.0,
        roi_outer_radius=100.0,
    )

    assert prediction_calls == [(9,)]
    assert planner_calls == [
        ((9,), (9,), ()),
        ((9,), (9,), (1,)),
        ((9,), (9,), (1,)),
    ]
