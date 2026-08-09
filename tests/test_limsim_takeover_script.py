# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for deterministic selection in the LimSim benchmark entry point."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import limsim_takeover_single as benchmark


def test_render_gif_removes_participant_after_trajectory_ends(monkeypatch, tmp_path):
    class FakeTrajectory:
        def __init__(self, states):
            self.history_states = states

        def has_state(self, frame):
            return frame in self.history_states

        def get_state(self, frame):
            return self.history_states[frame]

    class FakeVehicle:
        def __init__(self, states):
            self.color = "cyan"
            self.trajectory = FakeTrajectory(states)

    class FakeCamera:
        instances = []

        def __init__(self, **_):
            self.calls = []
            self._position = None
            self.instances.append(self)

        def update(
            self,
            frame,
            participants,
            participant_ids,
            previous_road_ids,
            previous_participant_ids,
            position,
        ):
            del participants, position
            current_participant_ids = set(participant_ids)
            self.calls.append((frame, set(previous_road_ids), set(previous_participant_ids)))
            geometry_data = {
                "participant_data": {
                    "participant_id_to_create": list(
                        current_participant_ids - previous_participant_ids
                    ),
                    "participant_id_to_remove": list(
                        previous_participant_ids - current_participant_ids
                    ),
                    "participants": [],
                }
            }
            return geometry_data, {"road"}, current_participant_ids

    class FakeAxes:
        def set_xlim(self, *_):
            pass

        def set_ylim(self, *_):
            pass

        def set_title(self, *_args, **_kwargs):
            pass

    class FakeCanvas:
        def draw(self):
            pass

        def tostring_rgb(self):
            return bytes((0, 0, 0))

        def get_width_height(self):
            return 1, 1

    class FakeRenderer:
        instances = []

        def __init__(self, **_):
            self.ax = FakeAxes()
            self.fig = SimpleNamespace(canvas=FakeCanvas())
            self.visible_participants = set()
            self.updates = []
            self.instances.append(self)

        def enable_trajectory_gradient(self):
            pass

        def update(self, geometry_data):
            participant_data = geometry_data["participant_data"]
            self.visible_participants -= set(participant_data["participant_id_to_remove"])
            self.visible_participants |= set(participant_data["participant_id_to_create"])
            self.updates.append(participant_data)

        def _remove_trajectory_lines(self):
            pass

        def destroy(self):
            pass

    state = SimpleNamespace(x=0.0, y=0.0)
    participants = {
        1: FakeVehicle({0: state, 1: state}),
        2: FakeVehicle({0: state}),
    }
    monkeypatch.setattr(benchmark, "Vehicle", FakeVehicle)
    monkeypatch.setattr(benchmark, "BEVCamera", FakeCamera)
    monkeypatch.setattr(benchmark, "MatplotlibRenderer", FakeRenderer)

    benchmark.render_gif(
        participants,
        SimpleNamespace(roadlines={}),
        [0, 1],
        ego_id=1,
        ego_plans={},
        out_path=tmp_path / "lifecycle.gif",
        fps=10,
        show_history=False,
    )

    camera = FakeCamera.instances[0]
    renderer = FakeRenderer.instances[0]
    assert camera.calls == [(0, set(), set()), (1, {"road"}, {1, 2})]
    assert renderer.updates[1]["participant_id_to_remove"] == [2]
    assert renderer.visible_participants == {1}


@pytest.mark.parametrize(
    ("last_successors", "expected_route"),
    [
        ({"C"}, ("A", "B", "C")),
        ({"C", "D"}, ("A", "B")),
    ],
)
def test_route_adapter_extends_only_unambiguous_topology(
    monkeypatch, last_successors, expected_route
):
    class FakeTrajectory:
        def __init__(self, state):
            self._state = state

        def has_state(self, frame):
            return frame == 0

        def get_state(self, frame):
            assert frame == 0
            return self._state

    class FakeVehicle:
        def __init__(self, state):
            self.trajectory = FakeTrajectory(state)

    state = SimpleNamespace(x=0.0, y=0.0, heading=0.0, speed=5.0)
    participants = {1: FakeVehicle(state)}
    map_ = SimpleNamespace(
        lanes={
            "A": SimpleNamespace(successors={"B"}),
            "B": SimpleNamespace(successors=last_successors),
            "C": SimpleNamespace(successors=set()),
            "D": SimpleNamespace(successors=set()),
        }
    )
    route_map = {}
    monkeypatch.setattr(benchmark, "Vehicle", FakeVehicle)
    monkeypatch.setattr(benchmark, "extract_lane_sequence", lambda *_args, **_kwargs: ["A", "B"])

    benchmark.update_local_route_map(
        participants,
        map_,
        frame=0,
        ego_id=1,
        roi_outer_radius=100.0,
        route_map=route_map,
        max_routes=10,
    )

    assert route_map == {1: expected_route}


def test_fixed_takeover_is_filtered_through_discovery_and_keeps_ego(monkeypatch):
    def _discover(info):
        first = dict(info, ego_id=100)
        selected = dict(info, ego_id=215)
        yield first, 404780
        yield selected, 405280

    monkeypatch.setattr(benchmark, "discover_clips", _discover)
    global_state = random.getstate()

    clips = benchmark.sample_clips(
        [{"dataset": "rounD", "file_id": 12}],
        num_clips=1,
        seed=1509,
        takeover_frame=405280,
    )

    assert [(info["ego_id"], frame) for info, frame in clips] == [(215, 405280)]
    assert random.getstate() == global_state


def test_planner_seed_is_stable_and_random_state_can_be_replayed():
    info = {"dataset": "rounD", "file_id": 12, "ego_id": 215}
    planner_seed = benchmark._planner_seed(42, "rounD", info, 405280, "fast_preview")

    assert planner_seed == 749248050
    assert planner_seed != benchmark._planner_seed(42, "rounD", info, 405280, "balanced_demo")

    benchmark._reset_planner_random_state(planner_seed)
    first = (random.random(), np.random.random())
    benchmark._reset_planner_random_state(planner_seed)
    second = (random.random(), np.random.random())
    assert first == second


@pytest.mark.parametrize(
    ("datasets", "file_id", "takeover_frame", "message"),
    [
        (["rounD", "inD"], 12, None, "exactly one dataset"),
        (["WOMD"], 12, None, "LevelX clips only"),
        (["rounD"], None, 405280, "requires --file-id"),
    ],
)
def test_levelx_debug_selector_boundaries(datasets, file_id, takeover_frame, message):
    with pytest.raises(ValueError, match=message):
        benchmark.cmd_benchmark(
            datasets=datasets,
            num_scenarios=1,
            config_names=["fast_preview"],
            file_id=file_id,
            takeover_frame=takeover_frame,
        )
