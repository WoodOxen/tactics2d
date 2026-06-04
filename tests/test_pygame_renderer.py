# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the pygame renderer compatibility layer."""


import os

import numpy as np
import pygame
import pytest
from shapely.geometry import LineString, Point, Polygon

from tactics2d.map.element import Area, Lane, Map, RoadLine
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State


@pytest.fixture(autouse=True)
def use_dummy_video_driver(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    yield
    pygame.quit()


def _build_map() -> Map:
    map_ = Map("pygame-test")
    map_.add_area(
        Area(
            "1",
            Polygon(
                [(-8, -8), (8, -8), (8, 8), (-8, 8)], holes=[[(-1, -1), (1, -1), (1, 1), (-1, 1)]]
            ),
            color="dark-gray",
        )
    )
    map_.add_area(Area("2", Polygon([(3, -1), (4, -1), (4, 1), (3, 1)]), type_="obstacle"))
    map_.add_lane(
        Lane(
            "3",
            left_side=LineString([(-6, -3), (6, -3)]),
            right_side=LineString([(-6, -5), (6, -5)]),
        )
    )
    map_.add_roadline(
        RoadLine("4", LineString([(-6, 0), (6, 0)]), type_="line_thick", color="white")
    )
    map_.add_roadline(RoadLine("5", LineString([(-6, 3), (6, 3)]), type_="roadline"))
    return map_


def _vehicle(id_, x, y, heading=0.0):
    vehicle = Vehicle(id_, length=4.0, width=2.0, color="light-turquoise")
    vehicle.add_state(State(frame=0, x=x, y=y, heading=heading))
    return vehicle


def _pedestrian(id_, x, y):
    pedestrian = Pedestrian(id_, color="light-blue")
    pedestrian.add_state(State(frame=0, x=x, y=y, heading=0.0))
    return pedestrian


def _cyclist(id_, x, y):
    cyclist = Cyclist(id_, color="light-orange")
    cyclist.add_state(State(frame=0, x=x, y=y, heading=0.0))
    return cyclist


def test_pygame_render_manager_smoke(monkeypatch):
    from tactics2d.renderer import RenderManager

    class DummySensor:
        id_ = 0
        off_screen = False
        window_size = (16, 16)

        def __init__(self):
            self.surface = pygame.Surface(self.window_size)
            self.bind_id = None

        def set_bind_id(self, bind_id):
            self.bind_id = bind_id

        def update(self, participants, participant_ids, frame=None, position=None, heading=None):
            self.surface.fill((255, 255, 255))

        def get_observation(self):
            return np.zeros((16, 16, 3), dtype=np.uint8)

    manager = RenderManager(fps=60, windows_size=(32, 32), off_screen=True)
    try:
        sensor = DummySensor()
        manager.add_sensor(sensor)
        manager.update({}, [], 0)
        manager.render()
        assert len(manager.get_observation()) == 1
        assert manager.graphic_driver == os.environ["SDL_VIDEODRIVER"]
    finally:
        manager.close()


def test_pygame_render_manager_binds_and_removes_missing_sensor():
    from tactics2d.renderer import RenderManager

    class TrackingSensor:
        def __init__(self, id_, off_screen=False):
            self.id_ = id_
            self.off_screen = off_screen
            self.window_size = (12, 12)
            self.surface = pygame.Surface(self.window_size)
            self.bind_id = None
            self.calls = []

        def set_bind_id(self, bind_id):
            self.bind_id = bind_id

        def update(self, participants, participant_ids, frame=None, position=None, heading=None):
            self.calls.append((frame, position, heading, tuple(participant_ids)))

        def get_observation(self):
            return np.ones((12, 12, 3), dtype=np.uint8) * self.id_

    manager = RenderManager(fps=30, windows_size=(48, 48), layout_style="block", off_screen=True)
    try:
        sensor = TrackingSensor(1)
        hidden_sensor = TrackingSensor(2, off_screen=True)
        vehicle = _vehicle(10, 0, 0, heading=0.2)

        manager.add_sensor(sensor, main_sensor=True)
        manager.add_sensor(hidden_sensor)
        manager.bind(1, 10)
        manager.update({10: vehicle}, [10], 0)

        assert manager.is_bound(1)
        assert manager.get_bind_id(1) == 10
        assert isinstance(sensor.calls[-1][1], Point)
        assert sensor.calls[-1][2] == pytest.approx(0.2)
        assert len(manager.get_observation()) == 2

        manager.update({10: vehicle}, [10], 100)
        assert not manager.is_bound(1)
        assert len(manager.get_observation()) == 1

        manager.remove_sensor(2)
        assert manager.get_observation() == []
    finally:
        manager.close()


def test_pygame_render_manager_rejects_unknown_layout():
    from tactics2d.renderer import RenderManager

    with pytest.raises(ValueError):
        RenderManager(layout_style="diagonal")


def test_top_down_camera_renders_map_and_participants():
    from tactics2d.renderer import TopDownCamera

    map_ = _build_map()
    participants = {
        1: _vehicle(1, 0, 0),
        2: _pedestrian(2, 2, 2),
        3: _cyclist(3, -2, 2),
        4: _vehicle(4, 100, 100),
    }
    camera = TopDownCamera(
        0, map_, perception_range=(10, 10, 10, 10), window_size=(80, 80), off_screen=True
    )

    camera.update(participants, [1, 2, 3, 4], frame=0)
    first_observation = camera.get_observation()
    assert first_observation.shape == (80, 80, 3)
    assert camera.map_rendered

    camera.update(participants, [1, 2, 3, 4], frame=0, position=Point(0, 0), heading=0.0)
    bound_observation = camera.get_observation()
    assert bound_observation.shape == (80, 80, 3)
    assert len(np.unique(bound_observation.reshape(-1, 3), axis=0)) > 1


def test_single_line_lidar_detects_obstacles_and_renders_points():
    from tactics2d.renderer import SingleLineLidar

    map_ = _build_map()
    lidar = SingleLineLidar(
        1,
        map_,
        perception_range=10.0,
        freq_scan=10.0,
        freq_detect=80.0,
        window_size=(80, 80),
        off_screen=False,
    )
    lidar.set_bind_id(1)
    participants = {1: _vehicle(1, 0, 0), 2: _vehicle(2, 5, 0)}

    lidar.update(participants, [1, 2], frame=0, position=Point(0, 0), heading=0.0)
    observation = lidar.get_observation()

    assert observation.shape == (8,)
    assert np.isfinite(observation).any()
    assert lidar.surface.get_size() == (80, 80)


def test_single_line_lidar_reports_no_hits_without_obstacles():
    from tactics2d.renderer import SingleLineLidar

    map_ = Map("empty-lidar")
    map_.add_roadline(RoadLine("1", LineString([(-1, 0), (1, 0)])))
    lidar = SingleLineLidar(
        2, map_, perception_range=5.0, freq_scan=10.0, freq_detect=40.0, window_size=(40, 40)
    )

    lidar.update({}, [], frame=0, position=Point(0, 0), heading=0.0)

    assert np.isinf(lidar.get_observation()).all()
