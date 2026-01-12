# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map generator."""


import sys

sys.path.append(".")
sys.path.append("..")

import logging

import pytest
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO)

from tactics2d.map.element import Area, Map
from tactics2d.map.generator import ParkingLotGenerator, RacingTrackGenerator
from tactics2d.participant.trajectory import State
from tactics2d.renderer import MatplotlibRenderer
from tactics2d.sensor import BEVCamera


@pytest.mark.map_generator
def test_parking_lot_generator():
    map_generator = ParkingLotGenerator()
    map_ = Map(name="parking_lot", scenario_type="parking")
    start_state, target_area, target_heading = map_generator.generate(map_)

    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    position = Point(0, 0)
    geometry_data, _, _ = camera.update(0, None, None, None, None, position)

    matplotlib_renderer = MatplotlibRenderer(
        (boundary[0], boundary[1]),
        (boundary[2], boundary[3]),
        resolution=((boundary[1] - boundary[0]) * 100, (boundary[3] - boundary[2]) * 100),
    )

    matplotlib_renderer.update(geometry_data, [position.x, position.y])
    matplotlib_renderer.save_single_frame(save_to="./tests/runtime/parking_lot.png")

    assert isinstance(start_state, State), "start_state should be a State object."
    assert isinstance(target_area, Area), "target_area should be a Area object."
    assert isinstance(target_heading, float), "target_heading should be a float."


@pytest.mark.map_generator
def test_racing_track_generator():
    map_generator = RacingTrackGenerator()
    map_ = Map(name="racing_track", scenario_type="racing")
    map_generator.generate(map_)

    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    position = Point(0, 0)
    geometry_data, _, _ = camera.update(0, None, None, None, None, position)

    matplotlib_renderer = MatplotlibRenderer(
        (boundary[0], boundary[1]),
        (boundary[2], boundary[3]),
        resolution=((boundary[1] - boundary[0]) * 10, (boundary[3] - boundary[2]) * 10),
    )
    matplotlib_renderer.update(geometry_data, position)
    matplotlib_renderer.save_single_frame(save_to="./tests/runtime/racing_track.png")

    assert isinstance(map_.customs["start_state"], State), "start_state should be a State object."


# if __name__ == "__main__":
#     map_generator = ParkingLotGenerator()
#     map_ = Map(name="parking_lot", scenario_type="parking")
