##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_map_generator.py
# @Description: This script is used to test the map generators in the map module.
# @Author: Yueyuan Li
# @Version: 1.0.0


import sys

sys.path.append(".")
sys.path.append("..")

import logging

import pytest

logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

from tactics2d.map.element import Area, Map
from tactics2d.map.generator import ParkingLotGenerator, RacingTrackGenerator
from tactics2d.participant.trajectory import State
from tactics2d.traffic import ScenarioDisplay


@pytest.mark.map_generator
def test_parking_lot_generator():
    map_generator = ParkingLotGenerator()
    map_ = Map(name="parking_lot", scenario_type="parking")
    start_state, target_area, target_heading = map_generator.generate(map_)

    fig, ax = plt.subplots()
    scenario_display = ScenarioDisplay()
    scenario_display.display_map(map_, ax=ax)
    boundary = map_.boundary
    logging.info(f"Boundary: {boundary}")
    ax.set_xlim(boundary[0], boundary[1])
    ax.set_ylim(boundary[2], boundary[3])
    ax.set_aspect("equal")
    fig.savefig("./test/runtime/parking_lot.png")

    assert isinstance(start_state, State), "start_state should be a State object."
    assert isinstance(target_area, Area), "target_area should be a Area object."
    assert isinstance(target_heading, float), "target_heading should be a float."


@pytest.mark.map_generator
def test_racing_track_generator():
    map_generator = RacingTrackGenerator()
    map_ = Map(name="racing_track", scenario_type="racing")
    map_generator.generate(map_)

    fig, ax = plt.subplots()
    scenario_display = ScenarioDisplay()
    scenario_display.display_map(map_, ax=ax)
    boundary = map_.boundary
    logging.info(f"Boundary: {boundary}")
    ax.set_xlim(boundary[0], boundary[1])
    ax.set_ylim(boundary[2], boundary[3])
    ax.set_aspect("equal")
    fig.savefig("./test/runtime/racing_track.png")

    assert isinstance(map_.customs["start_state"], State), "start_state should be a State object."


# if __name__ == "__main__":
#     map_generator = ParkingLotGenerator()
#     map_ = Map(name="parking_lot", scenario_type="parking")
