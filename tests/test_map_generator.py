import sys

sys.path.append(".")
sys.path.append("..")

import logging

import pytest

logging.basicConfig(level=logging.INFO)

from tactics2d.map.element import Area, Map
from tactics2d.map.generator import ParkingLotGenerator, RacingTrackGenerator
from tactics2d.participant.trajectory import State


@pytest.mark.map_generator
def test_parking_lot_generator():
    map_generator = ParkingLotGenerator()
    map_ = Map(name="parking_lot", scenario_type="parking")
    start_state, target_area, target_heading = map_generator.generate(map_)

    assert isinstance(start_state, State), "start_state should be a State object."
    assert isinstance(target_area, Area), "target_area should be a Area object."
    assert isinstance(target_heading, float), "target_heading should be a float."


@pytest.mark.map_generator
def test_racing_track_generator():
    map_generator = RacingTrackGenerator()
    map_ = Map(name="racing_track", scenario_type="racing")
    map_generator.generate(map_)

    assert isinstance(map_.customs["start_state"], State), "start_state should be a State object."


# if __name__ == "__main__":
#     map_generator = ParkingLotGenerator()
#     map_ = Map(name="parking_lot", scenario_type="parking")
