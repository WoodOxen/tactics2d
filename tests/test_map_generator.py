import sys

sys.path.append(".")
sys.path.append("..")

import logging

import pytest

logging.basicConfig(level=logging.INFO)

from tactics2d.map.element import Area, Map
from tactics2d.map.generator import ParkingLotGenerator, RacingTrackGenerator
from tactics2d.participant.trajectory import State
from tactics2d.traffic import scenario_display

# @pytest.mark.map_generator
# def test_parking_lot_generator():
#     map_generator = ParkingLotGenerator()
#     map_ = Map(name="parking_lot", scenario_type="parking")
#     start_state, target_area, target_heading = map_generator.generate(map_)

#     assert isinstance(start_state, State), "start_state should be a State object."
#     assert isinstance(target_area, Area), "target_area should be a Area object."
#     assert isinstance(target_heading, float), "target_heading should be a float."

# map_boundary = map_.boundary
# if (
#     map_boundary[1] - map_boundary[0] <= 2 * map_generator.map_margin
#     or map_boundary[3] - map_boundary[2] <= 2 * map_generator.map_margin
# ):
#     logging.error("The start state is the same as the target area.")


@pytest.mark.map_generator
def test_racing_track_generator():
    map_generator = RacingTrackGenerator()
    map_ = Map(name="racing_track", scenario_type="racing")
    map_generator.generate(map_)

    assert isinstance(map_.customs["start_state"], State), "start_state should be a State object."


# if __name__ == "__main__":
#     map_generator = ParkingLotGenerator()
#     map_ = Map(name="parking_lot", scenario_type="parking")
