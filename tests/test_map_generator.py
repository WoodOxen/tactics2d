import sys

sys.path.append(".")
sys.path.append("..")

import pytest
import logging

logging.basicConfig(level=logging.INFO)

from tactics2d.map.element import Map, Area
from tactics2d.map.generator import RacingTrackGenerator, ParkingLotGenerator
from tactics2d.trajectory.element import State


def test_parking_lot_generator():
    map_generator = ParkingLotGenerator()
    map_ = Map(name="parking_lot", scenario_type="parking")
    start_state, target_area, target_heading = map_generator.generate(map_)

    assert isinstance(start_state, State), "start_state should be a State object."
    assert isinstance(target_area, Area), "target_area should be a Area object."
    assert isinstance(target_heading, float), "target_heading should be a float."

    map_boundary = map_.boundary
    if (
        map_boundary[1] - map_boundary[0] <= 2 * map_generator.map_margin
        or map_boundary[3] - map_boundary[2] <= 2 * map_generator.map_margin
    ):
        logging.error("The start state is the same as the target area.")

    return map_


def test_racing_track_generator():
    return


if __name__ == "__main__":
    map_ = test_parking_lot_generator()
