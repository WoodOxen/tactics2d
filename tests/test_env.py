import sys

sys.path.append(".")
sys.path.append("..")

import pytest
import logging

logging.basicConfig(level=logging.INFO)
# from matplotlib.patches import Polygon
# import matplotlib.pyplot as plt

from tactics2d.participant.element import Vehicle
from tactics2d.map.element import Map
from tactics2d.map.generator import RacingTrackGenerator, ParkingLotGenerator
from tactics2d.envs import RacingEnv, ParkingEnv


@pytest.mark.skip(reason="not implemented")
def test_parking_env():
    return


@pytest.mark.skip(reason="not implemented")
def test_racing_env():
    return


if __name__ == "__main__":
    vehicle = Vehicle(
        id_=0,
        type_="sedan",
        steering_angle_range=(-0.75, 0.75),
        steering_velocity_range=(-0.5, 0.5),
        speed_range=(-10, 100),
        accel_range=(-1, 1),
    )
    # print(vehicle.length, vehicle.width)
    map_ = Map(name="ParkingLot", scenario_type="parking")
    map_generator = ParkingLotGenerator((vehicle.length, vehicle.width), 0.5)
    start_state, obstacles = map_generator.generate(map_)
    vehicle.reset(start_state)

    fig, ax = plt.subplots()
    for obstacle in obstacles:
        ax.add_patch(Polygon(list(obstacle.shape.coords), color="green"))
    ax.add_patch(Polygon(list(vehicle.get_pose().coords), color="blue"))
    ax.add_patch(Polygon(map_.areas[0].shape(outer_only=True), color="yellow"))

    ax.autoscale_view()
    ax.set_aspect("equal")
    plt.show()
