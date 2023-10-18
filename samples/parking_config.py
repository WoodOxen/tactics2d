import numpy as np

from shapely.geometry import LinearRing
from tactics2d.participant.element.defaults import VEHICLE_MODEL
from tactics2d.physics.single_track_kinematics import SingleTrackKinematics

# TODO: these parameters should also be modified in parking.py when changed
lidar_num = 120  #
lidar_range = 20.0  #
max_speed = 2  #
VALID_STEER = [-0.75, 0.75]

# the vehicle parameter
vehicle_type = "medium_car"

WHEEL_BASE = VEHICLE_MODEL[vehicle_type]["wheel_base"]  # wheelbase
LENGTH = VEHICLE_MODEL[vehicle_type]["length"]
WIDTH = VEHICLE_MODEL[vehicle_type]["width"]
VehicleBox = LinearRing(
    [
        [0.5 * LENGTH, -0.5 * WIDTH],
        [0.5 * LENGTH, 0.5 * WIDTH],
        [-0.5 * LENGTH, 0.5 * WIDTH],
        [-0.5 * LENGTH, -0.5 * WIDTH],
    ]
)
physic_model = SingleTrackKinematics(
    dist_front_hang=0.5 * LENGTH - VEHICLE_MODEL[vehicle_type]["front_overhang"],
    dist_rear_hang=0.5 * LENGTH - VEHICLE_MODEL[vehicle_type]["rear_overhang"],
    steer_range=tuple(VALID_STEER),
    speed_range=(-max_speed, max_speed),
)

# action space of action mask
PRECISION = 10
discrete_actions = []
for i in np.arange(
    VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1] / PRECISION), -VALID_STEER[-1] / PRECISION
):
    discrete_actions.append([i, max_speed])
for i in np.arange(
    VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1] / PRECISION), -VALID_STEER[-1] / PRECISION
):
    discrete_actions.append([i, -max_speed])
N_DISCRETE_ACTION = len(discrete_actions)
