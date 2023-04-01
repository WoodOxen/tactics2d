import sys

sys.path.append(".")
sys.path.append("..")

import os
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from shapely.geometry import LineString, LinearRing
from shapely.affinity import affine_transform, rotate
import pygame
import pytest

from tactics2d.trajectory.element import State
from tactics2d.physics import SingleTrackKinematics, SingleTrackDynamics


# Prototype: Volkswagen Polo(AW/BZ) (https://en.wikipedia.org/wiki/Volkswagen_Polo)
VEHICLE_PARAMS = {
    "length": 4.053,
    "width": 1.751,
    "height": 1.461,
    "wheel_base": 2.548,
    "front_overhang": 0.824,
    "rear_overhang": 0.681,
    "steer_range": (-0.5236, 0.5236),
    "angular_velocity_range": (-1.0472, 1.0472),
    "speed_range": (-2.78, 52.8),
    "accel_range": (-9.8, 3.517),
    "0_100_km/h": 12,
}

# fmt: off
ACTION_LIST = [
    (0, 0), (0, 1), (0, -1), (0.3, 0), (0.3, 0.5), (0.3, -0.5),
    (0, 2), (0, -2), (0, 5), (0, -5), (-0.3, 0), (-0.3, 0.5), (-0.3, -0.5),
    (1, 0), (1, 0.5), (1, -0.5), (-1, 0), (-1, 0.5), (-1, -0.5),
]
# fmt: on


class Visualizer:
    vehicle_bbox = [
        [0.5 * VEHICLE_PARAMS["length"], -0.5 * VEHICLE_PARAMS["width"]],
        [0.5 * VEHICLE_PARAMS["length"], 0.5 * VEHICLE_PARAMS["width"]],
        [-0.5 * VEHICLE_PARAMS["length"], 0.5 * VEHICLE_PARAMS["width"]],
        [-0.5 * VEHICLE_PARAMS["length"], -0.5 * VEHICLE_PARAMS["width"]],
    ]

    front_axle = [
        [
            0.5 * VEHICLE_PARAMS["length"] - VEHICLE_PARAMS["front_overhang"],
            0.5 * (VEHICLE_PARAMS["width"] + 0.01),
        ],
        [
            0.5 * VEHICLE_PARAMS["length"] - VEHICLE_PARAMS["front_overhang"],
            -0.5 * (VEHICLE_PARAMS["width"] + 0.01),
        ],
    ]

    rear_axle = [
        [
            -0.5 * VEHICLE_PARAMS["length"] + VEHICLE_PARAMS["rear_overhang"],
            0.5 * (VEHICLE_PARAMS["width"] + 0.01),
        ],
        [
            -0.5 * VEHICLE_PARAMS["length"] + VEHICLE_PARAMS["rear_overhang"],
            -0.5 * (VEHICLE_PARAMS["width"] + 0.01),
        ],
    ]

    # wheel order: left front, right front, left rear, right rear
    # fmt: off
    wheels = [
        ([front_axle[0][0] + 0.225, front_axle[0][1]], [front_axle[0][0] - 0.225, front_axle[0][1]]),
        ([front_axle[1][0] + 0.225, front_axle[1][1]], [front_axle[1][0] - 0.225, front_axle[1][1]]),
        ([rear_axle[0][0] + 0.225, rear_axle[0][1]], [rear_axle[0][0] - 0.225, rear_axle[0][1]]),
        ([rear_axle[1][0] + 0.225, rear_axle[1][1]], [rear_axle[1][0] - 0.225, rear_axle[1][1]]),
    ]
    # fmt: on

    def __init__(self, fps = 60):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()
        self.fps = fps

    def _scale(self, geometry, scale_factor = 10) -> list:
        point_list = np.array(list(geometry.coords))
        point_list = point_list * scale_factor
        return point_list

    def _draw_vehicle(self, state: State, action: tuple):
        steer, _ = action

        # draw vehicle bounding box
        transform_matrix = [
            np.cos(state.heading),
            -np.sin(state.heading),
            np.sin(state.heading),
            np.cos(state.heading),
            state.x,
            state.y,
        ]
        vehicle_bbox = affine_transform(LinearRing(self.vehicle_bbox), transform_matrix)

        pygame.draw.polygon(
            self.screen, (0, 245, 255, 100), self._scale(vehicle_bbox)
        )

        # draw axles
        front_axle = affine_transform(LineString(self.front_axle), transform_matrix)
        rear_axle = affine_transform(LineString(self.rear_axle), transform_matrix)
        pygame.draw.lines(self.screen, (0, 0, 0), False, self._scale(front_axle))
        pygame.draw.lines(self.screen, (0, 0, 0), False, self._scale(rear_axle))

        # draw wheels
        for wheel in self.wheels:
            wheel = affine_transform(LineString(wheel), transform_matrix)
            wheel = rotate(wheel, steer)
            pygame.draw.lines(self.screen, (0, 0, 0), False, self._scale(wheel), 2)

    def update(self, state: State, action: tuple, trajectory: list):
        self.screen.fill((255, 255, 255))
        self._draw_vehicle(state, action)
        pygame.draw.lines(self.screen, (100, 100, 100), False, self._scale(LineString(trajectory)), 1)
        pygame.display.update()
        self.clock.tick(self.fps)


# @pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display server")
# @pytest.mark.parametrize(
#     "steer_range, speed_range, angular_velocity_range, accel_range, delta_t", []
# )
def test_single_track_kinematic(
    steer_range, speed_range, angular_velocity_range, accel_range, delta_t
):
    vehicle_model = SingleTrackKinematics(
        VEHICLE_PARAMS["width"] / 2 - VEHICLE_PARAMS["front_overhang"],
        VEHICLE_PARAMS["width"] / 2 - VEHICLE_PARAMS["rear_overhang"],
        steer_range=steer_range,
        speed_range=speed_range,
        angular_velocity_range=angular_velocity_range,
        accel_range=accel_range,
        delta_t=delta_t,
    )

    step = 0.1
    visualizer = Visualizer(10)
    state = State(frame=0, x=40, y=10, heading=0, speed=0)
    trajectory = [(state.x, state.y)]

    for action in ACTION_LIST:
        for _ in range(20):
            new_state, true_action = vehicle_model.step(state, action, step)
            trajectory.append((new_state.x, new_state.y))
            visualizer.update(new_state, true_action, trajectory)
            state = new_state


if __name__ == "__main__":
    test_single_track_kinematic(
        steer_range=VEHICLE_PARAMS["steer_range"],
        speed_range=VEHICLE_PARAMS["speed_range"],
        angular_velocity_range=VEHICLE_PARAMS["angular_velocity_range"],
        accel_range=VEHICLE_PARAMS["accel_range"],
        delta_t=0.1,
    )
