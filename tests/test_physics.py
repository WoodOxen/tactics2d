import sys

sys.path.append(".")
sys.path.append("..")

import os

RENDER = "DISPLAY" in os.environ

import time
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from shapely.geometry import LineString, LinearRing
from shapely.affinity import affine_transform, rotate
import pygame
import pytest

from tactics2d.trajectory.element import State
from tactics2d.physics import PointMass, SingleTrackKinematics, SingleTrackDynamics


# Prototype: Volkswagen Polo(AW/BZ) (https://en.wikipedia.org/wiki/Volkswagen_Polo)
VEHICLE_PARAMS = {
    "length": 4.053,
    "width": 1.751,
    "height": 1.461,
    "wheel_base": 2.548,
    "front_overhang": 0.824,
    "rear_overhang": 0.681,
    "steer_range": (-0.5236, 0.5236),
    "speed_range": (-2.78, 52.8),
    "accel_range": (-9.8, 3.517),
    "0_100_km/h": 12,
}

# fmt: off
ACTION_LIST = [
    ((0, 0), 10), ((0, 2), 20), ((0, -2), 20), 
    ((0.1, 0), 10), ((0.1, 0.5), 20), ((0.1, -0.5), 20),
    ((0, 5), 20), ((0, -5), 20),
    ((-0.1, 0), 10), ((-0.1, 0.5), 20), ((-0.1, -0.5), 20),
    ((0.2, 0.5), 20), ((0.2, -0.5), 20), ((-0.2, 0.5), 20), ((-0.2, -0.5), 20),
    ((0, 1), 10), ((1, 0), 20), ((-1, 0), 20),
    # (0, 2), (0, -2), (0, 5), (0, -5), (-0.3, 0), (-0.3, 0.5), (-0.3, -0.5),
    # (1, 0), (1, 0.5), (1, -0.5), (-1, 0), (-1, 0.5), (-1, -0.5),
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
            0.5 * VEHICLE_PARAMS["width"] + 0.1,
        ],
        [
            0.5 * VEHICLE_PARAMS["length"] - VEHICLE_PARAMS["front_overhang"],
            -0.5 * VEHICLE_PARAMS["width"] - 0.1,
        ],
    ]

    rear_axle = [
        [
            -0.5 * VEHICLE_PARAMS["length"] + VEHICLE_PARAMS["rear_overhang"],
            0.5 * VEHICLE_PARAMS["width"] + 0.1,
        ],
        [
            -0.5 * VEHICLE_PARAMS["length"] + VEHICLE_PARAMS["rear_overhang"],
            -0.5 * VEHICLE_PARAMS["width"] - 0.1,
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

    def __init__(self, fps=60):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 1200))
        self.clock = pygame.time.Clock()
        self.font = pygame.freetype.SysFont(pygame.freetype.get_default_font(), 16)
        self.fps = fps

    def _scale(self, geometry, scale_factor=20) -> list:
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

        pygame.draw.polygon(self.screen, (0, 245, 255, 100), self._scale(vehicle_bbox))

        # draw axles
        front_axle = affine_transform(LineString(self.front_axle), transform_matrix)
        rear_axle = affine_transform(LineString(self.rear_axle), transform_matrix)
        pygame.draw.lines(self.screen, (0, 0, 0), False, self._scale(front_axle))
        pygame.draw.lines(self.screen, (0, 0, 0), False, self._scale(rear_axle))

        # draw wheels
        for wheel, steer_ in zip(self.wheels, [steer, steer, 0, 0]):
            wheel = affine_transform(LineString(wheel), transform_matrix)
            wheel = rotate(wheel, steer_, use_radians=True)
            pygame.draw.lines(self.screen, (0, 0, 0), False, self._scale(wheel), 2)

    def update(self, state: State, action: tuple, true_action: tuple, trajectory: list):
        self.screen.fill((255, 255, 255))
        self._draw_vehicle(state, true_action)
        pygame.draw.lines(
            self.screen, (100, 100, 100), False, self._scale(LineString(trajectory)), 1
        )

        infos = [
            f"frame = {state.frame}",
            f"state: x = {state.x:.2f}, y = {state.y:.2f}, heading = {state.heading:.2f}, speed = {state.speed:.2f}",
            f"actions: steer = {action[0]:.2f}, accel = {action[1]:.2f}",
            f"true actions: steer = {true_action[0]:.2f}, accel = {true_action[1]:.2f}",
        ]
        for i, info in enumerate(infos):
            self.font.render_to(self.screen, (30, 10 + i * 20), info, (0, 0, 0))
        pygame.display.update()
        self.clock.tick(self.fps)

    def quit(self):
        pygame.quit()


def execute_actions(vehicle_model):
    step = 0.1

    # f = open(f"./tests/test_{vehicle_model.abbrev}_trajectory.csv", "w")
    # f.write("frame, x, y, heading, vx, vy, speed, ax, ay, accel")

    if RENDER:
        visualizer = Visualizer(10)
        t1 = time.time()

    state = State(frame=0, x=10, y=10, heading=0, speed=0)
    trajectory = [(state.x, state.y)]

    for action, duration in ACTION_LIST:
        for _ in range(duration):
            # f.write(repr(state) + "\n")
            new_state, true_action = vehicle_model.step(state, action, step)
            trajectory.append((new_state.x, new_state.y))
            state = new_state
            if RENDER:
                visualizer.update(new_state, action, true_action, trajectory)

    if RENDER:
        t2 = time.time()
        n_frame = np.sum([n_frame for _, n_frame in ACTION_LIST])
        visualizer.quit()
        logging.info(f"Rendering took {t2 - t1:.2f} seconds.")
        logging.info(f"The average fps is {n_frame / (t2 - t1): .2f} Hz.")


@pytest.mark.physics
@pytest.mark.parametrize(
    "steer_range, speed_range, accel_range, delta_t",
    [
        (
            VEHICLE_PARAMS["steer_range"],
            (0, VEHICLE_PARAMS["speed_range"][1]),
            VEHICLE_PARAMS["accel_range"],
            0.005,
        )
    ],
)
def test_point_mass(steer_range, speed_range, accel_range, delta_t):
    vehicle_model = PointMass(
        steer_range=steer_range, speed_range=speed_range, accel_range=accel_range, delta_t=delta_t
    )

    execute_actions(vehicle_model)


@pytest.mark.physics
@pytest.mark.parametrize(
    "steer_range, speed_range, accel_range, delta_t",
    [
        (
            VEHICLE_PARAMS["steer_range"],
            (0, VEHICLE_PARAMS["speed_range"][1]),
            VEHICLE_PARAMS["accel_range"],
            0.005,
        )
    ],
)
def test_single_track_kinematic(steer_range, speed_range, accel_range, delta_t):
    vehicle_model = SingleTrackKinematics(
        VEHICLE_PARAMS["width"] / 2 - VEHICLE_PARAMS["front_overhang"],
        VEHICLE_PARAMS["width"] / 2 - VEHICLE_PARAMS["rear_overhang"],
        steer_range=steer_range,
        speed_range=speed_range,
        accel_range=accel_range,
        delta_t=delta_t,
    )

    execute_actions(vehicle_model)


# if __name__ == "__main__":
#     test_single_track_kinematic(
#     # test_point_mass(
#         steer_range=VEHICLE_PARAMS["steer_range"],
#         speed_range=(0, VEHICLE_PARAMS["speed_range"][1]),
#         accel_range=VEHICLE_PARAMS["accel_range"],
#         delta_t=0.005,
#     )
