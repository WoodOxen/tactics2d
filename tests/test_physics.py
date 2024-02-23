##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_physics.py
# @Description: This file implements the test cases for the physics models.
# @Author: Yueyuan Li
# @Version: 1.0.0

import sys

sys.path.append(".")
sys.path.append("..")

import os

RENDER = "DISPLAY" in os.environ

import logging
import time

logging.basicConfig(level=logging.INFO)

import numpy as np
import pygame
import pytest
from shapely import hausdorff_distance
from shapely.affinity import affine_transform, rotate
from shapely.geometry import LineString

from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State
from tactics2d.physics import (
    PointMass,
    SingleTrackDrift,
    SingleTrackDynamics,
    SingleTrackKinematics,
)

# fmt: off
PEDESTRIAN_ACTION_LIST = [
    ((0, 0), 100), # stop for 0.1 second
    ((1, 0), 500), # accelerate for 0.5 second along x-axis, expected to reach 0.5 m/s
    ((-1, 0), 500), # decelerate for 0.5 second along x-axis, expected to reach 0 m/s
    ((1, 0), 500), # accelerate for 0.5 second along x-axis, expected to reach 0.5 m/s
    ((0, 1), 500), # accelerate for 0.5 second along y-axis, expected to reach  0.707 m/s
    ((0, -1), 500), # accelerate for 0.5 second along y-axis, expected to reach  0.5 m/s
    ((1, 1), 500), ((2, 2), 500), ((-2, -2), 2000), ((-1, 2), 500), ((2, -1), 500)
]

VEHICLE_ACTION_LIST = [
    ((0, 0), 1000),
    ((1, 0), 1000), ((-1, 0), 1000),
    ((4, 0), 1000), ((-4, 0), 1000),
    ((15, 0), 2000), ((-15, 0), 500),
    ((1, 0), 1000),
    ((0.1, 0.3), 5000), ((0.1, -0.3), 5000),
    ((0.1, 0.6), 5000), ((0.1, -0.6), 5000),
]
# fmt: on


class Visualizer:
    def __init__(self, vehicle, fps=60):
        self.width = vehicle.width
        self.length = vehicle.length
        self.bbox = vehicle.geometry

        self.front_axle = [
            [0.5 * self.length - vehicle.front_overhang, 0.5 * self.width + 0.1],
            [0.5 * self.length - vehicle.front_overhang, -0.5 * self.width - 0.1],
        ]

        self.rear_axle = [
            [-0.5 * self.length + vehicle.rear_overhang, 0.5 * self.width + 0.1],
            [-0.5 * self.length + vehicle.rear_overhang, -0.5 * self.width - 0.1],
        ]

        # wheel order: left front, right front, left rear, right rear
        self.wheels = [
            (
                [self.front_axle[0][0] + 0.225, self.front_axle[0][1]],
                [self.front_axle[0][0] - 0.225, self.front_axle[0][1]],
            ),
            (
                [self.front_axle[1][0] + 0.225, self.front_axle[1][1]],
                [self.front_axle[1][0] - 0.225, self.front_axle[1][1]],
            ),
            (
                [self.rear_axle[0][0] + 0.225, self.rear_axle[0][1]],
                [self.rear_axle[0][0] - 0.225, self.rear_axle[0][1]],
            ),
            (
                [self.rear_axle[1][0] + 0.225, self.rear_axle[1][1]],
                [self.rear_axle[1][0] - 0.225, self.rear_axle[1][1]],
            ),
        ]

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
        _, steer = action

        # draw vehicle bounding box
        transform_matrix = [
            np.cos(state.heading),
            -np.sin(state.heading),
            np.sin(state.heading),
            np.cos(state.heading),
            state.x,
            state.y,
        ]
        pose = affine_transform(self.bbox, transform_matrix)

        pygame.draw.polygon(self.screen, (0, 245, 255, 100), self._scale(pose))

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
            f"actions: accel = {action[0]:.2f}, steer = {action[1]:.2f}",
            f"true actions: accel = {true_action[0]:.2f}, steer = {true_action[1]:.2f}",
        ]
        for i, info in enumerate(infos):
            self.font.render_to(self.screen, (30, 10 + i * 20), info, (0, 0, 0))
        pygame.display.update()
        self.clock.tick(self.fps)

    def quit(self):
        pygame.quit()


@pytest.mark.physics
@pytest.mark.parametrize(
    "speed_range, accel_range, interval, delta_t",
    [
        ([0, 5], [0, 2], 100, 5),
        ([-5, 5], [-2, 2], 9, 5),
        ([5, 5], [2, 2], 50, 3),
        (5, 2, 100, 5),
        (-5, -2, 100, 5),
        (None, None, 100, 5),
    ],
)
def test_point_mass(speed_range, accel_range, interval, delta_t):
    model_newton = PointMass(speed_range, accel_range, interval, delta_t, "newton")
    model_euler = PointMass(speed_range, accel_range, interval, delta_t, "euler")
    initial_state = State(frame=0, x=10, y=10, heading=0, speed=0)

    last_state_newton = initial_state
    last_state_euler = initial_state
    line_newton = [[last_state_newton.x, last_state_newton.y]]
    line_euler = [[last_state_euler.x, last_state_euler.y]]
    cnt = 0
    t1 = time.time()
    for action, duration in PEDESTRIAN_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            state_newton = model_newton.step(last_state_newton, action, interval)
            line_newton.append([state_newton.x, state_newton.y])
            last_state_newton = state_newton
            cnt += 1
    t2 = time.time()

    for action, duration in PEDESTRIAN_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            state_euler = model_euler.step(last_state_euler, action, interval)
            line_euler.append([state_euler.x, state_euler.y])
            last_state_euler = state_euler
    t3 = time.time()

    assert hausdorff_distance(LineString(line_newton), LineString(line_euler)) < 0.01
    logging.info("The average fps for Newton's method is {:.2f} Hz.".format(cnt / (t2 + 1e-6 - t1)))
    logging.info("The average fps for Euler's method is {:.2f} Hz.".format(cnt / (t3 + 1e-6 - t2)))


@pytest.mark.physics
@pytest.mark.parametrize("interval, delta_t", [(9, 5), (50, 3), (100, 5)])
def test_single_track_kinematic(interval, delta_t):
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    physics_model_constrained = SingleTrackKinematics(
        lf=vehicle.length / 2 - vehicle.front_overhang,
        lr=vehicle.length / 2 - vehicle.rear_overhang,
        steer_range=vehicle.steer_range,
        speed_range=vehicle.speed_range,
        accel_range=vehicle.accel_range,
        interval=interval,
        delta_t=delta_t,
    )

    physics_model = SingleTrackKinematics(
        lf=vehicle.length / 2 - vehicle.front_overhang,
        lr=vehicle.length / 2 - vehicle.rear_overhang,
        interval=interval,
        delta_t=delta_t,
    )

    assert physics_model_constrained.lf == 4.284 / 2 - 0.880
    assert physics_model_constrained.lr == 4.284 / 2 - 0.767
    logging.info(f"{vehicle.steer_range}, {vehicle.speed_range}, {vehicle.accel_range}")

    state_constrained = State(frame=0, x=10, y=10, heading=0, speed=0)
    state = State(frame=0, x=10, y=10, heading=0, speed=0)
    trajectory = [(state.x, state.y)]
    if RENDER:
        # visualizer = Visualizer(vehicle, int(1000/interval))
        visualizer = Visualizer(vehicle)
        t1 = time.time()

    for action, duration in VEHICLE_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            state, _, _ = physics_model.step(
                state_constrained,
                action[0] + np.random.uniform(4, 5),
                action[1] + np.random.uniform(0.8, 1.0),
                interval,
            )
            assert (
                physics_model_constrained.verify_state(state, state_constrained, interval) is False
            )

            state_constrained, real_accel, real_steer = physics_model_constrained.step(
                state_constrained, action[0], action[1], interval
            )
            trajectory.append((state_constrained.x, state_constrained.y))

            if RENDER:
                visualizer.update(state_constrained, action, (real_accel, real_steer), trajectory)

    if RENDER:
        t2 = time.time()
        n_frame = int(np.sum([n_frame for _, n_frame in VEHICLE_ACTION_LIST]) / interval)
        visualizer.quit()
        logging.info(
            "The average fps for single track kinematics model is {:.2f} Hz.".format(
                n_frame / (t2 + -t1)
            )
        )


@pytest.mark.physics
@pytest.mark.parametrize(
    "speed_range, accel_range, interval, delta_t",
    [
        ([0, 5], [0, 2], 100, 5),
        ([-5, 5], [-2, 2], 9, 5),
        ([5, 5], [2, 2], 50, 3),
        (5, 2, 100, 5),
        (-5, -2, 100, 5),
        (None, None, 100, 5),
    ],
)
def test_point_mass(speed_range, accel_range, interval, delta_t):
    model_newton = PointMass(speed_range, accel_range, interval, delta_t, "newton")
    model_euler = PointMass(speed_range, accel_range, interval, delta_t, "euler")
    initial_state = State(frame=0, x=10, y=10, heading=0, speed=0)

    last_state_newton = initial_state
    last_state_euler = initial_state
    line_newton = [[last_state_newton.x, last_state_newton.y]]
    line_euler = [[last_state_euler.x, last_state_euler.y]]
    cnt = 0
    t1 = time.time()
    for action, duration in PEDESTRIAN_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            state_newton = model_newton.step(last_state_newton, action, interval)
            line_newton.append([state_newton.x, state_newton.y])
            last_state_newton = state_newton
            cnt += 1
    t2 = time.time()

    for action, duration in PEDESTRIAN_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            state_euler = model_euler.step(last_state_euler, action, interval)
            line_euler.append([state_euler.x, state_euler.y])
            last_state_euler = state_euler
    t3 = time.time()

    assert hausdorff_distance(LineString(line_newton), LineString(line_euler)) < 0.01
    logging.info("The average fps for Newton's method is {:.2f} Hz.".format(cnt / (t2 + 1e-6 - t1)))
    logging.info("The average fps for Euler's method is {:.2f} Hz.".format(cnt / (t3 + 1e-6 - t2)))


@pytest.mark.parametrize("interval, delta_t", [(9, 5), (50, 3), (100, 5)])
def test_single_track_dynamics(interval, delta_t):
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    physics_model_constrained = SingleTrackDynamics(
        lf=vehicle.length / 2 - vehicle.front_overhang,
        lr=vehicle.length / 2 - vehicle.rear_overhang,
        mass=vehicle.kerb_weight,
        mass_height=vehicle.height / 2,
        steer_range=vehicle.steer_range,
        speed_range=vehicle.speed_range,
        accel_range=vehicle.accel_range,
        interval=interval,
        delta_t=delta_t,
    )

    physics_model = SingleTrackDynamics(
        lf=vehicle.length / 2 - vehicle.front_overhang,
        lr=vehicle.length / 2 - vehicle.rear_overhang,
        mass=vehicle.kerb_weight,
        mass_height=vehicle.height / 2,
    )

    state_constrained = State(frame=0, x=10, y=10, heading=0, speed=0)
    trajectory = [(state_constrained.x, state_constrained.y)]
    if RENDER:
        # visualizer = Visualizer(vehicle, int(1000/interval))
        visualizer = Visualizer(vehicle)
        t1 = time.time()

    for action, duration in VEHICLE_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            state, _, _ = physics_model.step(
                state_constrained,
                action[0] + np.random.uniform(4.5, 6),
                action[1] + np.random.uniform(0.8, 1.0),
                interval,
            )
            assert (
                physics_model_constrained.verify_state(state, state_constrained, interval) is False
            )

            state_constrained, real_accel, real_steer = physics_model_constrained.step(
                state_constrained, action[0], action[1], interval
            )
            trajectory.append((state_constrained.x, state_constrained.y))
            if RENDER:
                visualizer.update(state_constrained, action, (real_accel, real_steer), trajectory)

    if RENDER:
        t2 = time.time()
        n_frame = int(np.sum([n_frame for _, n_frame in VEHICLE_ACTION_LIST]) / interval)
        visualizer.quit()
        logging.info(
            "The average fps for single track dynamics model is {:.2f} Hz.".format(
                n_frame / (t2 + -t1)
            )
        )


@pytest.mark.physics
@pytest.mark.parametrize("interval, delta_t", [(9, 5), (50, 3), (100, 5)])
def test_single_track_drift(interval, delta_t):
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    physics_model_constrained = SingleTrackDrift(
        lf=vehicle.length / 2 - vehicle.front_overhang,
        lr=vehicle.length / 2 - vehicle.rear_overhang,
        mass=vehicle.kerb_weight,
        mass_height=vehicle.height / 2,
        steer_range=vehicle.steer_range,
        speed_range=vehicle.speed_range,
        accel_range=vehicle.accel_range,
        interval=interval,
        delta_t=delta_t,
    )

    state_constrained = State(frame=0, x=10, y=10, heading=0, speed=0)
    omega_wf = 0
    omega_wr = 0
    trajectory = [(state_constrained.x, state_constrained.y)]
    if RENDER:
        # visualizer = Visualizer(vehicle, int(1000/interval))
        visualizer = Visualizer(vehicle)
        t1 = time.time()

    for action, duration in VEHICLE_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            (
                state_constrained,
                omega_wf,
                omega_wr,
                real_accel,
                real_steer,
            ) = physics_model_constrained.step(
                state_constrained, omega_wf, omega_wr, action[0], action[1], interval
            )
            trajectory.append((state_constrained.x, state_constrained.y))
            if RENDER:
                visualizer.update(state_constrained, action, (real_accel, real_steer), trajectory)

    if RENDER:
        t2 = time.time()
        n_frame = int(np.sum([n_frame for _, n_frame in VEHICLE_ACTION_LIST]) / interval)
        visualizer.quit()
        logging.info(
            "The average fps for single track dynamics model is {:.2f} Hz.".format(
                n_frame / (t2 + -t1)
            )
        )


@pytest.mark.physics
@pytest.mark.parametrize("interval, delta_t", [(9, 5), (50, 3), (100, 5)])
def test_deviation(interval, delta_t):
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    kinematics_model = SingleTrackKinematics(
        lf=vehicle.length / 2 - vehicle.front_overhang,
        lr=vehicle.length / 2 - vehicle.rear_overhang,
        steer_range=vehicle.steer_range,
        speed_range=vehicle.speed_range,
        accel_range=vehicle.accel_range,
        interval=interval,
        delta_t=delta_t,
    )
    dynamics_model = SingleTrackDynamics(
        lf=vehicle.length / 2 - vehicle.front_overhang,
        lr=vehicle.length / 2 - vehicle.rear_overhang,
        mass=vehicle.kerb_weight,
        mass_height=vehicle.height / 2,
        steer_range=vehicle.steer_range,
        speed_range=vehicle.speed_range,
        accel_range=vehicle.accel_range,
        interval=interval,
        delta_t=delta_t,
    )
    drift_model = SingleTrackDrift(
        lf=vehicle.length / 2 - vehicle.front_overhang,
        lr=vehicle.length / 2 - vehicle.rear_overhang,
        mass=vehicle.kerb_weight,
        mass_height=vehicle.height / 2,
        steer_range=vehicle.steer_range,
        speed_range=vehicle.speed_range,
        accel_range=vehicle.accel_range,
        interval=interval,
        delta_t=delta_t,
    )

    state_kinematics = State(frame=0, x=10, y=10, heading=0, speed=0)
    state_dynamics = State(frame=0, x=10, y=10, heading=0, speed=0)
    state_drift = State(frame=0, x=10, y=10, heading=0, speed=0)
    omega_wf = 0
    omega_wr = 0
    trajectory_kinematics = [(state_kinematics.x, state_kinematics.y)]
    trajectory_dynamics = [(state_dynamics.x, state_dynamics.y)]
    trajectory_drift = [(state_drift.x, state_drift.y)]

    for action, duration in VEHICLE_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            state_kinematics, _, _ = kinematics_model.step(
                state_kinematics, action[0], action[1], interval
            )
            trajectory_kinematics.append((state_kinematics.x, state_kinematics.y))

            state_dynamics, _, _ = dynamics_model.step(
                state_dynamics, action[0], action[1], interval
            )
            trajectory_dynamics.append((state_dynamics.x, state_dynamics.y))

            state_drift, omega_wf, omega_wr, _, _ = drift_model.step(
                state_drift, omega_wf, omega_wr, action[0], action[1], interval
            )
            trajectory_drift.append((state_drift.x, state_drift.y))

    deviation1 = hausdorff_distance(
        LineString(trajectory_kinematics), LineString(trajectory_dynamics)
    )
    deviation2 = hausdorff_distance(LineString(trajectory_kinematics), LineString(trajectory_drift))
    deviation3 = hausdorff_distance(LineString(trajectory_dynamics), LineString(trajectory_drift))

    logging.info(f"The deviation between kinematics and dynamics model is {deviation1:.2f}")
    logging.info(f"The deviation between kinematics and drift model is {deviation2:.2f}")
    logging.info(f"The deviation between dynamics and drift model is {deviation3:.2f}")

    if RENDER:
        pygame.init()
        screen = pygame.display.set_mode((1200, 1200))
        screen.fill((255, 255, 255))
        pygame.draw.lines(screen, (100, 0, 0), False, np.array(trajectory_kinematics) * 20, 1)
        pygame.draw.lines(screen, (0, 100, 0), False, np.array(trajectory_dynamics) * 20, 1)
        pygame.draw.lines(screen, (0, 0, 100), False, np.array(trajectory_drift) * 20, 1)
        pygame.display.update()
        pygame.time.wait(3000)
        pygame.quit()
