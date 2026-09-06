# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for physics."""


import sys

sys.path.append(".")
sys.path.append("..")

import os

RENDER = "DISPLAY" in os.environ

import logging
import math

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
# Pedestrian actions for point mass model testing
# Format: ((accel_x, accel_y), duration_ms)
# Accelerations are in m/s², durations in milliseconds
PEDESTRIAN_ACTION_LIST = [
    ((0, 0), 100), # stop for 0.1 second
    ((1, 0), 500), # accelerate for 0.5 second along x-axis, expected to reach 0.5 m/s
    ((-1, 0), 500), # decelerate for 0.5 second along x-axis, expected to reach 0 m/s
    ((1, 0), 500), # accelerate for 0.5 second along x-axis, expected to reach 0.5 m/s
    ((0, 1), 500), # accelerate for 0.5 second along y-axis, expected to reach  0.707 m/s
    ((0, -1), 500), # accelerate for 0.5 second along y-axis, expected to reach  0.5 m/s
    ((1, 1), 500), ((2, 2), 500), ((-2, -2), 2000), ((-1, 2), 500), ((2, -1), 500)
]

# Vehicle actions for kinematic/dynamic/drift model testing
# Format: ((acceleration, steering), duration_ms)
# Acceleration in m/s², steering in radians, duration in milliseconds
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
    [(None, None, 100, 5), ([-5, 5], [-2, 2], 9, 5), ([0, 5], [0, 2], 100, 5)],
)
def test_point_mass(speed_range, accel_range, interval, delta_t):
    """The newton and euler backends must agree on the same trajectory (within tolerance)."""
    model_newton = PointMass(speed_range, accel_range, interval, delta_t, "newton")
    model_euler = PointMass(speed_range, accel_range, interval, delta_t, "euler")
    initial_state = State(frame=0, x=10, y=10, heading=0, speed=0)

    state = initial_state
    line_newton = [[state.x, state.y]]
    for action, duration in PEDESTRIAN_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            state = model_newton.step(state, action, interval)
            line_newton.append([state.x, state.y])

    state = initial_state
    line_euler = [[state.x, state.y]]
    for action, duration in PEDESTRIAN_ACTION_LIST:
        for _ in np.arange(0, duration, interval):
            state = model_euler.step(state, action, interval)
            line_euler.append([state.x, state.y])

    distance = hausdorff_distance(LineString(line_newton), LineString(line_euler))
    assert distance < 0.01, f"Hausdorff distance {distance:.6f} exceeds threshold 0.01"


@pytest.mark.physics
def test_step_size_insensitivity():
    """RK4-based models must produce nearly identical trajectories regardless of delta_t.

    A step-size-dependent integrator (e.g. the previous first-order Euler) would drift by
    several centimeters to meters here, so this guards against regressing the integrators.
    """
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    lf = vehicle.length / 2 - vehicle.front_overhang
    lr = vehicle.length / 2 - vehicle.rear_overhang

    def simulate(model, action_list):
        state = State(frame=0, x=10, y=10, heading=0, speed=0)
        trajectory = [(state.x, state.y)]
        for action, duration in action_list:
            for _ in np.arange(0, duration, 100):
                if isinstance(model, PointMass):
                    state = model.step(state, action, 100)
                else:
                    state, _, _ = model.step(state, action[0], action[1], 100)
                trajectory.append((state.x, state.y))
        return state, trajectory

    def make(model_cls, delta_t, **extra):
        if model_cls is PointMass:
            return model_cls(interval=100, delta_t=delta_t, backend="euler")
        return model_cls(lf=lf, lr=lr, interval=100, delta_t=delta_t, **extra)

    # (model, action list, extra kwargs, delta_ts to compare against the delta_t=5 baseline)
    specs = [
        (SingleTrackKinematics, VEHICLE_ACTION_LIST, {}, (20, 50)),
        (
            SingleTrackDynamics,
            VEHICLE_ACTION_LIST,
            dict(mass=vehicle.kerb_weight, mass_height=vehicle.height / 2),
            (20,),
        ),
        (PointMass, PEDESTRIAN_ACTION_LIST, {}, (20, 50)),
    ]

    for model_cls, actions, extra, delta_ts in specs:
        ref_state, ref_traj = simulate(make(model_cls, 5, **extra), actions)
        for delta_t in delta_ts:
            state, trajectory = simulate(make(model_cls, delta_t, **extra), actions)
            dist = hausdorff_distance(LineString(ref_traj), LineString(trajectory))
            end_dist = math.hypot(state.x - ref_state.x, state.y - ref_state.y)
            assert dist < 1e-2, (
                f"{model_cls.__name__} delta_t={delta_t} vs 5: Hausdorff distance "
                f"{dist:.6f} exceeds 1e-2"
            )
            assert end_dist < 1e-2, (
                f"{model_cls.__name__} delta_t={delta_t} vs 5: endpoint distance "
                f"{end_dist:.6f} exceeds 1e-2"
            )


@pytest.mark.physics
def test_interval_invariance():
    """With delta_t=5, the trajectory must not depend on how the horizon is split into step() calls."""
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    lf = vehicle.length / 2 - vehicle.front_overhang
    lr = vehicle.length / 2 - vehicle.rear_overhang
    total_ms = 600
    accel, delta = 2.0, 0.4

    def simulate(interval):
        model = SingleTrackKinematics(lf=lf, lr=lr, interval=interval, delta_t=5)
        state = State(frame=0, x=0, y=0, heading=0, speed=5.0)
        t = 0
        while t < total_ms:
            state, _, _ = model.step(state, accel, delta, interval)
            t += interval
        return state

    ref = simulate(5)
    for interval in (10, 50, 100, 300, 600):
        state = simulate(interval)
        assert abs(state.x - ref.x) < 1e-12, f"interval={interval} vs 5: x {state.x} != {ref.x}"
        assert abs(state.y - ref.y) < 1e-12, f"interval={interval} vs 5: y {state.y} != {ref.y}"
        assert (
            abs(state.heading - ref.heading) < 1e-12
        ), f"interval={interval} vs 5: heading {state.heading} != {ref.heading}"


@pytest.mark.physics
def test_drift_converges_at_fine_step():
    """SingleTrackDrift is numerically stiff (a fast mode around 800 s^-1 makes explicit RK4
    stable only below ~3.5 ms), so only fine step sizes converge. This guards the persistent
    yaw-rate/slip-angle state and the RK4 integrator: before the fix, even delta_t=1 ms
    integrated the wrong (per-step-reset) dynamics and diverged ~6 m from the true ODE over 5 s.
    """
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    lf = vehicle.length / 2 - vehicle.front_overhang
    lr = vehicle.length / 2 - vehicle.rear_overhang

    def simulate(delta_t):
        model = SingleTrackDrift(
            lf=lf,
            lr=lr,
            mass=vehicle.kerb_weight,
            mass_height=vehicle.height / 2,
            interval=100,
            delta_t=delta_t,
        )
        state = State(frame=0, x=0, y=0, heading=0, speed=10.0, vx=10.0, vy=0)
        omega_wf = omega_wr = 10.0 / model.radius
        t = 0
        while t < 5000:
            state, omega_wf, omega_wr, _, _ = model.step(state, omega_wf, omega_wr, 1.0, 0.2, 100)
            t += 100
        return state

    ref = simulate(1)
    state = simulate(2)
    dist = math.hypot(state.x - ref.x, state.y - ref.y)
    assert dist < 0.2, f"drift delta_t=2 vs 1: endpoint distance {dist:.4f} exceeds 0.2"


@pytest.mark.physics
@pytest.mark.parametrize(
    "model_cls", [SingleTrackKinematics, SingleTrackDynamics, SingleTrackDrift]
)
def test_single_track_smoke(model_cls):
    """A range-constrained model of each class steps the full action list without NaN/errors.

    Replaces three near-identical tests: builds the model from the medium_car template with the
    vehicle's steer/speed/accel ranges, runs the whole VEHICLE_ACTION_LIST, asserts every state
    stays finite with increasing frames, and finishes with an interval=9 (delta_t=5) step that
    exercises the ``interval % delta_t`` remainder sub-step.
    """
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    lf = vehicle.length / 2 - vehicle.front_overhang
    lr = vehicle.length / 2 - vehicle.rear_overhang

    extra = {}
    if issubclass(model_cls, (SingleTrackDynamics, SingleTrackDrift)):
        extra = dict(mass=vehicle.kerb_weight, mass_height=vehicle.height / 2)

    model = model_cls(
        lf=lf,
        lr=lr,
        steer_range=vehicle.steer_range,
        speed_range=vehicle.speed_range,
        accel_range=vehicle.accel_range,
        interval=100,
        delta_t=5,
        **extra,
    )

    state = State(frame=0, x=10, y=10, heading=0, speed=0)
    states = [state]
    omega_wf = omega_wr = 0
    for action, duration in VEHICLE_ACTION_LIST:
        for _ in np.arange(0, duration, 100):
            if isinstance(model, SingleTrackDrift):
                state, omega_wf, omega_wr, _, _ = model.step(
                    state, omega_wf, omega_wr, action[0], action[1], 100
                )
            else:
                state, _, _ = model.step(state, action[0], action[1], 100)
            assert state.frame == states[-1].frame + 100
            assert math.isfinite(state.x) and math.isfinite(state.y)
            assert math.isfinite(state.speed) and math.isfinite(state.heading)
            states.append(state)

    # Remainder sub-step path (5 ms sub-steps plus a 4 ms leftover for interval=9).
    if isinstance(model, SingleTrackDrift):
        state, omega_wf, omega_wr, _, _ = model.step(state, omega_wf, omega_wr, 1.0, 0.1, 9)
    else:
        state, _, _ = model.step(state, 1.0, 0.1, 9)
    assert state.frame == states[-1].frame + 9
    assert math.isfinite(state.x) and math.isfinite(state.y) and math.isfinite(state.speed)

    if RENDER and model_cls is SingleTrackKinematics:
        visualizer = Visualizer(vehicle)
        trajectory = [(s.x, s.y) for s in states]
        for s in states:
            visualizer.update(s, (0, 0), (0, 0), trajectory)
        visualizer.quit()


@pytest.mark.physics
@pytest.mark.parametrize("model_cls", [SingleTrackKinematics, SingleTrackDynamics])
def test_verify_state_rejects_out_of_range(model_cls):
    """verify_state must reject a step pushed beyond the steer/accel range but accept an in-range one."""
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    lf = vehicle.length / 2 - vehicle.front_overhang
    lr = vehicle.length / 2 - vehicle.rear_overhang

    extra = {}
    if model_cls is SingleTrackDynamics:
        extra = dict(mass=vehicle.kerb_weight, mass_height=vehicle.height / 2)
    ranges = dict(
        steer_range=vehicle.steer_range,
        speed_range=vehicle.speed_range,
        accel_range=vehicle.accel_range,
    )
    constrained = model_cls(lf=lf, lr=lr, interval=100, delta_t=5, **ranges, **extra)
    free = model_cls(lf=lf, lr=lr, interval=100, delta_t=5, **extra)

    last = State(frame=0, x=10, y=10, heading=0, speed=5.0, vx=5.0, vy=0)

    # Steering far beyond max_steer (pi/6 ~= 0.52 rad) must be rejected by verify_state.
    bad, _, _ = free.step(last, 0.0, 1.2, 100)
    assert constrained.verify_state(bad, last) is False

    # Regression: an accelerating near-straight step at speed used to be rejected because the
    # reachability box paired the max speed with the extreme steering angle. With a straight
    # travel bound (Euclidean displacement <= arc length) it is accepted.
    fast_last = State(frame=0, x=0, y=0, heading=0, speed=10.0, vx=10.0, vy=0)
    good, real_accel, real_steer = constrained.step(fast_last, 1.0, 0.1, 100)
    assert real_accel == 1.0 and real_steer == 0.1
    assert constrained.verify_state(good, fast_last) is True

    # A teleport far beyond one step of travel is still rejected by the distance bound.
    teleport = State(
        frame=fast_last.frame + 100,
        x=fast_last.x + 100.0,
        y=fast_last.y,
        heading=fast_last.heading,
        speed=fast_last.speed,
    )
    assert constrained.verify_state(teleport, fast_last) is False


@pytest.mark.physics
def test_grip_cap():
    """SingleTrackKinematics.mu is an opt-in grip limit.

    When mu is None (default) the model stays the pure no-slip kinematic reference. When mu is
    set, the yaw rate is capped so the lateral acceleration v * phi_dot stays within mu * g,
    i.e. the vehicle understeers once the grip limit is reached instead of keeping an
    unlimited-grip turn.
    """
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    lf = vehicle.length / 2 - vehicle.front_overhang
    lr = vehicle.length / 2 - vehicle.rear_overhang
    g = 9.81

    # Opt-in: the default model carries no friction limit.
    model_default = SingleTrackKinematics(lf=lf, lr=lr)
    assert model_default.mu is None

    mu = 0.85
    mu_g = mu * g

    def simulate(mu_, v0=20.0, accel=0.0, steer=0.3, seconds=5.0, delta_t=5):
        """Return (final state, max measured |yaw rate| over one step)."""
        model = SingleTrackKinematics(lf=lf, lr=lr, mu=mu_, interval=100, delta_t=delta_t)
        state = State(frame=0, x=0, y=0, heading=0, speed=v0, vx=v0, vy=0)
        max_yaw = 0.0
        for _ in range(int(seconds * 1000 / 100)):
            prev = state.heading
            state, _, _ = model.step(state, accel, steer, 100)
            dh = (state.heading - prev + math.pi) % (2 * math.pi) - math.pi
            max_yaw = max(max_yaw, abs(dh) / 0.1)
        return state, max_yaw

    # v=20, delta=0.3 demands a_y ~= 46 m/s^2 >> mu*g=8.34: the turn rate must be capped.
    state_capped, max_yaw_capped = simulate(mu)
    state_free, max_yaw_free = simulate(None)

    # The lateral acceleration v * phi_dot never exceeds mu * g (v stays 20 here).
    assert max_yaw_capped * 20.0 <= mu_g + 1e-6
    # The geometric (unlimited-grip) model turns far faster in the same corner.
    assert max_yaw_free > 2.0
    assert max_yaw_capped < 0.5
    # The capped vehicle follows a wider arc, so the two endpoints differ by tens of meters.
    assert math.hypot(state_capped.x - state_free.x, state_capped.y - state_free.y) > 30.0

    # Below the grip limit (a_y ~= 0.95 m/s^2 here) the cap is inactive: identical to mu=None.
    state_low_cap, _ = simulate(mu, v0=5.0, steer=0.1, seconds=2.0)
    state_low_free, _ = simulate(None, v0=5.0, steer=0.1, seconds=2.0)
    assert math.hypot(state_low_cap.x - state_low_free.x, state_low_cap.y - state_low_free.y) < 1e-6

    # Step-size invariance is preserved even in the (piecewise) capped regime: the only
    # non-smooth point is the moment the demand crosses mu*g, so keep a loose bound.
    state_cap5, _ = simulate(mu, v0=5.0, accel=1.5, steer=0.3, seconds=5.0, delta_t=5)
    state_cap20, _ = simulate(mu, v0=5.0, accel=1.5, steer=0.3, seconds=5.0, delta_t=20)
    assert math.hypot(state_cap5.x - state_cap20.x, state_cap5.y - state_cap20.y) < 0.2


@pytest.mark.physics
def test_friction_ellipse():
    """SingleTrackDynamics.friction_ellipse couples longitudinal and lateral grip.

    Off by default (no behaviour change). When enabled, braking/acceleration reduces the
    lateral-response coefficient, so a braking-and-turning vehicle accumulates less yaw than
    without the ellipse; at zero longitudinal acceleration the two are identical.
    """
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    lf = vehicle.length / 2 - vehicle.front_overhang
    lr = vehicle.length / 2 - vehicle.rear_overhang

    def make_model(flag):
        return SingleTrackDynamics(
            lf=lf,
            lr=lr,
            mass=vehicle.kerb_weight,
            mass_height=vehicle.height / 2,
            interval=100,
            delta_t=2,
            friction_ellipse=flag,
        )

    base = make_model(False)
    assert base.friction_ellipse is False

    def simulate(flag, accel, steer=0.0, seconds=3.0):
        model = make_model(flag)
        state = State(frame=0, x=0, y=0, heading=0, speed=20.0, vx=20.0, vy=0)
        for _ in range(int(seconds * 1000 / 100)):
            state, _, _ = model.step(state, accel, steer, 100)
        return state

    # accel = 0: the ellipse reduces mu_lat to mu, so the flag must not change the trajectory.
    s_off = simulate(False, accel=0.0, steer=0.3)
    s_on = simulate(True, accel=0.0, steer=0.3)
    assert math.hypot(s_off.x - s_on.x, s_off.y - s_on.y) < 1e-12

    # Braking (-6 m/s^2, well within mu*g) while cornering: the ellipse leaves less lateral
    # capability, so the yaw build-up (and thus the lateral drift) is smaller.
    s_off_brk = simulate(False, accel=-6.0, steer=0.3)
    s_on_brk = simulate(True, accel=-6.0, steer=0.3)
    assert math.hypot(s_off_brk.x - s_on_brk.x, s_off_brk.y - s_on_brk.y) > 1.0
    assert abs(s_on_brk.heading) < abs(s_off_brk.heading)


@pytest.mark.physics
def test_build_physics_model():
    """Vehicle.build_physics_model wires the template geometry/mass into a physics model."""
    vehicle = Vehicle(0)
    vehicle.load_from_template("medium_car")
    expected_lf = vehicle.length / 2 - vehicle.front_overhang
    expected_lr = vehicle.length / 2 - vehicle.rear_overhang

    # Kinematics: geometry + ranges (+ optional mu), no mass parameters needed.
    kin = vehicle.build_physics_model(SingleTrackKinematics, mu=0.85)
    assert isinstance(kin, SingleTrackKinematics)
    assert kin.lf == expected_lf
    assert kin.lr == expected_lr
    assert kin.mu == 0.85
    assert kin.steer_range == vehicle.steer_range
    assert kin.speed_range == vehicle.speed_range
    assert kin.accel_range == vehicle.accel_range

    kin_default = vehicle.build_physics_model(SingleTrackKinematics)
    assert kin_default.mu is None

    # Dynamics: geometry + mass + optional mu / friction_ellipse; I_z keeps the class default.
    dyn = vehicle.build_physics_model(SingleTrackDynamics, mu=0.85, friction_ellipse=True)
    assert isinstance(dyn, SingleTrackDynamics)
    assert dyn.lf == expected_lf and dyn.lr == expected_lr
    assert dyn.mass == vehicle.kerb_weight
    assert dyn.mass_height == vehicle.height / 2
    assert dyn.mu == 0.85
    assert dyn.friction_ellipse is True
    assert dyn.I_z == SingleTrackDynamics(lf=1, lr=1, mass=1, mass_height=0.5).I_z

    # Drift: geometry + mass + wheel radius / I_z overrides.
    drift = vehicle.build_physics_model(SingleTrackDrift, I_z=2000.0, radius=0.33)
    assert isinstance(drift, SingleTrackDrift)
    assert drift.lf == expected_lf and drift.lr == expected_lr
    assert drift.mass == vehicle.kerb_weight
    assert drift.mass_height == vehicle.height / 2
    assert drift.I_z == 2000.0
    assert drift.radius == 0.33
