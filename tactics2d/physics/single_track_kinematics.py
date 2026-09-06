# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Single track kinematics implementation."""


import math
from typing import Tuple, Union

import numpy as np

from tactics2d.participant.trajectory import State

from .physics_model_base import PhysicsModelBase


class SingleTrackKinematics(PhysicsModelBase):
    r"""This class implements a kinematic single-track bicycle model for a traffic participant.

    The is a simplified model to simulate the traffic participant's physics. The assumptions in this implementation include:

    1. The traffic participant is operating in a 2D plane (x-y).
    2. The left and right wheels always have the same steering angle and speed, so they can be regarded as a single wheel.
    3. The traffic participant is a rigid body, so its geometry does not change during the simulation.
    4. The traffic participant is Front-Wheel Drive (FWD).

    This implementation version is based on the following paper. It regard the geometry center as the reference point.

    ![Kinematic Single Track Model](https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/kinematic_bicycle_model.png)

    ![Demo of the implementation (interval=100 ms, $\Delta t$=5 ms)](https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/tactics2d-single_track_kinematics.gif)

    !!! quote "Reference"
        Kong, Jason, et al. "Kinematic and dynamic vehicle models for autonomous driving control design." *2015 IEEE intelligent vehicles symposium* (IV). IEEE, 2015.

    !!! warning
        This model integrates the equations of motion with a 4th-order Runge-Kutta scheme, so the
        discretization error stays negligible for delta_t up to about 50 ms. Use a smaller delta_t
        (the default 5 ms) when maximum fidelity to the continuous model is required.

    Attributes:
        lf (float): The distance from the geometry center to the front axle center. The unit is meter.
        lr (float): The distance from the geometry center to the rear axle center. The unit is meter.
        mu (float, optional): The tire-road friction coefficient. Defaults to None. It is a dimensionless quantity.

            - When `mu` is None (default), the model is the pure no-slip kinematic bicycle: it assumes unlimited grip and reproduces the CommonRoad kinematic single-track reference exactly.
            - When `mu` is set (e.g. 0.85 for dry asphalt), the turn rate is capped so the demanded lateral acceleration `a_y = v * phi_dot` stays within `mu * g`, i.e. the curvature is reduced to the grip-limit value and the vehicle understeers once the friction limit is reached. This removes the kinematic model's "unlimited grip" assumption at the cost of an extra piecewise (soft) nonlinearity; use a smaller `delta_t` for fine accuracy near the limit.

        steer_range (Union[float, Tuple[float, float]], optional): The steering angle range. The valid input is a float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.

            - When the steer_range is a non-negative float, the steering angle is constrained to be within the range [-steer_range, steer_range].
            - When the steer_range is a tuple, the steering angle is constrained to be within the range [min steering angle, max steering angle].
            - When the steer_range is negative or the min steering angle is not less than the max steering angle, the steer_range is set to None.

        speed_range (Union[float, Tuple[float, float]], optional): The speed range. The valid input is a float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            - When the speed_range is a non-negative float, the speed is constrained to be within the range [-speed_range, speed_range].
            - When the speed_range is a tuple, the speed is constrained to be within the range [min speed, max speed].
            - When the speed_range is negative or the min speed is not less than the max speed, the speed_range is set to None.

        accel_range (Union[float, Tuple[float, float]], optional): The acceleration range. The valid input is a float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).

            - When the accel_range is a non-negative float, the acceleration is constrained to be within the range [-accel_range, accel_range].
            - When the accel_range is a tuple, the acceleration is constrained to be within the range [min acceleration, max acceleration].
            - When the accel_range is negative or the min acceleration is not less than the max acceleration, the accel_range is set to None.

        interval (int, optional): The time interval between the current state and the new state. The unit is millisecond. Defaults to None.
        delta_t (int, optional): The time step for the simulation. The unit is millisecond. Defaults to `_DELTA_T`(5 ms). The expected value is between `_MIN_DELTA_T`(1 ms) and `interval`. The model integrates with a 4th-order Runge-Kutta scheme, so discretization error stays negligible up to ~50 ms; the default 5 ms is retained for backward compatibility.
    """

    def __init__(
        self,
        lf: float,
        lr: float,
        mu: float = None,
        steer_range: Union[float, Tuple[float, float]] = None,
        speed_range: Union[float, Tuple[float, float]] = None,
        accel_range: Union[float, Tuple[float, float]] = None,
        interval: int = 100,
        delta_t: int = None,
    ):
        """Initialize the kinematic single-track model.

        Args:
            lf (float): The distance from the center of mass to the front axle center. The unit is meter.
            lr (float): The distance from the center of mass to the rear axle center. The unit is meter.
            mu (float, optional): The tire-road friction coefficient. It is a dimensionless quantity. Defaults to None. When set, the turn rate is capped so the lateral acceleration stays within `mu * g` (grip limit); when None (default), the pure no-slip kinematic model is used (unlimited grip, matches the CommonRoad reference exactly).
            steer_range (Union[float, Tuple[float, float]], optional): The range of steering angle. The valid input is a positive float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.
            speed_range (Union[float, Tuple[float, float]], optional): The range of speed. The valid input is a positive float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            accel_range (Union[float, Tuple[float, float]], optional): The range of acceleration. The valid input is a positive float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).
            interval (int, optional): The time interval between the current state and the new state. The unit is millisecond.
            delta_t (int, optional): The discrete time step for the simulation. The unit is millisecond.
        """
        self.lf = lf
        self.lr = lr
        self.wheel_base = lf + lr
        self.mu = mu

        if isinstance(steer_range, float):
            self.steer_range = None if steer_range < 0 else [-steer_range, steer_range]
        elif hasattr(steer_range, "__len__") and len(steer_range) == 2:
            if steer_range[0] >= steer_range[1]:
                self.steer_range = None
            else:
                self.steer_range = steer_range
        else:
            self.steer_range = None

        if isinstance(speed_range, float):
            self.speed_range = None if speed_range < 0 else [-speed_range, speed_range]
        elif hasattr(speed_range, "__len__") and len(speed_range) == 2:
            if speed_range[0] >= speed_range[1]:
                self.speed_range = None
            else:
                self.speed_range = speed_range
        else:
            self.speed_range = None

        if isinstance(accel_range, float):
            self.accel_range = None if accel_range < 0 else [-accel_range, accel_range]
        elif hasattr(accel_range, "__len__") and len(accel_range) == 2:
            if accel_range[0] >= accel_range[1]:
                self.accel_range = None
            else:
                self.accel_range = accel_range
        else:
            self.accel_range = None

        self.interval = interval

        if delta_t is None:
            self.delta_t = self._DELTA_T
        else:
            self.delta_t = max(delta_t, self._MIN_DELTA_T)
            if self.interval is not None:
                self.delta_t = min(self.delta_t, self.interval)

    def _step(self, state: State, accel: float, delta: float, interval: int) -> State:
        beta = math.atan(self.lr / self.wheel_base * math.tan(delta))  # slip angle
        dt = float(self.delta_t) / 1000
        n_steps = interval // self.delta_t
        remainder = interval % self.delta_t

        # Loop-invariant terms: delta and beta are constant across sub-steps.
        cos_beta = math.cos(beta)
        k_phi = math.tan(delta) * cos_beta / self.wheel_base
        speed_range = self.speed_range

        # Grip limit (mu * g) in m/s^2, if a friction coefficient was supplied.
        mu_g = None if self.mu is None else self.mu * self._G

        def _turn_rate(v):
            """Yaw rate at speed ``v``, capped so the lateral acceleration v * phi_dot
            stays within mu * g when a grip limit is set. Without a limit (mu=None) or at
            zero/negative speed this is the pure geometric rate v * k_phi, so the default
            model reproduces the reference kinematics exactly.
            """
            rate = v * k_phi
            if mu_g is None or v <= 0.0:
                return rate
            limit = mu_g / v
            if rate > limit:
                return limit
            if rate < -limit:
                return -limit
            return rate

        def _rk4_step(x, y, phi, v, h):
            """Advance one sub-step of time h with a 4th-order Runge-Kutta scheme.

            The equations of motion are
                dx/dt = v * cos(phi + beta), dy/dt = v * sin(phi + beta),
                dphi/dt = turn_rate(v), dv/dt = accel.
            Since dv/dt is constant, its stage slopes are identical and the velocity
            update reduces to v += accel * h (integrated exactly).
            """
            # Stage 1
            k1x = v * math.cos(phi + beta)
            k1y = v * math.sin(phi + beta)
            k1p = v * k_phi if mu_g is None else _turn_rate(v)
            # Stage 2
            v2 = v + 0.5 * h * accel
            p2 = phi + 0.5 * h * k1p
            k2x = v2 * math.cos(p2 + beta)
            k2y = v2 * math.sin(p2 + beta)
            k2p = v2 * k_phi if mu_g is None else _turn_rate(v2)
            # Stage 3
            v3 = v + 0.5 * h * accel
            p3 = phi + 0.5 * h * k2p
            k3x = v3 * math.cos(p3 + beta)
            k3y = v3 * math.sin(p3 + beta)
            k3p = v3 * k_phi if mu_g is None else _turn_rate(v3)
            # Stage 4
            v4 = v + h * accel
            p4 = phi + h * k3p
            k4x = v4 * math.cos(p4 + beta)
            k4y = v4 * math.sin(p4 + beta)
            k4p = v4 * k_phi if mu_g is None else _turn_rate(v4)

            x += h / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            y += h / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
            phi += h / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
            v += h * accel
            return x, y, phi, v

        x, y = state.location
        phi = state.heading
        v = state.speed

        # Main steps with standard delta_t
        for _ in range(n_steps):
            x, y, phi, v = _rk4_step(x, y, phi, v, dt)

            if speed_range is not None:
                if v < speed_range[0]:
                    v = speed_range[0]
                elif v > speed_range[1]:
                    v = speed_range[1]

        # Remainder step if any
        if remainder > 0:
            x, y, phi, v = _rk4_step(x, y, phi, v, float(remainder) / 1000)

            if speed_range is not None:
                if v < speed_range[0]:
                    v = speed_range[0]
                elif v > speed_range[1]:
                    v = speed_range[1]

        state = State(
            frame=state.frame + interval,
            x=x,
            y=y,
            heading=phi % (2 * math.pi),
            vx=v * math.cos(phi),
            vy=v * math.sin(phi),
            speed=v,
            accel=accel,
        )

        return state

    def step(self, state: State, accel: float, delta: float, interval: int = None) -> State:
        """This function updates the state of the traffic participant with the Kinematic Single-Track Model.

        Args:
            state (State): The current state of the traffic participant.
            accel (float): The acceleration of the traffic participant. The unit is meter per second squared (m/s$^2$).
            delta (float): The steering angle of the traffic participant. The unit is radian.
            interval (int): The time interval between the current state and the new state. The unit is millisecond.

        Returns:
            next_state (State): The new state of the traffic participant.
            accel (float): The acceleration that is applied to the traffic participant.
            delta (float): The steering angle that is applied to the traffic participant.
        """
        accel = np.clip(accel, *self.accel_range) if self.accel_range is not None else accel
        delta = np.clip(delta, *self.steer_range) if self.steer_range is not None else delta
        interval = interval if interval is not None else self.interval

        next_state = self._step(state, accel, delta, interval)

        return next_state, accel, delta

    def verify_state(self, state: State, last_state: State, interval: int = None) -> bool:
        """This function provides a very rough check for the state transition.

        Uses the shared single-track reachability check of `PhysicsModelBase`; when a grip
        limit (`mu`) is set the extreme-steer heading advance is capped like in `_step`.

        Args:
            state (State): The current state of the traffic participant.
            last_state (State): The last state of the traffic participant.
            interval (int, optional): The time interval between the last state and the new state. The unit is millisecond.

        Returns:
            True if the new state is valid, False otherwise.
        """
        return self._verify_kinematic_state(
            state, last_state, interval, grip=(self.mu * self._G if self.mu is not None else None)
        )
