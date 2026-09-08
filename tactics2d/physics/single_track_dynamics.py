# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Single track dynamics implementation."""


import math
from typing import Tuple, Union

import numpy as np

from tactics2d.participant.trajectory import State

from .physics_model_base import PhysicsModelBase


class SingleTrackDynamics(PhysicsModelBase):
    r"""This class implements a dynamic single-track model for a vehicle.

    The dynamic single-track model is a simplified model to simulate the vehicle dynamics. It combines the front and rear wheels into a single wheel, and the vehicle is assumed to be a point mass.

    ![Demo of the implementation (interval=100 ms, $\Delta t$=5 ms)](https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/tactics2d-single_track_dynamics.gif)

    !!! quote "Reference"
        The dynamic single-track model is based on Chapter 7 of the following reference:
        [CommonRoad: Vehicle Models (2020a)](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf)

    !!! info "Parameter sources"
        Defaults follow CommonRoad: Vehicle Models (2020a), vehicle ID 1 (Ford Escort):
        `I_z = 1538.85` kg·m$^2$; cornering stiffness `cf = cr = 20.89` rad$^{-1}$ is derived
        from the Pacejka tire data as `-p_ky1 / p_dy1`; `mu` is a physical tire-road friction
        coefficient (see the attribute docs). `lf`, `lr`, `mass` and `mass_height` come from the
        participant template.

    Attributes:
        lf (float): The distance from the geometry center to the front axle center. The unit is meter.
        lr (float): The distance from the geometry center to the rear axle center. The unit is meter.
        steer_range (Union[float, Tuple[float, float]], optional): The steering angle range. The valid input is a float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.

            - When the steer_range is a non-negative float, the steering angle is constrained to be within the range [-steer_range, steer_range].
            - When the steer_range is a tuple, the steering angle is constrained to be within the range [min steering angle, max steering angle].
            - When the steer_range is negative or the min steering angle is not less than the max steering angle, the steer_range is set to None.

        mass (float): The mass of the vehicle. The unit is kilogram.
        mass_height (float): The height of the center of mass from the ground. The unit is meter.
        mu (float): The friction coefficient. It is a dimensionless quantity. Defaults to 0.7. This is a physical tire-road friction coefficient (dry asphalt ≈ 0.7–0.9); the CommonRoad ST model this class is based on uses the magic-formula ``p_dy1`` (1.0489) in its place, so set ``mu = p_dy1`` to match the reference exactly.
        friction_ellipse (bool): Whether to couple longitudinal and lateral grip through a friction ellipse. Defaults to False. When enabled, the lateral-response coefficient used by the high-speed branch is reduced to ``mu * sqrt(1 - (a / (mu * g))**2)``, so accelerating or braking consumes part of the available friction and full-throttle/hard-brake manoeuvres (|a| >= mu * g) leave no lateral capability; at a = 0 the behaviour is unchanged.
        I_z (float): The moment of inertia of the vehicle. The unit is kilogram per meter squared (kg/m$^2$). Defaults to 1538.85 (CommonRoad vehicle ID 1, Ford Escort).
        cf (float): The cornering stiffness of the front wheel. The unit is 1/rad. Defaults to 20.89 (= -p_ky1/p_dy1 derived from the Pacejka tire data).
        cr (float): The cornering stiffness of the rear wheel. The unit is 1/rad. Defaults to 20.89 (= -p_ky1/p_dy1 derived from the Pacejka tire data).

        speed_range (Union[float, Tuple[float, float]], optional): The speed range. The valid input is a float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            - When the speed_range is a non-negative float, the speed is constrained to be within the range [-speed_range, speed_range].
            - When the speed_range is a tuple, the speed is constrained to be within the range [min speed, max speed].
            - When the speed_range is negative or the min speed is not less than the max speed, the speed_range is set to None.

        accel_range (Union[float, Tuple[float, float]], optional): The acceleration range. The valid input is a float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).

            - When the accel_range is a non-negative float, the acceleration is constrained to be within the range [-accel_range, accel_range].
            - When the accel_range is a tuple, the acceleration is constrained to be within the range [min acceleration, max acceleration].
            - When the accel_range is negative or the min acceleration is not less than the max acceleration, the accel_range is set to None.

        interval (int, optional): The time interval between the current state and the new state. The unit is millisecond. Defaults to None.
        delta_t (int, optional): The time step for the simulation. The unit is millisecond. Defaults to `_DELTA_T`(5 ms). The expected value is between `_MIN_DELTA_T`(1 ms) and `interval`. The model integrates with a 4th-order Runge-Kutta scheme, so discretization error stays negligible up to ~20 ms; the default 5 ms is retained for backward compatibility.
    """

    def __init__(
        self,
        lf: float,
        lr: float,
        mass: float,
        mass_height: float,
        mu: float = 0.7,
        friction_ellipse: bool = False,
        I_z: float = 1538.85,
        cf: float = 20.89,
        cr: float = 20.89,
        steer_range: Union[float, Tuple[float, float]] = None,
        speed_range: Union[float, Tuple[float, float]] = None,
        accel_range: Union[float, Tuple[float, float]] = None,
        interval: int = 100,
        delta_t: int = None,
    ):
        """Initializes the single-track dynamics model.

        Args:
            lf (float): The distance from the center of mass to the front axle center. The unit is meter.
            lr (float): The distance from the center of mass to the rear axle center. The unit is meter.
            mass (float): The mass of the vehicle. The unit is kilogram. You can use the curb weight of the vehicle as an approximation.
            mass_height (float): The height of the center of mass from the ground. The unit is meter. You can use half of the vehicle height as an approximation.
            mu (float): The friction coefficient. It is a dimensionless quantity.
            friction_ellipse (bool): Whether to couple longitudinal and lateral grip through a friction ellipse. Defaults to False. When True, the lateral-response coefficient used by the high-speed branch is reduced to ``mu * sqrt(1 - (a / (mu * g))**2)``.
            I_z (float): The moment of inertia of the vehicle. The unit is kilogram per meter squared (kg/m$^2$).
            cf (float): The cornering stiffness of the front wheel. The unit is 1/rad.
            cr (float): The cornering stiffness of the rear wheel. The unit is 1/rad.
            steer_range (Union[float, Tuple[float, float]], optional): The range of steering angle. The valid input is a positive float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.
            speed_range (Union[float, Tuple[float, float]], optional): The range of speed. The valid input is a positive float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            accel_range (Union[float, Tuple[float, float]], optional): The range of acceleration. The valid input is a positive float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).
            interval (int, optional): The time interval between the current state and the new state. The unit is millisecond.
            delta_t (int, optional): The discrete time step for the simulation. The unit is millisecond.
        """
        self.lf = lf
        self.lr = lr
        self.wheel_base = lf + lr
        self.mass = mass
        self.mass_height = mass_height
        self.mu = mu
        self.friction_ellipse = friction_ellipse
        self.I_z = I_z
        self.cf = cf
        self.cr = cr

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

    def _step(self, state: State, accel: Tuple[float, float], delta: float, interval: int) -> State:
        dt = float(self.delta_t) / 1000
        n_steps = interval // self.delta_t
        remainder = interval % self.delta_t

        factor_f = (self._G * self.lr - accel * self.mass_height) / self.wheel_base
        factor_r = (self._G * self.lf + accel * self.mass_height) / self.wheel_base

        # Friction-ellipse coupling: when enabled, longitudinal acceleration consumes part of
        # the available friction, so the lateral-response coefficient is reduced to
        #   mu * sqrt(1 - (a / (mu * g))^2)
        # (0 when |a| >= mu * g). accel is constant within one step(), so mu_lat is constant.
        if self.friction_ellipse and self.mu > 0:
            _ratio = accel / (self.mu * self._G)
            mu_lat = self.mu * math.sqrt(max(0.0, 1.0 - _ratio * _ratio))
        else:
            mu_lat = self.mu

        # Precompute constant combinations for efficiency
        lf_cf_factor_f = self.lf * self.cf * factor_f
        lr_cr_factor_r = self.lr * self.cr * factor_r
        lf2_cf_factor_f = self.lf**2 * self.cf * factor_f
        lr2_cr_factor_r = self.lr**2 * self.cr * factor_r
        cf_factor_f = self.cf * factor_f
        cr_factor_r = self.cr * factor_r

        # Loop-invariant terms: delta is constant across sub-steps.
        tan_delta = math.tan(delta)
        cos_delta = math.cos(delta)
        cos2_delta = cos_delta**2
        denom2 = (1 + tan_delta * self.lr / self.wheel_base) ** 2
        speed_range = self.speed_range

        x, y = state.location
        phi = state.heading
        v = state.speed
        # d_phi (yaw rate) and beta (slip angle) are persistent states carried by the State
        # object between step() calls; fall back to the kinematic initial condition when absent.
        d_phi = state.yaw_rate if state.yaw_rate is not None else v / self.wheel_base * tan_delta
        beta = (
            state.slip_angle
            if state.slip_angle is not None
            else math.atan(self.lr / self.lf * tan_delta)
        )

        def _deriv(v, phi, beta, d_phi):
            """Right-hand side of the dynamics ODE at a given state.

            Returns (dx/dt, dy/dt, dphi/dt, dv/dt, dd_phi, d_beta), treating d_phi and beta
            as state variables with time derivatives dd_phi and d_beta.
            """
            dx = v * math.cos(phi + beta)
            dy = v * math.sin(phi + beta)
            f_phi = d_phi
            f_v = accel

            # Use a safe velocity to avoid division by zero
            v_safe = v if abs(v) > 1e-6 else (1e-6 if v >= 0 else -1e-6)

            if abs(v) >= 0.1:
                dd_phi = (
                    mu_lat
                    * self.mass
                    / self.I_z
                    * (
                        lf_cf_factor_f * delta
                        + (lr_cr_factor_r - lf_cf_factor_f) * beta
                        - (lf2_cf_factor_f + lr2_cr_factor_r) * d_phi / v_safe
                    )
                )
                d_beta = (
                    mu_lat
                    / v_safe
                    * (
                        cf_factor_f * delta
                        - (cr_factor_r + cf_factor_f) * beta
                        + (lr_cr_factor_r - lf_cf_factor_f) * d_phi / v_safe
                    )
                    - d_phi
                )
            else:
                cos_beta = math.cos(beta)
                sin_beta = math.sin(beta)
                d_beta = self.lr / denom2 / self.wheel_base / cos2_delta * delta
                dd_phi = (
                    1
                    / self.wheel_base
                    * (
                        accel * cos_beta * tan_delta
                        - v * sin_beta * tan_delta * d_beta
                        + v * cos_beta / cos2_delta * delta
                    )
                )

            return dx, dy, f_phi, f_v, dd_phi, d_beta

        def _rk4_step(x, y, phi, v, d_phi, beta, h):
            """Advance one sub-step of time h with a 4th-order Runge-Kutta scheme."""
            k1 = _deriv(v, phi, beta, d_phi)
            k2 = _deriv(
                v + 0.5 * h * k1[3],
                phi + 0.5 * h * k1[2],
                beta + 0.5 * h * k1[5],
                d_phi + 0.5 * h * k1[4],
            )
            k3 = _deriv(
                v + 0.5 * h * k2[3],
                phi + 0.5 * h * k2[2],
                beta + 0.5 * h * k2[5],
                d_phi + 0.5 * h * k2[4],
            )
            k4 = _deriv(v + h * k3[3], phi + h * k3[2], beta + h * k3[5], d_phi + h * k3[4])

            x += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            y += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            phi += h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            v += h / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            d_phi += h / 6.0 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
            beta += h / 6.0 * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])
            return x, y, phi, v, d_phi, beta

        # Main steps with standard delta_t
        for _ in range(n_steps):
            x, y, phi, v, d_phi, beta = _rk4_step(x, y, phi, v, d_phi, beta, dt)

            if speed_range is not None:
                if v < speed_range[0]:
                    v = speed_range[0]
                elif v > speed_range[1]:
                    v = speed_range[1]

        # Remainder step if any
        if remainder > 0:
            x, y, phi, v, d_phi, beta = _rk4_step(
                x, y, phi, v, d_phi, beta, float(remainder) / 1000
            )

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
            speed=v,
            accel=accel,
            yaw_rate=d_phi,
            slip_angle=beta,
        )

        return state

    def step(self, state: State, accel: float, delta: float, interval: int = None) -> State:
        """This function updates the state of the vehicle based on the dynamics single-track model.

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

        !!! info
        Uses the shared single-track reachability check of `PhysicsModelBase` (same rough
        check as the single track kinematics model).

        Args:
            state (State): The current state of the traffic participant.
            last_state (State): The last state of the traffic participant.
            interval (int, optional): The time interval between the last state and the new state. The unit is millisecond.

        Returns:
            True if the new state is valid, False otherwise.
        """
        return self._verify_kinematic_state(state, last_state, interval)
