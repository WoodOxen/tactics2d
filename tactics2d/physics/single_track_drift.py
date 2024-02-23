##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: single_track_drift.py
# @Description: This file implements a dynamic single-track model for a vehicle.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Tuple, Union

import numpy as np

from tactics2d.participant.trajectory import State

from .physics_model_base import PhysicsModelBase


class Tire:
    # longitudinal parameters
    p_cx1 = 1.6411  # shape factor for longitudinal force
    p_dx1 = 1.1739  # longitudinal friction coefficient mu_x at F_z0
    p_dx3 = 0.0  # variation of friction coefficient mu_x with camber
    p_ex1 = 0.4640  # longitudinal curvature at F_z0
    p_kx1 = 22.303  # longitudinal slip stiffness at F_z0
    p_hx1 = 1.2297e-3  # horizontal shift at F_z0
    p_vx1 = -8.8098e-6  # vertical shift at F_z0
    r_bx1 = 13.276  # slope factor for combined slip F_x reduction
    r_bx2 = -13.778  # variation of slope F_x reduction with kappa
    r_ex1 = 1.2568  # shape factor for combined slip F_x reduction
    r_cx1 = 0.6522  # curvature factor for combined F_x
    r_hx1 = 5.0722e-3  # shift factor for combined slip F_x reduction
    # lateral parameters
    p_cy1 = 1.3507  # shape factor for lateral force
    p_dy1 = 1.0489  # lateral friction coefficient mu_y
    p_dy3 = -2.8821  # variation of friction coefficient mu_y with squared camber
    p_ey1 = -7.4722e-3  # lateral curvature at F_z0
    p_ky1 = -21.920  # maximum value of stiffness
    p_hy1 = 2.6747e-3  # horizontal shift at F_z0
    p_hy3 = 3.1415e-2  # variation of shift with camber
    p_vy1 = 3.7318e-2  # vertical shift at F_z0
    p_vy3 = -0.3293  # variation of vertical shift with camber
    r_by1 = 7.1433  # slope factor for combined slip F_y reduction
    r_by2 = 9.1917  # variation of slope F_y reduction with alpha
    r_by3 = -2.7856e-2  # shift term for alpha in slope F_y reduction
    r_cy1 = 1.0719  # shape factor for combined F_y reduction
    r_ey1 = -0.2757  # curvature factor for combined F_y
    r_hy1 = 5.7448e-6  # shift factor for combined slip F_y reduction
    r_vy1 = -2.7825e-2  # kappa-induced side force at F_z0
    r_vy3 = -0.2756  # variation of S_vy_kappa/mu_y F_z with camber
    r_vy4 = 12.120  # variation of S_vy_kappa/mu_y F_z with alpha
    r_vy5 = 1.9  # variation of S_vy_kappa/mu_y F_z with kappa
    r_vy6 = -10.704  # variation of S_vy_kappa/mu_y F_z with arctan(kappa)


class SingleTrackDrift(PhysicsModelBase):
    """This class implements a dynamic single-track model for a vehicle.

    !!! warning
        This class was designed "as a simplification of the multi-body" model. Theoretically, it is applicable to the All-Wheel-Drive (AWD) vehicle. However, the tire model is so complicated that it is not fully tested in `tactics2d` v1.0.0. The current implementation is based on the MATLAB code provided by the CommonRoad project. Please use it with caution.

    !!! quote "Reference"
        The dynamic single-track model is based on Chapter 8 of the following reference:
        [CommonRoad: Vehicle Models (2020a)](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf)

        Pacejka, Hans. *Tire and vehicle dynamics.* Elsevier, 2005.

    Attributes:
        lf (float): The distance from the center of mass to the front axle. The unit is meter (m).
        lr (float): The distance from the center of mass to the rear axle. The unit is meter (m).
        mass (float): The mass of the vehicle. The unit is kilogram (kg).
        mass_height (float): The height of the center of mass. The unit is meter (m).
        radius (float): The effective radius of the wheel. The unit is meter (m). Defaults to 0.344.
        T_sb (float): The split parameter between the front and rear axles for the braking torque. Defaults to 0.76.
        T_se (float): The split parameter between the front and rear axles for the engine torque. Defaults to 1.
        tire (Any): The tire model. Default to the in-built tire model.
        I_z (float): The moment of inertia of the vehicle. The unit is kilogram meter squared (kg m^2). Defaults to 1500.
        I_yw (float): The moment of inertia of the wheel. The unit is kilogram meter squared (kg m^2). Defaults to 1.7.
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
        delta_t (int, optional): The default time interval between the current state and the new state, 5 milliseconds (ms). Defaults to None.
    """

    def __init__(
        self,
        lf: float,
        lr: float,
        mass: float,
        mass_height: float,
        radius: float = 0.344,
        T_sb: float = 0.76,
        T_se: float = 1,
        tire=Tire(),
        I_z: float = 1500,
        I_yw: float = 1.7,
        steer_range: Union[float, Tuple[float, float]] = None,
        speed_range: Union[float, Tuple[float, float]] = None,
        accel_range: Union[float, Tuple[float, float]] = None,
        interval: int = 100,
        delta_t: int = None,
    ):
        """Initializes the single-track drift model.

        Args:
            lf (float): The distance from the center of mass to the front axle center. The unit is meter.
            lr (float): The distance from the center of mass to the rear axle center. The unit is meter.
            mass (float): The mass of the vehicle. The unit is kilogram. You can use the curb weight of the vehicle as an approximation.
            mass_height (float): The height of the center of mass from the ground. The unit is meter. You can use half of the vehicle height as an approximation.
            radius (float, optional): The effective radius of the wheel. The unit is meter.
            T_sb (float): The split parameter between the front and rear axles for the braking torque.
            T_se (float): The split parameter between the front and rear axles for the engine torque.
            tire (Any): The tire model. The current implementation refers to the parameters in [CommonRoad: Vehicle Models (2020a)](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf). If you want to use a different tire model, you need to implement the tire model by yourself.
            I_z (float): The moment of inertia of the vehicle. The unit is kilogram meter squared (kg m^2).
            I_yw (float): The moment of inertia of the wheel. The unit is kilogram meter squared (kg m^2).
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
        self.radius = radius
        self.tire = tire
        self.T_sb = T_sb
        self.T_se = T_se
        self.I_z = I_z
        self.I_yw = I_yw

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

    def _pure_slip_longitudinal_tire_forces(
        self, kappa: float, gamma: float, F_z: float
    ) -> Tuple[float]:
        S_hx = self.tire.p_hx1
        S_vx = self.tire.p_vx1 * F_z
        kappa = -kappa

        kappa_x = kappa + S_hx
        mu_x = self.tire.p_dx1 * (1 - self.tire.p_dx3 * gamma**2)

        C_x = self.tire.p_cx1
        D_x = mu_x * F_z
        E_x = self.tire.p_ex1
        K_x = self.tire.p_kx1 * F_z
        B_x = K_x / (C_x * D_x + 1e-6)

        F_x = D_x * np.sin(
            C_x * np.arctan(B_x * kappa_x - E_x * (B_x * kappa_x - np.arctan(B_x * kappa_x))) + S_vx
        )

        return F_x

    def _pure_slip_lateral_tire_forces(
        self, alpha: float, gamma: float, F_z: float
    ) -> Tuple[float]:
        S_hy = np.sign(gamma) * (self.tire.p_hy1 + self.tire.p_hy3 * np.abs(gamma))
        S_vy = S_hy * F_z

        alpha_y = alpha + S_hy
        mu_y = self.tire.p_dy1 * (1 - self.tire.p_dy3 * gamma**2)

        C_y = self.tire.p_cy1
        D_y = mu_y * F_z
        E_y = self.tire.p_ey1
        K_y = self.tire.p_ky1 * F_z
        B_y = K_y / (C_y * D_y + 1e-6)

        F_y = D_y * np.sin(
            C_y * np.arctan(B_y * alpha_y - E_y * (B_y * alpha_y - np.arctan(B_y * alpha_y))) + S_vy
        )

        return F_y, mu_y

    def _combined_slip_longitudinal_tire_forces(
        self, kappa: float, alpha: float, F0_x: float
    ) -> Tuple[float]:
        S_hx_alpha = self.tire.r_hx1
        alpha_s = alpha + S_hx_alpha

        B_x_alpha = self.tire.r_bx1 * np.cos(np.arctan(self.tire.r_bx2 * kappa))
        C_x_alpha = self.tire.r_cx1
        E_x_alpha = self.tire.r_ex1
        D_x_alpha = F0_x / np.cos(
            C_x_alpha
            * np.arctan(
                B_x_alpha * S_hx_alpha
                - E_x_alpha * (B_x_alpha * S_hx_alpha - np.arctan(B_x_alpha * S_hx_alpha))
            )
        )

        F_x = D_x_alpha * np.cos(
            C_x_alpha
            * np.arctan(
                B_x_alpha * alpha_s
                - E_x_alpha * (B_x_alpha * alpha_s - np.arctan(B_x_alpha * alpha_s))
            )
        )

        return F_x

    def _combined_slip_lateral_tire_forces(
        self, kappa, alpha, gamma, mu_y, F_z, F0_y
    ) -> Tuple[float]:
        S_hy_kappa = self.tire.r_hy1
        kappa_s = kappa + S_hy_kappa

        B_y_kappa = self.tire.r_by1 * np.cos(np.arctan(self.tire.r_by2 * (alpha - self.tire.r_by3)))
        C_y_kappa = self.tire.r_cy1
        E_y_kappa = self.tire.r_ey1
        D_y_kappa = F0_y / np.cos(
            C_y_kappa
            * np.arctan(
                B_y_kappa * S_hy_kappa
                - E_y_kappa * (B_y_kappa * S_hy_kappa - np.arctan(B_y_kappa * S_hy_kappa))
            )
        )

        D_vy_kappa = (
            mu_y
            * F_z
            * (self.tire.r_vy1 + self.tire.r_vy3 * gamma)
            * np.cos(np.arctan(self.tire.r_vy4 * alpha))
        )
        S_vy_kappa = D_vy_kappa * np.sin(self.tire.r_vy5 * np.arctan(self.tire.r_vy6 * kappa))

        F_y = (
            D_y_kappa
            * np.cos(
                C_y_kappa
                * np.arctan(
                    B_y_kappa * kappa_s
                    - E_y_kappa * (B_y_kappa * kappa_s - np.arctan(B_y_kappa * kappa_s))
                )
            )
            + S_vy_kappa
        )

        return F_y

    def _tire_forces(
        self, v: float, delta: float, d_phi, beta: float, omega_wf: float, omega_wr: float
    ) -> Tuple[float]:
        # compute lateral tire slip angles:
        alpha_f = np.arctan((v * np.sin(beta) + d_phi * self.lf) / (v * np.cos(beta))) - delta
        alpha_r = np.arctan((v * np.sin(beta) - d_phi * self.lr) / (v * np.cos(beta)))

        # compute vertical tire forces
        F_zf = (self.mass * self._G * self.lr) / self.wheel_base
        F_zr = (self.mass * self._G * self.lf) / self.wheel_base

        # compute front and rear tire speeds
        u_wf = v * np.cos(beta) * np.cos(delta) + (v * np.sin(beta) + self.lf * d_phi) * np.sin(
            delta
        )
        u_wr = v * np.cos(beta)

        # computer longitudinal tire slip
        s_f = 1 - self.radius * omega_wf / u_wf
        s_r = 1 - self.radius * omega_wr / u_wr

        # compute tire forces using Pacejka's magic formula
        # pure slip longitudinal tire forces
        F0_xf = self._pure_slip_longitudinal_tire_forces(s_f, 0, F_zf)
        F0_xr = self._pure_slip_longitudinal_tire_forces(s_r, 0, F_zr)

        # pure slip lateral tire forces
        F0_yf, mu_yf = self._pure_slip_lateral_tire_forces(alpha_f, 0, F_zf)
        F0_yr, mu_yr = self._pure_slip_lateral_tire_forces(alpha_r, 0, F_zr)

        # combined slip longitudinal tire forces
        F_xf = self._combined_slip_longitudinal_tire_forces(s_f, alpha_f, F0_xf)
        F_xr = self._combined_slip_longitudinal_tire_forces(s_r, alpha_r, F0_xr)

        # combined slip lateral tire forces
        F_yf = self._combined_slip_lateral_tire_forces(s_f, alpha_f, 0, mu_yf, F_zf, F0_yf)
        F_yr = self._combined_slip_lateral_tire_forces(s_r, alpha_r, 0, mu_yr, F_zr, F0_yr)

        return F_xf, F_xr, F_yf, F_yr

    def _step(
        self,
        state: State,
        omega_wf: float,
        omega_wr: float,
        accel: float,
        delta: float,
        interval: int,
    ) -> State:
        # Completely refer to https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/MATLAB/vehiclemodels/vehicleDynamics_STD.m . I haven't fully understood the implementation yet.

        dts = [float(self.delta_t) / 1000] * (interval // self.delta_t)
        dts.append(float(interval % self.delta_t) / 1000)

        x, y = state.location
        v = state.speed
        phi = state.heading
        d_phi = v / self.wheel_base * np.tan(delta)
        beta = np.arctan(self.lr / self.lf * np.tan(delta))  # slip angle

        if accel > 0:
            T_B = 0
            T_E = self.mass * self.radius * accel
        else:
            T_B = self.mass * self.radius * accel
            T_E = 0

        for dt in dts:
            F_lf, F_lr, F_sf, F_sr = self._tire_forces(v, delta, d_phi, beta, omega_wf, omega_wr)

            dx = v * np.cos(phi + beta)
            dy = v * np.sin(phi + beta)

            if np.abs(v) >= 0.1:
                dv = (
                    1
                    / self.mass
                    * (
                        -F_sf * np.sin(delta - beta)
                        + F_sr * np.sin(beta)
                        + F_lr * np.cos(beta)
                        + F_lf * np.cos(delta - beta)
                    )
                )
                d_beta = -d_phi + 1 / (self.mass * v) * (
                    F_sf * np.cos(delta - beta)
                    + F_sr * np.cos(beta)
                    - F_lr * np.sin(beta)
                    + F_lf * np.sin(delta - beta)
                )
                dd_phi = (
                    1
                    / self.I_z
                    * (
                        F_sf * np.cos(delta) * self.lf
                        - F_sr * self.lr
                        + F_lf * np.sin(delta) * self.lf
                    )
                )
                d_phi += dd_phi * dt
                d_omega_wf = (
                    1 / self.I_yw * (-self.radius * F_lf + self.T_sb * T_B + self.T_se * T_E)
                )
                d_omega_wr = (
                    1
                    / self.I_yw
                    * (-self.radius * F_lr + (1 - self.T_sb) * T_B + (1 - self.T_se) * T_E)
                )
            else:
                dv = accel
                d_beta = (
                    self.lr
                    / (1 + np.tan(delta) * self.lr / self.wheel_base) ** 2
                    / self.wheel_base
                    / np.cos(delta) ** 2
                    * delta
                )
                dd_phi = (
                    1
                    / self.wheel_base
                    * (
                        accel * np.cos(beta) * np.tan(delta)
                        - v * np.sin(beta) * np.tan(delta) * d_beta
                        + v * np.cos(beta) / np.cos(delta) ** 2 * delta
                    )
                )
                d_phi += v * np.cos(beta) / self.wheel_base * np.tan(delta) * dt
                d_omega_wf = (
                    1
                    / (np.cos(delta) * self.radius)
                    * (
                        accel * np.cos(beta)
                        - v * np.sin(beta) * d_beta
                        + v * np.cos(beta) * np.tan(delta) * delta
                    )
                )
                d_omega_wr = 1 / self.radius * (accel * np.cos(beta) - v * np.sin(beta) * d_beta)

            x += dx * dt
            y += dy * dt
            v += dv * dt
            phi += d_phi * dt
            beta += d_beta * dt

            omega_wf += d_omega_wf * dt
            omega_wr += d_omega_wr * dt

            v = np.clip(v, *self.speed_range) if not self.speed_range is None else v

        state = State(
            frame=state.frame + interval,
            x=x,
            y=y,
            heading=np.mod(phi, 2 * np.pi),
            speed=v,
            accel=accel,
        )

        return state, omega_wf, omega_wr

    def step(
        self,
        state: State,
        omega_wf: float,
        omega_wr: float,
        accel: float,
        delta: float,
        interval: int = None,
    ) -> Tuple[State, float, float]:
        """This function updates the state of the traffic participant based on the single-track drift model.

        Args:
            state (State): The current state of the traffic participant.
            omega_wf (float): The angular velocity of the front wheel. The unit is radian per second (rad/s).
            omega_wr (float): The angular velocity of the rear wheel. The unit is radian per second (rad/s).
            accel (float): The acceleration of the traffic participant. The unit is meter per second squared (m/s$^2$).
            delta (float): The steering angle of the traffic participant. The unit is radian.
            interval (int): The time interval between the current state and the new state. The unit is millisecond.

        Returns:
            next_state (State): The new state of the traffic participant.
            next_omega_wf (float): The new angular velocity of the front wheel. The unit is radian per second (rad/s).
            next_omega_wr (float): The new angular velocity of the rear wheel. The unit is radian per second (rad/s).
            accel (float): The acceleration that is applied to the traffic participant.
            delta (float): The steering angle that is applied to the traffic participant.
        """
        accel = np.clip(accel, *self.accel_range) if not self.accel_range is None else accel
        delta = np.clip(delta, *self.steer_range) if not self.steer_range is None else delta
        interval = interval if interval is not None else self.interval

        next_state, next_omega_wf, next_omega_wr = self._step(
            state, omega_wf, omega_wr, accel, delta, interval
        )

        return next_state, next_omega_wf, next_omega_wr, accel, delta

    def verify_state(self, state: State, last_state: State, interval: int = None) -> bool:
        """This function provides a very rough check for the state transition.

        !!! info
        Uses the same rough check as the single track kinematics model.

        Args:
            state (State): The current state of the traffic participant.
            last_state (State): The last state of the traffic participant.
            interval (int, optional): The time interval between the last state and the new state. The unit is millisecond.

        Returns:
            True if the new state is valid, False otherwise.
        """
        interval = interval if interval is None else state.frame - last_state.frame
        dt = float(interval) / 1000
        last_speed = last_state.speed

        if None in [self.steer_range, self.speed_range, self.accel_range]:
            return True

        steer_range = np.array(self.steer_range)
        beta_range = np.arctan(self.lr / self.wheel_base * steer_range)

        # check that heading is in the range. heading_range may be larger than 2 * np.pi
        heading_range = np.mod(
            last_state.heading + last_speed / self.wheel_base * np.sin(beta_range) * dt, 2 * np.pi
        )
        if (
            heading_range[0] < heading_range[1]
            and not heading_range[0] <= state.heading <= heading_range[1]
        ):
            return False
        if heading_range[0] > heading_range[1] and not (
            heading_range[0] <= state.heading or state.heading <= heading_range[1]
        ):
            return False

        # check that speed is in the range
        speed_range = np.clip(last_speed + np.array(self.accel_range) * dt, *self.speed_range)
        if not speed_range[0] <= state.speed <= speed_range[1]:
            return False

        # check that x, y are in the range
        x_range = last_state.x + speed_range * np.cos(last_state.heading + beta_range) * dt
        y_range = last_state.y + speed_range * np.sin(last_state.heading + beta_range) * dt

        if not x_range[0] < state.x < x_range[1] or not y_range[0] < state.y < y_range[1]:
            return False

        return True
