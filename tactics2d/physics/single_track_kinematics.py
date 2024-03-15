##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: single_track_kinematics.py
# @Description: This file implements a kinematic single-track model for a traffic participant.
# @Author: Yueyuan Li
# @Version: 1.0.0

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
        This model will lose its accuracy when the time step is set too large or the traffic participant is made to travel at a high speed.

    Attributes:
        lf (float): The distance from the geometry center to the front axle center. The unit is meter.
        lr (float): The distance from the geometry center to the rear axle center. The unit is meter.
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
        delta_t (int, optional): The time step for the simulation. The unit is millisecond. Defaults to `_DELTA_T`(5 ms). The expected value is between `_MIN_DELTA_T`(1 ms) and `interval`. It is recommended to keep delta_t smaller than 5 ms.
    """

    def __init__(
        self,
        lf: float,
        lr: float,
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
            steer_range (Union[float, Tuple[float, float]], optional): The range of steering angle. The valid input is a positive float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.
            speed_range (Union[float, Tuple[float, float]], optional): The range of speed. The valid input is a positive float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            accel_range (Union[float, Tuple[float, float]], optional): The range of acceleration. The valid input is a positive float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).
            interval (int, optional): The time interval between the current state and the new state. The unit is millisecond.
            delta_t (int, optional): The discrete time step for the simulation. The unit is millisecond.
        """
        self.lf = lf
        self.lr = lr
        self.wheel_base = lf + lr

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
        beta = np.arctan(self.lr / self.wheel_base * np.tan(delta))  # slip angle
        dts = [float(self.delta_t) / 1000] * (interval // self.delta_t)
        dts.append(float(interval % self.delta_t) / 1000)

        x, y = state.location
        phi = state.heading
        v = state.speed

        for dt in dts:
            dx = v * np.cos(phi + beta)
            dy = v * np.sin(phi + beta)
            dv = accel
            dphi = v / self.wheel_base * np.tan(delta)

            x += dx * dt
            y += dy * dt
            phi += dphi * dt
            v += dv * dt

            v = np.clip(v, *self.speed_range) if not self.speed_range is None else v

        state = State(
            frame=state.frame + interval,
            x=x,
            y=y,
            heading=np.mod(phi, 2 * np.pi),
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
        accel = np.clip(accel, *self.accel_range) if not self.accel_range is None else accel
        delta = np.clip(delta, *self.steer_range) if not self.steer_range is None else delta
        interval = interval if interval is not None else self.interval

        next_state = self._step(state, accel, delta, interval)

        return next_state, accel, delta

    def verify_state(self, state: State, last_state: State, interval: int = None) -> bool:
        """This function provides a very rough check for the state transition.

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
