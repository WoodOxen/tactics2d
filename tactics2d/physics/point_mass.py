from typing import Tuple
from copy import deepcopy
import logging

import numpy as np

from .physics_model_base import PhysicsModelBase
from tactics2d.trajectory.element import State


class PointMass(PhysicsModelBase):
    """This class implements a point mass model for a traffic participant. The point mass model supposes that the mass of the object is concentrated at the center of the object. The state of the object is represented by its center position, velocity, and heading. The object is assumed to be operating in a 2D plane (x-y).

    This model is recommended to be used for pedestrians. If this model is used for bicycle and vehicles, the results may not be accurate.

    Attributes:
        steer_range (Tuple[float, float]): The range of the steering angle. The unit is radian. Defaults to None.
        speed_range (Tuple[float, float]): The range of the speed. The unit is meter per second (m/s). Defaults to None.
        accel_range (Tuple[float, float]): The range of the acceleration. The unit is meter per second squared (m/s$^2$). Defaults to None.
        interval (int): The time interval between the current state and the new state. The unit is millisecond. Defaults to None.
        delta_t (int): The discrete time step for the simulation. The unit is millisecond. Defaults to `DELTA_T`(5 ms). The expected value is between `MIN_DELTA_T`(1 ms) and `interval`.
        backend (str): The backend for the simulation. The default value is `newton`. The available choices are `newton` and `euler`.
    """

    backends = ["newton", "euler"]

    def __init__(
        self,
        steer_range: Tuple[float, float] = None,
        speed_range: Tuple[float, float] = None,
        accel_range: Tuple[float, float] = None,
        interval: int = None,
        delta_t: int = None,
        backend: str = "newton",
    ):
        self.steer_range = steer_range
        self.speed_range = speed_range
        self.accel_range = accel_range
        self.interval = interval

        if delta_t is None:
            self.delta_t = self.DELTA_T
        else:
            self.delta_t = max(delta_t, self.MIN_DELTA_T)
            if self.interval is not None:
                self.delta_t = min(self.delta_t, self.interval)

        if backend not in self.backends:
            logging.warning(f"Unsupported backend {backend}. Using `newton` instead.")
            self.backend = "newton"
        else:
            self.backend = backend

    def _step_newton(self, state: State, accel: float, steer: float, interval: int) -> State:
        vx, vy = state.velocity
        ax = accel * np.cos(state.heading + steer)
        ay = accel * np.sin(state.heading + steer)
        interval = float(interval) / 1000

        next_vx = vx + ax * interval
        next_vy = vy + ay * interval
        next_speed = np.linalg.norm([next_vx, next_vy])

        # When the speed has not exceeded the constraint, the next state can be calculated directly.
        if self.speed_range is None or self.speed_range[0] <= next_speed <= self.speed_range[1]:
            next_state = State(
                frame=state.frame + interval,
                x=state.x + vx * interval + 0.5 * ax * interval**2,
                y=state.y + vy * interval + 0.5 * ay * interval**2,
                heading=np.arctan2(next_vy, next_vx),
                vx=next_vx,
                vy=next_vy,
            )
        # When the speed has exceeded the constraint, the next state needs to be calculated in two steps.
        # 1. from current velocity to the constraint velocity
        # 2. the velocity is constant, but the heading is changing
        elif next_speed < self.speed_range[0]:
            a_ = ax**2 + ay**2
            b_ = 2 * (ax * vx + ay * vy)
            c_ = vx**2 + vy**2 - self.speed_range[0] ** 2
            t1 = (-b_ - np.sqrt(b_**2 - 4 * a_ * c_)) / (
                2 * a_
            )  # assume the minimal speed is positive
            t2 = interval - t1
            vx1 = vx + ax * t1
            vy1 = vy + ay * t1
            theta1 = np.arctan2(vy1, vx1)
            theta2 = steer * t2
            next_heading = theta1 + theta2
            R = self.accel_range[1] / theta2
            next_state = State(
                frame=state.frame + interval,
                x=state.x
                + vx * t1
                + 0.5 * ax * t1**2
                + R * (np.cos(next_heading + np.pi / 2) - np.cos(theta1 + np.pi / 2)),
                y=state.y
                + vy * t1
                + 0.5 * ay * t1**2
                + R * (np.sin(next_heading + np.pi / 2) - np.sin(theta1 + np.pi / 2)),
                heading=theta1 + theta2,
                vx=self.speed_range[0] * np.cos(theta1 + theta2),
                vy=self.speed_range[0] * np.sin(theta1 + theta2),
            )
        else:
            a_ = ax**2 + ay**2
            b_ = 2 * (ax * vx + ay * vy)
            c_ = vx**2 + vy**2 - self.speed_range[1] ** 2
            t1 = (-b_ + np.sqrt(b_**2 - 4 * a_ * c_)) / (2 * a_)
            t2 = interval - t1
            vx1 = vx + ax * t1
            vy1 = vy + ay * t1
            theta1 = np.arctan2(vy1, vx1)
            theta2 = steer * t2
            next_heading = theta1 + theta2
            R = self.accel_range[1] / theta2
            next_state = State(
                frame=state.frame + interval,
                x=state.x
                + vx * t1
                + 0.5 * ax * t1**2
                + R * (np.cos(next_heading + np.pi / 2) - np.cos(theta1 + np.pi / 2)),
                y=state.y
                + vy * t1
                + 0.5 * ay * t1**2
                + R * (np.sin(next_heading + np.pi / 2) - np.sin(theta1 + np.pi / 2)),
                heading=next_heading,
                vx=self.speed_range[1] * np.cos(next_heading),
                vy=self.speed_range[1] * np.sin(next_heading),
            )
        return next_state

    def _step_euler(self, state: State, accel: float, steer: float, interval: int = None) -> State:
        vx, vy = state.velocity
        x, y = state.location
        dts = [float(self.delta_t) / 1000] * (interval // self.delta_t)
        dts.append(float(interval % self.delta_t) / 1000)

        for dt in dts:
            vx += accel * np.cos(state.heading + steer) * dt
            vy += accel * np.sin(state.heading + steer) * dt
            speed = np.linalg.norm([vx, vy])
            speed_clipped = (
                np.clip(speed, *self.speed_range) if not self.speed_range is None else speed
            )
            if speed != speed_clipped:
                vx = speed_clipped * np.cos(state.heading + steer * dt)
                vy = speed_clipped * np.sin(state.heading + steer * dt)

            x += vx * dt
            y += vy * dt

        next_state = State(
            frame=state.frame + interval, x=x, y=y, heading=np.arctan2(vy, vx), vx=vx, vy=vy
        )

        return next_state

    def step(self, state: State, action: Tuple[float, float], interval: int = None) -> State:
        """This function updates the state of the traffic participant based on the point mass model.

        Args:
            state (State): The current state of the traffic participant.
            action (Tuple[float, float]): The first element is the acceleration, and the second element is the steering angle. The unit of the acceleration is meter per second squared (m/s$^2$). The unit of the steering angle is radian per second (rad/s).
            step (int): The time interval between the current state and the new state. The unit is millisecond. Defaults to None.

        Returns:
            State: A new state of the traffic participant.
        """
        if interval is None:
            interval = self.interval

        accel, steer = action
        accel = np.clip(accel, *self.accel_range)
        steer = np.clip(steer, *self.steer_range)

        if self.backend == "newton":
            next_state = self._step_newton(state, accel, steer, interval)
        elif self.backend == "euler":
            next_state = self._step_euler(state, accel, steer, interval)

        return next_state

    def verify_state(self, curr_state: State, prev_state: State) -> bool:
        return True
