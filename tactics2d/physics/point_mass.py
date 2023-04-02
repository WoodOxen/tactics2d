from typing import Tuple

import numpy as np

from .physics_model_base import PhysicsModelBase
from tactics2d.trajectory.element import State


MAX_DELTA_T = 0.005


class PointMass(PhysicsModelBase):

    abbrev = "PM"

    def __init__(
        self,
        steer_range: Tuple[float, float] = None,
        speed_range: Tuple[float, float] = None,
        accel_range: Tuple[float, float] = None,
        delta_t: float = MAX_DELTA_T,
    ):
        self.steer_range = steer_range
        self.speed_range = speed_range
        self.accel_range = accel_range
        self.delta_t = min(delta_t, MAX_DELTA_T)

    def _step(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        heading: float,
        steer: float,
        accel: float,
        dt: float,
    ):
        ax = accel * np.cos(heading + steer)
        ay = accel * np.sin(heading + steer)

        new_x = x + vx * dt
        new_y = y + vy * dt
        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        new_heading = np.arctan2(new_vy, new_vx)

        if self.speed_range is not None:
            speed = np.linalg.norm([new_vx, new_vy])
            speed_clipped = np.clip(speed, *self.speed_range)
            if speed_clipped != speed:
                k = speed_clipped / speed
                new_vx = k * new_vx
                new_vy = k * new_vy

        return new_x, new_y, new_vx, new_vy, new_heading

    def step(
        self, state: State, action: Tuple[float, float], step: float
    ) -> Tuple[State, tuple]:
        steer, accel = action
        x, y, heading = state.x, state.y, state.heading
        vx, vy = state.velocity

        if self.steer_range is not None:
            steer = np.clip(steer, *self.steer_range)

        if self.accel_range is not None:
            accel = np.clip(accel, *self.accel_range)

        dt = self.delta_t
        while dt <= step:
            x, y, vx, vy, heading = self._step(
                x, y, vx, vy, heading, steer, accel, self.delta_t
            )
            dt += self.delta_t

        if dt > step:
            x, y, vx, vy, heading = self._step(
                x, y, vx, vy, heading, steer, accel, step - (dt - self.delta_t)
            )

        new_state = State(
            state.frame + int(step * 1000), x=x, y=y, heading=heading, vx=vx, vy=vy
        )
        return new_state, (steer, accel)

    def verify_state(self, curr_state: State, prev_state: State) -> bool:
        return True
