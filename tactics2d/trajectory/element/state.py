from typing import Tuple

import numpy as np


class State:
    """_summary_

    Attributes:
        frame (int): The time stamp of the state. The default unit is millisecond (ms).
        x (float, optional): The x-axis coordinate of an object. The default unit is
            meter (m). Defaults to 0.
        y (float, optional): The y-axis coordinate of an object. The default unit is
            meter (m). Defaults to 0.
        heading (float, optional): The heading direction of an object. The heading information
            is parsed in an 2D Cardinal coordinate system counterclockwise. The default unit
            is radian. Defaults to 0.
        vx (float, optional): The velocity in the x-axis. The default unit is
            meter per second (m/s). Defaults to None.
        vy (float, optional): The velocity in the y-axis. The default unit is
            meter per second (m/s). Defaults to None.
        ax (float, optional): The acceleration in the x-axis. The default unit is
            meter per second squared (m/s^2). Defaults to None.
        ay (float, optional): The acceleration in the y-axis. The default unit is
            meter per second squared (m/s^2). Defaults to None.
    """

    def __init__(
        self,
        frame: int,
        x: float = 0,
        y: float = 0,
        heading: float = 0,
        vx: float = None,
        vy: float = None,
        speed: float = None,
        ax: float = None,
        ay: float = None,
        accel: float = None,
    ):
        self.frame = frame
        self.x = x
        self.y = y
        self.heading = heading
        self.vx = vx
        self.vy = vy
        self._speed = speed
        self.ax = ax
        self.ay = ay
        self._accel = accel

    @property
    def location(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        if not None in [self.vx, self.vy]:
            return (self.vx, self.vy)
        if None not in [self.speed, self.heading]:
            return (self.speed * np.cos(self.heading), self.speed * np.sin(self.heading))
        return None

    @property
    def speed(self):
        if self._speed is not None:
            return self._speed
        if None not in [self.vx, self.vy]:
            self._speed = np.linalg.norm([self.vx, self.vy])
            return self._speed

        return None

    @property
    def accel(self):
        if self._accel is not None:
            return self._accel
        if None not in [self.ax, self.ay]:
            self._accel = np.linalg.norm([self.ax, self.ay])
            return self._accel

        return None

    def __repr__(self):
        return f"{self.frame}, {self.x}, {self.y}, {self.heading}, {self.vx}, {self.vy}, {self.speed}, {self.ax}, {self.ay}, {self.accel}"

    def __str__(self):
        return f"State(frame={self.frame}, x={self.x}, y={self.y}, heading={self.heading}, vx={self.vx}, vy={self.vy}, speed={self.speed}, ax={self.ax}, ay={self.ay}, accel={self.accel})"

    def set_velocity(self, vx: float, vy: float):
        self.vx = vx
        self.vy = vy

    def set_speed(self, speed: float):
        self._speed = speed

    def set_accel(self, ax: float, ay: float):
        self.ax = ax
        self.ay = ay
        self._accel = np.linalg.norm([self.ax, self.ay])
