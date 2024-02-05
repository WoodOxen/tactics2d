##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: This file defines the state element of a trajectory.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Tuple

import numpy as np


class State:
    """This class defines the state element of a trajectory in a 2D Cartesian coordinate system.

    Attributes:
        frame (int): The time stamp. The default unit is millisecond.
        x (float, optional): The x-axis coordinate. The default unit is meter. Defaults to 0.
        y (float, optional): The y-axis coordinate. The default unit is meter. Defaults to 0.
        heading (float, optional): The heading direction. The default unit is radian. Defaults to 0.
        vx (float, optional): The velocity in the x-axis. The default unit is meter per second (m/s). Defaults to None.
        vy (float, optional): The velocity in the y-axis. The default unit is meter per second (m/s). Defaults to None.
        ax (float, optional): The acceleration in the x-axis. The default unit is meter per second squared (m/s$^2$). Defaults to None.
        ay (float, optional): The acceleration in the y-axis. The default unit is meter per second squared (m/s$^2$). Defaults to None.
        location (Tuple[float, float], read-only): The location. The default unit is meter. Defaults to (0, 0).
        speed (float, read-only): The scalar speed value. The default unit is meter per second (m/s). If the initialized speed is not None, this property will return this original value. If vx and vy are not None but speed is None, then speed will be calculated based on vx and vy. Otherwise, the property will return None.
        velocity (Tuple[float, float], read-only): The velocity vector. The default unit is meter per second (m/s). If vx and vy are available, the property will return (vx, vy). If vx and vy are not available but speed and heading are available, the property will return (speed * cos(heading), speed * sin(heading)). Otherwise, the property will return None.
        accel (float, read-only): The scalar acceleration value. The default unit is meter per second squared (m/s$^2$). If the initialized acceleration is not None, this property will return this original value. If ax and ay are not None but accel is None, then accel will be calculated based on ax and ay. Otherwise, the property will return None.
        acceleration (Tuple[float, float], read-only): The acceleration vector. The default unit is meter per second squared (m/s$^2$). If ax and ay are available, the property will return (ax, ay). Otherwise, the property will return None.
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
        """Initialize the state of a traffic participant.

        Args:
            frame (int): The time stamp. The default unit is millisecond.
            x (float, optional): The x-axis coordinate. The default unit is meter. Defaults to 0.
            y (float, optional): The y-axis coordinate. The default unit is meter. Defaults to 0.
            heading (float, optional): The heading direction. The default unit is radian. Defaults to 0.
            vx (float, optional): The velocity in the x-axis. The default unit is meter per second (m/s). Defaults to None.
            vy (float, optional): The velocity in the y-axis. The default unit is meter per second (m/s). Defaults to None.
            speed (float, optional): The scalar speed value. The default unit is meter per second (m/s). Defaults to None.
            ax (float, optional): The acceleration in the x-axis. The default unit is meter per second squared (m/s$^2$). Defaults to None.
            ay (float, optional): The acceleration in the y-axis. The default unit is meter per second squared (m/s$^2$). Defaults to None.
            accel (float, optional): The scalar acceleration value. The default unit is meter per second squared (m/s$^2$). Defaults to None.
        """
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
    def speed(self):
        if self._speed is not None:
            return self._speed
        if None not in [self.vx, self.vy]:
            self._speed = np.linalg.norm([self.vx, self.vy])
            return self._speed

        return None

    @property
    def velocity(self) -> Tuple[float, float]:
        if not None in [self.vx, self.vy]:
            return (self.vx, self.vy)
        if None not in [self.speed, self.heading]:
            return (self.speed * np.cos(self.heading), self.speed * np.sin(self.heading))

        return None

    @property
    def accel(self) -> float:
        if not None in [self.ax, self.ay]:
            return np.linalg.norm([self.ax, self.ay])

        return None

    @property
    def acceleration(self) -> Tuple[float, float]:
        if not None in [self.ax, self.ay]:
            return (self.ax, self.ay)

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
