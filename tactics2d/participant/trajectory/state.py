##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: This file defines the state element of a trajectory.
# @Author: Yueyuan Li
# @Version: 0.1.8rc1

from typing import Any, Tuple

import numpy as np


class State:
    """This class defines the state element of a trajectory in a 2D Cartesian coordinate system.

    !!! note
        Given that the attributes of this class are commonly utilized across various applications, their types will be verified upon assignment. Whenever feasible, they will be converted to the appropriate type.

    Attributes:
        frame (int): The time stamp. The unit is millisecond (ms).
        x (float): The x-axis coordinate. The unit is meter.
        y (float): The y-axis coordinate. The unit is meter.
        heading (float): The heading direction. The unit is radian.
        vx (float): The velocity in the x-axis. The unit is meter per second (m/s).
        vy (float): The velocity in the y-axis. The unit is meter per second (m/s).
        speed (float): The scalar speed value. The unit is meter per second (m/s). This attribute can be set while initialization or by `set_speed`. If it is not set while vx and vy are available, speed will be obtained by sqrt(vx$^2$+vy$^2$). Otherwise, the speed will be None.
        ax (float): The acceleration in the x-axis. The unit is meter per second squared (m/s$^2$).
        ay (float): The acceleration in the y-axis. The unit is meter per second squared (m/s$^2$).
        accel (float): The scalar acceleration value. The unit is meter per second squared (m/s$^2$). This attribute can be set while initialization or by `set_accel`. If it is not set while ax and ay are available, accel will be obtained by sqrt(ax$^2$+ay$^2$). Otherwise, the accel will be None.
        location (Tuple[float, float]): The location. The unit is meter. This attribute is **read-only**.
        velocity (Tuple[float, float]): The velocity vector (vx, vy). The unit is meter per second (m/s). If vx and vy are not available but speed and heading are available, the velocity will be obtained by (speed * cos(heading), speed * sin(heading)). Otherwise, the velocity will be None. This attribute is **read-only**.
        acceleration (Tuple[float, float]): The acceleration vector (ax, ay). The unit is meter per second squared (m/s$^2$). If ax and ay are not available, the acceleration will be None. This attribute is **read-only**.
    """

    __annotations__ = {
        "frame": int,
        "x": float,
        "y": float,
        "heading": float,
        "vx": float,
        "vy": float,
        "_speed": float,
        "ax": float,
        "ay": float,
        "_accel": float,
    }

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
            frame (int): The time stamp. The unit is millisecond (ms).
            x (float, optional): The x-axis coordinate. The unit is meter.
            y (float, optional): The y-axis coordinate. The unit is meter.
            heading (float, optional): The heading direction. The unit is radian.
            vx (float, optional): The velocity in the x-axis. The unit is meter per second (m/s).
            vy (float, optional): The velocity in the y-axis. The unit is meter per second (m/s).
            speed (float, optional): The scalar speed value. The unit is meter per second (m/s).
            ax (float, optional): The acceleration in the x-axis. The unit is meter per second squared (m/s$^2$).
            ay (float, optional): The acceleration in the y-axis. The unit is meter per second squared (m/s$^2$).
            accel (float, optional): The scalar acceleration value. The unit is meter per second squared (m/s$^2$).

        Raises:
            ValueError: If the type of the input value cannot be converted to the expected type, the function will raise a ValueError.
        """
        setattr(self, "frame", frame)
        setattr(self, "x", x)
        setattr(self, "y", y)
        setattr(self, "heading", heading)
        setattr(self, "vx", vx)
        setattr(self, "vy", vy)
        setattr(self, "_speed", speed)
        setattr(self, "ax", ax)
        setattr(self, "ay", ay)
        setattr(self, "_accel", accel)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.__annotations__:
            if __value is None:
                super().__setattr__(__name, None)
            elif isinstance(__value, self.__annotations__[__name]):
                super().__setattr__(__name, __value)
            else:
                try:
                    super().__setattr__(__name, self.__annotations__[__name](__value))
                except:
                    raise ValueError(
                        f"Failed to convert {__value} to the expected type of {__name}: ({self.__annotations__[__name]})."
                    )

    def __str__(self):
        return f"{self.__class__.__name__}(frame={self.frame}, x={self.x}, y={self.y}, heading={self.heading}, vx={self.vx}, vy={self.vy}, speed={self.speed}, ax={self.ax}, ay={self.ay}, accel={self.accel})"

    @property
    def location(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def speed(self):
        if not self._speed is None:
            return self._speed
        if not None in [self.vx, self.vy]:
            self._speed = np.linalg.norm([self.vx, self.vy])
            return self._speed

        return None

    @property
    def velocity(self) -> Tuple[float, float]:
        if not None in [self.vx, self.vy]:
            return (self.vx, self.vy)
        if not None in [self.speed, self.heading]:
            return (self.speed * np.cos(self.heading), self.speed * np.sin(self.heading))

        return None

    @property
    def accel(self) -> float:
        if not None in [self.ax, self.ay]:
            return np.linalg.norm([self.ax, self.ay])
        elif not self.acceleration is None:
            return np.linalg.norm(self.acceleration)

        return None

    @property
    def acceleration(self) -> Tuple[float, float]:
        if not None in [self.ax, self.ay]:
            return (self.ax, self.ay)
        elif not None in [self._accel, self.heading]:
            return (self._accel * np.cos(self.heading), self._accel * np.sin(self.heading))

        return None

    def set_heading(self, heading: float):
        """This function sets the heading direction (radian)."""
        self.heading = heading

    def set_velocity(self, vx: float, vy: float):
        """This function sets vx, vy."""
        self.vx = vx
        self.vy = vy

    def set_speed(self, speed: float):
        """This function sets the speed."""
        self._speed = speed

    def set_accel(self, ax: float, ay: float):
        """This function sets ax, ay and calculate the scalar value of acceleration."""
        self.ax = ax
        self.ay = ay
        self._accel = np.linalg.norm([self.ax, self.ay])
