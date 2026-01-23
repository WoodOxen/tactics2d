# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base controller implementation."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from tactics2d.participant.trajectory.state import State


class ControllerBase(ABC):
    """Abstract base class for all controllers in Tactics2D.

    Controllers are responsible for converting high-level goals into low-level
    control commands (steering, acceleration). Each controller implements a
    specific control algorithm (e.g., PID, pure pursuit, IDM) and must define
    how to compute control outputs given the current state and optional
    additional inputs.
    """

    @abstractmethod
    def step(self, ego_state: State, **kwargs) -> Tuple[float, float]:
        """Compute control commands for the ego vehicle.

        This is the main interface for controllers. Subclasses must implement
        this method to compute steering and acceleration commands based on
        the current ego state and any additional required inputs.

        Args:
            ego_state (State): Current state of the ego vehicle.
            **kwargs: Additional inputs required by specific controllers
                (e.g., leading_state, waypoints, wheel_base).

        Returns:
            Tuple[float, float]: (steering, acceleration) commands.
                steering: Steering angle in radians.
                acceleration: Acceleration in m/s².
        """
        pass

    def reset(self) -> None:
        """Reset the controller's internal state.

        This method should be called when the simulation resets or the
        controller is reused. Default implementation does nothing.
        """
        pass

    def configure(self, **kwargs) -> None:
        """Configure controller parameters.

        Update controller parameters dynamically. Subclasses should override
        this to validate and apply parameter updates.

        Args:
            **kwargs: Parameter names and values to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"Controller {self.__class__.__name__} has no parameter '{key}'"
                )

    @staticmethod
    def create_style_interpolator(
        y_left: float, y_right: float, x_left: float = -1.0, x_right: float = 1.0
    ):
        """Create a linear interpolator for driving style adjustment.

        Args:
            y_left: Parameter value at style_id = x_left (typically -1.0).
            y_right: Parameter value at style_id = x_right (typically 1.0).
            x_left: Left bound of style_id range. Default -1.0.
            x_right: Right bound of style_id range. Default 1.0.

        Returns:
            interp1d: Linear interpolator with bounds_error=False and
                fill_value=(y_left, y_right).
        """
        return interp1d(
            [x_left, x_right],
            [y_left, y_right],
            kind="linear",
            bounds_error=False,
            fill_value=(y_left, y_right),
        )

    def __repr__(self) -> str:
        """String representation of the controller."""
        return f"{self.__class__.__name__}(control_type={self.control_type})"
