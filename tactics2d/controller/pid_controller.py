# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""PID controller implementation."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from tactics2d.participant.trajectory.state import State

from .controller_base import ControllerBase


class PIDController(ControllerBase):
    """PID controller for combined lateral and longitudinal vehicle control.

    This controller implements PID control for both steering (lateral) and
    acceleration (longitudinal) control. It supports driving style adjustment
    through linear interpolation of PID parameters.

    The controller supports three control modes:
    - 'combined': Both lateral and longitudinal control (default)
    - 'lateral': Only steering control (acceleration = 0)
    - 'longitudinal': Only acceleration control (steering = 0)

    Attributes:
        dt (float): Control time step in seconds.
        control_mode (str): Control mode - 'combined', 'lateral', or 'longitudinal'.
        kp_lat (float): Proportional gain for lateral control.
        ki_lat (float): Integral gain for lateral control.
        kd_lat (float): Derivative gain for lateral control.
        max_steering (float): Maximum steering angle in radians.
        kp_lon (float): Proportional gain for longitudinal control.
        ki_lon (float): Integral gain for longitudinal control.
        kd_lon (float): Derivative gain for longitudinal control.
        max_accel (float): Maximum acceleration in m/s².
        min_accel (float): Minimum acceleration (maximum deceleration) in m/s².
    """

    def __init__(
        self,
        dt: float = 0.05,
        control_mode: str = "combined",
        # Lateral control parameters
        kp_lat: float = 1.5,
        ki_lat: float = 0.2,
        kd_lat: float = 0.5,
        max_steering: float = 0.5,  # ~28.6 degrees
        # Longitudinal control parameters
        kp_lon: float = 2.0,
        ki_lon: float = 0.3,
        kd_lon: float = 0.4,
        max_accel: float = 3.0,
        min_accel: float = -5.0,
        # Optional advanced parameters
        derivative_filter_alpha: float = 0.1,
    ):
        """Initialize PID controller with parameters.

        Args:
            dt: Control time step in seconds. Default 0.05.
            control_mode: Control mode - 'combined', 'lateral', or 'longitudinal'.
                Default 'combined'.
            kp_lat: Proportional gain for lateral control. Default 1.5.
            ki_lat: Integral gain for lateral control. Default 0.2.
            kd_lat: Derivative gain for lateral control. Default 0.5.
            max_steering: Maximum steering angle in radians. Default 0.5 (~28.6°).
            kp_lon: Proportional gain for longitudinal control. Default 2.0.
            ki_lon: Integral gain for longitudinal control. Default 0.3.
            kd_lon: Derivative gain for longitudinal control. Default 0.4.
            max_accel: Maximum acceleration in m/s². Default 3.0.
            min_accel: Minimum acceleration (maximum deceleration) in m/s².
                Default -5.0.
            derivative_filter_alpha: Low-pass filter coefficient for derivative term.
                Range (0, 1]. Smaller values = more filtering. Default 0.1.

        Raises:
            ValueError: If control_mode is invalid or parameters are out of valid ranges.
        """
        # Validate control mode
        valid_modes = {"combined", "lateral", "longitudinal"}
        if control_mode not in valid_modes:
            raise ValueError(f"control_mode must be one of {valid_modes}, got '{control_mode}'")

        # Validate parameters
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if max_steering <= 0:
            raise ValueError(f"max_steering must be positive, got {max_steering}")
        if max_accel <= 0:
            raise ValueError(f"max_accel must be positive, got {max_accel}")
        if min_accel >= 0:
            raise ValueError(f"min_accel must be negative (deceleration), got {min_accel}")
        if max_accel <= min_accel:
            raise ValueError(
                f"max_accel ({max_accel}) must be greater than min_accel ({min_accel})"
            )
        if derivative_filter_alpha <= 0 or derivative_filter_alpha > 1:
            raise ValueError(
                f"derivative_filter_alpha must be in range (0, 1], got {derivative_filter_alpha}"
            )

        # Store parameters
        self.dt = dt
        self.control_mode = control_mode
        self.kp_lat = kp_lat
        self.ki_lat = ki_lat
        self.kd_lat = kd_lat
        self.max_steering = max_steering
        self.kp_lon = kp_lon
        self.ki_lon = ki_lon
        self.kd_lon = kd_lon
        self.max_accel = max_accel
        self.min_accel = min_accel
        self._derivative_filter_alpha = derivative_filter_alpha

        # Create driving style interpolators (minimal configuration)
        # Conservative to aggressive ranges
        self._kp_lat_interpolator = self.create_style_interpolator(1.0, 2.0)
        self._kp_lon_interpolator = self.create_style_interpolator(1.5, 2.5)
        self._max_steering_interpolator = self.create_style_interpolator(0.4, 0.6)
        self._max_accel_interpolator = self.create_style_interpolator(2.5, 3.5)
        self._min_accel_interpolator = self.create_style_interpolator(-4.0, -6.0)

        # Internal state for PID controllers
        self._lat_integral = 0.0
        self._lat_prev_error = 0.0
        self._lat_prev_derivative = 0.0
        self._lon_integral = 0.0
        self._lon_prev_error = 0.0
        self._lon_prev_derivative = 0.0

        # Low-pass filter for derivative term (optional, simple implementation)

    def update_driving_style(self, style_id: float) -> None:
        """Update controller parameters based on driving style.

        Args:
            style_id: Driving style index in range [-1.0, 1.0].
                -1.0: Conservative driving style (smooth, cautious)
                0.0: Normal driving style (default parameters)
                1.0: Aggressive driving style (responsive, aggressive)
                Values outside range are extrapolated linearly.

        Raises:
            TypeError: If style_id is not int or float.
        """
        if not isinstance(style_id, (int, float)):
            raise TypeError("style_id must be int or float")

        # Update parameters using interpolators
        self.kp_lat = float(self._kp_lat_interpolator(style_id))
        self.kp_lon = float(self._kp_lon_interpolator(style_id))
        self.max_steering = float(self._max_steering_interpolator(style_id))
        self.max_accel = float(self._max_accel_interpolator(style_id))
        self.min_accel = float(self._min_accel_interpolator(style_id))

    def _compute_pid(
        self,
        error: float,
        integral: float,
        prev_error: float,
        prev_derivative: float,
        kp: float,
        ki: float,
        kd: float,
        output_limits: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """Compute PID control output with anti-windup and filtered derivative.

        Args:
            error: Current error signal.
            integral: Current integral accumulator.
            prev_error: Previous error for derivative calculation.
            prev_derivative: Previous filtered derivative value.
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            output_limits: Optional tuple (min, max) for output clamping.
                If provided, anti-windup adjusts integral when output saturates.

        Returns:
            Tuple[float, float, float, float]:
                output: PID control output.
                new_integral: Updated integral accumulator.
                new_prev_error: Updated previous error.
                new_derivative: Updated filtered derivative.
        """
        # Proportional term
        p_term = kp * error

        # Derivative term with low-pass filtering
        raw_derivative = (error - prev_error) / self.dt if self.dt > 0 else 0.0
        derivative = (
            self._derivative_filter_alpha * raw_derivative
            + (1 - self._derivative_filter_alpha) * prev_derivative
        )
        d_term = kd * derivative

        # Calculate output without integral term (for anti-windup check)
        output_no_integral = p_term + d_term

        # Apply output limits if provided for anti-windup
        saturated = False
        if output_limits is not None:
            min_output, max_output = output_limits
            # Check if output would saturate even without integral term
            if output_no_integral > max_output:
                saturated = True
                output_no_integral = max_output
            elif output_no_integral < min_output:
                saturated = True
                output_no_integral = min_output

        # Integral term with anti-windup
        if not saturated:
            # Only integrate if output is not saturated
            integral += error * self.dt
        else:
            # Leaky integration to prevent windup when saturated
            integral *= 0.99

        i_term = ki * integral

        # Final output
        output = output_no_integral + i_term

        # Apply limits again in case i_term pushed output beyond limits
        if output_limits is not None:
            min_output, max_output = output_limits
            output = np.clip(output, min_output, max_output)

        return output, integral, error, derivative

    def _clamp_output(self, output: float, min_val: float, max_val: float) -> float:
        """Clamp output to specified limits.

        Args:
            output: Control output to clamp.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Returns:
            Clamped output value.
        """
        return np.clip(output, min_val, max_val)

    def _compute_lateral_error(self, ego_state: State, **kwargs) -> float:
        """Compute lateral control error.

        Supports two error calculation methods:
        1. Heading error: Difference between current heading and target heading.
        2. Cross-track error: Lateral distance from reference path.

        Args:
            ego_state: Current ego vehicle state.
            **kwargs: Must contain either 'target_heading' (float) or
                'cross_track_error' (float).

        Returns:
            Lateral error in radians (for heading error) or meters (for cross-track error).

        Raises:
            ValueError: If neither target_heading nor cross_track_error is provided.
        """
        if "target_heading" in kwargs:
            target_heading = kwargs["target_heading"]
            if not isinstance(target_heading, (int, float)):
                raise TypeError("target_heading must be numeric")
            # Normalize angle difference to [-pi, pi]
            error = target_heading - ego_state.heading
            error = np.arctan2(np.sin(error), np.cos(error))
            return float(error)
        elif "cross_track_error" in kwargs:
            cross_track_error = kwargs["cross_track_error"]
            if not isinstance(cross_track_error, (int, float)):
                raise TypeError("cross_track_error must be numeric")
            return float(cross_track_error)
        else:
            raise ValueError(
                "Lateral control requires either 'target_heading' or 'cross_track_error' in kwargs"
            )

    def _compute_longitudinal_error(self, ego_state: State, **kwargs) -> float:
        """Compute longitudinal control error.

        Args:
            ego_state: Current ego vehicle state.
            **kwargs: Must contain 'target_speed' (float).

        Returns:
            Speed error in m/s (target_speed - current_speed).

        Raises:
            ValueError: If target_speed is not provided.
            TypeError: If target_speed is not numeric.
        """
        if "target_speed" not in kwargs:
            raise ValueError("Longitudinal control requires 'target_speed' in kwargs")

        target_speed = kwargs["target_speed"]
        if not isinstance(target_speed, (int, float)):
            raise TypeError("target_speed must be numeric")

        current_speed = ego_state.speed if ego_state.speed is not None else 0.0
        return float(target_speed - current_speed)

    def step(self, ego_state: State, **kwargs) -> Tuple[float, float]:
        """Compute PID control commands for combined lateral and longitudinal control.

        Args:
            ego_state: Current state of the ego vehicle.
            **kwargs: Additional inputs required for control:
                For lateral control: 'target_heading' (float) or 'cross_track_error' (float)
                For longitudinal control: 'target_speed' (float)
                Optional: 'wheel_base' (float) for lateral control calculations
                    (used to convert cross-track error to steering angle if needed).

        Returns:
            Tuple[float, float]: (steering_angle, acceleration) commands.
                steering_angle: Steering angle in radians.
                acceleration: Acceleration in m/s².

        Raises:
            ValueError: If required inputs are missing or invalid.
            TypeError: If input types are incorrect.
        """
        steering = 0.0
        acceleration = 0.0

        # Lateral control (if enabled)
        if self.control_mode in ["combined", "lateral"]:
            try:
                lat_error = self._compute_lateral_error(ego_state, **kwargs)

                # Compute PID control for lateral error
                lat_output, self._lat_integral, self._lat_prev_error, self._lat_prev_derivative = (
                    self._compute_pid(
                        lat_error,
                        self._lat_integral,
                        self._lat_prev_error,
                        self._lat_prev_derivative,
                        self.kp_lat,
                        self.ki_lat,
                        self.kd_lat,
                    )
                )

                # Convert lateral output to steering angle
                # If error is heading error (radians), output is already in radians
                # If error is cross-track error (meters), we need to convert
                # Simple conversion: steering = output * (2.0 / wheel_base)
                # where wheel_base defaults to typical car value
                if "cross_track_error" in kwargs:
                    wheel_base = kwargs.get("wheel_base", 2.637)  # Default car wheelbase
                    if wheel_base <= 0:
                        raise ValueError(f"wheel_base must be positive, got {wheel_base}")
                    # Convert cross-track error control to steering angle
                    # Simple bicycle model approximation: steering = arctan(2*L*error/distance^2)
                    # For small errors, approximate as linear
                    steering = lat_output * (2.0 / wheel_base)
                else:
                    # Heading error control: output is already steering angle
                    steering = lat_output

                # Clamp steering to physical limits
                steering = self._clamp_output(steering, -self.max_steering, self.max_steering)

            except (ValueError, TypeError) as e:
                if self.control_mode == "lateral":
                    # In lateral-only mode, re-raise the error
                    raise
                # In combined mode, continue with zero steering
                steering = 0.0

        # Longitudinal control (if enabled)
        if self.control_mode in ["combined", "longitudinal"]:
            try:
                lon_error = self._compute_longitudinal_error(ego_state, **kwargs)

                # Compute PID control for longitudinal error
                lon_output, self._lon_integral, self._lon_prev_error, self._lon_prev_derivative = (
                    self._compute_pid(
                        lon_error,
                        self._lon_integral,
                        self._lon_prev_error,
                        self._lon_prev_derivative,
                        self.kp_lon,
                        self.ki_lon,
                        self.kd_lon,
                        output_limits=(self.min_accel, self.max_accel),
                    )
                )

                # Clamp acceleration to physical limits
                acceleration = self._clamp_output(lon_output, self.min_accel, self.max_accel)

            except (ValueError, TypeError) as e:
                if self.control_mode == "longitudinal":
                    # In longitudinal-only mode, re-raise the error
                    raise
                # In combined mode, continue with zero acceleration
                acceleration = 0.0

        return steering, acceleration

    def reset(self) -> None:
        """Reset the controller's internal state.

        Clears integral accumulators and previous error values.
        """
        self._lat_integral = 0.0
        self._lat_prev_error = 0.0
        self._lat_prev_derivative = 0.0
        self._lon_integral = 0.0
        self._lon_prev_error = 0.0
        self._lon_prev_derivative = 0.0

    def configure(self, **kwargs) -> None:
        """Configure controller parameters.

        Args:
            **kwargs: Parameter names and values to update.
                Supported parameters: dt, control_mode, kp_lat, ki_lat, kd_lat,
                max_steering, kp_lon, ki_lon, kd_lon, max_accel, min_accel,
                derivative_filter_alpha.

        Raises:
            AttributeError: If parameter name is not recognized.
            ValueError: If parameter value is invalid.
        """
        # Map external parameter names to internal attribute names
        param_map = {"derivative_filter_alpha": "_derivative_filter_alpha"}

        # First, validate all parameters before applying any changes
        for key, value in kwargs.items():
            internal_key = param_map.get(key, key)
            if not hasattr(self, internal_key):
                raise AttributeError(f"PIDController has no parameter '{key}'")

            # Validate specific parameters
            if key == "dt" and value <= 0:
                raise ValueError(f"dt must be positive, got {value}")
            elif key == "control_mode" and value not in {"combined", "lateral", "longitudinal"}:
                raise ValueError(
                    f"control_mode must be 'combined', 'lateral', or 'longitudinal', got '{value}'"
                )
            elif key == "max_steering" and value <= 0:
                raise ValueError(f"max_steering must be positive, got {value}")
            elif key == "max_accel" and value <= 0:
                raise ValueError(f"max_accel must be positive, got {value}")
            elif key == "min_accel" and value >= 0:
                raise ValueError(f"min_accel must be negative (deceleration), got {value}")
            elif key == "derivative_filter_alpha" and (value <= 0 or value > 1):
                raise ValueError(f"derivative_filter_alpha must be in range (0, 1], got {value}")

        # Check max_accel > min_accel if both are being updated
        if "max_accel" in kwargs and "min_accel" in kwargs:
            max_val = kwargs["max_accel"]
            min_val = kwargs["min_accel"]
            if max_val <= min_val:
                raise ValueError(
                    f"max_accel ({max_val}) must be greater than min_accel ({min_val})"
                )

        # Apply all updates
        for key, value in kwargs.items():
            internal_key = param_map.get(key, key)
            setattr(self, internal_key, value)
