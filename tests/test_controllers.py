# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for vehicle controllers."""

import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import pytest

from tactics2d.controller import (
    AccelerationController,
    ControllerBase,
    IDMController,
    PIDController,
    PurePursuitController,
)
from tactics2d.participant.trajectory.state import State


class TestControllerBase:
    """Test ControllerBase abstract class."""

    def test_abstract_methods(self):
        """Test that ControllerBase has required abstract methods."""
        # Should not be able to instantiate abstract class
        with pytest.raises(TypeError):
            ControllerBase()

    def test_create_style_interpolator(self):
        """Test create_style_interpolator static method."""

        # Use a concrete subclass to test the static method
        class ConcreteController(ControllerBase):
            def step(self, ego_state: State, **kwargs):
                return 0.0, 0.0

        controller = ConcreteController()
        interpolator = controller.create_style_interpolator(1.0, 2.0)

        # Test interpolation at bounds
        assert interpolator(-1.0) == 1.0
        assert interpolator(1.0) == 2.0

        # Test interpolation in middle
        assert interpolator(0.0) == 1.5

        # Test extrapolation
        assert interpolator(-2.0) == 1.0  # Clamped to left value
        assert interpolator(2.0) == 2.0  # Clamped to right value


@pytest.mark.controller
class TestAccelerationController:
    """Tests for AccelerationController."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        controller = AccelerationController()
        assert controller.target_speed == 5.0
        assert controller.kp == 3.5
        assert controller.max_accel == 1.5
        assert controller.min_accel == -4.0

        # Custom initialization
        controller = AccelerationController(target_speed=10.0)
        assert controller.target_speed == 10.0

        # Invalid initialization
        with pytest.raises(ValueError, match="target_speed must be non-negative"):
            AccelerationController(target_speed=-1.0)

    def test_update_driving_style(self):
        """Test driving style adjustment."""
        controller = AccelerationController()

        # Conservative style
        controller.update_driving_style(-1.0)
        assert controller.kp == 4.5  # From interpolator (4.5, 2.5)
        assert controller.speed_factor == 0.8  # From interpolator (0.8, 1.2)

        # Aggressive style
        controller.update_driving_style(1.0)
        assert controller.kp == 2.5
        assert controller.speed_factor == 1.2

    def test_step_cruise_control(self):
        """Test step method with cruise control (no leading vehicle)."""
        controller = AccelerationController(target_speed=10.0)
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=5.0, accel=0.0)

        steering, acceleration = controller.step(ego_state)

        # AccelerationController should output zero steering
        assert steering == 0.0

        # Acceleration should be within limits
        assert controller.min_accel <= acceleration <= controller.max_accel

        # Should accelerate to reach target speed
        assert acceleration > 0.0

    def test_step_adaptive_cruise_control(self):
        """Test step method with adaptive cruise control (with leading vehicle)."""
        controller = AccelerationController(target_speed=10.0)
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=5.0, accel=0.0)
        front_state = State(frame=0, x=10.0, y=0, heading=0, speed=4.0, accel=0.0)

        steering, acceleration = controller.step(ego_state, front_state=front_state)

        assert steering == 0.0
        assert controller.min_accel <= acceleration <= controller.max_accel

        # With leading vehicle slower and ahead, should decelerate
        assert acceleration < 0.0


@pytest.mark.controller
class TestIDMController:
    """Tests for IDMController."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        controller = IDMController()
        assert controller.desired_speed == 10.0
        assert controller.time_headway == 1.5
        assert controller.min_spacing == 2.0
        assert controller.max_acceleration == 1.0
        assert controller.comfortable_deceleration == 3.0
        assert controller.delta == 4.0

        # Custom initialization
        controller = IDMController(
            desired_speed=15.0,
            time_headway=2.0,
            min_spacing=3.0,
            max_acceleration=2.0,
            comfortable_deceleration=4.0,
            delta=2.0,
        )
        assert controller.desired_speed == 15.0
        assert controller.time_headway == 2.0
        assert controller.min_spacing == 3.0
        assert controller.max_acceleration == 2.0
        assert controller.comfortable_deceleration == 4.0
        assert controller.delta == 2.0

    def test_step_free_flow(self):
        """Test step method with free flow (no leading vehicle)."""
        controller = IDMController(desired_speed=10.0)

        # Ego at lower speed
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=5.0)
        steering, acceleration = controller.step(ego_state)

        assert steering == 0.0  # IDM is longitudinal only
        assert acceleration > 0.0  # Should accelerate toward desired speed
        assert -controller.comfortable_deceleration <= acceleration <= controller.max_acceleration

        # Ego at desired speed
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=10.0)
        steering, acceleration = controller.step(ego_state)

        assert acceleration == 0.0  # At desired speed, acceleration should be zero

    def test_step_car_following(self):
        """Test step method with car following."""
        controller = IDMController(desired_speed=10.0)

        # Leading vehicle ahead at safe distance
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=5.0)
        leading_state = State(frame=0, x=20.0, y=0, heading=0, speed=6.0)

        steering, acceleration = controller.step(ego_state, leading_state)

        assert steering == 0.0
        assert -controller.comfortable_deceleration <= acceleration <= controller.max_acceleration

        # Leading vehicle too close
        leading_state = State(frame=0, x=3.0, y=0, heading=0, speed=6.0)
        steering, acceleration = controller.step(ego_state, leading_state)

        assert acceleration < 0.0  # Should decelerate

    def test_configure(self):
        """Test configure method."""
        controller = IDMController()

        controller.configure(desired_speed=12.0, max_acceleration=1.5)
        assert controller.desired_speed == 12.0
        assert controller.max_acceleration == 1.5

        # Test invalid parameter
        with pytest.raises(AttributeError, match="has no parameter"):
            controller.configure(invalid_param=1.0)


@pytest.mark.controller
class TestPurePursuitController:
    """Tests for PurePursuitController."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        controller = PurePursuitController()
        assert controller.min_pre_aiming_distance == 10.0
        assert controller.interval == 1.0
        assert controller._longitudinal_control.target_speed == 5.0

        # Custom initialization
        controller = PurePursuitController(min_pre_aiming_distance=5.0, target_speed=8.0)
        assert controller.min_pre_aiming_distance == 5.0
        assert controller._longitudinal_control.target_speed == 8.0

        # Invalid initialization
        with pytest.raises(ValueError, match="min_pre_aiming_distance must be positive"):
            PurePursuitController(min_pre_aiming_distance=0)
        with pytest.raises(ValueError, match="target_speed must be non-negative"):
            PurePursuitController(target_speed=-1.0)

    def test_update_driving_style(self):
        """Test driving style adjustment."""
        controller = PurePursuitController()

        controller.update_driving_style(-1.0)
        assert controller.interval == 2.0  # From interpolator (2.0, 1.0)

        controller.update_driving_style(1.0)
        assert controller.interval == 1.0

    def test_step(self):
        """Test step method with waypoints."""
        from shapely.geometry import LineString

        controller = PurePursuitController(target_speed=10.0)
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=5.0, accel=0.0)

        # Create a simple straight path
        waypoints = LineString([(0, 0), (100, 0)])

        steering, acceleration = controller.step(ego_state, waypoints=waypoints, wheel_base=2.5)

        # Steering should be computed
        assert isinstance(steering, float)
        assert isinstance(acceleration, float)

        # Acceleration should be within limits of the internal AccelerationController
        accel_controller = controller._longitudinal_control
        assert accel_controller.min_accel <= acceleration <= accel_controller.max_accel


@pytest.mark.controller
class TestPIDController:
    """Tests for PIDController."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        controller = PIDController()
        assert controller.dt == 0.05
        assert controller.control_mode == "combined"
        assert controller.kp_lat == 1.5
        assert controller.ki_lat == 0.2
        assert controller.kd_lat == 0.5
        assert controller.max_steering == 0.5
        assert controller.kp_lon == 2.0
        assert controller.ki_lon == 0.3
        assert controller.kd_lon == 0.4
        assert controller.max_accel == 3.0
        assert controller.min_accel == -5.0

        # Custom initialization
        controller = PIDController(
            dt=0.1,
            control_mode="lateral",
            kp_lat=2.0,
            ki_lat=0.1,
            kd_lat=0.3,
            max_steering=0.4,
            kp_lon=1.5,
            ki_lon=0.2,
            kd_lon=0.5,
            max_accel=2.5,
            min_accel=-4.0,
            derivative_filter_alpha=0.2,
        )
        assert controller.dt == 0.1
        assert controller.control_mode == "lateral"
        assert controller.kp_lat == 2.0
        assert controller.max_steering == 0.4
        assert controller._derivative_filter_alpha == 0.2

        # Invalid initialization
        with pytest.raises(ValueError, match="control_mode must be one of"):
            PIDController(control_mode="invalid")
        with pytest.raises(ValueError, match="dt must be positive"):
            PIDController(dt=0)
        with pytest.raises(ValueError, match="max_steering must be positive"):
            PIDController(max_steering=0)
        with pytest.raises(ValueError, match="max_accel must be positive"):
            PIDController(max_accel=0)

    def test_update_driving_style(self):
        """Test driving style adjustment."""
        controller = PIDController()

        # Conservative style
        controller.update_driving_style(-1.0)
        assert controller.kp_lat == 1.0  # From interpolator (1.0, 2.0)
        assert controller.kp_lon == 1.5  # From interpolator (1.5, 2.5)
        assert controller.max_steering == 0.4  # From interpolator (0.4, 0.6)
        assert controller.max_accel == 2.5  # From interpolator (2.5, 3.5)
        assert controller.min_accel == -4.0  # From interpolator (-4.0, -6.0)

        # Aggressive style
        controller.update_driving_style(1.0)
        assert controller.kp_lat == 2.0
        assert controller.kp_lon == 2.5
        assert controller.max_steering == 0.6
        assert controller.max_accel == 3.5
        assert controller.min_accel == -6.0

    def test_step_combined_control(self):
        """Test step method with combined control."""
        controller = PIDController(control_mode="combined")
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=5.0)

        # Test with heading error and speed error
        steering, acceleration = controller.step(ego_state, target_heading=0.1, target_speed=10.0)

        assert -controller.max_steering <= steering <= controller.max_steering
        assert controller.min_accel <= acceleration <= controller.max_accel

        # Test with cross-track error
        steering, acceleration = controller.step(
            ego_state, cross_track_error=0.5, target_speed=10.0, wheel_base=2.5
        )

        assert -controller.max_steering <= steering <= controller.max_steering
        assert controller.min_accel <= acceleration <= controller.max_accel

    def test_step_lateral_only(self):
        """Test step method with lateral-only control."""
        controller = PIDController(control_mode="lateral")
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=5.0)

        steering, acceleration = controller.step(ego_state, target_heading=0.2)

        assert -controller.max_steering <= steering <= controller.max_steering
        assert acceleration == 0.0  # Should be zero in lateral-only mode

    def test_step_longitudinal_only(self):
        """Test step method with longitudinal-only control."""
        controller = PIDController(control_mode="longitudinal")
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=5.0)

        steering, acceleration = controller.step(ego_state, target_speed=10.0)

        assert steering == 0.0  # Should be zero in longitudinal-only mode
        assert controller.min_accel <= acceleration <= controller.max_accel

    def test_reset(self):
        """Test reset method."""
        controller = PIDController()
        ego_state = State(frame=0, x=0, y=0, heading=0, speed=5.0)

        # Run a few steps to accumulate internal state
        for _ in range(5):
            controller.step(ego_state, target_heading=0.1, target_speed=10.0)

        # Reset
        controller.reset()

        # Check internal state is cleared
        assert controller._lat_integral == 0.0
        assert controller._lat_prev_error == 0.0
        assert controller._lat_prev_derivative == 0.0
        assert controller._lon_integral == 0.0
        assert controller._lon_prev_error == 0.0
        assert controller._lon_prev_derivative == 0.0

    def test_configure(self):
        """Test configure method."""
        controller = PIDController()

        controller.configure(kp_lat=3.0, max_steering=0.4)
        assert controller.kp_lat == 3.0
        assert controller.max_steering == 0.4

        controller.configure(derivative_filter_alpha=0.3)
        assert controller._derivative_filter_alpha == 0.3

        # Test invalid parameter
        with pytest.raises(AttributeError, match="has no parameter"):
            controller.configure(invalid_param=1.0)

        # Test invalid value
        with pytest.raises(ValueError, match="dt must be positive"):
            controller.configure(dt=-0.1)
