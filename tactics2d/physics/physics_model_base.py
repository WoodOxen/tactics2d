# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physics model base implementation."""


import math
from abc import ABC, abstractmethod

import numpy as np

from tactics2d.participant.trajectory import State, Trajectory


class PhysicsModelBase(ABC):
    """This abstract class defines the essential interfaces required to specify a physics kinematics/dynamics model for a traffic participant.

    Please feel free to inherent this class to implement your own physics model.

    Attributes:
        _DELTA_T (int): The default time interval between the current state and the new state, 5 milliseconds (ms).
        _MIN_DELTA_T (int): The minimum time interval between the current state and the new state, 1 millisecond (ms).
        _G (float): The gravitational acceleration, 9.81 m/s^2.
    """

    _DELTA_T: int = 5
    _MIN_DELTA_T: int = 1
    _G = 9.81  # gravitational acceleration, m/s^2

    @abstractmethod
    def step(self, state: State, action: tuple, interval: int = None) -> State:
        """This abstract function defines an interface to update the state of the traffic participant based on the physics model.

        Args:
            state (State): The current state of the traffic participant.
            action (tuple): The action to be applied to the traffic participant.
            interval (int): The time interval between the current state and the new state. The unit is millisecond.

        Returns:
            A new state of the traffic participant.
        """

    @abstractmethod
    def verify_state(self, state: State, last_state: State, interval: int = None) -> bool:
        """This abstract function defines an interface to verify the validity of the new state based on the physics model.

        Args:
            state (State): The new state of the traffic participant.
            last_state (State): The last state of the traffic participant.
            interval (int): The time interval between the last state and the new state. The unit is millisecond.

        Returns:
            True if the new state is valid, False otherwise.
        """

    def verify_states(self, trajectory: Trajectory) -> bool:
        """This function verifies a sequence of states over time based on the physics model. The default implementation calls verify_state() for each state in the sequence. However, this function is expected to be overridden to implement more efficient verification.

        Args:
            trajectory (Trajectory): The trajectory of the traffic participant.

        Returns:
            True if the trajectory is valid, False otherwise.
        """
        if trajectory.stable_freq is True:
            interval = 1000 / trajectory.fps

        last_state = trajectory.history_states[trajectory.frames[0]]
        for frame in trajectory.frames[1:]:
            state = trajectory.history_states[frame]
            interval = interval if trajectory.stable_freq else state.frame - last_state.frame
            if self.verify_state(state, last_state, interval) is False:
                return False

        return True

    def _verify_kinematic_state(
        self, state: State, last_state: State, interval: int = None, *, grip: float = None
    ) -> bool:
        """Shared "very rough" reachability check for the single-track models.

        Checks that a proposed state is consistent with a one-step transition of a bicycle
        model given the steer/speed/accel ranges:

        - the heading advance is within what the extreme steering angles can produce
          (optionally grip-limited by ``grip`` = ``mu * g`` in m/s^2, as used by the kinematics
          grip cap);
        - the speed is reachable within one step of the acceleration range (speed-clamped);
        - the travel distance is plausible. Steering only changes the direction, not the arc
          length, and the Euclidean displacement (chord) never exceeds the straight-line travel
          distance, so the position is bounded by the max straight travel of the step
          instead of an axis-aligned box that wrongly paired the max speed with the extreme
          steering angle (which rejected legitimate accelerating near-straight steps).

        Args:
            state (State): The candidate state to check.
            last_state (State): The previous state.
            interval (int, optional): Time between the two states in ms; derived from the
                frame difference when None.
            grip (float, optional): Grip-limit ``mu * g`` (m/s^2). When set and the vehicle
                moves forward, the extreme-steer heading advance is capped like the model's
                ``_step`` does.

        Returns:
            True if the transition looks reachable, False otherwise.
        """
        interval = state.frame - last_state.frame if interval is None else interval
        # Handle zero interval case
        if interval == 0:
            return True  # No time elapsed, state should be valid
        dt = float(interval) / 1000
        last_speed = last_state.speed

        if None in [self.steer_range, self.speed_range, self.accel_range]:
            return True

        steer_range = np.array(self.steer_range)
        beta_range = np.arctan(self.lr / self.wheel_base * steer_range)

        # Per-step yaw advance for the extreme steering angles. When a grip limit (grip) is
        # set, cap the yaw magnitude like _step does (|phi_dot| <= grip / v).
        yaw_step = last_speed / self.wheel_base * np.sin(beta_range) * dt
        if grip is not None and last_speed > 0:
            yaw_limit = grip / last_speed * dt
            yaw_step = np.clip(yaw_step, -yaw_limit, yaw_limit)

        # check that heading is in the range. heading_range may be larger than 2 * np.pi
        heading_range = np.mod(last_state.heading + yaw_step, 2 * np.pi)
        if (
            heading_range[0] < heading_range[1]
            and not heading_range[0] <= state.heading <= heading_range[1]
        ):
            return False
        if heading_range[0] > heading_range[1] and not (
            heading_range[0] <= state.heading or state.heading <= heading_range[1]
        ):
            return False

        # check that speed is reachable within one step of the accel range (speed-clamped)
        speed_end = np.clip(last_speed + np.array(self.accel_range) * dt, *self.speed_range)
        if not speed_end[0] <= state.speed <= speed_end[1]:
            return False

        # check that the travel distance is plausible: the straight-line travel distance of
        # a step (arc length) bounds the Euclidean displacement (chord <= arc), independent of
        # steering.
        d_lo = 0.5 * (last_speed + speed_end[0]) * dt
        d_hi = 0.5 * (last_speed + speed_end[1]) * dt
        travel_max = max(abs(last_speed * dt), abs(d_lo), abs(d_hi))
        if math.hypot(state.x - last_state.x, state.y - last_state.y) > travel_max:
            return False

        return True
