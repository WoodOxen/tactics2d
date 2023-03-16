import copy

import numpy as np
from shapely.geometry import Point

from .vehicle_physics_base import VehiclePhysicsBase
from tactics2d.trajectory.element import State


class KinematicSingleTrack(VehiclePhysicsBase):
    """An implementation of the Kinematic Single-Track Model.

    Use the center of vehicle's rear wheels as the origin of local coordinate system.

    Assume the vehicle is front-wheel-only drive.

    Attributes:

    """

    def __init__(
        self,
        wheel_base: float,
        step_len: float,
        n_step: int,
        speed_range: list,
        angle_range: list,
    ):
        self.wheel_base = wheel_base
        self.step_len = step_len
        self.n_step = n_step
        self.speed_range = speed_range
        self.angle_range = angle_range
        self.n_iter = 20

    def step2(self, state: State, action: list) -> State:
        """Update the state of a vehicle with the Kinematic Single-Track Model.

        Args:
            state (list): [x, y, car_angle, speed, steering]
            action (list): [steer, accel].
            step (float, optional): the step length for each simulation.
            n_step (int): number of step of updating the physical state. This value is decide by
                (physics simulation step length : rendering step length).

        """
        new_state = copy.deepcopy(state)
        x, y = new_state.loc.x, new_state.loc.y
        steer, accel = action
        new_state.steering = steer
        new_state.steering = np.clip(new_state.steering, *self.angle_range)

        # TODO: check correctness
        for _ in range(self.n_step):
            x += new_state.speed * np.cos(new_state.heading) * self.step_len
            y += new_state.speed * np.sin(new_state.heading) * self.step_len
            new_state.heading += (
                new_state.speed * np.tan(new_state.steering) / self.wheel_base
            ) * self.step_len
            new_state.speed += accel * self.step_len
            # new_state.steering += angular_speed * self.step_len
            new_state.speed = np.clip(new_state.speed, *self.speed_range)
            # if new_state.speed<0:
            #     print(self.speed_range, new_state.speed)

        new_state.loc = Point(x, y)
        return new_state

    def step(self, state: State, action: list) -> State:
        """Update the state of a vehicle with the Kinematic Single-Track Model.

        Args:
            state (list): [x, y, car_angle, speed, steering]
            action (list): [steer, speed].
            step (float, optional): the step length for each simulation.
            n_step (int): number of step of updating the physical state. This value is decide by
                (physics simulation step length : rendering step length).

        """
        # accel = np.clip(accel, *self.accel_range)
        # angular_speed = np.clip(angular_speed, *self.angular_speed_range)
        new_state = copy.deepcopy(state)
        x, y = new_state.loc.x, new_state.loc.y
        steer, speed = action
        new_state.steering = steer
        new_state.speed = speed
        new_state.speed = np.clip(new_state.speed, *self.speed_range)
        new_state.steering = np.clip(new_state.steering, *self.angle_range)

        # TODO: check correctness
        for _ in range(self.n_step):
            x += new_state.speed * np.cos(new_state.heading) * self.step_len
            y += new_state.speed * np.sin(new_state.heading) * self.step_len
            new_state.heading += (
                new_state.speed
                * np.tan(new_state.steering)
                / self.wheel_base
                * self.step_len
            )
            # new_state.speed += accel * self.step_len
            # new_state.steering += angular_speed * self.step_len
            # new_state.speed = np.clip(new_state.speed, *self.speed_range)
            # if new_state.speed<0:
            #     print(self.speed_range, new_state.speed)

        new_state.loc = Point(x, y)
        return new_state

    def verify_state(self, curr_state: State, prev_state: State, interval: float) -> bool:
        return True
