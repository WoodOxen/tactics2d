from typing import Callable, List
import random
from enum import Enum
import copy

import numpy as np
from shapely.geometry import Point, LinearRing
from shapely.affinity import affine_transform


WHEEL_BASE = 2.8  # wheelbase
FRONT_HANG = 0.96  # front hang length
REAR_HANG = 0.929  # rear hang length
WIDTH = 1.942  # width

VehicleBox = LinearRing([
    (-REAR_HANG, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE,  WIDTH/2),
    (-REAR_HANG,  WIDTH/2)])

COLOR_POOL = [
    (30, 144, 255, 255), # dodger blue
    (255, 127, 80, 255), # coral
    (255, 215, 0, 255) # gold
]

VALID_SPEED = [-2.5, 2.5]
VALID_STEER = [-0.5, 0.5]
VALID_ACCEL = [-1.0, 1.0]
VALID_ANGULAR_SPEED = [-0.5, 0.5]

NUM_STEP = 10
STEP_LENGTH = 5e-2


class Status(Enum):
    CONTINUE = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5


class State:
    def __init__(self, raw_state: list):
        self.loc: Point = Point(raw_state[:2])
        self.heading: float = raw_state[2]
        self.speed: float = raw_state[3]
        self.steering: float = raw_state[4]

    def create_box(self) -> LinearRing:
        cos_theta = np.cos(self.heading)
        sin_theta = np.sin(self.heading)
        mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
        return affine_transform(VehicleBox, mat)


class KSModel(object):
    """Update the state of a vehicle by the Kinematic Single-Track Model.

    Kinematic Single-Track Model use the vehicle's current speed, heading, location, 
    acceleration, and velocity of steering angle as input. Then it returns the estimation of 
    speed, heading, steering angle and location after a small time step.

    Use the center of vehicle's rear wheels as the origin of local coordinate system.

    Assume the vehicle is front-wheel-only drive.
    """
    def __init__(
        self, 
        wheel_base: float,
        step_len: float,
        n_step: int,
        speed_range: list,
        angle_range: list
    ):
        self.wheel_base = wheel_base
        self.step_len = step_len
        self.n_step = n_step
        self.speed_range = speed_range
        self.angle_range = angle_range


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
            new_state.heading += \
                 (new_state.speed * np.tan(new_state.steering) / self.wheel_base) * self.step_len 
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
            new_state.heading += \
                 new_state.speed * np.tan(new_state.steering) / self.wheel_base * self.step_len 
            # new_state.speed += accel * self.step_len
            # new_state.steering += angular_speed * self.step_len
            # new_state.speed = np.clip(new_state.speed, *self.speed_range)
            # if new_state.speed<0:
            #     print(self.speed_range, new_state.speed)

        
        new_state.loc = Point(x, y)
        return new_state


class Vehicle:
    """_summary_
    """
    def __init__(
        self,
        wheel_base: float = WHEEL_BASE,
        step_len: float = STEP_LENGTH,
        n_step: int = NUM_STEP,
        speed_range: list = VALID_SPEED, 
        angle_range: list = VALID_STEER
    ) -> None:

        self.initial_state: list = None
        self.state: State = None
        self.box: LinearRing = None
        self.trajectory: List[Point] = []
        self.kinetic_model: Callable = \
            KSModel(wheel_base, step_len, n_step, speed_range, angle_range)
        self.color = COLOR_POOL[0]#random.sample(COLOR_POOL, 1)[0]
        self.v_max = None
        self.v_min = None

    def reset(self, initial_state: State):
        """
        Args:
            init_pos (list): [x0, y0, theta0]
        """
        self.initial_state = initial_state
        self.state = self.initial_state
        # self.color = random.sample(COLOR_POOL, 1)[0]
        self.v_max = self.initial_state.speed
        self.v_min = self.initial_state.speed
        self.box = self.state.create_box()
        self.trajectory.clear()
        self.trajectory.append(self.state.loc)
        # self.color = random.sample(COLOR_POOL, 1)[0]

    def step(self, action: np.ndarray):
        """
        Args:
            action (list): [steer, acceleration]
        """
        self.state = self.kinetic_model.step(self.state, action)
        self.box = self.state.create_box()
        self.trajectory.append(self.state.loc)
        self.v_max = self.state.speed if self.state.speed > self.v_max else self.v_max
        self.v_min = self.state.speed if self.state.speed < self.v_min else self.v_min