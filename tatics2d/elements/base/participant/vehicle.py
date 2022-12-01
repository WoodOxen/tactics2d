from typing import Callable, List
from enum import Enum
import copy

import numpy as np
from shapely.geometry import Point, LineString, LinearRing
from shapely.affinity import affine_transform

from Tacktics2D.elements.base.participant.state import State


class Vehicle(object):
    """_summary_

    Attrs:
        id (int): object id, which must be unique
        length (float, optional): the length of the object. Defaults to 0.
        width (float, optional): the width of the object. Defaults to 0.
        initial_state (State, optional): the initial state of the object. Defaults to None.
        speed_range (tuple, optional): 
        accel_range (tuple, optional): 
        emerg_decel (float, optional): maximum deceleration of the object. Unit m/s^2. 
            Defaults to None.
        updater (, optional):
    """
    def __init__(
        self, id: int, type: str, 
        length: float = 0, width: float = 0, initial_state: State = None,
        speed_range: tuple = None, accel_range: tuple = None, emergency_decel: float = None,
        updater = None
    ):

        self.id = id
        self.type = type
        self.length = length
        self.width = width
        self.initial_state = initial_state
        self.speed_range = speed_range
        self.accel_range = accel_range
        self.emergency_decel = emergency_decel
        self.updater = updater

        self.current_state = initial_state
        self.state_list: List[State] = [initial_state]

    def add_state(self, state: State):
        idx = len(self.state_list) - 1
        if state.t > self.state_list[idx].t:
            self.state_list.append(state)
        else:
            while idx >= 0:
                if state.t == self.state_list[idx].t:
                    UserWarning("Object %d's state at time %f is overwritten." % (self.id, state.t))
                    self.state_list[idx] = state.t
                    break
                elif state.t < self.state_list[idx].t:
                    if idx > 0:
                        if state.t > self.state_list[idx-1].t:
                            self.state_list.insert(idx, state)
                            break
                        else:
                            idx -= 1
                    elif idx == 0:
                        self.state_list.insert(idx, state)

    def get_state(self, time_stamp: float = None) -> State:
        """Obtain the object's state at the requested time stamp. 
        If the time stamp is not specified, the function will return current state.
        If the time stamp is given but not found, the function will return None.
        """
        if time_stamp is None:
            return self.current_state
        else:
            for state in self.state_list:
                if time_stamp == state.t:
                    return state
            print("There is no state record for object %d at time %f" % (self.id, time_stamp))
            return

    def get_location(self, time_stamp: float = None) -> tuple:
        """Return the location coordination of the object at a given time in tuple format.
        If the time stamp is not specified, the function will return the location coordination of current state.
        """
        state = self.get_state(time_stamp)
        if state is not None:
            return state.get_location()
        return

    def get_velocity(self, time_stamp: float = None) -> tuple:
        """Return the velocity vector of the object at a given time in tuple format.
        If the time stamp is not specified, the function will return the velocity vector of current state.
        """
        state = self.get_state(time_stamp)
        if state is not None:
            return state.get_velocity()
        return

    def get_speed(self, time_stamp: float = None) -> float:
        """Return the speed value sqrt(vx^2+vy^2) of the object at a given time as float
        If the time stamp is not specified, the function will return the speed value of current state.
        """
        state = self.get_state(time_stamp)
        if state is not None:
            return state.get_speed()

    def get_heading(self, time_stamp: float = None) -> float:
        state = self.get_state(time_stamp)
        if state is not None:
            return state.heading
        return

    def get_acceleration(self, time_stamp: float = None) -> tuple:
        """Return the acceleration vector of the object at a given time in tuple format
        If the time is not specified, the function will return the acceleration vector of current state.
        If the acceleration at the specified time is not recorded, the function will return None.
        """
        state = self.get_state(time_stamp)
        if state is not None:
            acceleration = state.get_acceleration()
            if acceleration is not None:
                return acceleration
            else:
                print("Object %d's state record at time %f does not include acceleration." \
                    % (self.id, time_stamp))
        return

    def get_acceleration_scalar(self, time_stamp: float = None) -> float:
        """Return the value of acceleration sqrt(ax^2+ay^2) of the object at a given time as float
        If the time is not specified, the function will return the acceleration scalar value of current state.
        If the acceleration at the specified time is not recorded, the function will return None.
        """
        state = self.get_state(time_stamp)
        if state is not None:
            acceleration = state.get_acceleration_scalar()
            if acceleration is not None:
                return acceleration
            else:
                print("Object %d's state record at time %f does not include acceleration." \
                    % (self.id, time_stamp))
        return

    def get_trajectory(self) -> LineString:
        """
        """
        trajectory = []
        for state in self.state_list:
            trajectory.append(state.get_location())
        return LineString(trajectory)

    def reset(self, initial_state: State = None):
        """Reset the object to a given initial state.
        If the initial state is not specified, the object will be reset to the same initial state as previous.
        """
        if initial_state is not None:
            self.initial_state = initial_state
        self.current_state = self.initial_state
        self.state_list.clear()
        self.state_list.append(self.initial_state)


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
