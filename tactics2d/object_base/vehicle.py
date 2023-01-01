from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing
from shapely.affinity import affine_transform

from tactics2d.object_base.state import State


class Vehicle(object):
    """_summary_

    Attributes:
    """
    def __init__(
        self, id: str, type: str, initial_state: State = None,
        length: float = None, width: float = None, height: float = None,
        steering_angle_range: Tuple[float, float] = None, 
        steering_velocity_range: Tuple[float, float] = None, 
        speed_range: Tuple[float, float] = None,
        accel_range: Tuple[float, float] = None,
        comfort_accel_range: Tuple[float, float] = None,
        physics = None
    ):

        self.id = id
        self.type = type
        self.length = length
        self.width = width
        self.height = height
        self.bbox = LinearRing([
            [0,0], [self.length, 0], [self.length, self.width], [0, self.width]
        ])
        self.steering_angle_range = steering_angle_range
        self.steering_velocity_range = steering_velocity_range
        self.speed_range = speed_range
        self.accel_range = accel_range
        self.comfort_accel_range = comfort_accel_range
        self.physics = physics

        self.initial_state = initial_state
        self.current_state = initial_state
        self.history_state = []
        self.add_state(self.initial_state)

    @property
    def position(self):
        return self.current_state.x, self.current_state.y

    @property
    def state(self):
        return self.current_state

    @property
    def velocity(self):
        return self.current_state.vx, self.current_state.vy
    
    @property
    def speed(self):
        return np.linalg.norm([self.current_state.vx, self.current_state.y])

    def get_state(self, time_stamp: float = None) -> State:
        """Obtain the object's state at the requested time stamp.

        If the time stamp is not specified, the function will return current state.
        If the time stamp is given but not found, the function will return None.
        """
        if time_stamp is None:
            return self.current_state
        else:
            state = self.history_state.find(time_stamp)
        return state

    def verify_state(self, state1, state2, time_interval) -> bool:
        """Check if the state change is allowed by the vehicle's physical model.

        Args:
            state1 (_type_): _description_
            state2 (_type_): _description_
            time_interval (_type_): _description_

        Returns:
            bool: _description_
        """
        return True

    def add_state(self, state):
        return

    def update_state(self, action):
        """_summary_
        """
        self.current_state = self.physics.update(self.current_state, action)
        self.add_state(self.current_state)

    def get_trajectory(self, length: int = None):
        
        return

    def get_bbox(self) -> LinearRing:
        state = self.current_state
        transform_matrix = [
            np.cos(state.heading), -np.sin(state.heading), 
            np.sin(state.heading), np.cos(state.heading),
            state.x - self.length/2, state.y - self.width/2
        ]
        bbox = affine_transform(self.bbox, transform_matrix)
        return bbox

    def reset(self, state: State = None):
        """Reset the object to a given state. If the initial state is not specified, the object 
        will be reset to the same initial state as previous.
        """
        if state is not None:
            self.current_state = state
            self.initial_state = state
        else:
            self.current_state = self.initial_state
        self.history_state.clear()