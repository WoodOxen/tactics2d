from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing
from shapely.affinity import affine_transform

from tactics2d.trajectory.element.state import State


class Vehicle(object):
    """_summary_

    Attributes:
    """
    def __init__(
        self, id_: int, type_: str,
        length: float = None, width: float = None, height: float = None,
        steering_angle_range: Tuple[float, float] = None,
        steering_velocity_range: Tuple[float, float] = None,
        speed_range: Tuple[float, float] = None,
        accel_range: Tuple[float, float] = None,
        comfort_accel_range: Tuple[float, float] = None,
    ):

        self.id_ = id_
        self.type_ = type_
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

        self.trajectory = None
        self.physics = None

    @property
    def current_state(self):
        return self.trajectory.get_state()

    @property
    def location(self):
        return (self.current_state.x, self.current_state.y)

    @property
    def heading(self):
        return self.current_state.heading

    @property
    def velocity(self):
        return (self.current_state.vx, self.current_state.vy)

    @property
    def speed(self):
        return self.current_state.speed

    @property
    def accel(self):
        return self.current_state.accel

    def _verify_state(self, state1, state2, time_interval) -> bool:
        """Check if the state change is allowed by the vehicle's physical model.

        Args:
            state1 (_type_): _description_
            state2 (_type_): _description_
            time_interval (_type_): _description_

        Returns:
            bool: _description_
        """
        return True

    def update_state(self, action):
        """_summary_
        """
        self.current_state = self.physics.update(self.current_state, action)
        self.add_state(self.current_state)

    def bind_trajectory(self, trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()

    def bind_physics(self, physics):
        self.physics = physics

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