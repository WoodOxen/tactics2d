from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing
from shapely.affinity import rotate

from tactics2d.trajectory.element.state import State
from tactics2d.trajectory.element.trajectory import Trajectory


class Vehicle(object):
    """_summary_

    Attributes:
        id_ ():
        type_ ():
        length (float, optional): The length of the vehicle. The default unit is meter (m).
        width (float, optional): The width of the vehicle. The default unit is meter (m).
        height (float, optional): The height of the vehicle. The default unit is meter (m).
        steering_angle_range (Tuple[float, float], optional):
        steering_velocity_range (Tuple[float, float], optional):
        speed_range (Tuple[float, float], optional):
        accel_range (Tuple[float, float], optional):
        comfort_accel_range (Tuple[float, float], optional):
        body_type ()
    """
    def __init__(
        self, id_: int, type_: str,
        length: float = None, width: float = None, height: float = None, color = None,
        steering_angle_range: Tuple[float, float] = None,
        steering_velocity_range: Tuple[float, float] = None,
        speed_range: Tuple[float, float] = None,
        accel_range: Tuple[float, float] = None,
        comfort_accel_range: Tuple[float, float] = None,
        body_type = None, trajectory = None
    ):

        self.id_ = id_
        self.type_ = type_
        self.length = length
        self.width = width
        self.height = height
        self.color = color
        self.steering_angle_range = steering_angle_range
        self.steering_velocity_range = steering_velocity_range
        self.speed_range = speed_range
        self.accel_range = accel_range
        self.comfort_accel_range = comfort_accel_range
        self.bind_physics(body_type)
        self.bind_trajectory(trajectory)

        self.bbox = LinearRing([
            [0.5 * self.length, -0.5 * self.width],
            [0.5 * self.length, 0.5 * self.width],
            [-0.5 * self.length, 0.5 * self.width],
            [-0.5 * self.length, -0.5 * self.width, ]
        ])

    @property
    def current_state(self) -> State:
        return self.trajectory.get_state()

    @property
    def location(self):
        return self.current_state.location

    @property
    def heading(self) -> float:
        return self.current_state.heading

    @property
    def velocity(self):
        return (self.current_state.vx, self.current_state.vy)

    @property
    def speed(self) -> float:
        return self.current_state.speed

    @property
    def accel(self):
        return self.current_state.accel

    @property
    def pose(self) -> LinearRing:
        """The vehicle's bounding box which is rotated and moved based on the current state."""
        return rotate(self.bbox, self.heading, origin = self.location)

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

    def bind_trajectory(self, trajectory: Trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()

    def bind_physics(self, body_type = None):
        self.body_type = self.body_type if body_type is None else body_type
        if self.body_type:
            pass

    def update_state(self, action):
        """_summary_
        """
        self.current_state = self.physics.update(self.current_state, action)
        self.add_state(self.current_state)

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