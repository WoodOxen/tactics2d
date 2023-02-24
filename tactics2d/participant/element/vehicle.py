from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing
from shapely.affinity import affine_transform

from .participant_base import ParticipantBase
from tactics2d.trajectory.element.trajectory import State, Trajectory


class Vehicle(ParticipantBase):
    """_summary_

    Attributes:
        id_ ():
        type_ (str, optional):
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
        self, id_: int, type_: str = "car",
        length: float = 4.76, width: float = 1.85, height: float = 1.43, color = None,
        steering_angle_range: Tuple[float, float] = None,
        steering_velocity_range: Tuple[float, float] = None,
        speed_range: Tuple[float, float] = None,
        accel_range: Tuple[float, float] = None,
        comfort_accel_range: Tuple[float, float] = None,
        body_type = None, trajectory: Trajectory = None
    ):
        
        super().__init__(id_, type_, length, width, height)

        self.color = color
        self.steering_angle_range = steering_angle_range
        self.steering_velocity_range = steering_velocity_range
        self.speed_range = speed_range
        self.accel_range = accel_range
        self.comfort_accel_range = comfort_accel_range
        self.body_type = body_type
        self.bind_trajectory(trajectory)

        self.bbox = LinearRing([
            [0.5 * self.length, -0.5 * self.width],
            [0.5 * self.length, 0.5 * self.width],
            [-0.5 * self.length, 0.5 * self.width],
            [-0.5 * self.length, -0.5 * self.width, ]
        ])

    @property
    def pose(self) -> LinearRing:
        """The vehicle's bounding box which is rotated and moved based on the current state."""
        transform_matrix = [
            np.cos(self.heading), -np.sin(self.heading),
            np.sin(self.heading), np.cos(self.heading),
            self.location[0], self.location[1]
        ]
        return affine_transform(self.bbox, transform_matrix)

    def _verify_state(self) -> bool:
        """Check if the state change is allowed by the vehicle's physical model.

        Args:
            state1 (_type_): _description_
            state2 (_type_): _description_
            time_interval (_type_): _description_

        Returns:
            bool: _description_
        """
        return True
    
    def _verify_trajectory(self, trajectory: Trajectory) -> bool:
        return True

    def bind_trajectory(self, trajectory: Trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()

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