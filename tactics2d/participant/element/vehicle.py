from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing
from shapely.affinity import affine_transform

from .participant_base import ParticipantBase
from tactics2d.trajectory.element.trajectory import State, Trajectory

from .defaults import VEHICLE_MODEL


class Vehicle(ParticipantBase):
    """_summary_

    Attributes:
        id_ (int): The unique identifier of the vehicle.
        type_ (str, optional):
        color (tuple, optional): The color of the vehicle. Expressed by a tuple with 3 integers.
        length (float, optional): The length of the vehicle. The default unit is meter (m). 
            Defaults to None.
        width (float, optional): The width of the vehicle. The default unit is meter (m). 
            Defaults to None.
        height (float, optional): The height of the vehicle. The default unit is meter (m). 
            Defaults to None.
        kerb_weight: (float, optional): The weight of the vehicle. The default unit is 
            kilogram (kg). Defaults to None.
        steering_angle_range (Tuple[float, float], optional):
        steering_velocity_range (Tuple[float, float], optional):
        speed_range (Tuple[float, float], optional):
        accel_range (Tuple[float, float], optional):
        comfort_accel_range (Tuple[float, float], optional):
        body_type ()
    """

    def __init__(
        self, id_: int, type_: str = "sedan", color: tuple = None, kerb_weight: float = None,
        length: float = None, width: float = None, height: float = None,
        wheel_base: float = None, front_hang: float = None, rear_hang: float = None,
        steering_angle_range: Tuple[float, float] = None,
        steering_velocity_range: Tuple[float, float] = None,
        speed_range: Tuple[float, float] = None,
        accel_range: Tuple[float, float] = None,
        comfort_accel_range: Tuple[float, float] = None,
        body_type=None, trajectory: Trajectory = None,
    ):
        super().__init__(id_, type_)

        self.color = color

        attribs = [
            "length", "width", "height", "kerb_weight",
            "wheel_base", "front_hang", "rear_hang",
            "steering_angle_range", "steering_velocity_range", "speed_range",
            "accel_range","comfort_accel_range"
        ]
        for attrib in attribs:
            if locals()[attrib] is None:
                if self.type_ in VEHICLE_MODEL and attrib in VEHICLE_MODEL[type_]:
                    setattr(self, attrib, VEHICLE_MODEL[type_][attrib])
                else:
                    setattr(self, attrib, None)
            else:
                setattr(self, attrib, locals()[attrib])

        self.bind_trajectory(trajectory)

        self.bbox = LinearRing(
            [
                [0.5 * self.length, -0.5 * self.width], [0.5 * self.length, 0.5 * self.width],
                [-0.5 * self.length, 0.5 * self.width], [-0.5 * self.length, -0.5 * self.width],
            ]
        )

    def get_pose(self, frame: int = None) -> LinearRing:
        """Get the vehicle's bounding box which is rotated and moved based on the current state.
        """
        state = self.trajectory.get_state(frame)
        transform_matrix = [
            np.cos(state.heading), -np.sin(state.heading),
            np.sin(state.heading), np.cos(state.heading),
            state.location[0], state.location[1],
        ]
        return affine_transform(self.bbox, transform_matrix)

    def _verify_state(self, curr_state: State, prev_state: State, interval: float) -> bool:
        return True

    def _verify_trajectory(self, trajectory: Trajectory):
        return True

    def bind_trajectory(self, trajectory: Trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()

    def update_state(self, action):
        """_summary_"""
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
