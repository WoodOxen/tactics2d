from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing
from shapely.affinity import affine_transform

from .participant_base import ParticipantBase
from tactics2d.trajectory.element.trajectory import State, Trajectory
from tactics2d.physics import PointMass, SingleTrackKinematics
from tactics2d.physics import PointMass, SingleTrackKinematics

from .defaults import VEHICLE_MODEL


class Vehicle(ParticipantBase):
    """_summary_

    Attributes:
        id_ (int): The unique identifier of the vehicle.
        type_ (str, optional):
        length (float, optional): The length of the vehicle. The default unit is meter (m).
            Defaults to None.
        width (float, optional): The width of the vehicle. The default unit is meter (m).
            Defaults to None.
        height (float, optional): The height of the vehicle. The default unit is meter (m).
            Defaults to None.
        color (tuple, optional): The color of the vehicle. Expressed by a tuple with 3 integers.
        kerb_weight: (float, optional): The weight of the vehicle. The default unit is
            kilogram (kg). Defaults to None.
        steer_range (Tuple[float, float], optional): The range of the steering angle. The unit is radian. Defaults to None.
        speed_range (Tuple[float, float], optional): The range of the vehicle speed. The unit is meter per second. Defaults to None.
        accel_range (Tuple[float, float], optional): The range of the vehicle acceleration. The unit is meter per second squared. Defaults to None.
        comfort_accel_range (Tuple[float, float], optional): The range of the vehicle acceleration that is comfortable for the driver.
        physics_model ()
        steer_range (Tuple[float, float], optional): The range of the steering angle. The unit is radian. Defaults to None.
        speed_range (Tuple[float, float], optional): The range of the vehicle speed. The unit is meter per second. Defaults to None.
        accel_range (Tuple[float, float], optional): The range of the vehicle acceleration. The unit is meter per second squared. Defaults to None.
        comfort_accel_range (Tuple[float, float], optional): The range of the vehicle acceleration that is comfortable for the driver.
        physics_model ()
    """

    def __init__(
        self,
        id_: int,
        type_: str,
        length: float = None,
        width: float = None,
        height: float = None,
        color: tuple = None,
        kerb_weight: float = None,
        wheel_base: float = None,
        front_overhang: float = None,
        rear_overhang: float = None,
        speed_range: Tuple[float, float] = None,
        steer_range: Tuple[float, float] = (-np.pi / 6, np.pi / 6),
        accel_range: Tuple[float, float] = None,
        comfort_accel_range: Tuple[float, float] = None,
        physics_model=None,
        trajectory: Trajectory = None,
    ):
        super().__init__(id_, type_, length, width, height, color, trajectory)

        attribs = [
            "length",
            "width",
            "height",
            "kerb_weight",
            "wheel_base",
            "front_overhang",
            "rear_overhang",
            "front_overhang",
            "rear_overhang",
        ]

        for attrib in attribs:
            setattr(self, attrib, locals()[attrib])
            if getattr(self, attrib) is None:
                try:
                    setattr(self, attrib, VEHICLE_MODEL[self.type_][attrib])
                except:
                    pass

        self.speed_range = speed_range
        self.steer_range = steer_range
        self.accel_range = accel_range
        self.comfort_accel_range = comfort_accel_range

        if self.speed_range is None:
            try:
                self.speed_range = (0, VEHICLE_MODEL[self.type_]["max_speed"])
            except:
                pass

        if self.accel_range is None:
            try:
                self.accel_range = (
                    VEHICLE_MODEL[self.type_]["max_decel"],
                    VEHICLE_MODEL[self.type_]["max_accel"],
                )
            except:
                pass

        if self.comfort_accel_range is None:
            try:
                self.comfort_accel_range = (
                    VEHICLE_MODEL[self.type_]["max_comfort_decel"],
                    VEHICLE_MODEL[self.type_]["max_accel"],
                )
            except:
                pass

        self.physics_model = physics_model
        if self.physics_model is None:
            if None not in (self.width, self.front_overhang, self.rear_overhang):
                dist_front_hang = 0.5 * self.width - self.front_overhang
                dist_rear_hang = 0.5 * self.width - self.rear_overhang
                self.physics_model = SingleTrackKinematics(
                    dist_front_hang,
                    dist_rear_hang,
                    self.steer_range,
                    self.speed_range,
                    self.accel_range,
                )
            elif self.wheel_base is not None:
                self.physics_model = SingleTrackKinematics(
                    0.5 * self.wheel_base,
                    0.5 * self.wheel_base,
                    self.steer_range,
                    self.speed_range,
                    self.accel_range,
                )
            else:
                self.physics_model = PointMass(
                    self.steer_range, self.speed_range, self.accel_range
                )

        self.bbox = LinearRing(
            [
                [0.5 * self.length, -0.5 * self.width],
                [0.5 * self.length, 0.5 * self.width],
                [-0.5 * self.length, 0.5 * self.width],
                [-0.5 * self.length, -0.5 * self.width],
            ]
        )

    def add_state(self, state: State):
        if self.physics_model.verify_state(state, self.trajectory.current_state):
            self.trajectory.append_state(state)
        else:
            raise RuntimeError("Invalid state.")

    def _verify_trajectory(self, trajectory: Trajectory):
        for i in range(1, len(trajectory)):
            if not self.physics_model.verify_state(
                trajectory.get_state(trajectory.frames[i]),
                trajectory.get_state(trajectory.frames[i - 1]),
            ):
                return False
        return True

    def bind_trajectory(self, trajectory: Trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()

    def get_pose(self, frame: int = None) -> LinearRing:
        state = self.trajectory.get_state(frame)
        transform_matrix = [
            np.cos(state.heading),
            -np.sin(state.heading),
            np.sin(state.heading),
            np.cos(state.heading),
            state.location[0],
            state.location[1],
        ]
        return affine_transform(self.bbox, transform_matrix)

    def update(self, action: np.ndarray, step: float):
        """Update the agent's state with the given action."""
        current_state, _ = self.physics_model.step(self.current_state, action, step)
        self.add_state(current_state)
