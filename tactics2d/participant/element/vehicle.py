from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing, LineString
from shapely.affinity import translate, affine_transform

from .participant_base import ParticipantBase
from tactics2d.trajectory.element.trajectory import State, Trajectory
from tactics2d.physics import PointMass, SingleTrackKinematics

from .defaults import VEHICLE_MODEL


class Vehicle(ParticipantBase):
    """This class defines a four-wheeled vehicle with its common properties.

    The location of the vehicle refers to its center. The vehicle center is defined by its physical model. If the vehicle is front-wheel driven, which means its engine straddling the front axle, the vehicle center is the midpoint of the front axle. If the vehicle is rear-wheel driven, which means its engine straddling the rear axle, the vehicle center is the midpoint of the rear axle. If the vehicle is four-wheel driven, which means its engine straddling both the front and rear axles, the vehicle center is the midpoint of the wheel base. If the vehicle is all-wheel driven, which means its engine straddling all the axles, the vehicle center is the midpoint of the wheel base.

    <img src="https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/vehicle_driven.png" alt="Front-wheel driven, rear-wheel driven, four-wheel driven, and all-wheel driven vehicles"/>

    <img src="https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/vehicle_centers.png" alt="Vehicle center definition for different vehicle types."/>

    Attributes:
        id_ (int): The unique identifier of the vehicle.
        type_ (str): The type of the vehicle.
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
        physics_model (): Defaults to None.
        steer_range (Tuple[float, float], optional): The range of the steering angle. The unit is radian. Defaults to None.
        speed_range (Tuple[float, float], optional): The range of the vehicle speed. The unit is meter per second. Defaults to None.
        accel_range (Tuple[float, float], optional): The range of the vehicle acceleration. The unit is meter per second squared. Defaults to None.
        comfort_accel_range (Tuple[float, float], optional): The range of the vehicle acceleration that is comfortable for the driver.
    """

    attributes = {
        "color": tuple,
        "kerb_weight": float,
        "front_overhang": float,
        "wheel_base": float,
        "rear_overhang": float,
        "speed_range": tuple,
        "steer_range": tuple,
        "accel_range": tuple,
        "comfort_accel_range": tuple,
        "driven_type": str,
        "physics_model": None,
    }
    default_vehicle_types = set(VEHICLE_MODEL.keys())
    default_driven_types = {
        "front_wheel_driven",
        "rear_wheel_driven",
        "four_wheel_driven",
        "all_wheel_driven",
    }

    def __init__(
        self,
        id_: int,
        type_: str = "sedan",
        **kwargs,
    ):
        super().__init__(id_, type_, **kwargs)

        attribute_dict = {**self.default_attributes, **self.attributes}

        if self.type_ in self.default_vehicle_types:
            for attr in attribute_dict:
                if getattr(self, attr) is None and attr in VEHICLE_MODEL[self.type_]:
                    setattr(self, attr, VEHICLE_MODEL[self.type_][attr])

            if self.speed_range is None:
                self.speed_range = (0, VEHICLE_MODEL[self.type_]["max_speed"])

            if self.accel_range is None:
                self.accel_range = (
                    VEHICLE_MODEL[self.type_]["max_decel"],
                    VEHICLE_MODEL[self.type_]["max_accel"],
                )

        if self.steer_range is None:
            self.steer_range = (-np.pi / 6, np.pi / 6)

        if self.physics_model is None:
            if None not in (self.width, self.front_overhang, self.rear_overhang):
                dist_front_hang = 0.5 * self.length - self.front_overhang
                dist_rear_hang = 0.5 * self.length - self.rear_overhang
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
                self.physics_model = PointMass(self.steer_range, self.speed_range, self.accel_range)

        if self.driven_type not in self.default_driven_types:
            self.driven_type = "rear_wheel_driven"

        if not None in [self.width, self.length]:
            self.bbox = LinearRing(
                [
                    [self.length, 0.5 * self.width],
                    [self.length, -0.5 * self.width],
                    [0, -0.5 * self.width],
                    [0, 0.5 * self.width],
                ]
            )

            if self.driven_type == "front_wheel_driven" and self.front_overhang is not None:
                self.center_shift = self.front_overhang - self.length
            elif self.driven_type == "rear_wheel_driven" and self.rear_overhang is not None:
                self.center_shift = -self.rear_overhang
            elif not None in [self.wheel_base, self.rear_overhang]:
                self.center_shift = -self.rear_overhang - 0.5 * self.wheel_base
            else:
                self.center_shift = -0.5 * self.length

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
        bbox = translate(self.bbox, self.center_shift)
        return affine_transform(bbox, transform_matrix)

    def get_trace(self, frame_range: tuple = None):
        states = self.trajectory.get_states(frame_range)
        trace = None
        if len(states) == 0:
            pass
        elif len(states) == 1:
            trace = self.get_pose(states[0].frame)
        else:
            center_line = []
            start_pose = np.array(list(self.get_pose(states[0].frame).coords))
            end_pose = np.array(list(self.get_pose(states[-1].frame).coords))
            start_point = tuple(np.mean(start_pose[2:4], axis=0))  # the midpoint of the rear
            end_point = tuple(np.mean(end_pose[0:2], axis=0))  # the midpoint of the front
            center_line.append(start_point)
            for state in states:
                trajectory.append(state.location)
            center_line.append(end_point)
            trajectory = LineString(trajectory)

            left_bound = trajectory.offset_curve(self.width / 2)
            right_bound = trajectory.offset_curve(-self.width / 2)

            trace = LinearRing(list(left_bound.coords) + list(reversed(list(right_bound.coords))))

        return trace

    def update(self, action: np.ndarray, step: float):
        """Update the agent's state with the given action."""
        current_state, _ = self.physics_model.step(self.current_state, action, step)
        self.add_state(current_state)
