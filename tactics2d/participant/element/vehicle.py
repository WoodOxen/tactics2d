##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: vehicle.py
# @Description: This file defines a class for a four-wheeled vehicle.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Any, Tuple
import logging

import numpy as np
from shapely.geometry import LinearRing, LineString
from shapely.affinity import translate, affine_transform

from .participant_base import ParticipantBase
from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.physics import SingleTrackKinematics

from .participant_template import VEHICLE_TEMPLATE


class Vehicle(ParticipantBase):
    """This class defines a four-wheeled vehicle with its common properties.

    Attributes:
        id_ (int): The unique identifier of the vehicle.
        type_ (str): The type of the vehicle. Defaults to "medium_car".
        trajectory (Trajectory): The trajectory of the vehicle. Defaults to an empty trajectory.
        color (tuple): The color of the vehicle. The color of the traffic participant. This attribute will be left to the sensor module to verify and convert to the appropriate type. You can refer to [Matplotlib's way](https://matplotlib.org/stable/users/explain/colors/colors.html) to specify validate colors. Defaults to light-turquoise (43, 203, 186).
        length (float): The length of the vehicle. The default unit is meter. Defaults to None.
        width (float): The width of the vehicle. The default unit is meter. Defaults to None.
        height (float): The height of the vehicle. The default unit is meter. Defaults to None.
        kerb_weight: (float): The weight of the vehicle. The default unit is kilogram (kg). Defaults to None.
        wheel_base (float): The wheel base of the vehicle. The default unit is meter. Defaults to None.
        front_overhang (float): The front overhang of the vehicle. The default unit is meter. Defaults to None.
        rear_overhang (float): The rear overhang of the vehicle. The default unit is meter. Defaults to None.
        driven_mode: (str): The driven way of the vehicle. The available options are "FWD", "RWD", "4WD", and "AWD". Defaults to "FWD".
        max_steer (float): The maximum approach angle of the vehicle. The unit is radian. Defaults to $\pi$/6.
        max_speed (float): The maximum speed of the vehicle. The unit is meter per second. Defaults to 55.56 (= 200 km/h).
        max_accel (float): The maximum acceleration of the vehicle. The unit is meter per second squared. Defaults to 3.0.
        max_decel (float): The maximum deceleration of the vehicle. The unit is meter per second squared. Defaults to 10.
        steer_range (Tuple[float, float]): The range of the vehicle steering angle. The unit is radian. Defaults to (-$\pi$/6, $\pi$/6).
        speed_range (Tuple[float, float]): The range of the vehicle speed. The unit is meter per second (m/s). Defaults to (-16.667, 55.556) (= -60~200 km/h).
        accel_range (Tuple[float, float]): The range of the vehicle acceleration. The unit is meter per second squared. Defaults to (-10, 3).
        verify (bool): Whether to verify the trajectory to bind or the state to add. Defaults to False.
        physics_model (PhysicsModelBase): The physics model of the cyclist. Defaults to SingleTrackKinematics.
        shape (float): The shape of the cyclist. It is represented as a bounding box with its original point located at the mass center. This attribute is **read-only**.
        current_state (State): The current state of the traffic participant. This attribute is **read-only**.
    """

    __annotations__ = {
        "type_": str,
        "length": float,
        "width": float,
        "height": float,
        "kerb_weight": float,
        "wheel_base": float,
        "front_overhang": float,
        "rear_overhang": float,
        "driven_mode": str,
        "max_steer": float,
        "max_speed": float,
        "max_accel": float,
        "max_decel": float,
        "verify": bool,
    }
    _default_color = (43, 203, 186, 255)  # light-turquoise
    _driven_modes = {"FWD", "RWD", "4WD", "AWD"}

    def __init__(
        self, id_: Any, type_: str = "medium_car", trajectory: Trajectory = None, **kwargs
    ):
        """Initialize the vehicle.

        Args:
            id_ (int): The unique identifier of the vehicle.
            type_ (str, optional): The type of the vehicle.
            trajectory (Trajectory, optional): The trajectory of the vehicle.

        Keyword Args:
            length (float, optional): The length of the vehicle. The default unit is meter. Defaults to None.
            width (float, optional): The width of the vehicle. The default unit is meter. Defaults to None.
            height (float, optional): The height of the vehicle. The default unit is meter. Defaults to None.
            color (tuple, optional): The color of the vehicle. Defaults to None.
            kerb_weight (float, optional): The kerb weight of the vehicle. The default unit is kilogram (kg). Defaults to None.
            wheel_base (float, optional): The wheel base of the vehicle. The default unit is meter. Defaults to None.
            front_overhang (float, optional): The front overhang of the vehicle. The default unit is meter. Defaults to None.
            rear_overhang (float, optional): The rear overhang of the vehicle. The default unit is meter. Defaults to None.
            max_steer (float, optional): The maximum steering angle of the vehicle. The unit is radian. Defaults to $\pi$ / 6.
            max_speed (float, optional): The maximum speed of the vehicle. The unit is meter per second. Defaults to 55.556 (=200 km/h).
            max_accel (float, optional): The maximum acceleration of the vehicle. The unit is meter per second squared. Defaults to 3.0.
            max_decel (float, optional): The maximum deceleration of the vehicle. The unit is meter per second squared. Defaults to 10.0.
            verify (bool): Whether to verify the trajectory to bind or the state to add.
            physics_model (PhysicsModelBase): The physics model of the cyclist. Defaults to None. If the physics model is a custom model, it should be an instance of the [`PhysicsModelBase`](../api/physics.md/#PhysicsModelBase) class.
            driven_mode (str, optional): The driven way of the vehicle. The available options are ["FWD", "RWD", "4WD", "AWD"]. Defaults to "FWD".
        """

        super().__init__(id_, type_, trajectory, **kwargs)

        self.max_steer = np.pi / 6 if self.max_steer is None else self.max_steer
        self.max_speed = 55.556 if self.max_speed is None else self.max_speed
        self.max_accel = 3.0 if self.max_accel is None else self.max_accel
        self.max_decel = 10.0 if self.max_decel is None else self.max_decel

        self.speed_range = (-16.667, self.max_speed)
        self.steer_range = (-self.max_steer, self.max_steer)
        self.accel_range = (-self.max_accel, self.max_accel)
        self.driven_mode = "FWD" if self.driven_mode is None else self.driven_mode

        if self.verify:
            if "physics_model" not in kwargs or kwargs["physics_model"] is None:
                self.physics_model = SingleTrackKinematics()
            else:
                self.physics_model = kwargs["physics_model"]

        if not None in [self.length, self.width]:
            self._bbox = LinearRing(
                [
                    [0.5 * self.length, -0.5 * self.width],
                    [0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, -0.5 * self.width],
                ]
            )
        else:
            self._bbox = None

    @property
    def geometry(self) -> LinearRing:
        return self._bbox

    def load_from_template(
        self, type_name: str, overwrite: bool = False, template: dict = VEHICLE_TEMPLATE
    ):
        """Load the vehicle properties from the template.

        Args:
            type_name (str): The type of the vehicle.
            overwrite (bool, optional): Whether to overwrite the existing properties. Defaults to False.
            template (dict, optional): The template of the vehicle. Defaults to VEHICLE_TEMPLATE.
        """
        if type_name in template:
            for key in template[type_name]:
                if getattr(self, key, None) is None or overwrite:
                    setattr(self, key, value)
        else:
            logging.warning(
                f"{type_name} is not in the vehicle template. The default values will be used."
            )

    def add_state(self, state: State):
        """This function adds a state to the vehicle.

        Args:
            state (State): The state to add.
        """
        if not self.verify or self.physics_model is None:
            self.trajectory.add_state(state)
        elif self.physics_model.verify_state(state, self.trajectory.current_state):
            self.trajectory.append_state(state)
        else:
            raise RuntimeError(
                "Invalid state checked by the physics model %s."
                % (self.physics_model.__class__.__name__)
            )

    def bind_trajectory(self, trajectory: Trajectory):
        """This function binds a trajectory to the vehicle.

        Args:
            trajectory (Trajectory): The trajectory to bind.

        Raises:
            TypeError: If the input trajectory is not of type [`Trajectory`](#tactics2d.participant.trajectory.Trajectory).
        """
        if not isinstance(trajectory, Trajectory):
            raise TypeError("The trajectory must be an instance of Trajectory.")

        if self.verify:
            if not self._verify_trajectory(trajectory):
                self.trajectory = Trajectory(self.id_)
                logging.warning(
                    f"The trajectory is invalid. Vehicle {self.id_} is not bound to the trajectory."
                )
            else:
                self.trajectory = trajectory
        else:
            self.trajectory = trajectory
            logging.info(f"Vehicle {self.id_} is bound to a trajectory without verification.")

    def get_pose(self, frame: int = None) -> LinearRing:
        """This function gets the pose of the vehicle at the requested frame.

        Args:
            frame (int, optional): The frame to get the vehicle's pose.
        
        Returns:
            pose (LinearRing): The vehicle's bounding box which is rotated and moved based on the current state.
        """
        state = self.trajectory.get_state(frame)
        transform_matrix = [
            np.cos(state.heading),
            -np.sin(state.heading),
            np.sin(state.heading),
            np.cos(state.heading),
            state.location[0],
            state.location[1],
        ]
        return affine_transform(self._bbox, transform_matrix)

    def get_pose_new(self, **kwargs) -> LinearRing:
        if kwargs is None or "frame" in kwargs:
            state = self.trajectory.get_state(kwargs["frame"])
        elif "state" in kwargs and isinstance(kwargs["state"], State):
            state = kwargs["state"]
        elif "heading" in kwargs:
            if "location" in kwargs:
                state = State(0, kwargs["location"][0], kwargs["location"][1], kwargs["heading"])
            elif "x" in kwargs and "y" in kwargs:
                state = State(0, kwargs["x"], kwargs["y"], kwargs["heading"])
        else:
            raise NotImplementedError("Invalid arguments.")

        transform_matrix = [
            np.cos(state.heading),
            -np.sin(state.heading),
            np.sin(state.heading),
            np.cos(state.heading),
            state.location[0],
            state.location[1],
        ]
        bbox = translate(self.bbox_new, self.center_shift)
        return affine_transform(bbox, transform_matrix)

    def get_trace(self, frame_range: Tuple[int, int] = None) -> LinearRing:
        """This function gets the trace of the vehicle within the requested frame range.

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range. The first element is the start frame, and the second element is the end frame. The default unit is millisecond (ms).
        
        Returns:
            The trace of the cyclist within the requested frame range.
        """
        states = self.get_states(frame_range)
        trace = None
        if len(states) == 0:
            pass
        elif len(states) == 1:
            trace = self.get_pose_new(frame=states[0].frame)
        else:
            center_line = []
            start_pose = np.array(list(self.get_pose_new(frame=states[0].frame).coords))
            end_pose = np.array(list(self.get_pose_new(frame=states[-1].frame).coords))
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

    def update(self, action: Tuple[float, float], interval: int = None):
        """This function updates the vehicle's state based on the input action command.

        Args:
            action (Tuple[float, float]): The action to be applied to the vehicle. The action is a two-element tuple [steer, accel]. The steer is the steering angle, and the accel is the acceleration. The unit of the steer is radian, and the unit of the accel is meter per second squared (m/s$^2$).
            interval (int, optional): The time interval to execute the action. The default unit is millisecond (ms). 
        """
        current_state, _ = self.physics_model.step(self.current_state, action, interval)
        self.add_state(current_state)
