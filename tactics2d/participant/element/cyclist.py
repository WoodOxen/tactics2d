##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: cyclist.py
# @Description: This file defines a class for a cyclist participant.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Any, Tuple
import logging

import numpy as np
from shapely.geometry import LineString, LinearRing
from shapely.affinity import affine_transform

from .participant_base import ParticipantBase
from tactics2d.participant.trajectory import Trajectory
from tactics2d.physics import SingleTrackKinematics

from .participant_template import CYCLIST_TEMPLATE


class Cyclist(ParticipantBase):
    """This class defines a cyclist with its common properties.

    Attributes:
        id_ (Any): The unique identifier of the cyclist.
        type_ (str): The type of the cyclist. Defaults to "cyclist".
        trajectory (Trajectory): The trajectory of the cyclist. Defaults to an empty trajectory.
        color (Any): The color of the cyclist. This attribute will be left to the sensor module to verify and convert to the appropriate type. You can refer to [Matplotlib's way](https://matplotlib.org/stable/users/explain/colors/colors.html) to specify validate colors. Defaults to light-orange (253, 150, 68).
        length (float): The length of the cyclist. The unit is meter. Defaults to None.
        width (float): The width of the cyclist. The unit is meter. Defaults to None.
        height (float): The height of the cyclist. The unit is meter. Defaults to None.
        max_steer (float): The maximum steering angle of the cyclist. The unit is radian. Defaults to 1.05.
        max_speed (float): The maximum speed of the cyclist. The unit is meter per second. Defaults to 22.78.
        max_accel (float): The maximum acceleration of the cyclist. The unit is meter per second squared. Defaults to 5.8.
        max_decel (float): The maximum deceleration of the cyclist. The unit is meter per second squared. Defaults to 7.8.
        steer_range (Tuple[float, float]): The steering angle range of the cyclist. The unit is radian. Defaults to (-1.05, 1.05).
        speed_range (Tuple[float, float]): The speed range of the cyclist. The unit is meter per second. Defaults to (0, 22.78).
        accel_range (Tuple[float, float]): The acceleration range of the cyclist. The unit is meter per second squared. Defaults to (-7.8, 5.8).
        verify (bool): Whether to verify the trajectory to bind or the state to add. Defaults to False.
        physics_model (PhysicsModelBase): The physics model of the cyclist. Defaults to SingleTrackKinematics.
        geometry (float): The geometry shape of the cyclist. It is represented as a bounding box with its original point located at the center. This attribute is **read-only**.
        current_state (State): The current state of the cyclist. This attribute is **read-only**.
    """

    __annotations__ = {
        "type_": str,
        "length": float,
        "width": float,
        "height": float,
        "max_steer": float,
        "max_speed": float,
        "max_accel": float,
        "max_decel": float,
        "verify": bool,
    }
    _default_color = (253, 150, 68, 255)  # light-orange

    def __init__(self, id_: Any, type_: str = "cyclist", trajectory: Trajectory = None, **kwargs):
        """Initialize a cyclist participant. `tactics2d` treat motorcyclists as cyclists by default.

        Args:
            id_ (Any): The unique identifier of the cyclist.
            type_ (str, optional): The type of the cyclist.
            trajectory (Trajectory, optional): The trajectory of the cyclist.

        Keyword Args:
            color (Any): The color of the cyclist. This argument will be left to the sensor module to verify and convert to the appropriate type.
            length (float): The length of the cyclist. The unit is meter.
            width (float): The width of the cyclist. The unit is meter.
            height (float): The height of the cyclist. The unit is meter.
            max_steer (float): The maximum steering angle of the cyclist. The unit is radian.
            max_speed (float): The maximum speed of the cyclist. The unit is meter per second.
            max_accel (float): The maximum acceleration of the cyclist. The unit is meter per second squared.
            verify (bool): Whether to verify the trajectory to bind or the state to add.
            physics_model (PhysicsModelBase): The physics model of the cyclist. Defaults to None. If the physics model is a custom model, it should be an instance of the [`PhysicsModelBase`](../api/physics.md/#PhysicsModelBase) class.
        """
        super().__init__(id_, type_, trajectory, **kwargs)

        self.load_from_template(type_ if type_ in CYCLIST_TEMPLATE else "cyclist")

        self.steer_range = (-self.max_steer, self.max_steer)
        self.speed_range = (0, self.max_speed)
        self.accel_range = (-self.max_decel, self.max_accel)

        if not self.verify:
            self.physics_model = None
        elif not "physics_model" in kwargs or kwargs["physics_model"] is None:
            self.physics_model = SingleTrackKinematics()
        else:
            self.physics_model = kwargs["physics_model"]

        self._bbox = LinearRing(
            [
                [0.5 * self.length, -0.5 * self.width],
                [0.5 * self.length, 0.5 * self.width],
                [-0.5 * self.length, 0.5 * self.width],
                [-0.5 * self.length, -0.5 * self.width],
            ]
        )

    @property
    def geometry(self) -> LinearRing:
        return self._bbox

    def load_from_template(
        self, type_name: str, overwrite: bool = True, template: dict = CYCLIST_TEMPLATE
    ):
        """This function automatically complete the missing attributes of the instance based on the template.

        Args:
            type_name (str): The name of the template parameter set to load.
            overwrite (bool, optional): Whether to overwrite attributes that are not None with the template's value.
            template (dict, CYCLIST_TEMPLATE): The template to load from. The available template names are ["cyclist", "moped", "motorcycle"]. You can check the details by calling [`tactics2d.participant.element.list_cyclist_templates()`](#tactics2d.participant.element.list_cyclist_templates).
        """
        if type_name in template:
            for key, value in template[type_name].items():
                if getattr(self, key, None) is None or overwrite:
                    setattr(self, key, value)
        else:
            logging.warning(
                f"{type_name} is not in the cyclist template. Cannot auto-complete the empty attributes"
            )

    def bind_trajectory(self, trajectory: Trajectory):
        """This function binds a trajectory to the cyclist.

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
                    f"The trajectory is invalid. Cyclist {self.id_} is not bound to the trajectory."
                )
            else:
                self.trajectory = trajectory
        else:
            self.trajectory = trajectory
            logging.debug(f"Cyclist {self.id_} is bound to a trajectory without verification.")

    def get_pose(self, frame: int = None) -> LinearRing:
        """This function gets the pose of the cyclist at the requested frame.

        Args:
            frame (int, optional): The frame to get the cyclist's pose.

        Returns:
            pose (LinearRing): The cyclist's bounding box which is rotated and moved based on the current state.
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
        pose = affine_transform(self._bbox, transform_matrix)

        return pose

    def get_trace(self, frame_range: Tuple[int, int] = None) -> LinearRing:
        """This function gets the trace of the cyclist within the requested frame range.

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range. The first element is the start frame, and the second element is the end frame. The unit is millisecond (ms).

        Returns:
            The trace of the cyclist within the requested frame range.
        """
        center_line = LineString(self.trajectory.get_trace(frame_range))
        buffer = center_line.buffer(self.width / 2, cap_style="square")
        return LinearRing(buffer.exterior.coords)
