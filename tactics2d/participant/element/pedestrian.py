##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: pedestrian.py
# @Description: This file defines a class for a pedestrian.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Any, Tuple
import logging

from shapely.geometry import LineString, LinearRing

from .participant_base import ParticipantBase
from tactics2d.participant.trajectory import Trajectory
from tactics2d.physics import PointMass

from .participant_template import PEDESTRIAN_TEMPLATE


class Pedestrian(ParticipantBase):
    """This class defines a pedestrian with its common properties.

    Attributes:
        id_ (Any): The unique identifier of the pedestrian.
        type_ (str): The type of the pedestrian. Defaults to "adult_male".
        trajectory (Trajectory): The trajectory of the pedestrian. Defaults to an empty trajectory.
        color (Any): The color of the pedestrian. This attribute will be left to the sensor module to verify and convert to the appropriate type. You can refer to [Matplotlib's way](https://matplotlib.org/stable/users/explain/colors/colors.html) to specify validate colors. Defaults to light-blue (69, 170, 242).
        length (float): The length of the pedestrian. The unit is meter. Defaults to None.
        width (float): The width of the pedestrian. The unit is meter. Defaults to 0.4.
        height (float): The height of the pedestrian. The unit is meter. Defaults to None.
        max_speed (float): The maximum speed of the pedestrian. The unit is meter per second. Defaults to 7.0.
        max_accel (float): The maximum acceleration of the pedestrian. The unit is meter per second squared. Defaults to 1.5.
        speed_range (Tuple[float, float]): The speed range of the pedestrian. The unit is meter per second. Defaults to (-7.0, 7.0).
        accel_range (Tuple[float, float]): The acceleration range of the pedestrian. The unit is meter per second squared. Defaults to (-1.5, 1.5).
        verify (bool): Whether to verify the trajectory to bind or the state to add. Defaults to False.
        physics_model (PhysicsModelBase): The physics model of the pedestrian. Defaults to PointMass.
        geometry (float): The geometry shape of the pedestrian. It is represented as the radius of a circle. This attribute is **read-only**.
        current_state (State): The current state of the pedestrian. This attribute is **read-only**.
    """

    __annotations__ = {
        "type_": str,
        "length": float,
        "width": float,
        "height": float,
        "max_speed": float,
        "max_accel": float,
        "verify": bool,
    }
    _default_color = (69, 170, 242, 255)  # light-blue

    def __init__(
        self, id_: Any, type_: str = "adult_male", trajectory: Trajectory = None, **kwargs
    ):
        """Initialize the pedestrian.

        Args:
            id_ (Any): The unique identifier of the pedestrian.
            type_ (str, optional): The type of the pedestrian.
            trajectory (Trajectory, optional): The trajectory of the pedestrian.

        Keyword Args:
            color (Any): The color of the pedestrian. This attribute will be left to the sensor module to verify and convert to the appropriate type.
            length (float): The length of the pedestrian. The unit is meter.
            width (float): The width of the pedestrian. The unit is meter.
            height (float): The height of the pedestrian. The unit is meter.
            max_speed (float): The maximum speed of the pedestrian. The unit is meter per second.
            max_accel (float): The maximum acceleration of the pedestrian. The unit is meter per second squared.
            verify (bool): Whether to verify the trajectory to bind or the state to add.
            physics_model (PhysicsModelBase): The physics model of the pedestrian. Defaults to PointMass. If the physics model is a custom model, it should be an instance of the [`PhysicsModelBase`](../api/physics.md/#PhysicsModelBase) class.
        """
        super().__init__(id_, type_, trajectory, **kwargs)

        self.load_from_template(type_ if type_ in PEDESTRIAN_TEMPLATE else "adult_male")

        self.speed_range = (-self.max_speed, self.max_speed)
        self.accel_range = (-self.max_accel, self.max_accel)

        if not "physics_model" in kwargs or kwargs["physics_model"] is None:
            self.physics_model = PointMass(
                speed_range=self.speed_range, accel_range=self.accel_range
            )
        else:
            self.physics_model = kwargs["physics_model"]

        self._radius = self.width / 2

    @property
    def geometry(self):
        return self._radius

    def load_from_template(self, type_name: str, overwrite: bool = True, template: dict = None):
        """This function automatically complete the missing attributes of the instance based on the template.

        Args:
            type_name (str): The name of the template parameter set to load.
            overwrite (bool, optional): Whether to overwrite attributes that are not None with the template's value.
            template (dict, optional): The template to load from. The available template names are ["adult_male", "adult_female", "children_six_year_old", "children_ten_year_old"]. You can check the details by calling [`tactics2d.participant.element.list_pedestrian_templates()`](#tactics2d.participant.element.list_pedestrian_templates).
        """
        if template is None:
            template = PEDESTRIAN_TEMPLATE

        if type_name in template:
            for key, value in template[type_name].items():
                if overwrite or getattr(self, key) is None:
                    setattr(self, key, value)
        else:
            logging.warning(
                f"{type_name} is not in the template. Cannot auto-complete the empty attributes"
            )

    def bind_trajectory(self, trajectory: Trajectory):
        """This function binds a trajectory to the pedestrian.

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
                    f"The trajectory is invalid. Pedestrian {self.id_} is not bound to the trajectory."
                )
            else:
                self.trajectory = trajectory
        else:
            self.trajectory = trajectory
            logging.debug(f"Pedestrian {self.id_} is bound to a trajectory without verification.")

    def get_pose(self, frame: int = None) -> Tuple[Tuple[float, float], float]:
        """This function gets the pose of the pedestrian at the requested frame.

        Args:
            frame (int, optional): The requested frame. The unit is millisecond (ms).

        Returns:
            location (Tuple[float, float]): The location of the pedestrian.
            radius (float): The radius of the pedestrian.
        """
        location = self.trajectory.get_state(frame).location
        return (location, self._radius)

    def get_trace(self, frame_range: Tuple[int, int] = None) -> LinearRing:
        """This function gets the trace of the pedestrian within the requested frame range.

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range. The first element is the start frame, and the second element is the end frame. The unit is millisecond (ms).

        Returns:
            The trace of the pedestrian within the requested frame range.
        """
        center_line = LineString(self.trajectory.get_trace(frame_range))
        buffer = center_line.buffer(self.geometry)
        return LinearRing(buffer.exterior.coords)
