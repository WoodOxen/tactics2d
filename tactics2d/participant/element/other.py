##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: other.py
# @Description: This file defines a class for a traffic participant of an unknown type/undefined.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Union, Tuple

import numpy as np
from shapely.geometry import Point, LinearRing, LineString
from shapely.affinity import affine_transform

from tactics2d.participant.trajectory.trajectory import State

from .participant_base import ParticipantBase
from tactics2d.participant.trajectory import Trajectory


class Other(ParticipantBase):
    """This class defines a dynamic traffic participant of an *other* type.

    Attributes:
        id_ (Any): The unique identifier of the traffic participant.
        type_ (str): The type of the traffic participant. Defaults to "unknown".
        trajectory (Trajectory): The trajectory of the traffic participant. Defaults to an empty trajectory.
        color (Any): The color of the traffic participant. This attribute will be left to the sensor module to verify and convert to the appropriate type. You can refer to [Matplotlib's way](https://matplotlib.org/stable/users/explain/colors/colors.html) to specify validate colors. Defaults to black (0, 0, 0).
        length (float): The length of the traffic participant. The default unit is meter. Defaults to None.
        width (float): The width of the traffic participant. The default unit is meter. Defaults to None.
        height (float): The height of the traffic participant. The default unit is meter. Defaults to None.
        physics_model (PhysicsModelBase): The physics model of the traffic participant. Defaults to None.
        shape (LinearRing): The shape of the traffic participant. This attribute is **read-only**. If both length and width are available, the shape will be a rectangle. If only length or width is available, the shape will be a square. Otherwise, the shape will be None.
        current_state (State): The current state of the traffic participant. This attribute is **read-only**.
    """

    def __init__(self, id_, type_: str = "unknown", trajectory: Trajectory = None, **kwargs):
        """Initialize the traffic participant of an *other* type.

        Args:
            id_ (Any): The unique identifier of the traffic participant.
            type_ (str, optional): The type of the traffic participant.
            trajectory (Trajectory, optional): The trajectory of the traffic participant.

        Keyword Args:
            color (Any, optional): The color of the traffic participant. This argument will be left to the sensor module to verify and convert to the appropriate type.
            length (float, optional): The length of the traffic participant. The default unit is meter.
            width (float, optional): The width of the traffic participant. The default unit is meter.
            height (float, optional): The height of the traffic participant. The default unit is meter.
        """
        super().__init__(id_, type_, trajectory, **kwargs)

    @property
    def shape(self):
        if not self.length is None and not self.width is None:
            return LinearRing(
                [
                    [0.5 * self.length, -0.5 * self.width],
                    [0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, -0.5 * self.width],
                ]
            )
        elif not self.length is None:
            return LinearRing(
                [
                    [0.5 * self.length, -0.5 * self.length],
                    [0.5 * self.length, 0.5 * self.length],
                    [-0.5 * self.length, 0.5 * self.length],
                    [-0.5 * self.length, -0.5 * self.length],
                ]
            )
        elif not self.width is None:
            return LinearRing(
                [
                    [0.5 * self.width, -0.5 * self.width],
                    [0.5 * self.width, 0.5 * self.width],
                    [-0.5 * self.width, 0.5 * self.width],
                    [-0.5 * self.width, -0.5 * self.width],
                ]
            )
        else:
            return None

    def _verify_state(self, state: State) -> bool:
        return True

    def _verify_trajectory(self, trajectory: Trajectory) -> bool:
        return True

    def bind_trajectory(self, trajectory: Trajectory = None):
        """This function is used to bind a trajectory to the traffic participant.

        Args:
            trajectory (Trajectory, optional): The trajectory of the traffic participant.

        Raises:
            TypeError: If the input trajectory is not of type [`Trajectory`](#tactics2d.participant.trajectory.Trajectory).
        """
        if not isinstance(trajectory, Trajectory):
            raise TypeError(
                f"Expected a trajectory of type 'Trajectory', but got {type(trajectory)}."
            )

        self.trajectory = trajectory if not trajectory is None else Trajectory(id_=self.id_)

    def get_pose(self, frame: int = None) -> Union[Point, LinearRing]:
        """This function gets the outfigure of the traffic participant at the requested frame.

        Args:
            frame (int, optional): The time stamp of the requested pose. The default unit is millisecond (ms).

        Returns:
           pose (Union[Point, LinearRing]): The outfigure of the traffic participant at the requested frame. If the shape of the traffic participant is available, the pose will be a LinearRing that describe the outfigure of the traffic participant at the requested frame. Otherwise, the pose will be a Point that describe the location of the traffic participant at the requested frame.
        """
        shape = self.shape
        if shape is None:
            return Point(self.trajectory.get_state(frame).location)

        transform_matrix = [
            np.cos(self.trajectory.get_state(frame).heading),
            -np.sin(self.trajectory.get_state(frame).heading),
            np.sin(self.trajectory.get_state(frame).heading),
            np.cos(self.trajectory.get_state(frame).heading),
            self.trajectory.get_state(frame).location[0],
            self.trajectory.get_state(frame).location[1],
        ]
        return affine_transform(shape, transform_matrix)

    def get_trace(self, frame_range: Tuple[int, int] = None) -> Union[LineString, LinearRing]:
        """This function gets the trace of the traffic participant within the requested frame range.

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range. The first element is the start frame, and the second element is the end frame. The default unit is millisecond (ms). If the frame range is None, the trace of the whole trajectory will be returned.

        Returns:
            trace (Union[LineString, LinearRing]): The trace of the traffic participant.

                - If the width of the traffic participant is available, the trace will be a LinearRing that describe the area that the traffic participant occupied during the requested frame range with width/2 as the trace's width.
                - If the width is absent while the length is available, the trace will be a LinearRing that describe the area that the traffic participant occupied during the requested frame range with length/2 as the trace's width.
                - If the shape of the traffic participant is unavailable, the trace will be a LineString that describe the center line of the traffic participant during the requested frame range.
        """
        trace = LineString(self.trajectory.get_trace(frame_range))
        if not self.width is None:
            trace = trace.buffer(self.width / 2, cap_style="square")
            trace = LinearRing(trace.exterior.coords)
        elif not self.length is None:
            trace = trace.buffer(self.length / 2, cap_style="square")
            trace = LinearRing(trace.exterior.coords)

        return trace
