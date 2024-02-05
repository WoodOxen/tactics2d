##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: participant_base.py
# @Description: This file defines an abstract class for a traffic participant.
# @Author: Yueyuan Li
# @Version: 1.0.0

from abc import ABC, abstractmethod
from typing import Any, Union
import logging

from tactics2d.participant.trajectory.trajectory import State, Trajectory


class ParticipantBase(ABC):
    """This class defines the essential interfaces required to describe a dynamic traffic participant.

    Please feel free to inherent this class to implement your own traffic participant.

    Attributes:
        id_ (int): The unique identifier of the traffic participant.
        type_ (str): The type of the traffic participant. Defaults to None.
        length (float, optional): The length of the traffic participant. The default unit is meter (m). Defaults to None.
        width (float, optional): The width of the traffic participant. The default unit is meter (m). Defaults to None.
        height (float, optional): The height of the traffic participant. The default unit is meter (m). Defaults to None.
        color (Any, optional): The color of the traffic participant. Defaults to None.
        trajectory (Trajectory): The trajectory of the traffic participant. Defaults to None.

    """

    __annotations__ = {
        "id_": int,
        "type_": str,
        "length": float,
        "width": float,
        "height": float,
        "color": Any,
    }

    def __init__(self, id_: int, type_: str, trajectory: Trajectory = None, **kwargs):
        """Initialize the traffic participant.

        Args:
            id_ (int): The unique identifier of the traffic participant.
            type_ (str): The type of the traffic participant.
            trajectory (Trajectory, optional): The trajectory of the traffic participant. Defaults to None.

        Keyword Args:
            length (float, optional): The length of the traffic participant. The default unit is meter (m). Defaults to None.
            width (float, optional): The width of the traffic participant. The default unit is meter (m). Defaults to None.
            height (float, optional): The height of the traffic participant. The default unit is meter (m). Defaults to None.
            color (Any, optional): The color of the traffic participant. Defaults to None.
        """
        setattr(self, "id_", id_)
        setattr(self, "type_", type_)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.trajectory = Trajectory(id_=self.id_)
        if trajectory is not None:
            self.bind_trajectory(kwargs["trajectory"])
    
    def __setattr__(self, __name: str, __value: Any):
        if __name in self.__annotations__:
            if isinstance(__value, self.__annotations__[__name]):
                super().__setattr__(__name, __value)
            else:
                try:
                    super().__setattr__(__name, self.__annotations__[__name](__value))
                except:
                    super().__setattr__(__name, None)
                    logging.warning(
                        f"The value of attribute {__name} cannot be converted to the required type so it is set to None."
                    )

    def is_active(self, frame: int) -> bool:
        """This function checks if the participant has state information at the requested frame.
        
        Args:
            frame (int): The requested frame. The default unit is millisecond (ms).
        
        Returns:
            bool: True if the participant has state information at the requested frame, False otherwise.
        """
        if frame < self.trajectory.first_frame or frame > self.trajectory.last_frame:
            return False
        return True

    @abstractmethod
    def _verify_trajectory(self, trajectory: Trajectory) -> bool:
        """This function verifies the trajectory of the traffic participant. It should be overridden to implement the verification logic.

        Args:
            trajectory (Trajectory): The trajectory of the traffic participant.
        
        Returns:
            bool: True if the trajectory is valid, False otherwise.
        """

    @abstractmethod
    def bind_trajectory(self, trajectory: Trajectory):
        """This function binds a trajectory with the traffic participant. It should be overridden to implement the binding logic."""

    @abstractmethod
    def get_pose(self, frame: int = None):
        """Get the traffic participant's pose at the requested frame. If the frame is not specified, the function will return the current pose. If the frame is requested but not found, the function will raise a TrajectoryKeyError."""

    def get_states(self, frame_range=None) -> list:
        """Get the traffic participant's states within the requested frame range.

        Args:
            frame_range (_type_, optional): The requested frame range. If the frame range is
                not specified, the function will return all states. If the frame range is a tuple,
                the function will return the states within the requested range. If the frame range
                is a list, the function will return the states at the requested frames. If the frame
                range is an element same as the frame id, the function will return a list only containing
                the state. Defaults to None.

        Returns:
            list: A list of the traffic participant's states.

        Raises:
            TypeError: The frame range must be a tuple, or a list, or an element same as the frame id.
            ValueError: The frame range must be a tuple with two elements.
            KeyError: Any requested frame is not found in the trajectory.
        """
        frames = self.trajectory.frames
        states = []
        if frame_range is None:
            for frame in frames:
                states.append(self.trajectory.get_state(frame))
        elif isinstance(frame_range, tuple):
            if len(frame_range) == 2:
                start_frame, end_frame = frame_range
                for frame in frames:
                    if frame >= start_frame and frame <= end_frame:
                        states.append(self.trajectory.get_state(frame))
            else:
                raise ValueError("The frame range must be a tuple with two elements.")
        elif isinstance(frame_range, list):
            for frame in frame_range:
                states.append(self.trajectory.get_state(frame))
        else:
            try:
                states.append(self.trajectory.get_state(frame_range))
            except:
                raise TypeError(
                    "The frame range must be a tuple, or a list, or an element same as the frame id."
                )

        return states

    def get_trajectory(self, frame_range=None) -> list:
        """Get the traffic participant's trajectory within the requested frame range.

        Args:
            frame_range (_type_, optional): The requested frame range. If the frame range is
                not specified, the function will return the whole trajectory. If the frame range
                is a tuple, the function will return the trajectory within the requested range. If the
                frame range is a list, the function will return the trajectory at the requested frames.
                If the frame range is an element same as the frame id, the function will return
                a list only containing the location. Defaults to None.

        Returns:
            list: A list of the traffic participant's history locations.

        Raises:
            TypeError: The frame range must be a tuple, or a list, or an element same as the frame id.
            ValueError: The frame range must be a tuple with two elements.
            KeyError: Any requested frame is not found in the trajectory.
        """
        frames = self.trajectory.frames
        trajectory = []
        if frame_range is None:
            for frame in frames:
                trajectory.append(self.trajectory.get_location(frame).location)
        elif isinstance(frame_range, tuple):
            if len(frame_range) == 2:
                start_frame, end_frame = frame_range
                for frame in frames:
                    if frame >= start_frame and frame <= end_frame:
                        trajectory.append(self.trajectory.get_location(frame).location)
            else:
                raise ValueError("The frame range must be a tuple with two elements.")
        elif isinstance(frame_range, list):
            for frame in frame_range:
                trajectory.append(self.trajectory.get_location(frame).location)
        else:
            try:
                trajectory.append(self.trajectory.get_location(frame_range).location)
            except:
                raise TypeError(
                    "The frame range must be a tuple, or a list, or an element same as the frame id."
                )

        return trajectory

    @abstractmethod
    def get_trace(self, frame_range=None):
        """Get the region boundary that the traffic participant has occupied within the requested frame range.

        Args:
            frame_range (_type_, optional): The requested frame range. Defaults to None.
        """

    def reset(self, state: State = None, keep_trajectory: bool = False):
        """Reset the object to a requested state. If the initial state is not specified, the object
                will be reset to the same initial state as previous.

        Args:
            state (State, optional): The initial state of the object. Defaults to None.
            keep_trajectory (bool, optional): Whether to keep the record of history trajectory.
                This argument only works when the state is not specified. When the state is
                not None, the trajectory will be reset to the new state.
                Defaults to False.
        """
        self.trajectory.reset(state, keep_trajectory)
