##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: participant_base.py
# @Description: This file defines an abstract class for a traffic participant.
# @Author: Yueyuan Li
# @Version: 1.0.0

from abc import ABC, abstractmethod
from typing import Tuple, List
import logging

from tactics2d.participant.trajectory import State, Trajectory


class ParticipantBase(ABC):
    """This class defines the essential interfaces required to describe a dynamic traffic participant.

    Please feel free to inherent this class to implement your own traffic participant.

    !!! note
        Given that the attributes of this class are commonly utilized across various applications, their types will be verified upon assignment. Whenever feasible, they will be converted to the appropriate type.

    Attributes:
        id_ (Any): The unique identifier of the traffic participant.
        type_ (str): The type of the traffic participant.
        trajectory (Trajectory): The trajectory of the traffic participant.
        color (Any): The color of the traffic participant. This attribute will be left to the sensor module to verify and convert to the appropriate type. You can refer to [Matplotlib's way](https://matplotlib.org/stable/users/explain/colors/colors.html) to specify validate colors. Defaults to black (0, 0, 0).
        length (float): The length of the traffic participant. The default unit is meter. Defaults to None.
        width (float): The width of the traffic participant. The default unit is meter. Defaults to None.
        height (float): The height of the traffic participant. The default unit is meter. Defaults to None.
        verify (bool): Whether to verify the trajectory to bind or the state to add. Defaults to False.
        physics_model (PhysicsModelBase): The physics model of the traffic participant. Defaults to None.
        shape (Any): The shape of the traffic participant. *It should be overridden to implement the shape logic.* This attribute is **read-only**.
        current_state (State): The current state of the traffic participant. This attribute is **read-only**.
    """

    __annotations__ = {
        "type_": str,
        "length": float,
        "width": float,
        "height": float,
        "verify": bool,
    }
    _default_color = (0, 0, 0)

    def __init__(self, id_, type_: str, trajectory: Trajectory = None, **kwargs):
        """Initialize the traffic participant.

        Args:
            id_ (Any): The unique identifier of the traffic participant.
            type_ (str): The type of the traffic participant.
            trajectory (Trajectory, optional): The trajectory of the traffic participant.

        Keyword Args:
            color (Any): The color of the traffic participant. This argument will be left to the sensor module to verify and convert to the appropriate type.
            length (float): The length of the traffic participant. The default unit is meter.
            width (float): The width of the traffic participant. The default unit is meter.
            height (float): The height of the traffic participant. The default unit is meter.
            verify (bool): Whether to verify the trajectory to bind or the state to add. Defaults to False.
        """
        setattr(self, "id_", id_)
        setattr(self, "type_", type_)

        if not trajectory is None:
            self.bind_trajectory(trajectory)
        else:
            self.trajectory = Trajectory(id_=self.id_)

        if not "color" in kwargs or kwargs["color"] is None:
            self.color = self._default_color
        else:
            self.color = kwargs["color"]

        for key in self.__annotations__.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, None)

        self.verify = False if self.verify is None else self.verify
        self.physics_model = None

    def __setattr__(self, __name: str, __value: Any):
        if __name in self.__annotations__:
            if __value is None:
                super().__setattr__(__name, None)
            elif isinstance(__value, self.__annotations__[__name]):
                super().__setattr__(__name, __value)
            else:
                try:
                    if isinstance(__annotations__[__name], tuple):
                        for __type in __annotations__[__name]:
                            if isinstance(__value, __type):
                                super().__setattr__(__name, __type(__value))
                                break
                    else:
                        super().__setattr__(__name, self.__annotations__[__name](__value))
                except:
                    super().__setattr__(__name, None)
                    logging.warning(f"Cannot set {__name} to {__value}. Set to None instead.")
        else:
            super().__setattr__(__name, __value)

    def _verify_state(self, state: State) -> bool:
        """This function verifies whether the input state is feasible considering the traffic participant's dynamics.

        Args:
            state (State): The state to be verified.

        Returns:
            bool: True if the state is valid, False otherwise.
        """
        if self.verify:
            return self.physics_model.verify_state(state, self.current_state)
        return True

    def _verify_trajectory(self, trajectory: Trajectory) -> bool:
        """This function verifies the trajectory of the traffic participant.

        Args:
            trajectory (Trajectory): The trajectory of the traffic participant.

        Returns:
            bool: True if the trajectory is valid, False otherwise.
        """
        if self.verify:
            return self.physics_model.verify_states(trajectory)
        return True

    @property
    @abstractmethod
    def shape(self):
        """The shape of the traffic participant. *It should be overridden in implementation.*"""

    @property
    def current_state(self) -> State:
        return self.trajectory.get_state()

    @abstractmethod
    def bind_trajectory(self, trajectory: Trajectory = None):
        """This function binds a trajectory to the traffic participant. *It should be overridden in implementation.*

        Args:
            trajectory (Trajectory): A trajectory to be bound to the traffic participant.
        """

    @abstractmethod
    def get_pose(self, frame: int = None):
        """This function gets the pose of the traffic participant at the requested frame. *It should be overridden in implementation.*

        Args:
            frame (int, optional): The requested frame. The default unit is millisecond (ms).
        """

    @abstractmethod
    def get_trace(self, frame_range: Tuple[int, int] = None):
        """This function gets the trace of the traffic participant within the requested frame range. *It should be overridden in implementation.*

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range. The first element is the start frame, and the second element is the end frame. The default unit is millisecond (ms).
        """

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

    def add_state(self, state: State):
        """This function wraps the `add_state` method of the trajectory. It does some class-specific checks before adding the state to the trajectory.

        Args:
            state (State): The state to be added to the trajectory.
        """
        self.trajectory.add_state(state)

    def get_state(self, frame: int = None) -> State:
        """This function returns the state of the participant at the requested frame.

        Args:
            frame (int, optional): The requested frame. The default unit is millisecond (ms). If the frame is not specified, the function will return the current state.

        Returns:
            State: The state of the participant at the requested frame.

        Raises:
            KeyError: The requested frame is not found in the trajectory.
        """
        if frame is None:
            return self.current_state
        if frame not in self.history_states:
            raise KeyError(f"Time stamp {frame} is not found in the trajectory {self.id_}.")
        return self.history_states[frame]

    def get_states(self, frame_range: Tuple[int] = None, frames: List[int] = None) -> List[State]:
        """Get the traffic participant's states within the requested frame range.

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range. The first element is the start frame, and the second element is the end frame. The default unit is millisecond (ms).
            frames (List[int], optional): The requested frames. The default unit is millisecond (ms). If the frames are not in the ascending order, they will be sorted before the states are returned.

        Returns:
            List[State]: A list of the traffic participant's states. If the frame_range is specified, the function will return the states within the requested range. If the frames is specified, the function will return the states at the requested frames. If neither is specified, the function will return all states.

        Raises:
            ValueError: The `frame_range` must be a tuple with two elements.
            KeyError: Any requested frame in `frames` is not found in the trajectory.
        """

        if frame_range is None and frames is None:
            return [self.trajectory.get_state(frame) for frame in self.trajectory.frames]
        elif frame_range is not None:
            if len(frame_range) == 2:
                start_frame, end_frame = frame_range
                return [
                    self.trajectory.get_state(frame)
                    for frame in self.trajectory.frames
                    if frame >= start_frame and frame <= end_frame
                ]
            else:
                raise ValueError("The frame range must be a tuple with two elements.")
        elif frames is not None:
            frames = sorted(frames)
            return [self.trajectory.get_state(frame) for frame in frames]

    def reset(self, state: State = None, keep_trajectory: bool = False):
        """Reset the object to a requested state. If the initial state is not specified, the object will be reset to the same initial state as previous.

        Args:
            state (State, optional): The initial state of the object.
            keep_trajectory (bool, optional): Whether to keep the record of history trajectory.
                This argument only works when the state is not specified. When the state is
                not `None`, the trajectory will be reset to the new state.
                Defaults to False.
        """
        self.trajectory.reset(state, keep_trajectory)
