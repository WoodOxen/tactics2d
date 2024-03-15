##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: trajectory.py
# @Description: This file defines a trajectory data structure.
# @Author: Yueyuan Li
# @Version: 1.0.0

import logging
from typing import Any, List, Tuple

import numpy as np

from .state import State


class Trajectory:
    """This class defines a trajectory data structure.

    Attributes:
        id_ (Any): The unique identifier of the trajectory.
        fps (float): The frequency of the trajectory.
        stable_freq (bool): Whether the trajectory has a stable frequency.
        frames (List[int]): The list of time stamps of the trajectory. This attribute is **read-only**.
        initial_state (State): The initial state of the trajectory. If the trajectory is empty, it will be None. This attribute is **read-only**.
        history_states (dict): The dictionary of the states of the trajectory. The key is the time stamp and the value is the state. This attribute is **read-only**.
        last_state (State): The last state of the trajectory. If the trajectory is empty, it will be None. This attribute is **read-only**.
        first_frame (int): The first frame of the trajectory. The unit is millisecond (ms). If the trajectory is empty, it will be None. This attribute is **read-only**.
        last_frame (int): The last frame of the trajectory. The unit is millisecond (ms). If the trajectory is empty, it will be None. This attribute is **read-only**.
        average_speed (float): The average speed of the trajectory. The unit is m/s. This attribute is **read-only**.
    """

    def __init__(self, id_: Any, fps: float = None, stable_freq: bool = True):
        """Initialize the trajectory.

        Args:
            id_ (Any): The unique identifier of the trajectory.
            fps (float, optional): The frequency of the trajectory.
            stable_freq (bool, optional): The flag indicating whether the trajectory has a stable frequency.
        """
        self.id_ = id_
        self.fps = fps
        self.stable_freq = stable_freq

        self._history_states = {}
        self._frames = []
        self._current_state = None

    def __len__(self):
        return len(self._frames)

    @property
    def frames(self) -> List[int]:
        return self._frames

    @property
    def history_states(self):
        return self._history_states

    @property
    def initial_state(self):
        if len(self) == 0:
            return None
        return self._history_states[self._frames[0]]

    @property
    def last_state(self):
        if len(self) == 0:
            return None
        return self._history_states[self._frames[-1]]

    @property
    def first_frame(self):
        if len(self) == 0:
            return None
        return self._frames[0]

    @property
    def last_frame(self):
        if len(self) == 0:
            return None
        return self._frames[-1]

    @property
    def average_speed(self):
        return np.mean([state.speed for state in self._history_states.values()])

    def get_state(self, frame: int = None) -> State:
        """This function get the object's state at the requested frame.

        Args:
            frame (int, optional): The time stamp of the requested state. The unit is millisecond (ms).

        Returns:
            The state of the object at the requested frame. If the frame is None, the current state will be returned.

        Raises:
            KeyError: If the requested frame is not found in the trajectory.
        """
        if frame is None:
            return self._current_state
        if frame not in self._history_states:
            raise KeyError(f"Time stamp {frame} is not found in the trajectory {self.id_}.")
        return self._history_states[frame]

    def add_state(self, state: State):
        """This function adds a state to the trajectory.

        Args:
            state (State): The state to be added to the trajectory.

        Raises:
            ValueError: If the input state is not a valid State object.
            KeyError: If the time stamp of the state is earlier than the last time stamp in the trajectory.
        """
        # check input type first
        if not isinstance(state, State):
            raise ValueError(f"The input state is not a valid State object.")

        if state.frame in self._history_states:
            self._history_states[state.frame] = state
            logging.warning(
                f"State at time stamp {state.frame} is already in trajectory {self.id_}. It will be overwritten."
            )
        if len(self._frames) > 0 and state.frame < self._frames[-1]:
            raise KeyError(
                f"Trying to insert an early time stamp {state.frame} happening \
                    before the last stamp {self._frames[-1]} in trajectory {self.id_}"
            )

        if len(self._history_states) > 1:
            current_interval = state.frame - self._frames[-1]
            last_interval = self._frames[-1] - self._frames[-2]
            if current_interval != last_interval and self.stable_freq:
                self.stable_freq = False
                logging.warning(f"The time interval of the trajectory {self.id_} is uneven.")

        self._frames.append(state.frame)
        self._history_states[state.frame] = state
        self._current_state = state

    def get_trace(self, frame_range: Tuple[int, int] = None) -> list:
        """This function gets the trace of the trajectory within the requested frame range.

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range. The first element is the start frame, and the second element is the end frame. The unit is millisecond (ms).

        Returns:
            trace (list): A list of locations. If the frame range is None, the trace of the whole trajectory will be returned. If the trajectory is empty, an empty list will be returned.
        """
        start_frame = self.first_frame if frame_range is None else frame_range[0]
        end_frame = self.last_frame if frame_range is None else frame_range[1]
        trace = []

        for frame in self.frames:
            if start_frame <= frame <= end_frame:
                trace.append(self.get_state(frame).location)

        return trace

    def reset(self, state: State = None, keep_history: bool = False):
        """This function resets the trajectory.

        Args:
            state (State, optional): The state to be set as the current state. If it is None, the history initial state will be set as the current state.
            keep_history (bool, optional): The flag indicating whether the history states will be kept.
        """
        if state is None:
            initial_state = self.initial_state
            if not keep_history:
                self._history_states.clear()
                self._frames.clear()
                self.add_state(initial_state)
            else:
                self._current_state = initial_state
        else:
            self._history_states.clear()
            self._frames.clear()
            self.add_state(state)
