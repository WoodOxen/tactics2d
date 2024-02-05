##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: trajectory.py
# @Description: This file defines a class for a trajectory data structure.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import List
import warnings

import numpy as np

from .state import State


class Trajectory(object):
    """This class serves as the foundational implementation of a trajectory data structure.

    Attributes:
        id_ (int): The id of the trajectory.
        fps (float, optional): The frequency of the trajectory. Defaults to None.
        stable_freq (bool, optional): Whether the trajectory has a stable frequency.
            Defaults to True.
        current_state (State, optional): The current state of the trajectory. This attribute
            will automatically update when the trajectory is appended with a new state.
        history_states (dict, optional): The history states of the trajectory. The key is the
            frame of the state and the value is the state itself.
        frames (list, optional): The list of frames of the trajectory. The value of frames
            must be monotonically increasing. The frames are integers and the default unit is
            millisecond (ms).
        initial_state (State, read-only): The initial state of the trajectory.
        last_state (State, read-only): The last state of the trajectory.
        first_frame (int, read-only): The first frame of the trajectory. The unit is millisecond (ms).
        last_frame (int, read-only): The last frame of the trajectory. The unit is millisecond (ms).
        average_speed (float, read-only): The average speed of the trajectory. The unit is m/s.
    """

    def __init__(self, id_: int, fps: float = None, stable_freq: bool = True):
        self.id_ = id_
        self.current_state = None
        self.history_states = {}
        self.frames = []
        self.fps = fps
        self.stable_freq = stable_freq

    def __len__(self):
        return len(self.frames)

    @property
    def initial_state(self):
        if len(self.frames) == 0:
            return None
        return self.history_states[self.frames[0]]

    @property
    def last_state(self):
        if len(self.frames) == 0:
            return None
        return self.history_states[self.frames[-1]]

    @property
    def first_frame(self):
        return self.frames[0]

    @property
    def last_frame(self):
        return self.frames[-1]

    @property
    def average_speed(self):
        return np.mean([state.speed for state in self.history_states.values()])

    def get_state(self, frame: int = None) -> State:
        """Obtain the object's state at the requested frame.

        If the frame is not specified, the function will return the current state. If the
            frame is given but not found, the function will raise a KeyError.
        """
        if frame is None:
            return self.current_state
        if frame not in self.history_states:
            raise KeyError(f"Time stamp {frame} is not found in the trajectory {self.id_}.")
        return self.history_states[frame]

    def append_state(self, state: State):
        if state.frame in self.history_states:
            raise KeyError(
                f"State at time stamp {state.frame} is already in trajectory {self.id_}."
            )
        if len(self.frames) > 0 and state.frame < self.frames[-1]:
            raise KeyError(
                f"Trying to insert an early time stamp {state.frame} happening \
                    before the last stamp {self.frames[-1]} in trajectory {self.id_}"
            )

        if len(self.history_states) > 1:
            current_interval = state.frame - self.frames[-1]
            last_interval = self.frames[-1] - self.frames[-2]
            if current_interval != last_interval and self.stable_freq:
                self.stable_freq = False
                warnings.warn(f"The time interval of the trajectory {self.id_} is uneven.")

        self.frames.append(state.frame)
        self.history_states[state.frame] = state
        self.current_state = state

    def reset(self, state: State = None, keep_history: bool = False):
        if state is None:
            initial_state = self.initial_state
            if not keep_history:
                self.history_states.clear()
                self.frames.clear()
                self.append_state(initial_state)
            else:
                self.current_state = initial_state
        else:
            self.history_states.clear()
            self.frames.clear()
            self.append_state(state)
