from typing import List
import warnings

import numpy as np

from .state import State


class TrajectoryKeyError(KeyError):
    pass


class Trajectory(object):
    """_summary_

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
    def trace(self) -> List[tuple]:
        trace = []
        for frame in self.frames:
            trace.append(self.history_states[frame].location)
        return trace

    @property
    def average_speed(self):
        return np.mean([state.speed for state in self.history_states.values()])

    def get_state(self, frame: int = None) -> State:
        """Obtain the object's state at the requested time stamp.

        If the time stamp is not specified, the function will return current state.
        If the time stamp is given but not found, the function will return None.
        """
        if frame is None:
            return self.current_state
        if frame not in self.history_states:
            raise TrajectoryKeyError(
                f"Time stamp {frame} is not found in the trajectory {self.id_}."
            )
        return self.history_states[frame]

    def append_state(self, state: State):
        if state.frame in self.history_states:
            raise TrajectoryKeyError(
                f"State at time stamp {state.frame} is already in trajectory {self.id_}."
            )
        if len(self.frames) > 0 and state.frame < self.frames[-1]:
            raise TrajectoryKeyError(
                f"Trying to insert an early time stamp {state.frame} happening \
                    before the last stamp {self.frames[-1]} in trajectory {self.id_}"
            )

        if len(self.history_states) > 1:
            current_interval = state.frame - self.frames[-1]
            last_interval = self.frames[-1] - self.frames[-2]
            if current_interval != last_interval and self.fixed_freq:
                self.fixed_freq = False
                warnings.warn(
                    f"The time interval of the trajectory {self.id_} is uneven."
                )

        self.frames.append(state.frame)
        self.history_states[state.frame] = state
        self.current_state = state

    def reset(self):
        self.current_state = None
        self.history_states.clear()
        self.frames.clear()
