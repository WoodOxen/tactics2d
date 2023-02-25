import warnings

import numpy as np

from .state import State


class TrajectoryKeyError(KeyError):
    pass


class Trajectory(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, id_: int, fps: float = None, fixed_freq: bool = True):
        self.id_ = id_
        self.current_state = None
        self.history_states = {}
        self.frames = []
        self.fps = fps
        self.fixed_freq = fixed_freq

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
    def trace(self):
        trace = []
        for frame in self.frames:
            trace.append(list(self.history_states[frame].location))
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
