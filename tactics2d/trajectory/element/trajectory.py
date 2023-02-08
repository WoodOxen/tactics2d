import warnings

from tactics2d.trajectory.element.state import State


class Trajectory(object):
    def __init__(self, id_: str) -> None:
        self.id_ = id_
        self.current_state = None
        self.history_states = {}
        self.time_stamps = []
        self.even_interval = True

    @property
    def initial_state(self):
        if len(self.time_stamps) == None:
            return None
        return self.history_states[self.time_stamps[0]]

    @property
    def last_state(self):
        if len(self.time_stamps) == None:
            return None
        return self.history_states[self.time_stamps[-1]]

    @property
    def trace(self):
        trace = []
        for time_stamp in self.time_stamps:
            trace.append(list(self.history_states[time_stamp].location))
        return trace

    def get_state(self, time_stamp: float = None) -> State:
        """Obtain the object's state at the requested time stamp.

        If the time stamp is not specified, the function will return current state.
        If the time stamp is given but not found, the function will return None.
        """
        if time_stamp is None:
            return self.current_state
        if time_stamp not in self.history_state:
            raise KeyError(
                f"Time stamp {time_stamp} is not found in the trajectory {self.id_}.")
        return self.history_states[time_stamp]

    def append_state(self, state: State):
        if state.time_stamp in self.history_state:
            raise KeyError(
                f"State at time stamp {state.time_stamp} is already in trajectory {self.id_}.")
        if state.time_stamp < self.time_stamps[-1]:
            raise KeyError(
                f"Trying to insert an early time stamp {state.time_stamp} happening \
                    before the last stamp {self.time_stamps[-1]} in trajectory {self.id_}")

        if len(self.history_states) > 1:
            current_interval = state.time_stamp - self.time_stamps[-1]
            last_interval = self.time_stamps[-1] - self.time_stamps[-2]
            if current_interval  != last_interval and self.even_interval:
                self.even_interval = False
                warnings.warn(f"The time interval of the trajectory {self.id} is uneven.")

        self.time_stamps.append(state.time_stamp)
        self.history_states[state.time_stamp] = state
        self.current_state = state