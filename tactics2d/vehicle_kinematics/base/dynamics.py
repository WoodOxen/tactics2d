from abc import ABC, abstractmethod
from typing import Tuple

from tactics2d.vehicle_dynamics.base.state import State


class Dynamics(ABC):
    def __init__(
        self, time_step: float, 
        steering_angle_range: Tuple[float, float] = None, 
        speed_range: Tuple[float, float] = None,
        accel_range: Tuple[float, float] = None,
    ):
        self.time_step = time_step
        self.steering_angle_range = steering_angle_range
        self.speed_range = speed_range
        self.accel_range = accel_range

    @abstractmethod
    def update(self, state: State, action) -> State:
        """Generate a new state based on current state and the given action.

        Args:
            state (State): _description_
            action (_type_): _description_

        Returns:
            State: _description_
        """

    @abstractmethod
    def validate(self, prev_state: State, curr_state: State) -> bool:
        """Verify whether the vehicle can move from previous state to current state within the given time step based on the dynamic model's rules.

        Args:
            prev_state (State): _description_
            curr_state (State): _description_
        """