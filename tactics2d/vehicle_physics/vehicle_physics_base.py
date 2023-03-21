from abc import ABC, abstractmethod

from tactics2d.trajectory.element import State


class VehiclePhysicsBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def verify_state(self, curr_state: State, prev_state: State, interval: float) -> bool:
        """Check if the state change is allowed by the participant's physical constraints.

        Args:
            curr_state (State): The current state of the participant.
            prev_state (State): The previous state of the participant.
            interval (float): The time interval between the current state and the previous state.

        Returns:
            bool: True if the new state is valid, False otherwise.
        """
