from abc import ABC, abstractmethod

from tactics2d.trajectory.element import State


class PhysicsModelBase(ABC):
    @abstractmethod
    def step(self, state: State, action: tuple, step: float) -> State:
        """Update the state of a vehicle with the given action.

        Args:
            state (State): The current state of the vehicle.
            action (tuple): The action to be applied to the vehicle.
            step (float): The length of the step for the simulation. The unit is second.

        Returns:
            State: The new state of the vehicle.
        """

    @abstractmethod
    def verify_state(self, curr_state: State, prev_state: State) -> bool:
        """Check if the state change is allowed by the participant's physical constraints.

        Args:
            curr_state (State): The current state of the participant.
            prev_state (State): The previous state of the participant.

        Returns:
            bool: True if the new state is valid, False otherwise.
        """
