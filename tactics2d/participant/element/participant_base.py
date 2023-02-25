from abc import ABC, abstractmethod

from tactics2d.trajectory.element.trajectory import State, Trajectory


class ParticipantBase(ABC):
    """This class define an interface for all the traffic participants provided in tactics2d.

    Attributes:
        id_ (int): The unique identifier of the traffic participant.
        type_ (str): The type of the traffic participant. Defaults to None.
        length (float): The length of the traffic participant. The default unit is meter (m). Defaults to None.
        width (float): The width of the traffic participant. The default unit is meter (m). Defaults to None.
        height (float): The height of the traffic participant. The default unit is meter (m). Defaults to None.
        trajectory (Trajectory): The trajectory of the traffic participant. Defaults to None.
    """

    def __init__(
        self, id_: int, type_: str = None,
        length: float = None, width: float = None, height: float = None,
        trajectory: Trajectory = None,
    ):
        self.id_ = id_
        self.type_ = type_
        self.length = length
        self.width = width
        self.height = height

    @property
    def current_state(self) -> State:
        return self.trajectory.get_state()

    @property
    def location(self):
        return self.current_state.location

    @property
    def heading(self) -> float:
        return self.current_state.heading

    @property
    def velocity(self):
        return (self.current_state.vx, self.current_state.vy)

    @property
    def speed(self) -> float:
        return self.current_state.speed

    @property
    def accel(self):
        return self.current_state.accel

    @abstractmethod
    def _verify_state(self, curr_state: State, prev_state: State, interval: float) -> bool:
        """Check if the state change is allowed by the participant's physics constraints.

        Args:
            curr_state (State): 
            prev_state (State):
            interval (float): The time interval between the current state and the previous state.

        Returns:
            bool: True if the state change is valid, False otherwise.
        """

    @abstractmethod
    def _verify_trajectory(self, trajectory: Trajectory):
        """Check if the trajectory is valid based on the participant's physics constraints.

        Returns:
            bool: True if the trajectory is valid, False otherwise.
        """

    @abstractmethod
    def bind_trajectory(self, trajectory: Trajectory):
        """Bind a trajectory with the traffic participant."""
