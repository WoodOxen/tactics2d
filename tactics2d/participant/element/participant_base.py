from abc import ABC, abstractmethod

import numpy as np
from shapely.affinity import affine_transform

from tactics2d.trajectory.element.trajectory import State, Trajectory


class ParticipantBase(ABC):
    """This class define an interface for all the traffic participants provided in tactics2d.

    Attributes:
        id_ (int): The unique identifier of the traffic participant.
        type_ (str): The type of the traffic participant. Defaults to None.
        length (float): The length of the traffic participant. The default unit is meter (m).
            Defaults to None.
        width (float): The width of the traffic participant. The default unit is meter (m).
            Defaults to None.
        height (float): The height of the traffic participant. The default unit is meter (m).
            Defaults to None.
        trajectory (Trajectory): The trajectory of the traffic participant. Defaults to None.
    """

    def __init__(
        self,
        id_: int,
        type_: str,
        length: float = None,
        width: float = None,
        height: float = None,
        trajectory: Trajectory = None,
    ):
        self.id_ = id_
        self.type_ = type_
        self.length = length
        self.width = width
        self.height = height

        self.trajectory = Trajectory(id_=self.id_)
        if trajectory is not None:
            self.bind_trajectory(trajectory)

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

    def is_alive(self, frame: int) -> bool:
        """Check if the participant is in its simulation life cycle."""
        if frame < self.trajectory.first_frame or frame > self.trajectory.last_frame:
            return False
        return True

    @abstractmethod
    def _verify_trajectory(self, trajectory: Trajectory):
        """Check if the trajectory is allowed by the participant's physical constraints.

        Returns:
            bool: True if the trajectory is valid, False otherwise.
        """

    @abstractmethod
    def bind_trajectory(self, trajectory: Trajectory):
        """Bind a trajectory with the traffic participant."""

    def get_pose(self, frame: int = None):
        """Get the traffic participant's pose at the requested frame.

        If the frame is not specified, the function will return the current pose.
        If the frame is given but not found, the function will raise a TrajectoryKeyError.
        """
        state = self.trajectory.get_state(frame)
        transform_matrix = [
            np.cos(state.heading),
            -np.sin(state.heading),
            np.sin(state.heading),
            np.cos(state.heading),
            state.location[0],
            state.location[1],
        ]
        return affine_transform(self.bbox, transform_matrix)

    def reset(self, state: State = None, keep_trajectory: bool = False):
        """Reset the object to a given state. If the initial state is not specified, the object
                will be reset to the same initial state as previous.

        Args:
            state (State, optional): The initial state of the object. Defaults to None.
            keep_trajectory (bool, optional): Whether to keep the record of history trajectory.
                This argument only works when the state is not specified. When the state is
                not None, the trajectory will be reset to the new state.
                Defaults to False.
        """
        self.trajectory.reset(state, keep_trajectory)
