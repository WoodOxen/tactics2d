from typing import Tuple

from tactics2d.trajectory.element.trajectory import Trajectory


class ParticipantBase(object):
    def __init__(
        self, id_: int, type_: str = None,
        length: float = None, width: float = None, height: float = None,
        trajectory: Trajectory = None
    ):

        self.id_ = id_
        self.type_ = type_
        self.length = length
        self.width = width
        self.height = height
        self.bind_trajectory(trajectory)

    def _verify_state(self) -> bool:
        """Check if the state change is allowed by the vehicle's physical model.

        Args:
            state1 (_type_): _description_
            state2 (_type_): _description_
            time_interval (_type_): _description_

        Returns:
            bool: _description_
        """
        raise NotImplementedError

    def bind_trajectory(self, trajectory: Trajectory):
        self.trajectory = trajectory