from .participant_base import ParticipantBase
from tactics2d.trajectory.element.state import State
from tactics2d.trajectory.element.trajectory import Trajectory


class Cyclist(ParticipantBase):
    def __init__(
        self,
        id_: int,
        type_: str = None,
        length: float = None,
        width: float = None,
        height: float = None,
        trajectory: Trajectory = None,
    ):
        self.id_ = id_
        self.trajectory = None
        self.controller = None

        super().__init__(id_, type_, length, width, height)

        self.bind_trajectory(trajectory)

    def _verify_state(self, state: State) -> bool:
        return True

    def _verify_trajectory(self, trajectory: Trajectory):
        return True

    def bind_trajectory(self, trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()
