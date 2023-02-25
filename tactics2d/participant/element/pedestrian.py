from .participant_base import ParticipantBase
from tactics2d.trajectory.element.state import State
from tactics2d.trajectory.element.trajectory import Trajectory


class Pedestrian(ParticipantBase):
    def __init__(
        self, id_: int,
        type_: str = None,
        length: float = None,
        width: float = None,
        height: float = None,
        trajectory=None,
    ):
        super().__init__(id_, type_, length, width, height)

        self.controller = None

        self.bind_trajectory(trajectory)

    @property
    def current_state(self) -> State:
        return self.trajectory.get_state()

    @property
    def location(self):
        return self.current_state.location

    def _verify_state(self, curr_state: State, prev_state: State, interval: float) -> bool:
        return True

    def _verify_trajectory(self, trajectory: Trajectory):
        return True

    def bind_trajectory(self, trajectory: Trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()
