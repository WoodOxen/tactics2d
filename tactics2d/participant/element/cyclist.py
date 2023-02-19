from tactics2d.participant.element.participant_base import ParticipantBase


class Cyclist(ParticipantBase):
    def __init__(
            self, id_: int, type_: str = None,
            length: float = None, width: float = None, height: float = None,
        ):

        self.id_ = id_
        self.trajectory = None
        self.controller = None

        super().__init__(id_, type_, length, width, height)

    def _verify_state(self) -> bool:
        return True

    def bind_trajectory(self, trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()