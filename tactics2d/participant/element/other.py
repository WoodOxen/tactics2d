import numpy as np
from shapely.geometry import LinearRing
from shapely.affinity import affine_transform

from .participant_base import ParticipantBase
from tactics2d.trajectory.element.state import State
from tactics2d.trajectory.element.trajectory import Trajectory


class Other(ParticipantBase):
    """_summary_

    Attributes:

    """
    def __init__(
        self, id_: int, type_: str = None,
        length: float = None, width: float = None, height: float = None,
        shape: LinearRing = None,
        trajectory=None,
    ):
        super().__init__(id_, type_, length, width, height, trajectory)

        self._shape = shape

    @property
    def shape(self) -> LinearRing:
        if self._shape is None:
            self._shape = LinearRing(
                [
                    [0.5 * self.length, -0.5 * self.width], [0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, 0.5 * self.width], [-0.5 * self.length, -0.5 * self.width],
                ]
            )
        return self._shape

    def _verify_trajectory(self, trajectory: Trajectory) -> bool:
        return True

    def bind_trajectory(self, trajectory: Trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()
        
    def get_pose(self, frame: int = None) -> LinearRing:
        state = self.trajectory.get_state(frame)
        transform_matrix = [
            np.cos(state.heading), -np.sin(state.heading),
            np.sin(state.heading), np.cos(state.heading),
            state.location[0], state.location[1],
        ]
        return affine_transform(self.shape, transform_matrix)
