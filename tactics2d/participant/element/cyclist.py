import numpy as np
from shapely.geometry import LinearRing
from shapely.affinity import affine_transform

from .participant_base import ParticipantBase
from tactics2d.trajectory.element.state import State
from tactics2d.trajectory.element.trajectory import Trajectory


class Cyclist(ParticipantBase):
    def __init__(
        self,
        id_: int,
        type_: str = None,
        length: float = 1.60,
        width: float = 0.65,
        height: float = 1.70,
        trajectory: Trajectory = None,
    ):
        super().__init__(id_, type_, length, width, height)

        self.bbox = LinearRing(
            [
                [0.5 * self.length, -0.5 * self.width],
                [0.5 * self.length, 0.5 * self.width],
                [-0.5 * self.length, 0.5 * self.width],
                [-0.5 * self.length, -0.5 * self.width],
            ]
        )

    def get_pose(self, frame: int = None) -> LinearRing:
        """Get the cyclist's bounding box which is rotated and moved based on the current state."""
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

    def _verify_state(
        self, curr_state: State, prev_state: State, interval: float
    ) -> bool:
        return True

    def _verify_trajectory(self, trajectory: Trajectory):
        return True

    def bind_trajectory(self, trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()
