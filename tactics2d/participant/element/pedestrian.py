from shapely.geometry import LineString

from .participant_base import ParticipantBase
from tactics2d.trajectory.element.state import State
from tactics2d.trajectory.element.trajectory import Trajectory


class Pedestrian(ParticipantBase):
    """This class defines a pedestrian with its common properties.

    The area of the pedestrian is defined by a circle, and the radius of the circle is half of
    the width of the pedestrian.

    Attributes:
        id_ (int): The unique identifier of the pedestrian.
        type_ (str, optional): The type of the pedestrian. Defaults to None.
        length (float, optional): The length of the pedestrian. The default unit is meter (m).
            Defaults to None.
        width (float, optional): The width of the pedestrian. The default unit is meter (m).
            Defaults to None.
        height (float, optional): The height of the pedestrian. The default unit is meter (m).
            Defaults to None.
        color (tuple, optional): The color of the pedestrian. Expressed by a tuple with 3 integers.
        trajectory (Trajectory, optional): The trajectory of the pedestrian. Defaults to None.
    """

    def __init__(self, id_: int, type_: str = None, **kwargs):
        super().__init__(id_, type_, **kwargs)

        self.controller = None

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

    def get_pose(self, frame: int = None):
        return super().get_pose(frame)

    def get_trace(self, frame_range=None):
        """Get the trace of the pedestrian within the requested frame range.

        Args:
            frame_range (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        trajectory = self.get_trajectory(frame_range)
        trajectory = LineString(trajectory)
        buffer = trajectory.buffer(self.width / 2)
        return buffer
