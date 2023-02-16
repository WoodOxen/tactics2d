from tactics2d.trajectory.element.state import State
from tactics2d.trajectory.element.trajectory import Trajectory


class Pedestrian(object):
    def __init__(self, id_: int):
        self.id_ = id_
        self.trajectory = None
        self.controller = None

    def _verify_state(self, state1, state2, time_interval) -> bool:
        return True

    @property
    def current_state(self) -> State:
        return self.trajectory.get_state()

    @property
    def location(self):
        return self.current_state.location

    def bind_trajectory(self, trajectory: Trajectory):
        if self._verify_trajectory(trajectory):
            self.trajectory = trajectory
        else:
            raise RuntimeError()