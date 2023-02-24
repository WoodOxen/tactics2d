from tactics2d.trajectory.element.trajectory import State, Trajectory


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
    
    def _verify_trajectory(self):
        raise NotImplementedError

    def bind_trajectory(self, trajectory: Trajectory):
        raise NotImplementedError