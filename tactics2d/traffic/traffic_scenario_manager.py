from abc import ABC, abstractmethod

from shapely.geometry import Polygon

from .violation_detection import TrafficEvent


class TrafficScenarioManager(ABC):
    """The base class for scenario managers.

    The scenario manager is used to reset a scenario (including the map, agent, and
        participants), update the state of the traffic participants, and check the traffic events.

    Attributes:
        render_fps (int): The frame rate of the render manager.
        off_screen (bool): Whether to render the scenario off screen.
        max_step (int): The maximum time step. The simulation will be terminated if the time step exceeds this value.
        step_size (float): The time duration of each step. The default value is 1 / render_fps.
    """

    def __init__(self, render_fps: int, off_screen: bool, max_step: int, step_size: float = None):
        self.render_fps = render_fps
        self.off_screen = off_screen
        self.step_size = step_size if step_size is not None else 1 / render_fps  # TODO

        self.n_step = 0
        self.max_step = max_step
        self.status = TrafficEvent.NORMAL

        self.map_ = None
        self.participants = None
        self.render_manager = None

        self.agent = None

        self.status_checklist = []

    @abstractmethod
    def update(self, action):
        """Update the state of the traffic participants."""

    def get_observation(self):
        """Get the observation of the current state."""
        return self.render_manager.get_observation()

    def render(self):
        """Render the current state with the render manager."""
        self.render_manager.render()

    @abstractmethod
    def reset(self):
        """Reset the scenario."""

    def get_active_participants(self, frame: int) -> list:
        """Get the list of active participants at the given frame."""
        return [
            participant.id_ for participant in self.participants if participant.is_active(frame)
        ]

    def _check_time_exceeded(self):
        """Check if the simulation has reached the maximum time step."""
        if self.n_step > self.max_step:
            self.status = TrafficEvent.TIME_EXCEEDED

    def _check_retrograde(self):
        """Check if the agent is driving in the opposite direction of the lane."""
        raise NotImplementedError

    def _check_non_drivable(self):
        """Check if the agent is driving on the non-drivable area."""
        raise NotImplementedError

    def _check_outbound(self):
        """Check if the agent is outside the map boundary."""
        map_boundary = Polygon(
            [
                (self.map_.boundary[0], self.map_.boundary[2]),
                (self.map_.boundary[0], self.map_.boundary[3]),
                (self.map_.boundary[1], self.map_.boundary[3]),
                (self.map_.boundary[1], self.map_.boundary[2]),
            ]
        )

        if not map_boundary.contains(Polygon(self.agent.get_pose())):
            self.status = TrafficEvent.OUTSIDE_MAP

    def _check_collision(self):
        """Check if the agent collides with other participants or the static obstacles."""
        raise NotImplementedError

    @abstractmethod
    def _check_completed(self):
        """Check if the goal of this scenario is reached."""

    def check_status(self) -> TrafficEvent:
        """Detect different traffic events and return the status.

        If the status is normal, the simulation will continue. Otherwise, the simulation
            will be terminated and the status will be returned. If multiple traffic events
            happen at the same step, only the event with the highest priority will be returned.
        """
        if self.status != TrafficEvent.NORMAL:
            return self.status

        for checker in self.status_checklist:
            checker()
            if self.status != TrafficEvent.NORMAL:
                break

        return self.status
