##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: scenario_manager.py
# @Description: This script defines an abstract class for a scenario manager.
# @Author: Yueyuan Li
# @Version: 1.0.0

from abc import ABC, abstractmethod
from typing import Tuple

from .status import ScenarioStatus, TrafficStatus


class ScenarioManager(ABC):
    """This class implements an abstract scenario manager.

    The scenario manager is used to reset a scenario (including the map, agent, and participants), manage the state update the traffic participants, and check the traffic events.

    Attributes:
        max_step (int): The maximum time step of the scenario. Defaults to None. If not specified, the scenario will not be terminated by the time step.
        step_size (int): The time interval to update a physical step. The unit is millisecond (ms). Defaults to int(1000 / render_fps).
        render_fps (int): The frame rate of the rendering. The unit is Hz. Defaults to 60.
        off_screen (bool): Whether to display the rendering result on the screen. Defaults to False.
        cnt_step (int): The current time step of the scenario. Initialized as 0.
        scenario_status (ScenarioStatus): The high-level status of the scenario. Defaults to ScenarioStatus.NORMAL.
        traffic_status (TrafficStatus): The low-level status of the traffic. Defaults to TrafficStatus.NORMAL.
        map_ (Map): The map of the scenario. Defaults to None.
        participants (List[Participant]): The list of traffic participants. Defaults to None.
        render_manager (RenderManager): The render manager of the scenario. Defaults to None.
        agent (Agent): The agent vehicle in the scenario. Defaults to None.
        status_checklist (list): A list of function pointers to check the status of the scenario. Defaults to [].
    """

    def __init__(
        self,
        max_step: int = None,
        step_size: int = None,
        render_fps: int = 60,
        off_screen: bool = False,
    ):
        """Initialize the scenario manager.

        Args:
            max_step (int, optional): The maximum time step of the scenario.
            step_size (int, optional): The time interval to update a physical step. The unit is millisecond (ms). If not specified, the step size will be set to int(1000 / render_fps).
            render_fps (int): The frame rate of the rendering. The unit is Hz.
            off_screen (bool): Whether to display the rendering result on the screen.
        """
        self.render_fps = render_fps
        self.off_screen = off_screen
        self.max_step = max_step
        self.step_size = int(step_size) if step_size is not None else int(1000 / render_fps)

        self.cnt_step = 0

        self.scenario_status = ScenarioStatus.NORMAL
        self.traffic_status = TrafficStatus.NORMAL

        self.map_ = None
        self.participants = None
        self.render_manager = None

        self.agent = None

        self.status_checklist = []

    @abstractmethod
    def check_status(self) -> Tuple[ScenarioStatus, TrafficStatus]:
        """This function checks both the high-level scenario status and the low-level traffic status. *It should be overridden in implementation.*

        Returns:
            The high-level scenario status and the low-level traffic status.
        """

    @abstractmethod
    def update(self, action):
        """This function updates the states of the traffic participants (including the agent and other participants). *It should be overridden in implementation.*"""

    @abstractmethod
    def render(self):
        """This function renders the current state with the render manager. *It should be overridden in implementation.*"""

    @abstractmethod
    def reset(self):
        """This function resets the scenario. *It should be overridden in implementation.*"""

    def check_time_exceeded(self) -> bool:
        """This function check the high-level scenario status that whether the current time step has exceeded the maximum time step.

        Returns:
            True if the current time step is greater than the maximum time step. Otherwise, False.
        """
        if self.max_step is None:
            return False
        else:
            return self.cnt_step > self.max_step

    def get_active_participants(self, frame: int) -> list:
        """This function obtains the list of active participants at the given frame.

        Args:
            frame (int): The frame number. The unit is millisecond (ms).

        Returns:
            The list of active participant IDs.
        """
        return [
            participant.id_ for participant in self.participants if participant.is_active(frame)
        ]

    def get_observation(self):
        """This function gets the observation of the current state."""
        return self.render_manager.get_observation()
