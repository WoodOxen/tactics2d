##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: no_action.py
# @Description: This script defines the no-action event for the traffic scenario.
# @Author: Yueyuan Li
# @Version: 1.0.0


import numpy as np

from .event_base import EventBase


class NoAction(EventBase):
    """This class defines a detector to check whether the agent has no action for a long time.

    Attributes:
        last_action (Union[int, float, np.ndarray]): The last action of the agent.
        cnt_no_action (int): The counter of the no-action time period.
        max_step (int): The maximum tolerant time step for no action. Defaults to 100.
    """

    def __init__(self, max_step=100):
        """Initialize the attributes in the class.

        Args:
            max_no_action (int, optional): The maximum tolerant time step for no action.
        """
        self.last_action = None
        self.cnt_no_action = 0
        self.max_step = max_step

    def update(self, action) -> bool:
        """This function updates the no-action counter based on the given action.

        Args:
            action (Union[int, float, np.ndarray]): The action of the agent.

        Returns:
            If the no-action counter exceeds the maximum time step, return True; otherwise, return False.
        """
        if isinstance(action, int) or isinstance(action, float):
            if self.last_action is None:
                self.last_action = action
            elif np.abs(action) < 1e-5:
                self.cnt_no_action += 1
                self.last_action = action
            else:
                self.cnt_no_action = 0
                self.last_action = action
        else:
            action = np.array(action)
            if self.last_action is None:
                self.last_action = action
            elif np.linalg.norm(action - self.last_action) < 1e-5:
                self.cnt_no_action += 1
                self.last_action = action
            else:
                self.cnt_no_action = 0
                self.last_action = action

        return self.cnt_no_action > self.max_step

    def reset(self):
        """This function resets the no-action counter."""
        self.last_action = None
        self.cnt_no_action = 0
