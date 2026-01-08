##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: time_exceed.py
# @Description: This script defines the event to check whether the current time step has exceeded the maximum tolerant time step.
# @Author: Yueyuan Li
# @Version: 0.1.8rc1


from .event_base import EventBase


class TimeExceed(EventBase):
    """This class defines a detector to check whether the current time step has exceeded the maximum tolerant time step.

    Attributes:
        max_step (int): The maximum tolerant time step.
    """

    def __init__(self, max_step: int):
        """Initialize the TimeExceed class.

        Args:
            max_step (int): The maximum tolerant time step.
        """
        self.max_step = max_step
        self.cnt_step = 0

    def update(self):
        """This function updates the time step counter.

        Returns:
            If the time step counter exceeds the maximum time step, return True; otherwise, return False.
        """
        self.cnt_step += 1
        return self.cnt_step > self.max_step

    def reset(self):
        """This function resets the time step counter to 0."""
        self.cnt_step = 0
