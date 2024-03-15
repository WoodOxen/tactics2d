##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: event_base.py
# @Description: This script defines the event base class for the traffic scenario.
# @Author: Yueyuan Li
# @Version: 1.0.0


from abc import ABC, abstractmethod


class EventBase(ABC):
    """This class defines the essential interfaces required to implement a traffic event detector."""

    @abstractmethod
    def update(self, *args, **kwargs):
        """This function updates the event detector based on the given information."""

    @abstractmethod
    def reset(self):
        """This function resets the event detector."""
