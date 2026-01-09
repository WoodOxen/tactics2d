# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Event base implementation."""


from abc import ABC, abstractmethod


class EventBase(ABC):
    """This class defines the essential interfaces required to implement a traffic event detector."""

    @abstractmethod
    def update(self, *args, **kwargs):
        """This function updates the event detector based on the given information."""

    @abstractmethod
    def reset(self):
        """This function resets the event detector."""
