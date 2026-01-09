# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Off lane implementation."""


from .event_base import EventBase


class OffLane(EventBase):
    """This class defines a detector to check whether the agent is off the road."""

    def __init__(self) -> None:
        self.lanes = None

    def update(self, *args, **kwargs):
        return False

    def reset(self, lanes):
        self.lanes = lanes
