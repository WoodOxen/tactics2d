##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: off_lane.py
# @Description: This script defines the event to check whether the agent is off the road.
# @Author: Yueyuan Li
# @Version: 0.1.8rc1


from .event_base import EventBase


class OffLane(EventBase):
    """This class defines a detector to check whether the agent is off the road."""

    def __init__(self) -> None:
        self.lanes = None

    def update(self, *args, **kwargs):
        return False

    def reset(self, lanes):
        self.lanes = lanes
