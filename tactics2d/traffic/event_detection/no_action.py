##! python3
from __future__ import annotations

from typing import Any, Union

from shapely.geometry import Polygon

from .event_base import EventBase

# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: no_action.py
# @Description: This script defines the no-action event for the traffic scenario.
# @Author: Yueyuan Li
# @Version: 0.1.8rc1


class NoAction(EventBase):
    """This class defines a detector to check whether the agent has no action for a long time.

    Attributes:
        last_action (Union[int, float, Any]): The last action of the agent.
        cnt_no_action (int): The counter of the no-action time period.
        max_step (int): The maximum tolerant time step for no action. Defaults to 100.
    """

    def __init__(self, max_step=100):
        """Initialize the attributes in the class.

        Args:
            max_no_action (int, optional): The maximum tolerant time step for no action.
        """
        self.last_pose = None
        self.cnt_no_action = 0
        self.max_step = max_step

    def update(self, agent_pose: Polygon) -> bool:
        """This function updates the no-action counter based on the given agent pose.

        Args:
            agent_pose (Polygon): The current pose of the agent.

        Returns:
            If the no-action counter exceeds the maximum time step, return True; otherwise, return False.
        """
        if self.last_pose is None:
            self.last_pose = agent_pose
        else:
            intersection = agent_pose.intersection(self.last_pose)
            union = agent_pose.union(self.last_pose)
            iou = intersection.area / union.area
            if iou > 0.999:
                self.cnt_no_action += 1
            else:
                self.cnt_no_action = 0
            self.last_pose = agent_pose

        return self.cnt_no_action > self.max_step

    def reset(self):
        """This function resets the no-action counter."""
        self.last_action = None
        self.cnt_no_action = 0
