##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: completed.py
# @Description: This script defines the event detector to check whether the agent has arrived at a target area.
# @Author: Yueyuan Li
# @Version: 1.0.0


from tactics2d.map.element import Area

from .event_base import EventBase


class Arrival(EventBase):
    """This class is used to detect whether the agent has arrived at a target area.

    Attributes:
        target_area (Area): The target area that the agent needs to reach.
        threshold (float): The threshold of the intersection over union (IoU) to determine whether the agent has completed the task. Defaults to 0.95.
    """

    def __init__(self, target_area: Area = None, threshold: float = 0.95):
        """Initialize an instance for the class.

        Args:
            target_area (Area): The target area that the agent needs to reach.
            threshold (float, optional): The threshold of the intersection over union (IoU) to determine whether the agent has completed the task.
        """
        self.target_area = target_area
        self.threshold = threshold

    def update(self, agent_pose: Area):
        """This function updates the status of the task completion.

        Args:
            agent_pose (Polygon): The current pose of the agent.

        Returns:
            is_completed (bool): Whether the agent has completed the task.
            iou (float): The intersection over union (IoU) of the agent's pose and the target area.
        """
        intersection = agent_pose.intersection(self.target_area.geometry)
        union = agent_pose.union(self.target_area.geometry)
        iou = intersection.area / union.area
        is_completed = iou >= self.threshold

        return is_completed, iou

    def reset(self, target_area: Area = None):
        """This function resets the target area of the task."""
        self.target_area = target_area
