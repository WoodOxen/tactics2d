# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Out bound implementation."""


from shapely.geometry import Polygon

from .event_base import EventBase


class OutBound(EventBase):
    """This class defines a detector to check whether the agent is out of the map boundary.

    Attributes:
        map_boundary (Polygon): The boundary of the map.
        out_bound (bool): Whether the agent is out of the map boundary.
    """

    def __init__(self, boundary: tuple = None):
        """Initialize an instance for the class.

        Args:
            boundary (tuple): The boundary of the map. The boundary is represented by a tuple of four elements (xmin, xmax, ymin, ymax)
        """
        self.map_boundary = None
        if not boundary is None:
            self.map_boundary = Polygon(
                [
                    (boundary[0], boundary[2]),
                    (boundary[0], boundary[3]),
                    (boundary[1], boundary[3]),
                    (boundary[1], boundary[2]),
                ]
            )

    def update(self, agent_pose: Polygon) -> bool:
        """This function checks whether the agent is out of the map boundary.

        Args:
            agent_pose (Polygon): The current pose of the agent.

        Returns:
            If the agent is out of the map boundary, return True; otherwise, return False.
        """
        if self.map_boundary is None:
            return False
        return not self.map_boundary.contains(agent_pose)

    def reset(self, boundary: tuple = None):
        """This function reset the instance by updating the map boundary.

        Args:
            boundary (tuple): The boundary of the map. The boundary is represented by a tuple of four elements (xmin, xmax, ymin, ymax)
        """
        self.map_boundary = None
        if not boundary is None:
            self.map_boundary = Polygon(
                [
                    (boundary[0], boundary[2]),
                    (boundary[0], boundary[3]),
                    (boundary[1], boundary[3]),
                    (boundary[1], boundary[2]),
                ]
            )
