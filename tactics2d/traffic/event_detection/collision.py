# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collision implementation."""


from typing import Optional

from shapely.geometry import Polygon

from tactics2d.geometry import spatial

from .event_base import EventBase


class DynamicCollision(EventBase):
    """This class defines a detector to check whether the agent collides into other agents.

    Bodies are tested as oriented boxes through :func:`spatial.boxes_overlap`,
    which is exact for rectangles and needs no polygon construction, so the
    detector is usable in a per-frame loop over raw agent footprints.
    """

    def __init__(self, margin: float = 1.0):
        """Initialize the detector.

        Args:
            margin: Shrink factor applied to both bodies before testing, so
                contacts that merely graze are reported as collision-free.
                Defaults to 1.0.
        """
        super().__init__()
        self.margin = margin

    def update(self, agent_box, other_boxes, margin: Optional[float] = None) -> bool:
        """Check the agent's footprint against the other agents' footprints.

        Args:
            agent_box: The agent as ``(x, y, yaw, length, width)``. Units are m
                and rad.
            other_boxes: Footprints to test against, each laid out the same way.
            margin (float, optional): Shrink factor for this call, overriding
                the instance default. Defaults to None.

        Returns:
            True when the agent's footprint overlaps any of the others.
        """
        margin = self.margin if margin is None else margin
        return any(spatial.boxes_overlap(agent_box, other, margin) for other in other_boxes)

    def reset(self):
        return


class StaticCollision(EventBase):
    """This class defines a detector to check whether the agent collides into static objects."""

    def __init__(self, static_objects: list = None):
        self.static_objects = static_objects

    def update(self, agent_pose: Polygon) -> bool:
        collide = False
        for static_object in self.static_objects:
            if agent_pose.intersects(static_object.geometry):
                collide = True
                break
        return collide

    def reset(self, static_objects=None):
        self.static_objects = static_objects
