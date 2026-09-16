# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collision implementation."""


from shapely.geometry import Polygon
from shapely.strtree import STRtree

from tactics2d.map.element.map import HAS_STRTREE

from .event_base import EventBase


class DynamicCollision(EventBase):
    """This class defines a detector to check whether the agent collides into other agents."""

    def __init__(self):
        super().__init__()

    def update(self, agent_pose: Polygon, other_agents) -> bool:
        collide = False
        for other_agent in other_agents:
            other_agent_pose = other_agent.get_pose()
            if agent_pose.geometry.intersects(other_agent_pose.geometry):
                collide = True
                break
        return collide

    def reset(self):
        return


class StaticCollision(EventBase):
    """This class defines a detector to check whether the agent collides into static objects.

    A Shapely ``STRtree`` is built once over the static-object geometries and
    queried with the agent bounding box per update, so the exact ``intersects``
    check runs only on candidates instead of every static object.
    """

    def __init__(self, static_objects: list = None):
        super().__init__()
        self.static_objects = static_objects
        self._static_index = None
        self._static_geometries = []
        self._build_index()

    def _build_index(self):
        """(Re)build the spatial index over the current static objects."""
        self._static_geometries = [
            obj.geometry for obj in (self.static_objects or []) if obj.geometry is not None
        ]
        if HAS_STRTREE and self._static_geometries:
            self._static_index = STRtree(self._static_geometries)
        else:
            self._static_index = None

    def update(self, agent_pose: Polygon) -> bool:
        if self._static_index is not None:
            for idx in self._static_index.query(agent_pose):
                if agent_pose.intersects(self._static_geometries[idx]):
                    return True
            return False

        collide = False
        for static_object in self.static_objects:
            if agent_pose.intersects(static_object.geometry):
                collide = True
                break
        return collide

    def reset(self, static_objects=None):
        self.static_objects = static_objects
        self._build_index()
