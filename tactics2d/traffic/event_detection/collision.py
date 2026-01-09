# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collision implementation."""


from shapely.geometry import Polygon

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
