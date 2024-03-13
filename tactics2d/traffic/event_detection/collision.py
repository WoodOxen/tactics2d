##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: collision.py
# @Description: This file defines the collision events for traffic scenarios.
# @Author: Yueyuan Li
# @Version: 1.0.0

from shapely.geometry import Polygon

from .event_base import EventBase


class DynamicCollision(EventBase):
    """This class is used to detect whether the agent collides with a movable traffic participant, such as a vehicle or pedestrian."""

    def __init__(self):
        super().__init__()

    def update(self, agent_pose, other_agent_states) -> bool:
        collide = False
        for other_agent_state in other_agent_states:
            if agent_pose.geometry.intersects(other_agent_state.geometry):
                collide = True
                break
        return collide

    def reset(self):
        return


class StaticCollision(EventBase):
    """This class is used to detect whether the agent collides into a static object, such as a building or a wall."""

    def __init__(self, static_objects=None):
        self.collide = False
        self.static_objects = static_objects

    def update(self, agent_pose: Polygon) -> bool:
        return self.collide

    def reset(self, static_objects=None):
        self.collide = False
        self.static_objects = static_objects
