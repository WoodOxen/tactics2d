##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: off_route.py
# @Description: This script defines the event to check whether the agent is off the route.
# @Author: Yueyuan Li
# @Version: 1.0.0


from shapely.geometry import LineString, Point

from .event_base import EventBase


class OffRoute(EventBase):
    """This class defines a detector to check whether the agent is off the route.

    Attributes:
        threshold (float): The threshold to determine whether the agent is off the route. The unit is the same as the route.
        route (LineString): The route for the agent to follow.
    """

    def __init__(self, threshold) -> None:
        self.threshold = threshold
        self.route = None

    def update(self, location: Point):
        """This function will check whether the location of the agent is deviated from the route over the threshold.

        Args:
            location (Point): The location of the agent's center.
        """
        if self.route is None:
            raise ValueError("The route should be set before the event detection.")

        distance = self.route.distance(location)
        return distance > self.threshold

    def reset(self, route: LineString):
        """This function resets the event detector with the given route.

        Args:
            route (LineString): The route for the agent to follow.

        Raises:
            TypeError: The route should be a LineString or a list of points.
        """
        if not isinstance(route, LineString):
            try:
                route = LineString(route)
            except:
                raise TypeError("The route should be a LineString or a list of points.")

        self.route = route
