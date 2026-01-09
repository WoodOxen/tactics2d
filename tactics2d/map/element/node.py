# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Node implementation."""


from shapely.geometry import Point


class Node:
    """This class implements the map element *Node*.

    !!! quote "Reference"
        - [OpenStreetMap's description of a node](https://wiki.openstreetmap.org/wiki/Node)
        - [Lanelet2's description of a node](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md)

    The add operation of the node is defined as the addition of the coordinates of the node.

    The subtract operation of the node is defined as the subtraction of the coordinates of the node.

    Attributes:
        id_ (str): The id of the node.
        x (float): The x coordinate of the node.
        y (float): The y coordinate of the node.
        location (Point): The location of the node expressed in geometry format. This attribute is **read-only**.
    """

    def __init__(self, id_: str, x: float, y: float):
        """Initialize an instance of this class.

        Args:
            id_ (str): The id of the node.
            x (float): The x coordinate of the node.
            y (float): The y coordinate of the node.
        """
        self.id_ = id_
        self.x = x
        self.y = y

    @property
    def location(self):
        return Point(self.x, self.y)

    def __add__(self, other):
        new_node = Node(id_=None, x=self.x + other.x, y=self.y + other.y)
        return new_node

    def __sub__(self, other):
        new_node = Node(id_=None, x=self.x - other.x, y=self.y - other.y)
        return new_node
