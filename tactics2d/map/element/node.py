from shapely.geometry import Point


class Node:
    """This class implements the map element *node*.

    The add operation of the node is defined as the addition of the coordinates of the node.
    The subtract operation of the node is defined as the subtraction of the coordinates of the node.

    Attributes:
        id_ (str): The id of the node.
        x (float): The x coordinate of the node.
        y (float): The y coordinate of the node.
        location (Point): The location of the node expressed in geometry format.
    """

    def __init__(self, id_: str, x: float, y: float):
        self.id_ = id_
        self.x = x
        self.y = y
        self.location = Point(x, y)

    def __add__(self, other):
        new_node = Node(id_=None, x=self.x + other.x, y=self.y + other.y)
        return new_node

    def __sub__(self, other):
        new_node = Node(id_=None, x=self.x - other.x, y=self.y - other.y)
        return new_node
