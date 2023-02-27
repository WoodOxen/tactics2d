from shapely.geometry import Point


class Node(object):
    """Implementation of a node.

    Attributes:
        id_ (str): The id of the node.
        x (float): The x coordinate of the node.
        y (float): The y coordinate of the node.
    """

    def __init__(self, id_: str, x: float, y: float):
        self.id_ = id_
        self.x = x
        self.y = y
        self.location = Point(x, y)

    @property
    def coords(self) -> tuple:
        return (self.x, self.y)

    def __add__(self, other):
        new_node = Node(id_=None, x=self.x + other.x, y=self.y + other.y)
        return new_node

    def __sub__(self, other):
        new_node = Node(id_=None, x=self.x - other.x, y=self.y - other.y)
        return new_node
