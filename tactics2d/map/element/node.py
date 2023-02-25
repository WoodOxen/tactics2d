from shapely.geometry import Point


class Node(object):
    """tactics2d.map.element.Node

    Args:
        object (_type_): _description_
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
