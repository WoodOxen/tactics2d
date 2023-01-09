from shapely.geometry import Point


class Node(object):
    def __init__(self, id_: str, location: Point):
        self.id_ = id_
        self.location = location