from shapely.geometry import Point


class Node(object):
    def __init__(self, id: str, location: Point):
        self.id = id
        self.location = location