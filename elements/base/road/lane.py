from enum import Enum

from shapely.geometry import LineString

from Tacktics2D.elements.base.road.defaults import *


class Relationship(Enum):
    IRRELEVANT = 0
    PREDECESSOR = 1
    SUCCESSOR = 2
    LEFT_NEIGHBOR = 3
    RIGHT_NEIGHBOR = 4


class Lane(object):
    """_summary_

    Args:
        id (str): lane id
        left_side (LineString):
        right_side (LineString):
        subtype (str): lane type, which can be from the defaults or customized
        participant (list): Defaults to None.
        speed_limit (float, optional): The maximum speed is allowed on the lane. Unit m/s. Defaults to None.
    """
    def __init__(
        self, id: str, left_side: LineString, right_side: LineString, subtype: str = None, 
        participants: list = None, speed_limit: float = None
    ):

        self.id = id
        self.left_side = left_side
        self.right_side = right_side
        self.subtype = subtype

        if self.subtype in DEFAULT_LANE:
            default_attrs = DEFAULT_LANE[self.subtype]
            self.participants = participants \
                if participants is not None else default_attrs["participants"]
            self.speed_limit = speed_limit \
                if speed_limit is not None else default_attrs["speed_limit"]
        else:
            self.participants = participants
            self.speed_limit = speed_limit

        # lists of related lanes
        self.predecessors = []
        self.successors = []
        self.left_neighbors = []
        self.right_neighbors = []

    def add_related_lane(self, id: str, relationship: Relationship):
        """Add a related lane's id to the corresponding list

        Args:
            id (str): the related lane's id
            relationship (Relationship): the relationship of the related lane to the self lane
        """
        if id == self.id:
            UserWarning("Lane %d cannot be a related lane to itself." % self.id)
            return
        if relationship == Relationship.PREDECESSOR:
            self.predecessors.append(id)
        elif relationship == Relationship.SUCCESSOR:
            self.successors.append(id)
        elif relationship == Relationship.LEFT_NEIGHBOR:
            self.left_neighbors.append(id)
        elif relationship == Relationship.RIGHT_NEIGHBOR:
            self.right_neighbors.append(id)

    def is_related(self, id: str) -> Relationship:
        """Check if a given lane is related to the self lane

        Args:
            id (str): The given lane's id.
        """
        if id in self.predecessors:
            return Relationship.PREDECESSOR
        elif id in self.successors:
            return Relationship.SUCCESSOR
        elif id in self.left_neighbors:
            return Relationship.LEFT_NEIGHBOR
        elif id in self.right_neighbors:
            return Relationship.RIGHT_NEIGHBOR
    
    def get_shape(self) -> list:
        """Return the shape of the lane
        """
        left_side_list = list(self.left_side.coords)
        right_side_list = list(self.right_side.coords)
        return left_side_list + right_side_list[::-1]

    def get_starts(self) -> list:
        """Return the start points of the lane
        """
        return [self.left_side.coords[0], self.right_side.coords[0]]
    
    def get_ends(self) -> list:
        """Return the end points of the lane
        """
        return [self.left_side.coords[-1], self.right_side.coords[-1]]

