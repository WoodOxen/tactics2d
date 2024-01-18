from typing import Union
from enum import Enum
import logging

from shapely.geometry import LineString, LinearRing

from .defaults import LEGAL_SPEED_UNIT


class LaneRelationship(Enum):
    PREDECESSOR = 1
    SUCCESSOR = 2
    LEFT_NEIGHBOR = 3
    RIGHT_NEIGHBOR = 4


class Lane:
    """This class implements the lenelet2-style map element *Lane*.

    Detailed definition of lanelet2-style lane:
        [LaneletPrimitives](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md)

    Attributes:
        id_ (str): The unique identifier of the lane.
        left_side (LineString): The left side of the lane.
        right_side (LineString): The right side of the lane.
        line_ids (set, optional): The ids of the roadline components. Defaults to None.
        type_ (str): The type of the lane. The default value is `"lanelet"`.
        subtype (str, optional): The subtype of the lane. Defaults to None.
        location (str, optional): The location of the lane (urban, nonurban, etc.). Defaults to None.
        inferred_participants (list, optional): The allowing type of traffic participants that can pass the lane. Defaults to None.
        speed_limit (float, optional): The speed limit in this lane. Defaults to None.
        speed_limit_unit (str, optional): The unit of speed limit in this area. The valid units
            are `"km/h"`, `"mi/h"`, and `"m/s"`. Defaults to `"km/h"`.
        speed_limit_mandatory (bool, optional): Whether the speed limit is mandatory or
            not. Defaults to True.
        custom_tags (dict, optional): The custom tags of the lane. Defaults to None.
        predecessors (set): The ids of the available lanes before entering the current lane.
        successors (set): The ids of the available lanes after exiting the current lane.
        left_neighbors (set): The ids of the available lanes on the left side of the current
            lane.
        right_neighbors (set): The ids of the available lanes on the right side of the current
            lane.
        start (list, read-only): The start points of the lane.
        end (list, read-only): The end points of the lane.
        shape (list, read-only): The shape of the lane.
    """

    speed_limit_units = ["km/h", "mi/h", "m/s"]

    def __init__(
        self,
        id_: str,
        left_side: LineString,
        right_side: LineString,
        line_ids: set = None,
        type_: str = "lanelet",
        subtype: str = None,
        color: tuple = None,
        location: str = None,
        inferred_participants: list = None,
        speed_limit: float = None,
        speed_limit_unit: str = "km/h",
        speed_limit_mandatory: bool = True,
        custom_tags: dict = None,
    ):

        self.id_ = id_
        self.left_side = left_side
        self.right_side = right_side
        self.line_ids = line_ids
        self.type_ = type_
        self.subtype = subtype
        self.color = color
        self.location = location
        self.inferred_participants = inferred_participants
        self.speed_limit = speed_limit
        self.speed_limit_unit = speed_limit_unit
        self.speed_limit_mandatory = speed_limit_mandatory
        self.custom_tags = custom_tags

        if self.speed_limit_unit not in LEGAL_SPEED_UNIT:
            logging.warning(
                "Invalid speed limit unit %s. The legal units types are %s"
                % (self.speed_limit_unit, ", ".join(LEGAL_SPEED_UNIT))
            )

        if not self.left_side is None and not self.right_side is None:
            self.geometry = LinearRing(
                list(self.left_side.coords) + list(reversed(list(self.right_side.coords)))
            )
        else:
            self.geometry = None

        self.predecessors = set()
        self.successors = set()
        self.left_neighbors = set()
        self.right_neighbors = set()

    @property
    def starts(self) -> list:
        return [self.left_side.coords[0], self.right_side.coords[0]]

    @property
    def ends(self) -> list:
        return [self.left_side.coords[-1], self.right_side.coords[-1]]

    @property
    def shape(self) -> list:
        return list(self.polygon.coords)

    def is_related(self, id_: str) -> LaneRelationship:
        """Check if a given lane is related to the lane

        Args:
            id_ (str): The given lane's id.
        """
        if id_ in self.predecessors:
            return LaneRelationship.PREDECESSOR
        elif id_ in self.successors:
            return LaneRelationship.SUCCESSOR
        elif id_ in self.left_neighbors:
            return LaneRelationship.LEFT_NEIGHBOR
        elif id_ in self.right_neighbors:
            return LaneRelationship.RIGHT_NEIGHBOR

    def add_related_lane(self, id_: Union[str, list], relationship: LaneRelationship):
        """Add a related lane's id to the corresponding list

        Args:
            id_ (str): The related lane's id
            relationship (LaneRelationship): The relationship of the lanes
        """
        if id_ is None:
            return

        if isinstance(id_, str):
            if id_ == self.id_:
                logging.warning(f"Lane {self.id_} cannot be a related lane to itself.")
                return
            if relationship == LaneRelationship.PREDECESSOR:
                self.predecessors.add(id_)
            elif relationship == LaneRelationship.SUCCESSOR:
                self.successors.add(id_)
            elif relationship == LaneRelationship.LEFT_NEIGHBOR:
                self.left_neighbors.add(id_)
            elif relationship == LaneRelationship.RIGHT_NEIGHBOR:
                self.right_neighbors.add(id_)

        elif isinstance(id_, list):
            if self.id_ in id_:
                id_ = [i for i in id_ if i != self.id_]
                logging.warning(f"Lane {self.id_} cannot be a related lane to itself.")

            if relationship == LaneRelationship.PREDECESSOR:
                self.predecessors.update(id_)
            elif relationship == LaneRelationship.SUCCESSOR:
                self.successors.update(id_)
            elif relationship == LaneRelationship.LEFT_NEIGHBOR:
                self.left_neighbors.update(id_)
            elif relationship == LaneRelationship.RIGHT_NEIGHBOR:
                self.right_neighbors.update(id_)
