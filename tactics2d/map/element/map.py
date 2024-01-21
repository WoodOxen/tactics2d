import enum
import warnings

import numpy as np

from .area import Area
from .lane import Lane
from .node import Node
from .regulatory import Regulatory
from .roadline import RoadLine


class MapElement(enum.Enum):
    NODE = enum.auto()
    LANE = enum.auto()
    AREA = enum.auto()
    ROADLINE = enum.auto()
    REGULATORY = enum.auto()
    CUSTOM = enum.auto()


class Map:
    """This class implements a map to manage the road elements.

    Attributes:
        name (str, optional): The name of the map. Defaults to None.
        scenario_type (str, optional): The scenario type of the map. Defaults to None.
        country (str, optional): The country that the map is located in. Defaults to None.
        ids (set): The identifier of the elements in the map. All elements in the map should
            have a unique id. A conflict in ids will raise a KeyError.
        nodes (dict): The nodes in the map. This attribute needs to be set manually by trigger
            the "add_node" method.
        lanes (dict): The lanes in the map. This attribute needs to be set manually by trigger
            the "add_lane" method.
        areas (dict): The areas in the map. This attribute needs to be set manually by trigger
            the "add_area" method.
        roadlines (dict): The roadlines in the map. This attribute needs to be set manually by
            trigger the "add_roadline" method.
        regulations (dict): The regulations in the map. This attribute needs to be set manually
            by trigger the "add_regulatory" method.
        boundary (tuple, optional): The boundary of the map expressed in the form of
            (left, right, front, back). This attribute is automatically calculated when requested.
    """

    def __init__(self, name: str = None, scenario_type: str = None, country: str = None):
        """Initialize the attributes in the class.

        Args:
            name (str, optional): The name of the map. Defaults to None.
            scenario_type (str, optional): The scenario type of the map. Defaults to None.
            country (str, optional): The country that the map is located in. Defaults to None.
        """
        self.name = name
        self.scenario_type = scenario_type
        self.country = country

        self.ids = dict()
        self.nodes = dict()
        self.lanes = dict()
        self.areas = dict()
        self.roadlines = dict()
        self.regulations = dict()
        self.customs = dict()
        self._boundary = None

    @property
    def boundary(self):
        if self._boundary is None:
            x_min, x_max, y_min, y_max = (float("inf"), float("-inf"), float("inf"), float("-inf"))

            for node in self.nodes.values():
                x_min = min(x_min, node.x)
                x_max = max(x_max, node.x)
                y_min = min(y_min, node.y)
                y_max = max(y_max, node.y)

            for area in self.areas.values():
                area_coords = np.array(area.geometry.exterior.coords)
                x_min = min(x_min, np.min(area_coords[:, 0]))
                x_max = max(x_max, np.max(area_coords[:, 0]))
                y_min = min(y_min, np.min(area_coords[:, 1]))
                y_max = max(y_max, np.max(area_coords[:, 1]))

            self._boundary = (x_min - 10, x_max + 10, y_min - 10, y_max + 10)
        return self._boundary

    def add_node(self, node: Node):
        """Add a node to the map.

        Args:
            node (Node): The node to be added to the map.

        Raises:
            KeyError: If the id of the node is used by any other road element.
        """
        if node.id_ in self.ids:
            if node.id_ in self.nodes:
                warnings.warn(f"Node {node.id_} already exists! Replaced the node with new data.")
            else:
                raise KeyError(f"The id of Node {node.id_} is used by the other road element.")
        self.nodes[node.id_] = node
        self.ids[node.id_] = MapElement.NODE

    def add_roadline(self, roadline: RoadLine):
        """Add a roadline to the map.

        Args:
            roadline (RoadLine): The roadline to be added to the map.

        Raises:
            KeyError: If the id of the roadline is used by any other road element.
        """
        if roadline.id_ in self.ids:
            if roadline.id_ in self.roadlines:
                warnings.warn(
                    f"Roadline {roadline.id_} already exists! Replaced the roadline with new data."
                )
            else:
                raise KeyError(
                    f"The id of Roadline {roadline.id_} is used by the other road element."
                )

        self.roadlines[roadline.id_] = roadline
        self.ids[roadline.id_] = MapElement.ROADLINE

    def add_lane(self, lane: Lane):
        """Add a lane to the map.

        Args:
            lane (Lane): The lane to be added to the map.

        Raises:
            KeyError: If the id of the lane is used by any other road element.
        """
        if lane.id_ in self.ids:
            if lane.id_ in self.lanes:
                warnings.warn(f"Lane {lane.id_} already exists! Replacing the lane with new data.")
            else:
                raise KeyError(f"The id of Lane {lane.id_} is used by the other road element.")

        self.lanes[lane.id_] = lane
        self.ids[lane.id_] = MapElement.LANE

    def add_area(self, area: Area):
        """Add an area to the map.

        Args:
            area (Area): The area to be added to the map.

        Raises:
            KeyError: If the id of the area is used by any other road element.
        """
        if area.id_ in self.ids:
            if area.id_ in self.areas:
                warnings.warn(f"Area {area.id_} already exists! Replacing the area with new data.")
            else:
                raise KeyError(f"The id of Area {area.id_} is used by the other road element.")

        self.areas[area.id_] = area
        self.ids[area.id_] = MapElement.AREA

    def add_regulatory(self, regulatory: Regulatory):
        """Add a regulatory to the map.

        Args:
            regulatory (Regulatory): The regulatory to be added to the map.

        Raises:
            KeyError: If the id of the regulatory is used by any other road element.
        """
        if regulatory.id_ in self.ids:
            if regulatory.id_ in self.regulations:
                warnings.warn(
                    f"Regulatory {regulatory.id_} already exists! Replacing the regulatory with new data."
                )
            else:
                raise KeyError(
                    f"The id of Regulatory {regulatory.id_} is used by the other road element."
                )

        self.regulations[regulatory.id_] = regulatory
        self.ids[regulatory.id_] = MapElement.REGULATORY

    def get_by_id(self, id_: str):
        if not id_ in self.ids:
            warnings.warn(f"Cannot find element with id {id_}.")
            return None

        if self.ids[id_] == MapElement.NODE:
            return self.nodes[id_]
        elif self.ids[id_] == MapElement.LANE:
            return self.lanes[id_]
        elif self.ids[id_] == MapElement.AREA:
            return self.areas[id_]
        elif self.ids[id_] == MapElement.ROADLINE:
            return self.roadlines[id_]
        elif self.ids[id_] == MapElement.REGULATORY:
            return self.regulations[id_]
        elif self.ids[id_] == MapElement.CUSTOM:
            return self.customs[id_]

    def reset(self):
        """Reset the map by clearing all the road elements."""
        self.ids.clear()
        self.nodes.clear()
        self.lanes.clear()
        self.areas.clear()
        self.roadlines.clear()
        self.regulations.clear()
        self.customs.clear()
        self._boundary = None
