import warnings

import numpy as np

from .area import Area
from .lane import Lane
from .node import Node
from .regulatory import RegulatoryElement
from .roadline import RoadLine


class MapKeyError(KeyError):
    pass


class Map:
    """This class implements a map to manage the road elements.

    Attributes:
        name (str, optional): _description_. Defaults to None.
        scenario_type (str, optional): _description_. Defaults to None.
        country (str, optional): _description_. Defaults to None.
        ids (set): The identical index of the elements in the map. A conflict in ids will raise a
            MapKeyError.
        nodes (dict): The nodes in the map.
        lanes (dict): The lanes in the map.
        areas (dict): The areas in the map.
        roadlines (dict): The roadlines in the map.
        regulations (dict): The regulations in the map.
        boundary (tuple, optional): The boundary of the map expressed in the form of
            (left, right, front, back). This attribute is automatically calculated when requested.
    """

    def __init__(self, name: str = None, scenario_type: str = None, country: str = None):
        self.name = name
        self.scenario_type = scenario_type
        self.country = country

        self.ids = set()
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
            x_min, x_max, y_min, y_max = (
                float("inf"),
                float("-inf"),
                float("inf"),
                float("-inf"),
            )

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
        if node.id_ in self.ids:
            if node.id_ in self.nodes:
                warnings.warn(f"Node {node.id_} already exists! Replaced the node with new data.")
            else:
                raise MapKeyError(f"The id of Node {node.id_} is used by the other road element.")
        self.nodes[node.id_] = node

    def add_roadline(self, roadline: RoadLine):
        if roadline.id_ in self.ids:
            if roadline.id_ in self.roadlines:
                warnings.warn(
                    f"Roadline {roadline.id_} already exists! Replaced the roadline with new data."
                )
            else:
                raise MapKeyError(
                    f"The id of Roadline {roadline.id_} is used by the other road element."
                )
        self.ids.add(roadline.id_)
        self.roadlines[roadline.id_] = roadline

    def add_lane(self, lane: Lane):
        if lane.id_ in self.ids:
            if lane.id_ in self.lanes:
                warnings.warn(f"Lane {lane.id_} already exists! Replacing the lane with new data.")
            else:
                raise MapKeyError(f"The id of Lane {lane.id_} is used by the other road element.")
        self.ids.add(lane.id_)
        self.lanes[lane.id_] = lane

    def add_area(self, area: Area):
        if area.id_ in self.ids:
            if area.id_ in self.areas:
                warnings.warn(f"Area {area.id_} already exists! Replacing the area with new data.")
            else:
                raise MapKeyError(f"The id of Area {area.id_} is used by the other road element.")
        self.ids.add(area.id_)
        self.areas[area.id_] = area

    def add_regulatory(self, regulatory: RegulatoryElement):
        if regulatory.id_ in self.ids:
            if regulatory.id_ in self.regulations:
                warnings.warn(
                    f"Regulatory {regulatory.id_} already exists! Replacing the regulatory with new data."
                )
            else:
                raise MapKeyError(
                    f"The id of Regulatory {regulatory.id_} is used by the other road element."
                )
        self.ids.add(regulatory.id_)
        self.regulations[regulatory.id_] = regulatory

    def reset(self):
        self.ids.clear()
        self.nodes.clear()
        self.lanes.clear()
        self.areas.clear()
        self.roadlines.clear()
        self.regulations.clear()
        self.customs.clear()
        self._boundary = None
