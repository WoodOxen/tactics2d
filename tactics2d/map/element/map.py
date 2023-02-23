import warnings

from .area import Area
from .lane import Lane
from .node import Node
from .regulatory import RegulatoryElement
from .roadline import RoadLine


class  MapKeyError(KeyError):
    pass


class Map(object):
    """## tactics2d.map.element.Map

    Attributes:
        name (str, optional): _description_. Defaults to None.
        scenario_type (str, optional): _description_. Defaults to None.
        country (str, optional): _description_. Defaults to None.
    """
    def __init__(
        self, name: str = None, scenario_type: str = None, country: str = None
    ):

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
        self.boundary = None

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
                warnings.warn(f"Roadline {roadline.id_} already exists! Replaced the roadline with new data.")
            else:
                raise MapKeyError(f"The id of Roadline {roadline.id_} is used by the other road element.")
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
                warnings.warn()
            else:
                raise MapKeyError()
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
        self.boundary = None