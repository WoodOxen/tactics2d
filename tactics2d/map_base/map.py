import warnings

from tactics2d.map_base.node import Node
from tactics2d.map_base.roadline import RoadLine
from tactics2d.map_base.lane import Lane
from tactics2d.map_base.area import Area
from tactics2d.map_base.regulatory import RegulatoryElement


class Map(object):
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
        self._boundary = None

    def add_node(self, node: Node):
        if node.id_ in self.ids:
            if node.id_ in self.nodes:
                warnings.warn("Node %s already exists! Replacing the node with new data." % node.id_)
            else:
                raise ValueError()
        self.nodes[node.id_] = node

    def add_roadline(self, roadline: RoadLine):
        if roadline.id_ in self.ids:
            if roadline.id_ in self.roadlines:
                warnings.warn("Roadline %s already exists! Replacing the roadline with new data." % roadline.id_)
            else:
                raise ValueError()
        self.ids.add(roadline.id_)
        self.roadlines[roadline.id_] = roadline

    def add_lane(self, lane: Lane):
        if lane.id_ in self.ids:
            if lane.id_ in self.lanes:
                warnings.warn("Lane %s already exists! Replacing the lane with new data." % lane.id_)
            else:
                raise ValueError()
        self.ids.add(lane.id_)
        self.lanes[lane.id_] = lane

    def add_area(self, area: Area):
        if area.id_ in self.ids:
            if area.id_ in self.areas:
                warnings.warn("Area %s already exists! Replacing the area with new data." % area.id_)
            else:
                raise ValueError()
        self.ids.add(area.id_)
        self.areas[area.id_] = area

    def add_regulatory(self, regulatory: RegulatoryElement):
        if regulatory.id_ in self.ids:
            if regulatory.id_ in self.regulations:
                warnings.warn()
            else:
                raise ValueError()
        self.ids.add(regulatory.id_)
        self.regulations[regulatory.id_] = regulatory

    @property
    def boundary(self):
        return self._boundary
    
    @boundary.setter
    def boundary(self, boundary):
        self._boundary = boundary

    def reset(self):
        self.ids.clear()
        self.nodes.clear()
        self.lanes.clear()
        self.areas.clear()
        self.roadlines.clear()
        self.regulations.clear()
        self.customs.clear()
        self.boundary = None