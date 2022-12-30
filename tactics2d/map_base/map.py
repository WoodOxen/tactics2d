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

        self._ids = set()
        self._nodes = dict()
        self._lanes = dict()
        self._areas = dict()
        self._roadlines = dict()
        self._regulations = dict()
        self._customs = dict()
        self._boundary = None

    def add_node(self, node: Node):
        if node.id in self._ids:
            if node.id in self._nodes:
                warnings.warn("Node %s already exists! Replacing the node with new data." % node.id)
            else:
                raise ValueError()
        self._nodes[node.id] = node

    def add_roadline(self, roadline: RoadLine):
        if roadline.id in self._ids:
            if roadline.id in self._roadlines:
                warnings.warn("Roadline %s already exists! Replacing the roadline with new data." % roadline.id)
            else:
                raise ValueError()
        self._ids.add(roadline.id)
        self._roadlines[roadline.id] = roadline

    def add_lane(self, lane: Lane):
        if lane.id in self._ids:
            if lane.id in self._lanes:
                warnings.warn("Lane %s already exists! Replacing the lane with new data." % lane.id)
            else:
                raise ValueError()
        self._ids.add(lane.id)
        self._lanes[lane.id] = lane

    def add_area(self, area: Area):
        if area.id in self._ids:
            if area.id in self._areas:
                warnings.warn("Area %s already exists! Replacing the area with new data." % area.id)
            else:
                raise ValueError()
        self._ids.add(area.id)
        self._areas[area.id] = area

    def add_regulatory(self, regulatory: RegulatoryElement):
        if regulatory.id in self._ids:
            if regulatory.id in self._regulations:
                warnings.warn()
            else:
                raise ValueError()
        self._ids.add(regulatory.id)
        self._regulations[regulatory.id] = regulatory

    def set_boundary(self, boundary):
        self._boundary = boundary

    def get_boundary(self):
        return self._boundary

    def reset(self):
        self._ids.clear()
        self._nodes.clear()
        self._lanes.clear()
        self._areas.clear()
        self._roadlines.clear()
        self._regulations.clear()
        self._customs.clear()
        self._boundary = None