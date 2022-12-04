import warnings

from tactics2d.map_base.lane import Lane
from tactics2d.map_base.area import Area
from tactics2d.map_base.roadline import RoadLine

class Map(object):
    def __init__(
        self, name: str = None, scenario_type: str = None, country: str = None
    ):

        self.name = name
        self.scenario_type = scenario_type
        self.country = country

        self.ids = set()
        self.lanes = dict()
        self.areas = dict()
        self.roadlines = dict()
        self.regulations = dict()
        self.boundary = None

    def add_lane(self, lane: Lane):
        if lane.id in self.ids:
            if lane.id in self.lanes:
                warnings.warn("Lane %s already exists! Replacing the lane with new data." % lane.id)
            elif lane.id in self.areas:
                raise ValueError()
            elif lane.id in self.roadlines:
                raise ValueError()
            elif lane.id in self.regulations:
                raise ValueError()
        self.lanes[lane.id] = lane

    def add_area(self, area: Area):
        if area.id in self.ids:
            if area.id in self.lanes:
                raise ValueError()
            elif area.id in self.areas:
                warnings.warn("Area %s already exists! Replacing the area with new data." % area.id)
            elif area.id in self.roadlines:
                raise ValueError()
            elif area.id in self.regulations:
                raise ValueError()
        self.areas[area.id] = area

    def add_roadline(self, roadline: RoadLine):
        if roadline.id in self.ids:
            if roadline.id in self.lanes:
                raise ValueError()
            elif roadline.id in self.areas:
                raise ValueError()
            elif roadline.id in self.roadlines:
                warnings.warn("Roadline %s already exists! Replacing the roadline with new data." % roadline.id)
            elif roadline.id in self.regulations:
                raise ValueError()
        self.roadlines[roadline.id] = roadline

    def get_boundary(self):
        return self.boundary