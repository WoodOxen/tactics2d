
from shapely.geometry import Point,LinearRing
from shapely.affinity import affine_transform
import numpy as np

class Position:
    def __init__(self, raw_state: list):
        self.loc: Point = Point(raw_state[:2]) #(x,y)
        self.heading: float = raw_state[2]

    def create_box(self, VehicleBox:LinearRing) -> LinearRing:
        cos_theta = np.cos(self.heading)
        sin_theta = np.sin(self.heading)
        mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
        return affine_transform(VehicleBox, mat)

    def get_pos(self,):
        return (self.loc.x, self.loc.y, self.heading)