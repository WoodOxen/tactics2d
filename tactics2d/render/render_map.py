import numpy as np
import pyglet
from pyglet import shapes

from tactics2d.map_base.map import Map
from tactics2d.render.color_default import LANE_COLOR, AREA_COLOR


def get_lane_batch(map: Map, batch, group, color_dict: dict = LANE_COLOR):
    # lane_batch = pyglet.graphics.Batch()
    polygons = []
    for lane in map.lanes.values():
        lane_shape = np.array(lane.get_shape()).astype(np.int32).tolist()
        if lane.subtype in color_dict:
            color = color_dict[lane.subtype]
        polygons.append(shapes.Polygon(*lane_shape, color=color, batch=batch))
    return polygons


def render_map(map: Map):
    batches = []

    return
