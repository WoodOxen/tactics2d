import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import pyglet
from pyglet import shapes

from tactics2d.envs.racing import CarRacing
from tactics2d.render.camera import Camera
from tactics2d.render.render_map import get_lane_batch


N_CHECKPOINT = (10, 20) # the number of turns is ranging in 10-20
TRACK_WIDTH = 15 # the width of the track is ranging in 15m
TRACK_RAD = 800 # the maximum curvature radius
CURVE_RAD = (10, 150) # the curvature radius is ranging in 10-150m
TILE_LENGTH = 10 # the length of each tile
STEP_LIMIT = 20000 # steps
TIME_STEP = 0.01 # state update time step: 0.01 s/step


if __name__ == "__main__":
    env = CarRacing(render_mode="human")
    env.reset()
    gui_camera = Camera()
    fpv_camera = Camera()
    fpv_camera.zoom = 2
    batch = pyglet.graphics.Batch()

    # background_layer = pyglet.graphics.Group(0)
    map_layer = pyglet.graphics.Group(0)
    object_layer = pyglet.graphics.Group(1)

    lane = get_lane_batch(env.map, batch, map_layer)
    bbox = list(env.agent.get_bbox().coords)
    start_line = list(env.start_line.coords)
    line = pyglet.shapes.Line(start_line[0][0], start_line[0][1], start_line[1][0], start_line[1][1], color = (0, 255, 0), batch=batch)
    vehicle = pyglet.shapes.Polygon(*bbox, color=(255, 0, 0), batch=batch)

    print(vehicle.position)

    @env.window.event
    def on_draw():
        env.window.clear()
        # fpv_camera.position = vehicle.position
        # with fpv_camera:
        batch.draw()
    
    pyglet.app.run()

    # plt.plot(*LineString(center_points).xy)
    # plt.plot(*LineString(left_points).xy)
    # plt.plot(*LineString(right_points).xy)
    # plt.plot(*start_line.xy, color="red")
    # plt.plot(*finish_line.xy, color="blue")
    # plt.plot(*start_loc.xy, marker='o')
    

    # plt.axis(map.get_boundary())
    
    # plt.gca().set_aspect("equal")

    # plt.show()
