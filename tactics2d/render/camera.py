import numpy as np

import pyglet
from pyglet import gl


class Camera(object):
    def __init__(self, min_zoom=1, max_zoom=100):
        self.offset_x = 0
        self.offset_y = 0
        self.angle = 0
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self._zoom = np.clip(1, self.min_zoom, self.max_zoom)
    
    @property
    def zoom(self):
        return self._zoom
    
    @zoom.setter
    def zoom(self, zoom):
        self._zoom = np.clip(1, self.min_zoom, self.max_zoom)

    @property
    def position(self):
        return self.offset_x, self.offset_y
    
    @position.setter
    def position(self, position):
        self.offset_x, self.offset_y = position

    def begin(self):
        
        gl.glScalef(self._zoom, self._zoom, 1)


    def end(self):
        gl.glScalef(1 / self._zoom, 1 / self._zoom, 1)
        # gl.glTranslatef(self.offset_x * self._zoom, self.offset_y * self._zoom, 0)
        return
    
    def __enter__(self):
        self.begin()
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.end()
