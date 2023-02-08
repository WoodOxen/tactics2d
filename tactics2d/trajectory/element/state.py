import numpy as np


class State(object):
    def __init__(self, frame: int, x: float = 0, y: float = 0, heading: float = 0):
            self.frame = frame
            self.x = x
            self.y = y
            self.heading = heading
            self.vx = None
            self.vy = None
            self.v_norm = None
            self.ax = None
            self.ay = None
            self.a_norm = None

    @property
    def location(self):
          return (self.x, self.y)
    
    @property
    def velocity(self):
        if None in [self.vx, self.vy]:
            self.get_velocity()
            
        return (self.vx, self.vy)

    @property
    def speed(self):
        if self.v_norm is not None:
            return self.v_norm
        if None not in [self.vx, self.vy]:
            self.v_norm = np.linalg.norm([self.vx, self.vy])
            return self.v_norm
        return None
    
    @property
    def accel(self):
        if self.a_norm is not None:
            return self.a_norm
        if None not in [self.ax, self.ay]:
            self.a_norm = np.linalg.norm([self.ax, self.ay])
            return self.a_norm
        return None

    def set_velocity(self, vx: float, vy: float):
        self.vx = vx
        self.vy = vy

    def set_speed(self, speed: float):
        self.v_norm = speed

    def set_accel(self, accel: float):
        self.a_norm = accel