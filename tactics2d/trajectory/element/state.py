import numpy as np


class State(object):
    """_summary_

    Attributes:
        frame (int): _description_
        x (float, optional): _description_. Defaults to 0.
        y (float, optional): _description_. Defaults to 0.
        heading (float, optional): The heading direction of an object. The heading information is parsed in an 2D Cardinal coordinate system counterclockwise. The default unit is radian. Defaults to 0.
    """
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

    def set_accel(self, ax: float, ay: float):
        self.ax = ax
        self.ay = ay
        self.a_norm = np.linalg.norm([self.ax, self.ay])