class State(object):
    def __init__(
        self, timestamp: float, 
        heading: float, x: float, y: float, 
        vx: float, vy: float, ax: float = None, ay: float = None,   
    ):
    
        self.timestamp = timestamp
        self.heading = heading
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
