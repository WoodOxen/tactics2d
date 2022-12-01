from shapely.geometry import Point

class State(object):
    """This class is used to record the state of an dynamic object.

    Attrs:
        t (float): the time stamp
        loc (Point): the location coordination of the dynamic object
        v (Point): the velocity vector of the dynamic object
        heading (float): the heading angle of the dynamic object
        a (Point, optional): the acceleration vector of the dynamic object
        steering (float, optional): the steering angle of the dynamic object
    """
    def __init__(
        self, t: float, loc: Point, v: Point, heading: float,
        a: Point = None, steering: float = None
    ):

        self.t = t
        self.loc = loc
        self.v = v
        self.heading = heading
        self.a = a
        self.steering = steering

    def get_location(self) -> tuple:
        """Return the location coordination in tuple format
        """
        return list(self.loc.coords)[0]

    def get_velocity(self) -> tuple:
        """Return the velocity vector in tuple format
        """
        return list(self.v.coords)[0]

    def get_speed(self) -> float:
        """Return the speed value sqrt(vx^2+vy^2)
        """
        return Point(0, 0).distance(self.v)

    def get_acceleration(self) -> tuple:
        """Return the acceleration vector in tuple format
        """
        if self.a is not None:
            return list(self.a.coords)[0]
        return 

    def get_acceleration_scalar(self) -> float:
        """Return the value of acceleration sqrt(ax^2+ay^2)
        """
        if self.a is not None:
            return Point(0, 0).distance(self.a)
        return
