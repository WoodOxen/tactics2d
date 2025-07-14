##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: idm.py
# @Description: This script implements the Intelligent Driver Model (IDM).
# @Author: Tactics2D Team
# @Version: 0.1.9


import numpy as np
from scipy.integrate import RK45


class IDM:
    """This class implements the Intelligent Driver Model (IDM) for simulating vehicle behavior in traffic scenarios.

    The IDM is a car-following model that describes how vehicles adjust their speed based on the distance to the vehicle in front, their own speed, and desired speed. The original model only considers the single-path traffic flow.

    !!! quote "Reference"
        Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic states in empirical observations and microscopic simulations." Physical review E 62.2 (2000): 1805.
    """
    def __init__(self, s0: float=2.0, T: float=1.6, a: float=0.73, b: float=1.67, delta:int=4, t_bound: float=0.05):
        """Initialize the IDM parameters.

        Args:
            s0 (float, optional): The minimum desired net distance. Defaults to 2.0 m.
            T (float, optional): The safe time headway. Defaults to 1.6 s.
            a (float, optional): The acceleration parameter. Defaults to 0.73 m/s$^2$.
            b (float, optional): The comfortable deceleration. Defaults to 1.67 m/s$^2$
            delta (int, optional): The acceleration exponent. Defaults to 4.
            t_bound (float, optional): The time boundary which the integration won’t continue beyond it. Defaults to 0.05 s.
        """
        self.s0 = s0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.t_bound = t_bound
    
    def get_acceleration(self, ego_speed: float, v0: float, delta_v: float, s: float) -> float:
        """

        Args:
            ego_speed (float): The speed of ego vehicle. The unit is m/s.
            v0 (float): The desired speed for the ego vehicle. The unit is m/s.
            delta_v (float): The speed difference between ego vehicle and the leading vehicle.
            s (float): The distance from the front of ego vehicle to the front of the leading vehicle. The unit is m.

        Returns:
            float:
        """
        s_star = self.s0 + ego_speed*self.T +ego_speed*delta_v/ (2.0 * np.sqrt(self.a * self.b))
        acceleration = self.a * (1.0 - (ego_speed/v0) ** self.delta - (s_star / s) ** 2)
        return acceleration

    def get_speed(self, ego_speed: float, leading_vehicle_speed: float, v0: float, l: float, s: float) -> float:
        """Estimate ego vehicle’s future speed after a short time using the IDM acceleration model.

        Args:
            ego_speed (float): The speed of ego vehicle. The unit is m/s.
            leading_vehicle_speed (float): The speed of the leading vehicle or obstacle.
            v0 (float): The desired speed for the ego vehicle. The unit is m/s
            l (float): The length of the leading vehicle or obstacle. The unit is m.
            s (float): The distance from the front of ego vehicle to the front of the leading vehicle. The unit is m.

        Returns:
            float:
        """
        def idm_equation(t, x):
            ego_position, ego_speed = x
            delta_v = ego_speed - leading_vehicle_speed
            s_star = self.s0 + ego_speed*self.T +ego_speed*delta_v/ (2.0 * np.sqrt(self.a * self.b))
            # The maximum is needed to avoid numerical instability
            net_distance = max(0.1, s + t * leading_vehicle_speed - ego_position - l)
            dvdt = self.a * (1.0 - (ego_speed / v0) ** self.delta - (s_star / net_distance) ** 2)

            return [ego_speed, dvdt]
        
        # Set the initial conditions
        y0 = [0.0, ego_speed]
        # Integrate the differential equations using RK45
        rk45 = RK45(fun=idm_equation, t0=0.0, y0=y0, t_bound=self.t_bound)
        while rk45.status == "running":
            rk45.step()

        target_speed = rk45.y[1]

        return np.clip(target_speed, 0, np.inf)