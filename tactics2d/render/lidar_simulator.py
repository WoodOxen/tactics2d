import math

import numpy as np
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform

ORIGIN = Point((0,0))

class LidarSimlator():
    def __init__(self,
        lidar_range:float = 10.0,
        lidar_num:int = 120
    ) -> None:
        '''
        Args:
            lidar_range(float): the max distance that the obstacle can be dietected.
            lidar_num(int): the beam num of the lidar simulation.
        '''
        self.lidar_range = lidar_range
        self.lidar_num = lidar_num
        self.lidar_lines = []
        for a in range(lidar_num):
            self.lidar_lines.append(LineString(((0,0), (math.cos(a*math.pi/lidar_num*2)*lidar_range,\
                 math.sin(a*math.pi/lidar_num*2)*lidar_range))))

    def get_observation(self, ego_pos:list, obstacles:list):
        '''
        Get the lidar observation from the vehicle's view.

        Args:
            ego_state: (x, y, yaw)
            obstacles: the list of obstacles in map

        Return:
            lidar_obs(np.array): the lidar data in sequence of angle, with the length of lidar_num.
        '''

        rotated_obstacles = self._rotate_and_filter_obstacles(ego_pos, obstacles)
        lidar_obs = []
        for l in self.lidar_lines:
            min_distance = self.lidar_range
            for obs in rotated_obstacles:
                distance = l.intersection(obs).distance(ORIGIN)
                if distance>0.1 and distance<min_distance:
                    min_distance = distance
            lidar_obs.append(min_distance)
        return np.array(lidar_obs)

    def _rotate_and_filter_obstacles(self, ego_pos:tuple, obstacles:list):
        '''
        Rotate the obstacles around the vehicle and remove the obstalces which is out of lidar range.
        '''
        x, y, theta = ego_pos
        a = math.cos(theta)
        b = math.sin(theta)
        x_off = -x*a - y*b
        y_off = x*b - y*a
        affine_mat = [a, b, -b, a, x_off, y_off]
        rotated_obstacles = []
        for obs in obstacles:
            rotated_obs = affine_transform(obs, affine_mat)
            if rotated_obs.distance(ORIGIN) < self.lidar_range:
                rotated_obstacles.append(rotated_obs)
        return rotated_obstacles
