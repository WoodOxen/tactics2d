import sys
sys.path.append("../")
sys.path.append(".")
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

if __name__ == "__main__":
    from shapely.geometry import LinearRing
    import time
    import matplotlib.pyplot as plt

    obs1 = LinearRing(((1,1), (1,-1), (3,-1), (4,1), ))
    obs2 = LinearRing(((-5,1), (-5,-4), (-8,-1), (-9,1), ))
    obs3 = LinearRing(((0,4), (2, 6.3), (0.5, 7), (-3.3, 6.8)))

    car_pos = (10,5,0.9)
    lidar_range = 10.0
    lidar_num = 100
    lidar = LidarSimlator(lidar_range, lidar_num)
    OBSLIST = [obs1,obs2,obs3,obs1]

    t = time.time()
    lidar_view = lidar.get_observation(car_pos, OBSLIST)
    print(time.time()-t)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_xlabel('x')
    for i in range(len(lidar_view)):
        # line=plt.Line2D((0,0),(math.cos(i*math.pi/180)*lidar_view[i],math.sin(i*math.pi/180)*lidar_view[i]))
        ax.add_line(plt.Line2D((0,math.cos(i*math.pi/lidar_num*2)*lidar_view[i]), (0,math.sin(i*math.pi/lidar_num*2)*lidar_view[i])))
    for obs in lidar._rotate_and_filter_obstacles(car_pos, OBSLIST):
        ax.add_patch(plt.Polygon(xy=list(obs.coords), color='r'))
    plt.show()
