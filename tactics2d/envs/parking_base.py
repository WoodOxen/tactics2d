'The map_base for random parking env generator'


import sys
sys.path.append("../")
from math import pi, cos, sin
from typing import List

import numpy as np
from numpy.random import randn, random
from shapely.geometry import LinearRing
from shapely.affinity import affine_transform

# from env.vehicle import State,LENGTH,WIDTH
from tactics2d.object_base.position import Position
from tactics2d.map_base.map import Map
# from tactics2d.map_base.area import Area

WHEEL_BASE = 2.8  # wheelbase
FRONT_HANG = 0.96  # front hang length
REAR_HANG = 0.929  # rear hang length
LENGTH = WHEEL_BASE+FRONT_HANG+REAR_HANG
WIDTH = 1.942  # width
# TODO define the vehicle center
VEHICLE_BOX = LinearRing([
    (-REAR_HANG, -WIDTH/2),
    (FRONT_HANG + WHEEL_BASE, -WIDTH/2),
    (FRONT_HANG + WHEEL_BASE,  WIDTH/2),
    (-REAR_HANG,  WIDTH/2)])

MIN_DIST_TO_OBST = 0.1
MIN_PARA_PARK_LOT_LEN = LENGTH*1.25
MIN_BAY_PARK_LOT_WIDTH = WIDTH + 1.2
# the distance that the obstacles out of driving area is from dest
BAY_PARK_WALL_DIST = 7.0
PARA_PARK_WALL_DIST = 4.5

DEBUG = True
if DEBUG:
    import matplotlib.pyplot as plt

def create_vehicle_box(x,y,heading,vehicle_box):
    cos_theta = np.cos(heading)
    sin_theta = np.sin(heading)
    mat = [cos_theta, -sin_theta, sin_theta, cos_theta, x, y]
    return affine_transform(vehicle_box, mat)

def get_box_coords(x,y,heading,vehicle_box):
    vehicle_box = create_vehicle_box(x,y,heading,vehicle_box)
    return list(vehicle_box.coords)[:-1]

def random_gaussian_num(mean, std, clip_low, clip_high):
    rand_num = randn()*std + mean
    return np.clip(rand_num, clip_low, clip_high)

def random_uniform_num(clip_low, clip_high):
    rand_num = random()*(clip_high - clip_low) + clip_low
    return rand_num

def get_rand_pos(origin_x, origin_y, angle_min, angle_max, radius_min, radius_max):
    angle_mean = (angle_max+angle_min)/2
    angle_std = (angle_max-angle_min)/4
    angle_rand = random_gaussian_num(angle_mean, angle_std, angle_min, angle_max)
    radius_rand = random_gaussian_num((radius_min+radius_max)/2, (radius_max-radius_min)/4, radius_min, radius_max)
    return origin_x+cos(angle_rand)*radius_rand, origin_y+sin(angle_rand)*radius_rand

def gene_bay_park(vehicle_box):
    '''
    Generate the parameters that a bay parking case need.
    Returns
    ----------
        `start` (list): [x, y, yaw]
        `dest` (list): [x, y, yaw]
        `obstacles` (list): [ obstacle (`LinearRing`) , ...]
    '''
    origin = (0., 0.)
    bay_half_len = 15.
    generate_success = True
    # generate obstacle on back
    obstacle_back = LinearRing((
        (origin[0]+bay_half_len, origin[1]),
        (origin[0]+bay_half_len, origin[1]-1),
        (origin[0]-bay_half_len, origin[1]-1),
        (origin[0]-bay_half_len, origin[1])))
    if DEBUG:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlim(-20,20)
        ax.set_ylim(-20,20)

    # generate dest
    dest_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
    rb, _, _, lb  = get_box_coords(origin[0], origin[1], dest_yaw, vehicle_box)
    min_dest_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
    dest_x = origin[0]
    dest_y = random_gaussian_num(min_dest_y+0.4, 0.2, min_dest_y, min_dest_y+0.8)
    car_rb, car_rf, car_lf, car_lb  = get_box_coords(dest_x, dest_y, dest_yaw, vehicle_box)
    dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))
    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list([car_rb, car_rf, car_lf, car_lb]), color='r'))

    # generate obstacle on left
    # the obstacle can be another vehicle or just a simple obstacle
    if random()<0.5: # generate simple obstacle
        max_dist_to_obst = 1.0
        min_dist_to_obst = 0.4 + MIN_DIST_TO_OBST
        left_obst_rf = get_rand_pos(*car_lf, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        left_obst_rb = get_rand_pos(*car_lb, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_left = LinearRing((
            left_obst_rf,
            left_obst_rb,
            (origin[0]-bay_half_len, origin[1]),
            (origin[0]-bay_half_len, left_obst_rf[1])))
    else: # generate another vehicle as obstacle on left
        max_dist_to_obst = 1.0
        min_dist_to_obst = 0.4 + MIN_DIST_TO_OBST
        left_car_x = origin[0] - (WIDTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        left_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
        rb, _, _, lb  = get_box_coords(left_car_x, origin[1], left_car_yaw, vehicle_box)
        min_left_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        left_car_y = random_gaussian_num(min_left_car_y+0.4, 0.2, min_left_car_y, min_left_car_y+0.8)
        obstacle_left = create_vehicle_box( left_car_x, left_car_y, left_car_yaw, vehicle_box)

    # generate obstacle on right
    dist_dest_to_left_obst = dest_box.distance(obstacle_left)
    min_dist_to_obst = max(MIN_BAY_PARK_LOT_WIDTH-WIDTH-dist_dest_to_left_obst, 0)+MIN_DIST_TO_OBST
    max_dist_to_obst = 1.0
    if random()<0.5: # generate simple obstacle
        right_obst_lf = get_rand_pos(*car_rf, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        right_obst_lb = get_rand_pos(*car_rb, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_right = LinearRing((
            (origin[0]+bay_half_len, right_obst_lf[1]),
            (origin[0]+bay_half_len, origin[1]),
            right_obst_lb,
            right_obst_lf))
    else: # generate another vehicle as obstacle on right
        right_car_x = origin[0] + (WIDTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        right_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
        rb, _, _, lb  = get_box_coords(right_car_x, origin[1], right_car_yaw, vehicle_box)
        min_right_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        right_car_y = random_gaussian_num(min_right_car_y+0.4, 0.2, min_right_car_y, min_right_car_y+0.8)
        obstacle_right = create_vehicle_box(right_car_x, right_car_y, right_car_yaw, vehicle_box)
    # print(dist_dest_to_left_obst, dest_box.distance(obstacle_right), dest_box.distance(obstacle_right)+dist_dest_to_left_obst)
    dist_dest_to_right_obst = dest_box.distance(obstacle_right)
    if dist_dest_to_right_obst+dist_dest_to_left_obst<MIN_BAY_PARK_LOT_WIDTH-WIDTH or \
        dist_dest_to_left_obst<MIN_DIST_TO_OBST or \
        dist_dest_to_right_obst<MIN_DIST_TO_OBST:
        # print(dist_dest_to_left_obst, dest_box.distance(obstacle_right), dest_box.distance(obstacle_right)+dist_dest_to_left_obst)
        generate_success = False
    # check collision
    obstacles = [obstacle_back, obstacle_left, obstacle_right]
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False

    # generate obstacles out of start range
    max_obstacle_y = max([np.max(np.array(obs.coords)[:,1]) for obs in obstacles])+MIN_DIST_TO_OBST
    other_obstcales = []
    if random()<0.2: # in this case only a wall will be generate
        other_obstcales = [LinearRing((
        (origin[0]-bay_half_len, BAY_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST),
        (origin[0]+bay_half_len, BAY_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST),
        (origin[0]+bay_half_len, BAY_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1),
        (origin[0]-bay_half_len, BAY_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1)))]
    else:
        other_obstacle_range = LinearRing((
        (origin[0]-bay_half_len, BAY_PARK_WALL_DIST+max_obstacle_y),
        (origin[0]+bay_half_len, BAY_PARK_WALL_DIST+max_obstacle_y),
        (origin[0]+bay_half_len, BAY_PARK_WALL_DIST+max_obstacle_y+8),
        (origin[0]-bay_half_len, BAY_PARK_WALL_DIST+max_obstacle_y+8)))
        valid_obst_x_range = (origin[0]-bay_half_len+2, origin[0]+bay_half_len-2)
        valid_obst_y_range = (BAY_PARK_WALL_DIST+max_obstacle_y+2, BAY_PARK_WALL_DIST+max_obstacle_y+6)
        for _ in range(3):
            obs_x = random_uniform_num(*valid_obst_x_range)
            obs_y = random_uniform_num(*valid_obst_y_range)
            obs_yaw = random()*pi*2
            obs_coords = np.array(get_box_coords(obs_x, obs_y, obs_yaw, vehicle_box))
            obs = LinearRing(obs_coords+0.5*random(obs_coords.shape))
            if obs.intersects(other_obstacle_range):
                continue
            other_obstcales.append(obs)

    # merge two kind of obstacles
    obstacles.extend(other_obstcales)
    if DEBUG:
        for obs in obstacles:
            ax.add_patch(plt.Polygon(xy=list(obs.coords), color='b'))

    # generate start position
    start_box_valid = False
    valid_start_x_range = (origin[0]-bay_half_len/2, origin[0]+bay_half_len/2)
    valid_start_y_range = (max_obstacle_y+1, BAY_PARK_WALL_DIST+max_obstacle_y-1)
    while not start_box_valid:
        start_box_valid = True
        start_x = random_uniform_num(*valid_start_x_range)
        start_y = random_uniform_num(*valid_start_y_range)
        start_yaw = random_gaussian_num(0, pi/6, -pi/2, pi/2)
        start_yaw = start_yaw+pi if random()<0.5 else start_yaw
        start_box = create_vehicle_box(start_x, start_y, start_yaw, vehicle_box)
        # check collision
        for obst in obstacles:
            if obst.intersects(start_box):
                if DEBUG:
                    print('start box intersect with obstacles, will retry to generate.')
                start_box_valid = False
        # check overlap with dest box
        if dest_box.intersects(start_box):
            if DEBUG:
                print('start box intersect with dest box, will retry to generate.')
            start_box_valid = False

    # randomly drop the obstacles
    for obs in obstacles:
        if random()<0.1:
            obstacles.remove(obs)
    if DEBUG:
        ax.add_patch(plt.Polygon(xy=get_box_coords(start_x, start_y, start_yaw, vehicle_box), color='g'))
        print(generate_success)
        plt.show()
    # return if generate successfully
    if generate_success:
        return [start_x, start_y, start_yaw], [dest_x, dest_y, dest_yaw], obstacles
    else:
        # print(1)
        return gene_bay_park(vehicle_box)

def gene_parallel_park(vehicle_box):
    '''
    Generate the parameters that a parallel parking case need.
    
    Returns
    ----------
        `start` (list): [x, y, yaw]
        `dest` (list): [x, y, yaw]
        `obstacles` (list): [ obstacle (`LinearRing`) , ...]
    '''
    origin = (0., 0.)
    bay_half_len = 18.
    generate_success = True
    # generate obstacle on back
    obstacle_back = LinearRing((
        (origin[0]+bay_half_len, origin[1]),
        (origin[0]+bay_half_len, origin[1]-1),
        (origin[0]-bay_half_len, origin[1]-1),
        (origin[0]-bay_half_len, origin[1])))
    
    if DEBUG:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlim(-20,20)
        ax.set_ylim(-20,20)

    # generate dest
    dest_yaw = random_gaussian_num(0, pi/36, -pi/12, pi/12)
    rb, rf, _, _  = get_box_coords(origin[0], origin[1], dest_yaw, vehicle_box)
    min_dest_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
    dest_x = origin[0]
    dest_y = random_gaussian_num(min_dest_y+0.4, 0.2, min_dest_y, min_dest_y+0.8)
    car_rb, car_rf, car_lf, car_lb  = get_box_coords(dest_x, dest_y, dest_yaw, vehicle_box)
    dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))

    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list([car_rb, car_rf, car_lf, car_lb]), color='r'))

    # generate obstacle on left
    # the obstacle can be another vehicle or just a simple obstacle
    if random()<0.5: # generate simple obstacle
        max_dist_to_obst = 1.0
        min_dist_to_obst = 0.4 + MIN_DIST_TO_OBST
        left_obst_rf = get_rand_pos(*car_lb, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        left_obst_rb = get_rand_pos(*car_rb, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_left = LinearRing((
            left_obst_rf,
            left_obst_rb,
            (origin[0]-bay_half_len, origin[1]),
            (origin[0]-bay_half_len, left_obst_rf[1])))
    else: # generate another vehicle as obstacle on left
        max_dist_to_obst = 1.0
        min_dist_to_obst = 0.4 + MIN_DIST_TO_OBST
        left_car_x = origin[0] - (LENGTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        left_car_yaw = random_gaussian_num(0, pi/36, -pi/12, pi/12)
        rb, rf, _, _  = get_box_coords(left_car_x, origin[1], left_car_yaw, vehicle_box)
        min_left_car_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
        left_car_y = random_gaussian_num(min_left_car_y+0.4, 0.2, min_left_car_y, min_left_car_y+0.8)
        obstacle_left = create_vehicle_box(left_car_x, left_car_y, left_car_yaw, vehicle_box)

    # generate obstacle on right
    dist_dest_to_left_obst = dest_box.distance(obstacle_left)
    min_dist_to_obst = max(MIN_PARA_PARK_LOT_LEN-LENGTH-dist_dest_to_left_obst, 0)+MIN_DIST_TO_OBST
    max_dist_to_obst = 1.0
    if random()<0.5: # generate simple obstacle
        right_obst_lf = get_rand_pos(*car_lf, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        right_obst_lb = get_rand_pos(*car_rf, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_right = LinearRing((
            (origin[0]+bay_half_len, right_obst_lf[1]),
            (origin[0]+bay_half_len, origin[1]),
            right_obst_lb,
            right_obst_lf))
    else: # generate another vehicle as obstacle on right
        right_car_x = origin[0] + (LENGTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        right_car_yaw = random_gaussian_num(0, pi/36, -pi/12, pi/12)
        rb, rf, _, _  = get_box_coords(right_car_x, origin[1], right_car_yaw, vehicle_box)
        min_right_car_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
        right_car_y = random_gaussian_num(min_right_car_y+0.4, 0.2, min_right_car_y, min_right_car_y+0.8)
        obstacle_right = create_vehicle_box(right_car_x, right_car_y, right_car_yaw, vehicle_box)
    # print(dist_dest_to_left_obst, dest_box.distance(obstacle_right), dest_box.distance(obstacle_right)+dist_dest_to_left_obst)
    dist_dest_to_right_obst = dest_box.distance(obstacle_right)
    if dist_dest_to_right_obst+dist_dest_to_left_obst<LENGTH/4 or \
        dist_dest_to_left_obst<MIN_DIST_TO_OBST or \
        dist_dest_to_right_obst<MIN_DIST_TO_OBST:
        generate_success = False
    # check collision
    obstacles = [obstacle_back, obstacle_left, obstacle_right]
    for obst in obstacles:
        if obst.intersects(dest_box):
            print(dist_dest_to_left_obst, min_dist_to_obst)
            print('dest box intersect with obstacles!!!')
            generate_success = False

    # generate obstacles out of start range
    max_obstacle_y = max([np.max(np.array(obs.coords)[:,1]) for obs in obstacles])+MIN_DIST_TO_OBST
    other_obstcales = []
    if random()<0.2: # in this case only a wall will be generate
        other_obstcales = [LinearRing((
        (origin[0]-bay_half_len, PARA_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST),
        (origin[0]+bay_half_len, PARA_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST),
        (origin[0]+bay_half_len, PARA_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1),
        (origin[0]-bay_half_len, PARA_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1)))]
    else:
        other_obstacle_range = LinearRing((
        (origin[0]-bay_half_len, PARA_PARK_WALL_DIST+max_obstacle_y),
        (origin[0]+bay_half_len, PARA_PARK_WALL_DIST+max_obstacle_y),
        (origin[0]+bay_half_len, PARA_PARK_WALL_DIST+max_obstacle_y+8),
        (origin[0]-bay_half_len, PARA_PARK_WALL_DIST+max_obstacle_y+8)))
        valid_obst_x_range = (origin[0]-bay_half_len+2, origin[0]+bay_half_len-2)
        valid_obst_y_range = (PARA_PARK_WALL_DIST+max_obstacle_y+2, PARA_PARK_WALL_DIST+max_obstacle_y+6)
        for _ in range(3):
            obs_x = random_uniform_num(*valid_obst_x_range)
            obs_y = random_uniform_num(*valid_obst_y_range)
            obs_yaw = random()*pi*2
            obs_coords = np.array(get_box_coords(obs_x, obs_y, obs_yaw, vehicle_box))
            obs = LinearRing(obs_coords+0.5*random(obs_coords.shape))
            if obs.intersects(other_obstacle_range):
                continue
            other_obstcales.append(obs)

    # merge two kind of obstacles
    obstacles.extend(other_obstcales)
    if DEBUG:
        for obs in obstacles:
            ax.add_patch(plt.Polygon(xy=list(obs.coords), color='b'))
    
    # generate start position
    start_box_valid = False
    valid_start_x_range = (origin[0]-bay_half_len/2, origin[0]+bay_half_len/2)
    valid_start_y_range = (max_obstacle_y+1, PARA_PARK_WALL_DIST+max_obstacle_y-1)
    while not start_box_valid:
        start_box_valid = True
        start_x = random_uniform_num(*valid_start_x_range)
        start_y = random_uniform_num(*valid_start_y_range)
        start_yaw = random_gaussian_num(0, pi/6, -pi/2, pi/2)
        start_yaw = start_yaw+pi if random()<0.5 else start_yaw
        start_box = create_vehicle_box(start_x, start_y, start_yaw, vehicle_box)
        # check collision
        for obst in obstacles:
            if obst.intersects(start_box):
                if DEBUG:
                    print('start box intersect with obstacles, will retry to generate.')
                start_box_valid = False
        # check overlap with dest box
        if dest_box.intersects(start_box):
            if DEBUG:
                print('start box intersect with dest box, will retry to generate.')
            start_box_valid = False
    
    # flip the dest box so that the orientation of start matches the dest
    if cos(start_yaw)<0:
        dest_box_center = np.mean(np.array(dest_box.coords[:-1]), axis=0)
        dest_x = 2*dest_box_center[0] - dest_x
        dest_y = 2*dest_box_center[1] - dest_y
        dest_yaw += pi

    # randomly drop the obstacles
    for obs in obstacles:
        if random()<0.1:
            obstacles.remove(obs)
    if DEBUG:
        ax.add_patch(plt.Polygon(xy=get_box_coords(start_x, start_y, start_yaw, vehicle_box), color='g'))
        print(generate_success)
        plt.show()
    # return if success
    if generate_success:
        return [start_x, start_y, start_yaw], [dest_x, dest_y, dest_yaw], obstacles
    else:
        return gene_parallel_park(vehicle_box)

# TODO use class `Area` from tactics2d.map_base.area
class Area(object):
    def __init__(
        self,
        shape = None,
        subtype: str = None,
    ):
        self.shape = shape
        self.subtype = subtype

    def get_shape(self):
        return np.array(self.shape.coords)

class ParkingMapNormal(Map):
    def __init__(self,name="CarParking", scenario_type="parking"):
        super().__init__(name, scenario_type)

        self.start: Position = None
        self.dest: Position = None
        self.start_box:LinearRing = None
        self.dest_box:LinearRing = None
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.n_obstacle = 0
        self.obstacles:List[Area] = []
        self.vehicle_box:LinearRing = None
    
    def set_vehicle(self, vehicle_box):
        self.vehicle_box = vehicle_box

    def reset(self) -> Position:
        self.set_vehicle(VEHICLE_BOX) # TODO
        if random() > 0.5:
            start, dest, obstacles = gene_bay_park(self.vehicle_box)
            self.case_id = 0
        else:
            start, dest, obstacles = gene_parallel_park(self.vehicle_box)
            self.case_id = 1

        self.start = Position(start)
        self.start_box = self.start.create_box(self.vehicle_box)
        self.dest = Position(dest)
        self.dest_box = self.dest.create_box(self.vehicle_box)
        self.xmin = np.floor(min(self.start.loc.x, self.dest.loc.x) - 10)
        self.xmax = np.ceil(max(self.start.loc.x, self.dest.loc.x) + 10)
        self.ymin = np.floor(min(self.start.loc.y, self.dest.loc.y) - 10)
        self.ymax = np.ceil(max(self.start.loc.y, self.dest.loc.y) + 10)
        self.set_boundary([self.xmin, self.ymin, self.xmax, self.ymax])
        self.obstacles = list([Area(shape=obs, subtype="obstacle", \
            ) for obs in obstacles])
        self.n_obstacle = len(self.obstacles)

        return self.start
