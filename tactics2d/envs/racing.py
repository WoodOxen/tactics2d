from typing import Optional, Union
from enum import Enum
import time

import numpy as np
from shapely.geometry import Point, LineString
import gym
from gym import spaces

from tactics2d.map_base.node import Node
from tactics2d.map_base.roadline import RoadLine
from tactics2d.map_base.lane import Lane
from tactics2d.map_base.map import Map
from tactics2d.math.point import get_circle
from tactics2d.math.bezier import Bezier

STATE_W = 128
STATE_H = 128
VIDEO_W = 600
VIDEO_H = 400
WIN_W= 1000
WIN_H = 1000

FPS = 50

DISCRETE_ACTION = np.array([
    [0, 0], # do nothing
    [-0.3, 0], # steer left
    [0.3, 0], # steer right
    [0, 0.2], # accelerate
    [0, -0.8], # decelerate
])

# Track related configurations
N_CHECKPOINT = (10, 20) # the number of turns is ranging in 10-20
TRACK_WIDTH = 15 # the width of the track is ranging in 15m
TRACK_RAD = 800 # the maximum curvature radius
CURVE_RAD = (10, 150) # the curvature radius is ranging in 10-150m
TILE_LENGTH = 5 # the length of each tile
TOLERANT_TIME = 20000 # steps
TIME_STEP = 0.02 # simulate speed: 0.01s/step

class Status(Enum):
    CONTINUE = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5


class CarRacing(gym.Env):
    """
    ## Description
    An improved version of Box2D's CarRacing gym environment

    ## Action Space
    If continuous there are 2 actions
    - 0: steering, -1 is full left, +1 is full right
    - 1: acceleration, range [-1, 1], unit m^2/s

    If discrete there are 5 actions: 
    - 0: do nothing
    - 1: steer left
    - 2: steer right
    - 3: accelerate
    - 4: decelerate

    ## Observation Space

    A bird-eye view 128x128 RGB image of the car and the race track.

    ## Rewards

    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }
    def __init__(
        self, render_mode: Optional[str] = None, render_fps: int = FPS,
        verbose: bool =True, continuous: bool = True
    ):

        self.render_mode = "human" if render_mode is None else render_mode
        self.verbose = verbose
        self.continuous = continuous

        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-0.5, -1.].astype(np.float32),
                np.array(0.5, 1.).astype(np.float32))
            )
        else:
            self.action_space = spaces.Discrete(5) # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.bezier_generator = Bezier(2, 50)
        self.map = Map(name="CarRacing", scenario_type="racing")

    def _create_checkpoints(self):
        n_checkpoint = np.random.randint(*N_CHECKPOINT)
        noise = np.random.uniform(0, 2 * np.pi / n_checkpoint, n_checkpoint)
        alpha = 2 * np.pi * np.arange(n_checkpoint) / n_checkpoint + noise
        rad = np.random.uniform(TRACK_RAD / 5, TRACK_RAD, n_checkpoint)

        checkpoints = np.array([rad * np.cos(alpha), rad * np.sin(alpha)])
        success = False
        control_points = []

        for _ in range(100):
            glued_cnt = 0
            control_points.clear()

            for i in range(n_checkpoint):
                pt1 = checkpoints[:, i-1]
                pt2 = checkpoints[:, i]
                next_i = 0 if i+1 == n_checkpoint else i+1
                pt3 = checkpoints[:, next_i]

                t1 = np.random.uniform(low=1/4, high=1/2)
                t2 = np.random.uniform(low=1/4, high=1/2)
                pt1_ = (1-t1) * pt2 + t1 * pt1
                pt3_ = (1-t2) * pt2 + t2 * pt3
                _, radius = get_circle(pt1_, pt2, pt3_)
                if radius < CURVE_RAD[0]:
                    if rad[i] > rad[next_i]:
                        rad[next_i] += np.random.uniform(0., 10.)
                    else:
                        rad[next_i] -= np.random.uniform(0., 10.)
                    alpha[next_i] += np.random.uniform(0., 0.05)
                    checkpoints[:, next_i] = \
                        [rad[next_i]*np.cos(alpha[next_i]), rad[next_i]*np.sin(alpha[next_i])]
                elif radius > CURVE_RAD[1]:
                    if rad[i] > rad[next_i]:
                        rad[next_i] -= np.random.uniform(0., 10.)
                    else:
                        rad[next_i] += np.random.uniform(0., 10.)
                    alpha[next_i] -= np.random.uniform(0., 0.05)
                    checkpoints[:, next_i] = \
                        [rad[next_i]*np.cos(alpha[next_i]), rad[next_i]*np.sin(alpha[next_i])]
                else:
                    glued_cnt += 1
                    control_points.append([pt1_, pt3_])
            
            if glued_cnt == n_checkpoint:
                success = True
                break
        
        success = success and all(alpha == sorted(alpha))
        return checkpoints, control_points, success

    def _create_map(self) -> bool:
        """Generate a new map with random track configurations.
        """
        self.map.reset()
        t1 = time.time()

        # generate checkpoints of the track
        success = False
        while not success:
            checkpoints, control_points, success = self._create_checkpoints()
        n_checkpoints = checkpoints.shape[1]
        
        if self.verbose:
            print("Generated a new track.", end=" ")

        # find the start-finish straight and the start point
        straight_lens = []
        for i in range(n_checkpoints):
            straight_lens.append(Point(control_points[i][0]).distance(Point(control_points[i-1][1])))
        sorted_id = sorted(range(n_checkpoints), key=lambda i: straight_lens[i], reverse=True)
        start_id = None
        for i in range(3):
            start_id = sorted_id[i]
            if straight_lens[start_id] < 200:
                break
        
        start_point = LineString(
            [control_points[start_id][0], control_points[start_id-1][1]]
        ).interpolate(straight_lens[start_id]/3)

        # get center line
        points = []
        for i in range(n_checkpoints):
            points += self.bezier_generator.get_points(np.array([
                control_points[start_id - i - 1][1],
                checkpoints[:, start_id - i - 1],
                control_points[start_id - i - 1][0]
            ]))

        center = LineString([start_point] + points + [start_point])
        distance = center.length

        # split the track into tiles
        n_tile = int(np.ceil(distance/TILE_LENGTH))
        k = TRACK_WIDTH / 2 / TILE_LENGTH

        if self.verbose:
            print("The track is %dm long and has %d tiles." % (int(distance), n_tile), end=" ")

        center_points = []
        left_points = []
        right_points = []

        for i in range(n_tile+1):
            center_points.append(center.interpolate(TILE_LENGTH * i))
            if i > 0:
                pt1 = center_points[i-1]
                pt2 = center_points[i]
                x_diff = pt2.x - pt1.x
                y_diff = pt2.y - pt1.y
                left_points.append([pt1.x - k*y_diff, pt1.y + k*x_diff])
                right_points.append([pt1.x + k*y_diff, pt1.y - k*x_diff])

        # generate map
        for i in range(n_tile):
            tile = Lane(
                id="%04d" % i,
                left_side=LineString([left_points[i], left_points[i+1]]),
                right_side=LineString([right_points[i], right_points[i+1]]),
                subtype="road", inferred_participants=["vehicle:car", "vehicle:racingcar"]
            )
            self.map.add_lane(tile)

        bbox = center.bounds
        boundary = [bbox[0]-50, bbox[2]+50, bbox[1]-50,  bbox[3]+50]
        self.map.set_boundary(boundary)
        
        # record time cost
        t2 = time.time()
        if self.verbose:
            print("Generating process takes %.4fs." % (t2-t1))
    
    def reset(self):
        self.reward = 0.
        self.tile_visited_count = 0
        self.vehicle_on_track = True
        self._create_map()

    def _check_arrived(self):
        return 

    def _check_time_exceeded(self):
        return self.t > TOLERANT_TIME

    def step(self, action: Union[np.array, int]):
        return 
    
    def render(self):
        if self.render_mode not in self.metadata["render_mode"]:
            raise NotImplementedError


if __name__ == "__main__":
    import pygame

    action = np.array([0., 0.])
    
    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = -0.1
                if event.key == pygame.K_RIGHT:
                    action[0] = +0.1
                if event.key == pygame.K_UP:
                    action[1] = -0.1
                if event.key == pygame.K_DOWN:
                    action[1] = -0.1  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    global restart
                    restart = True