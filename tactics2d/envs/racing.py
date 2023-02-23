from typing import Optional, Union
import time

import numpy as np
from shapely.geometry import Point, LineString, LinearRing
# import gymnasium as gym
import gym
from gym import spaces

from tactics2d.utils.bezier import Bezier
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.trajectory.element import State, Trajectory
from tactics2d.traffic.traffic_event import TrafficEvent


STATE_W = 128
STATE_H = 128
VIDEO_W = 600
VIDEO_H = 400
WIN_W= 1000
WIN_H = 1000

FPS = 100
MAX_FPS = 200
STEP_LIMIT = 20000 # steps
TIME_STEP = 0.01 # state update time step: 0.01 s/step

DISCRETE_ACTION = np.array([
    [0, 0], # do nothing
    [-0.3, 0], # steer left
    [0.3, 0], # steer right
    [0, 0.2], # accelerate
    [0, -0.8], # decelerate
])


class RacingEnv(gym.Env):
    """
    ## tactics2d.envs.RacingEnv
    
    An improved version of Box2D's CarRacing gym environment.

    -  **Action Space**: 
        -  If continuous there are 2 actions:
            -  0: steering, -1 is full left, +1 is full right
            -  1: acceleration, range [-1, 1], unit $m^2/s$
        -  If discrete there are 5 actions: 
            -  0: do nothing
            -  1: steer left
            -  2: steer right
            -  3: accelerate
            -  4: decelerate
    -  **Observation Space**: A bird-eye view 128x128 RGB image of the car and the race track.
    -  **Rewards**: [TBD]
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    def __init__(
        self, render_mode: Optional[str] = "human", render_fps: int = FPS,
        verbose: bool = True, continuous: bool = True
    ):

        self.render_mode = render_mode
        if self.render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError
        self.render_fps = render_fps
        if render_fps > MAX_FPS:
            raise UserWarning()

        self.verbose = verbose
        self.continuous = continuous

        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-0.5, -1.]).astype(np.float32),
                np.array([0.5, 1.]).astype(np.float32)
            )
        else:
            self.action_space = spaces.Discrete(5) # do nothing, left, right, gas, brake
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.bezier_generator = Bezier(2, 50)
        self.map = Map(name="CarRacing", scenario_type="racing")
        self.agent = Vehicle(
            id_="0", type_="vehicle:racing", length=5, width=1.8, height=1.5,
            steering_angle_range=(-0.5, 0.5), steering_velocity_range=(-0.5, 0.5),
            speed_range=(-10, 100), accel_range=(-1, 1)
        )

    def _reset_map(self):
        """Reset the map by generating a new one with random track configurations.
        """
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

        # create the new map by the centerline
        center = LineString([start_point] + points + [start_point])
        n_tile = self._create_map(center)

        # record time cost
        t2 = time.time()
        if self.verbose:
            print("Generating process takes %.4fs." % (t2-t1))

        return n_tile

    def _reset_vehicle(self):
        heading_direction = [
            np.mean(list(self.finish_line.coords), axis=0),
            np.mean(list(self.start_line.coords), axis=0)
        ]
        heading_vec = heading_direction[1] - heading_direction[0]
        heading_angle = np.arctan2(heading_vec[1], heading_vec[0])
        heading_direction = LineString(heading_direction)
        start_loc = heading_direction.interpolate(heading_direction.length - self.agent.length/2)

        state = State(
            timestamp = self.n_step, heading = heading_angle, 
            x = start_loc.x, y = start_loc.y, vx=0, vy=0
        )
        self.agent.reset(state)
        if self.verbose:
            print("The racing car starts at (%.3f, %.3f), heading to %.3f rad." \
                % (start_loc.x, start_loc.y, heading_angle))
    
    def reset(self):
        self.n_step = 0
        n_tile = self._reset_map()
        self.tile_visited = [False] * n_tile
        self.tile_visited_count = 0
        self.start_tile = self.map.lanes["0000"]
        self.start_line = LineString(self.start_tile.get_ends())
        self.finish_line = LineString(self.start_tile.get_starts())
        self._reset_vehicle()
        self.window.set_size(int(self.map.boundary[1]), int(self.map.boundary[3]))

    def _check_retrograde(self):

        
        return TrafficEvent.VIOLATION_RETROGRADE

    def _check_non_drivable(self) -> bool:
        
        return TrafficEvent.VIOLATION_NON_DRIVABLE

    def _check_arrived(self):
        return TrafficEvent.ROUTE_COMPLETED

    def _check_outbound(self) -> bool:
        bound = self.map.boundary()
        boundary = LinearRing([
            [bound[0], bound[2]], [bound[0], bound[3]],
            [bound[1], bound[3]], [bound[1], bound[2]]
        ])
        if self.agent.pose.within(boundary):
            return TrafficEvent.NORMAL
        return TrafficEvent.OUTSIDE_MAP

    def _check_time_exceeded(self):
        if self.n_step < STEP_LIMIT:
            return TrafficEvent.NORMAL
        return TrafficEvent.TIME_EXCEED

    def _check_status(self):
        if self._check_retrograde() == TrafficEvent.VIOLATION_RETROGRADE:
            return TrafficEvent.VIOLATION_RETROGRADE
        if self._check_non_drivable() == TrafficEvent.VIOLATION_NON_DRIVABLE:
            return TrafficEvent.VIOLATION_NON_DRIVABLE
        if self._check_arrived() == TrafficEvent.ROUTE_COMPLETED:
            return TrafficEvent.ROUTE_COMPLETED
        if self._check_outbound() == TrafficEvent.OUTSIDE_MAP:
            return TrafficEvent.OUTSIDE_MAP
        if self._check_time_exceeded() == TrafficEvent.TIME_EXCEED:
            return TrafficEvent.TIME_EXCEED
        return TrafficEvent.NORMAL

    def _get_observation(self) -> np.ndarray:
        
        return

    def _get_reward(self):
        
        return

    def step(self, action: Union[np.array, int]):
        if not self.continuous:
            action = DISCRETE_ACTION[action]
        
        self.agent.update(action)
        # position = self._locate(self.agent)

        observation = self._get_observation()
        status = self._check_status()
        reward = self._get_reward()
        done = (status != TrafficEvent.NORMAL)

        info = {
            "status": status,
        }

        if done and self.verbose:
            print("The vehicle stops after %d steps. Stop reason: %s" % (self.n_step, str(status)))

        return observation, reward, done, info
    
    def render(self):
        if self.render_mode == "human":
            return
        elif self.render_mode == "rgb_array":
            return


if __name__ == "__main__":
    action = np.array([0., 0.])

    import pygame

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