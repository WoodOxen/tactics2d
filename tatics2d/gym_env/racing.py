# A custom version of CarRacing gym environment
# Mainly refer to 
# https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

from typing import Optional, Union, List

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled, InvalidAction
from shapely.geometry import Point, LinearRing, LineString, Polygon
from shapely.affinity import affine_transform
import pygame

from Tacktics2D.elements.base.road.lane import Lane
from Tacktics2D.elements.base.participant.vehicle import Vehicle


OBS_W = 128
OBS_H = 128
VIDEO_W = 600
VIDEO_H = 400
WIN_W= 1000
WIN_H = 1000

N_CHECKPOINT = 12
TRACK_RAD = 150
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 5
TRACK_DETAIL_STEP = 3.5

FPS = 50
TOLERANT_TIME = 10000

DISCRETE_ACTION = np.array([
    [0, 0], # do nothing
    [-0.3, 0], # steer left
    [0.3, 0], # steer right
    [0, 0.2], # accelerate
    [0, -0.8], # decelerate
])

BG_COLOR = (36, 128, 103)
TILE_COLOR = (120, 120, 120)

class Status(Enum):
    CONTINUE = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5


class RacingMap(object):
    def __init__(
        self, 
        track_rad: float = TRACK_RAD, # tile is heavily morphed circle with this radius
        track_width: float = TRACK_WIDTH,
        track_turn_rate:float = TRACK_TURN_RATE,
        track_detail_step:float = TRACK_DETAIL_STEP,
        n_checkpoint: int = N_CHECKPOINT,
        verbose: bool = True
    ):
        # default configs for randomly generate a new tile
        self.track_rad = track_rad
        self.track_width = track_width
        self.track_turn_rate = track_turn_rate
        self.track_detail_step = track_detail_step
        self.n_checkpoint = n_checkpoint
        self.verbose = verbose

        self.start = [0.0] * 3
        self.tiles: List[Lane] = []
        self.n_tile = 0
        self.tile_visited = []
        self.tile_visited_count = 0
        self.xmin, self.xmax = 0.0, 0.0
        self.ymin, self.ymax = 0.0, 0.0

    def _create_tile(self) -> bool:
        """Randomly generate a new tile. Update self.tiles and self.n_tile.

        Args:
            n_checkpoint (int, optional): Defaults to None.

        Returns:
            bool: report whether the tile is successfully generated
        """
        # Create checkpoints
        noise = np.random.uniform(0, 2 * np.pi / self.n_checkpoint, self.n_checkpoint)
        alpha = 2 * np.pi * np.arange(self.n_checkpoint) / self.n_checkpoint + noise
        rad = np.random.uniform(self.track_rad / 3, self.track_rad, self.n_checkpoint)

        alpha[0] = 0
        rad[0] = 1.5 * self.track_rad
        alpha[-1] = 2 * np.pi * (self.n_checkpoint-1) / self.n_checkpoint
        rad[-1] = 1.5 * self.track_rad
        start_alpha = - np.pi / self.n_checkpoint

        checkpoints = np.array([alpha, rad * np.cos(alpha), rad * np.sin(alpha)])
        
        # Go from one checkpoint to another to create tile
        x, y, beta = 1.5 * self.track_rad, 0, 0
        dest_i = 0
        laps = 0
        tiles = []
        no_freeze = 2500
        visited_other_side = False

        while no_freeze > 0 and laps < 5:
            alpha = np.arctan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * np.pi

            # Find destination from checkpoints
            while True: 
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[:, dest_i % self.n_checkpoint]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % self.n_checkpoint == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * np.pi
                continue

            # vector towards destination
            dest_dx = dest_x - x
            dest_dy = dest_y - y
            cos_beta = np.cos(beta)
            sin_beta = np.sin(beta)

            while beta - alpha > 1.5 * np.pi:
                beta -= 2 * np.pi
            while beta - alpha < -1.5 * np.pi:
                beta += 2 * np.pi
            prev_beta = beta
            proj = cos_beta * dest_dx + sin_beta * dest_dy
            if proj > 0.3:
                beta -= min(self.track_turn_rate, abs(5e-3 * proj))
            if proj < -0.3:
                beta += min(self.track_turn_rate, abs(5e-3 * proj))

            x -= sin_beta * self.track_detail_step
            y += cos_beta * self.track_detail_step

            tiles.append([alpha, (prev_beta + beta)/2, x, y])
            no_freeze -= 1

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        idx_range = []
        i = len(tiles)
        while len(idx_range) < 2:
            i -= 1
            if i == 0:
                return False #Failed
            if tiles[i-1][0] <= start_alpha < tiles[i][0]:
                idx_range.append(i)

        if self.verbose:
            print("Tile generation: %i..%i -> %i-tile tile" \
                % (idx_range[1], idx_range[0], idx_range[0] - idx_range[1]))

        tiles = tiles[idx_range[1] : idx_range[0] - 1]

        # Length of perpendicular jump to put together head and tail
        first_beta = tiles[0][1]
        first_perp_x = np.cos(first_beta)
        first_perp_y = np.sin(first_beta)
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (tiles[0][2] - tiles[-1][2]))
            + np.square(first_perp_y * (tiles[0][3] - tiles[-1][3]))
        )
        if well_glued_together > self.track_detail_step:
            return False

        # create tile
        self.n_tile = len(tiles)
        for i in range(self.n_tile):
            _, beta1, x1, y1 = tiles[i]
            _, beta2, x2, y2 = tiles[i - 1]
            point_list = [
                [x2 - self.track_width * np.cos(beta2), y2 - self.track_width * np.sin(beta2)],
                [x1 - self.track_width * np.cos(beta1), y1 - self.track_width * np.sin(beta1)],
                [x1 + self.track_width * np.cos(beta1), y1 + self.track_width * np.sin(beta1)],
                [x2 + self.track_width * np.cos(beta2), y2 + self.track_width * np.sin(beta2)],
            ]
            
            tile = Lane(shape=LinearRing(point_list), subtype="road")
            self.tiles.append(tile)

        # generate starting point
        self.start = State([tiles[0][2], tiles[0][3], tiles[0][1] + np.pi/2, 0, 0])

        # locate visualization range
        tiles = np.array(tiles)
        self.xmin = np.floor(tiles[:, 2].min() - 10)
        self.xmax = np.ceil(tiles[:, 2].max() + 10)
        self.ymin = np.floor(tiles[:, 3].min() - 10)
        self.ymax = np.ceil(tiles[:, 3].max() + 10)
        return True

    def reset(self, ) -> State:
        """Generate a new map and reset the map-related data."""
        self.n_tile = 0
        self.tiles.clear()
        success = False
        while not success:
            success = self._create_tile()
            if not success and self.verbose:
                print("Retry to generate tile (normal if there are not many instances of this message)")

        self.tile_visited = [False] * self.n_tile
        self.tile_visited[0] = True
        self.tile_visited_count = 1
                
        return self.start


class CarRacing(gym.Env):
    """
    Description:

    Action Space:
        If continuous:
            There are 3 actions: steer (-1 is full left, +1 is full right), gas, and brake.
        If discrete:
            There are 5 actions: do nothing, steer left, steer right, gas, brake.

    Observation Space:
        State consists of 96x96 pixels.

    Args:
        
    """

    metadata = {
        "render_mode": [
            "human", 
            "rgb_array",
        ]
    }

    def __init__(
        self, 
        n_checkpoint: int = N_CHECKPOINT,
        render_mode: str = None, 
        fps: int = FPS,
        verbose: bool =True, 
        continuous: bool =True
    ):
        super().__init__()
        
        self.n_checkpoint = n_checkpoint
        self.verbose = verbose
        self.continuous = continuous
        self.render_mode = "human" if render_mode is None else render_mode
        self.fps = fps
        self.screen: Optional[pygame.Surface] = None
        self.matrix = None
        self.clock = None
        self.is_open = True
        self.vehicle_on_track = True # False when vehicle drive across the destination(start) reversely
        self.t = 0.0
        
        self.map = RacingMap(verbose=self.verbose)
        self.vehicle = Vehicle(speed_range=[-2.5, 5])
        self.reward = 0.0
        self.prev_reward = 0.0

        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-0.5, -1]).astype(np.float32),
                np.array([0.5, 1]).astype(np.float32),
            ) # steer, acceleration
        else:
            self.action_space = spaces.Discrete(5) # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_W, OBS_H, 3), dtype=np.uint8
        )

    def reset(self) -> np.ndarray:
        # self._destroy() #TODO
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.vehicle_on_track = True
        self.t = 0.0

        initial_state = self.map.reset()
        self.vehicle.reset(initial_state)
        self.matrix = self._coord_transform_matrix()
        return self.step()[0]

    def _coord_transform_matrix(self) -> list:
        """Get the transform matrix that convert the real world coordinate to the pygame coordinate.
        """
        k1 = WIN_W/ (self.map.xmax - self.map.xmin)
        k2 = WIN_H / (self.map.ymax - self.map.ymin)
        k = k1 if k1 < k2 else k2
        bx = 0.5 * (WIN_W - k * (self.map.xmax + self.map.xmin))
        by = 0.5 * (WIN_H - k * (self.map.ymax + self.map.ymin))
        return [k, 0, 0, k, bx, by]

    def _coord_transform(self, object) -> list:
        transformed = affine_transform(object, self.matrix)
        return list(transformed.coords)
    
    def _detect_collision(self, sorted_idx: list):
        for idx in sorted_idx:
            tile_shape = self.map.tiles[idx].shape
            if self.vehicle.box.intersects(tile_shape):
                left_bound = LineString(list(tile_shape.coords)[0:2])
                right_bound = LineString(list(tile_shape.coords)[2:4])
                if self.vehicle.box.intersects(left_bound) or self.vehicle.box.intersects(right_bound):
                    return True
        return False
    
    def _check_arrived(self, sorted_idx: list):
        # check cross the start reversely
        if len(self.vehicle.trajectory)>1:
            start_direction = self.map.start.heading
            driving_direction = (self.vehicle.trajectory[-1].x - self.vehicle.trajectory[-2].x,\
                 self.vehicle.trajectory[-1].y - self.vehicle.trajectory[-2].y) 
            driving_line = LineString((self.vehicle.trajectory[-2], self.vehicle.trajectory[-1]))
            driving_reverse = True
            if np.cos(start_direction)*driving_direction[0]+np.cos(start_direction)*driving_direction[1]>0:
                driving_reverse = False
            if self.map.tiles[0].shape.intersects(driving_line) and driving_reverse:
                self.vehicle_on_track = False
            elif self.map.tiles[0].shape.intersects(driving_line) and not driving_reverse:
                self.vehicle_on_track = True

        # if the vehicle has driven across the start reversely,
        # then visit new tile is unavailable until it drives
        # across the start again.
        if not self.vehicle_on_track or len(self.vehicle.trajectory)<=1:
            arrived = False
            visit_new_tile_count = 0
        else:
            arrived = False
            loc_idx = -1
            visit_new_tile = False
            visit_new_tile_count = 0
            for idx in sorted_idx:
                if Polygon(self.map.tiles[idx].shape).contains(self.vehicle.box.centroid) and idx-self.map.tile_visited_count<20:
                    if not self.map.tile_visited[idx]:
                        self.map.tile_visited[idx] = True
                        self.map.tile_visited_count += 1
                        visit_new_tile = True
                        visit_new_tile_count += 1
                        loc_idx = idx
                    break
            if visit_new_tile:
                while not self.map.tile_visited[loc_idx-1]:
                    self.map.tile_visited[loc_idx-1] = True
                    self.map.tile_visited_count += 1
                    visit_new_tile_count += 1
                    loc_idx -= 1
            # judge arrive
            if self.map.tile_visited_count > self.map.n_tile-20 and \
                self.map.tiles[-1].shape.intersects(driving_line) and not driving_reverse:
                arrived = True
            elif self.map.tile_visited_count==self.map.n_tile:
                arrived = True

        return arrived, visit_new_tile_count
    
    def _check_time_exceeded(self):
        return self.t > TOLERANT_TIME
    
    def _check_status(self):
        sorted_idx = sorted(
            range(self.map.n_tile),
            key=lambda x: self.vehicle.box.centroid.distance(self.map.tiles[x].shape.centroid)
        )

        arrived, visit_new_tile_count = self._check_arrived(sorted_idx)
        if self._detect_collision(sorted_idx[:20]):
            return Status.COLLIDED, visit_new_tile_count
        if arrived:
            return Status.ARRIVED, visit_new_tile_count
        if self._check_time_exceeded():
            return Status.OUTTIME, visit_new_tile_count
        return Status.CONTINUE, visit_new_tile_count

    def _get_reward(self, visit_tile_count) -> float:
        time_cost = - np.tanh(self.t/self.map.n_tile/10) # [-1, 0]
        if visit_tile_count < 0: # when the vehicle sticks at a tile or goes back, [-inf, -1]
            tile_reward = visit_tile_count - 1
        else: # # when the vehicle visits a new tile [0, ~10]
            tile_reward = visit_tile_count *200 / self.map.n_tile
        if self.vehicle.v_max == self.vehicle.v_min:
            speed_reward = 0
        else:
            speed_reward = \
                0.5*(self.vehicle.state.speed - 0) / (self.vehicle.v_max - self.vehicle.v_min) 

        return time_cost + tile_reward + speed_reward

    def step(self, action: Union[np.array, int] = None):
        assert self.vehicle is not None
        if action is not None:
            if self.continuous:
                self.vehicle.step(action)
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.vehicle.step(DISCRETE_ACTION[action])

        self.t += 1. / self.fps
        observation = self.render(self.render_mode)

        step_reward = 0
        if action is not None:
            self.prev_reward = self.reward
        status, visit_new_tile_count = self._check_status()
        if status == Status.COLLIDED:
            step_reward = -200
        elif status == Status.OUTTIME:
            step_reward = -100
        elif status == Status.ARRIVED:
            step_reward = 200
        elif status == Status.CONTINUE:
            step_reward = self._get_reward(visit_new_tile_count)
        self.reward += step_reward

        done = (status != Status.CONTINUE)
        if done and self.verbose:
            print(f'The vehicle is stopped after {round(self.t,3)} secondes. Stop reason: {status}.' )
            print(f'Accumulated reward: {self.reward}')
        
        info = {
            "status": status,
        }

        return observation, step_reward, done, info

    def _render(self, surface: pygame.Surface):
        surface.fill(BG_COLOR)
        for tile in self.map.tiles:
            tile_color = TILE_COLOR if tile.color is None else tile.color
            pygame.draw.polygon(surface, tile_color, self._coord_transform(tile.shape))
        pygame.draw.polygon(surface, self.vehicle.color, self._coord_transform(self.vehicle.box))

    def _get_observation(self,  surface: pygame.Surface):
        angle = self.vehicle.state.heading
        old_center = surface.get_rect().center

        vehicle_center = np.array(self._coord_transform(self.vehicle.box.centroid)[0])

        dx = (vehicle_center[0]-old_center[0])
        dy = (vehicle_center[1]-old_center[1])

        # align the center of the observation with the center of the vehicle
        observation = pygame.Surface((WIN_W, WIN_H))
        observation.fill(BG_COLOR)
        observation.blit(surface, (int(-dx), int(-dy)))

        # Rotate and find the center of the vehicle
        capture = pygame.transform.rotate(observation, np.rad2deg(angle))
        rotate = pygame.Surface((WIN_W, WIN_H))
        rotate.blit(capture, capture.get_rect(center=old_center))

        observation = rotate.subsurface((
            (WIN_W-OBS_W)/2, (WIN_H-OBS_H)/2), (OBS_W, OBS_H))

        obs_str = pygame.image.tostring(observation, "RGB")
        observation = np.frombuffer(obs_str, dtype=np.uint8)
        observation = observation.reshape(self.observation_space.shape)

        return observation

    def render(self, mode: str = "human"):
        if mode not in self.metadata["render_mode"]:
            raise NotImplementedError

        if mode == "human":
            display_flags = pygame.SHOWN
        else:
            display_flags = pygame.HIDDEN

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags = display_flags)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._render(self.screen)
        observation = self._get_observation(self.screen)
        pygame.display.update()
        self.clock.tick(self.fps)

        return observation

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0])

    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -0.1
                if event.key == pygame.K_RIGHT:
                    a[0] = +0.1
                if event.key == pygame.K_UP:
                    a[1] = -0.1
                if event.key == pygame.K_DOWN:
                    a[1] = -0.1  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    global restart
                    restart = True

    env = CarRacing()
    is_open = True
    while is_open:
        env.reset()
        total_reward = 0.0
        n_step = 0
        
        restart = False
        status = Status.CONTINUE
        while status == Status.CONTINUE and is_open:
            register_input()
            observation, reward, status, is_open = env.step(a)
            total_reward += reward
            n_step += 1
            # is_open = env.render()

    env.close()

