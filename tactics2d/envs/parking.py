import sys
sys.path.append("../")
from typing import Optional, Union
import csv
import math

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled, InvalidAction
from shapely.geometry import LineString, LinearRing, Polygon
from shapely.affinity import affine_transform

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame`"
    )

from tactics2d.traffic_event.status import Status


OBS_W = 256
OBS_H = 256
VIDEO_W = 600
VIDEO_H = 400
WIN_W = 500
WIN_H = 500

FPS = 50
TOLERANT_TIME = 1000

DISCRETE_ACTION = np.array([
    [0, 0], # do nothing
    [-0.3, 0], # steer left
    [0.3, 0], # steer right
    [0, 0.2], # accelerate
    [0, -0.8], # decelerate
])

VALID_ACCEL = [-1.0, 1.0]
VALID_STEER = [-1.0, 1.0]

BG_COLOR = (255, 255, 255, 255)
START_COLOR = (100, 149, 237, 255)
DEST_COLOR = (69, 139, 0, 255)
OBSTACLE_COLOR = (150, 150, 150, 255)
TRAJ_COLOR = (10, 10, 150, 255)


class ParkingMap(object):
    default = {
        "path": '../data/Case%d.csv'
    }

    def __init__(self):

        self.case_id:int = None
        self.start: State = None
        self.dest: State = None
        self.start_box:LinearRing = None
        self.dest_box:LinearRing = None
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.n_obstacle = 0
        self.obstacles:List[Area] = []

    def reset(self, case_id: int = None, path: str = None) -> State:
        case_id = 19 # np.random.randint(1, 21) if case_id is None else case_id
        path = self.default["path"] if path is None else path

        if case_id == self.case_id:
            return self.start
        else:
            self.case_id = case_id
            fname = path % case_id
            self.obstacles.clear()

        with open(fname, 'r') as f:
            reader = csv.reader(f)
            tmp = list(reader)
            v = [float(i) for i in tmp[0]]

            self.start = State(v[:3]+[0,0])
            self.start_box = self.start.create_box()
            self.dest = State(v[3:6]+[0,0])
            self.dest_box = self.dest.create_box()
            self.xmin = np.floor(min(self.start.loc.x, self.dest.loc.x) - 10)
            self.xmax = np.ceil(max(self.start.loc.x, self.dest.loc.x) + 10)
            self.ymin = np.floor(min(self.start.loc.y, self.dest.loc.y) - 10)
            self.ymax = np.ceil(max(self.start.loc.y, self.dest.loc.y) + 10)

            self.n_obstacle = int(v[6])
            n_vertex = np.array(v[7:7 + self.n_obstacle], dtype=np.int32)
            vertex_start = 7 + self.n_obstacle + (np.cumsum(n_vertex, dtype=np.int32) - n_vertex) * 2
            for vs, nv in zip(vertex_start, n_vertex):
                obstacle = LinearRing(np.array(v[vs : vs+nv*2]).reshape((nv, 2), order='A'))
                self.obstacles.append(
                    Area(shape=obstacle, subtype="obstacle", color=(150, 150, 150, 255)))

        return self.start


class CarParking(gym.Env):
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
        render_mode: str = None,
        fps: int = FPS,
        verbose: bool =True, 
        continuous: bool =True
    ):
        super().__init__()

        self.verbose = verbose
        self.continuous = continuous
        self.render_mode = "human" if render_mode is None else render_mode
        self.fps = fps
        self.screen: Optional[pygame.Surface] = None
        self.matrix = None
        self.clock = None
        self.is_open = True
        self.t = 0.0
        self.k = None

        self.map = ParkingMap()
        self.vehicle = Vehicle()
        self.reward = 0.0
        self.prev_reward = 0.0

        if self.continuous:
            self.action_space = spaces.Box(
                np.array([VALID_STEER[0], VALID_ACCEL[0]]).astype(np.float32),
                np.array([VALID_STEER[1], VALID_ACCEL[1]]).astype(np.float32),
            ) # steer, acceleration
        else:
            self.action_space = spaces.Discrete(5) # do nothing, left, right, gas, brake
       
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_W, OBS_H, 3), dtype=np.uint8
        )

    def reset(self, case_id: int = None, data_dir: str = None) -> np.ndarray:
        self.reward = 0.0
        self.prev_reward = 0.0
        self.t = 0.0

        initial_state = self.map.reset(case_id, data_dir)
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
        self.k = k
        return [k, 0, 0, k, bx, by]

    def _coord_transform(self, object) -> list:
        transformed = affine_transform(object, self.matrix)
        return list(transformed.coords)

    def _detect_collision(self):
        return False
        for obstacle in self.map.obstacles:
            if self.vehicle.box.intersects(obstacle.shape):
                return True
        return False
    
    def _detect_outbound(self):
        vehicle_box = np.array(self._coord_transform(self.vehicle.box))
        if vehicle_box[:, 0].min() < 0:
            return True
        if vehicle_box[:, 0].max() > WIN_W:
            return True
        if vehicle_box[:, 1].min() < 0:
            return True
        if vehicle_box[:, 1].max() > WIN_H:
            return True
        return False

    def _check_arrived(self):
        vehicle_box = Polygon(self.vehicle.box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        if union_area / dest_box.area > 0.95:
            return True
        return False
    
    def _check_time_exceeded(self):
        return self.t > TOLERANT_TIME

    def _check_status(self):
        if self._detect_collision():
            return Status.COLLIDED
        if self._detect_outbound():
            return Status.OUTBOUND
        if self._check_arrived():
            return Status.ARRIVED
        if self._check_time_exceeded():
            return Status.OUTTIME
        return Status.CONTINUE

    def _get_reward2(self, prev_state: State, curr_state: State) -> float:
        time_cost = - np.tanh(self.t / 1000)

        dist_diff = curr_state.loc.distance(self.map.dest.loc)
        angle_diff = abs(curr_state.heading - self.map.dest.heading)
        if dist_diff == 0:
            k = 10 / angle_diff
        elif angle_diff == 0:
            k = 10 / dist_diff
        else:
            k = np.sqrt(10 / (dist_diff * angle_diff))
        dx =  dist_diff - prev_state.loc.distance(self.map.dest.loc)
        dtheta =  angle_diff - abs(prev_state.heading - self.map.dest.heading)
        distance_reward = k * dist_diff * dx
        angle_reward = k * angle_diff * dtheta
        print(k, dist_diff, angle_diff)
        print(time_cost , distance_reward , angle_reward)
        return time_cost + distance_reward + angle_reward

    def _get_reward(self, prev_state: State, curr_state: State) -> float:
        time_cost = - np.tanh(self.t / 10000)

        def get_angle_diff(angle1, angle2):
            # norm to 0 ~ pi
            return math.acos(math.cos(angle1 - angle2))

        dist_diff = curr_state.loc.distance(self.map.dest.loc)
        angle_diff = get_angle_diff(curr_state.heading, self.map.dest.heading)
        prev_dist_diff = prev_state.loc.distance(self.map.dest.loc)
        prev_angle_diff = get_angle_diff(prev_state.heading, self.map.dest.heading)
        dist_norm_ratio = self.map.dest.loc.distance(self.map.start.loc)
        angle_norm_ratio = math.pi
        dist_reward = math.exp(-dist_diff/dist_norm_ratio) - \
            math.exp(-prev_dist_diff/dist_norm_ratio)
        angle_reward = math.exp(-angle_diff/angle_norm_ratio) - \
            math.exp(-prev_angle_diff/angle_norm_ratio)

        dist_reward *= 10
        # reward mainly focus angle when is close to destinaiton
        angle_reward *= (5*max(dist_norm_ratio/dist_diff,10))
        # print(time_cost , dist_reward , angle_reward)

        vehicle_box = Polygon(self.vehicle.box)
        prev_vehicle_box = Polygon(prev_state.create_box())
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        prev_union_area = prev_vehicle_box.intersection(dest_box).area
        arriving_reward = (union_area-prev_union_area)/dest_box.area *100
        # print(time_cost , dist_reward , angle_reward , arriving_reward)
        return time_cost + dist_reward + angle_reward + arriving_reward
        

    def step(self, action: Union[np.ndarray, int] = None):
        assert self.vehicle is not None
        prev_state = self.vehicle.state
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

        self.t += 1 # . / self.fps
        observation = self.render(self.render_mode)

        step_reward = 0
        if action is not None:
            self.prev_reward = self.reward
        status = self._check_status()
        if status == Status.COLLIDED:
            step_reward = -50
        elif status == Status.OUTBOUND:
            step_reward = -50
        elif status == Status.OUTTIME:
            step_reward = -20
        elif status == Status.ARRIVED:
            step_reward = 50
        elif status == Status.CONTINUE:
            step_reward = self._get_reward(prev_state, self.vehicle.state)
        self.reward += step_reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_open = False

        return observation, step_reward, status, self.is_open

    def _render(self, surface: pygame.Surface):
        surface.fill(BG_COLOR)
        for obstacle in self.map.obstacles:
            pygame.draw.polygon(
                surface, OBSTACLE_COLOR, self._coord_transform(obstacle.shape))

        pygame.draw.polygon(
            surface, START_COLOR, self._coord_transform(self.map.start_box), width=1)
        pygame.draw.polygon(
            surface, DEST_COLOR, self._coord_transform(self.map.dest_box))#, width=1
        
        pygame.draw.polygon(
            surface, self.vehicle.color, self._coord_transform(self.vehicle.box))

        if len(self.vehicle.trajectory) > 1:
            pygame.draw.lines(
                surface, TRAJ_COLOR, False, 
                self._coord_transform(LineString(self.vehicle.trajectory[-10:]))
            )

    def _get_observation(self, surface: pygame.Surface):
        angle = self.vehicle.state.heading
        old_center = surface.get_rect().center

        # Rotate and find the center of the vehicle
        capture = pygame.transform.rotate(surface, np.rad2deg(angle))
        rotate = pygame.Surface((WIN_W, WIN_H))
        rotate.blit(capture, capture.get_rect(center=old_center))
        
        vehicle_center = np.array(self._coord_transform(self.vehicle.box.centroid)[0])
        dx = (vehicle_center[0]-old_center[0])*np.cos(angle) \
            + (vehicle_center[1]-old_center[1])*np.sin(angle)
        dy = -(vehicle_center[0]-old_center[0])*np.sin(angle) \
            + (vehicle_center[1]-old_center[1])*np.cos(angle)
        
        # align the center of the observation with the center of the vehicle
        observation = pygame.Surface((WIN_W, WIN_H))
    
        observation.fill(BG_COLOR)
        observation.blit(rotate, (int(-dx), int(-dy)))
        observation = observation.subsurface((
            (WIN_W-OBS_W)/2, (WIN_H-OBS_H)/2), (OBS_W, OBS_H))

        # add destination info when it is out of horizon
        if self.map.dest_box.centroid.distance(self.vehicle.box.centroid)>150/self.k:
            dest_direction_relative = math.atan2(self.map.dest_box.centroid.y-self.vehicle.box.centroid.y, \
                self.map.dest_box.centroid.x-self.vehicle.box.centroid.x) - self.vehicle.state.heading
            arrow_margin = 20
            if -OBS_H/OBS_W < math.tan(dest_direction_relative) <= OBS_H/OBS_W:
                arrow_pos_x = OBS_W - arrow_margin if math.cos(dest_direction_relative)>0 else arrow_margin
                arrow_pos_y = (arrow_pos_x-OBS_W/2) * math.tan(dest_direction_relative) + OBS_H/2
            else:
                arrow_pos_y = OBS_H - arrow_margin if math.sin(dest_direction_relative)>0 else arrow_margin
                arrow_pos_x = (arrow_pos_y-OBS_H/2) * math.cos(dest_direction_relative) / math.sin(dest_direction_relative) + OBS_W/2
            # create the arrow
            arrow_margin -= 2
            pt1 = (int(arrow_pos_x+math.cos(dest_direction_relative)*arrow_margin),\
                    int(arrow_pos_y+math.sin(dest_direction_relative)*arrow_margin))
            pt2 = (int(arrow_pos_x+math.cos(dest_direction_relative+1.7)*arrow_margin/2),\
                    int(arrow_pos_y+math.sin(dest_direction_relative+1.7)*arrow_margin/2))
            pt3 = (int(arrow_pos_x), int(arrow_pos_y))
            pt4 = (int(arrow_pos_x+math.cos(dest_direction_relative-1.7)*arrow_margin/2),\
                    int(arrow_pos_y+math.sin(dest_direction_relative-1.7)*arrow_margin/2))
            pygame.draw.polygon(observation, DEST_COLOR, [pt1, pt2, pt3, pt4])

    
        obs_str = pygame.image.tostring(observation, "RGB")
        observation = np.frombuffer(obs_str, dtype=np.uint8)
        observation = observation.reshape(self.observation_space.shape)

        return observation

    def render(self, mode: str = "human"):
        assert mode in self.metadata["render_mode"]
        assert self.vehicle is not None

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

    env = CarParking()
    is_open = True
    while is_open:
        env.reset()
        total_reward = 0.0
        n_step = 0
        restart = False
        status = Status.CONTINUE
        while status == Status.CONTINUE and is_open:
            register_input()
            observation, reward, status = env.step(a)
            total_reward += reward
            n_step += 1

        is_open = False

    env.close()

