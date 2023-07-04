import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.filters import minimum_filter1d
from shapely.geometry import LineString,Point

from tactics2d.participant.element.defaults import VEHICLE_MODEL
from tactics2d.physics.single_track_kinematics import SingleTrackKinematics
from tactics2d.trajectory.element.state import State

vehicle_type = 'medium_car'

WHEEL_BASE = VEHICLE_MODEL[vehicle_type]['wheel_base']  # wheelbase
LENGTH = VEHICLE_MODEL[vehicle_type]['length']
WIDTH = VEHICLE_MODEL[vehicle_type]['width']


from shapely.geometry import LinearRing
VehicleBox = LinearRing(
            [
                [0.5 * LENGTH, -0.5 * WIDTH],
                [0.5 * LENGTH, 0.5 * WIDTH],
                [-0.5 * LENGTH, 0.5 * WIDTH],
                [-0.5 * LENGTH, -0.5 * WIDTH],
            ]
        )

PRECISION = 10
step_speed = 1
VALID_STEER = [-0.75, 0.75]
discrete_actions = []
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, step_speed])
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, -step_speed])
N_DISCRETE_ACTION = len(discrete_actions)

physic_model = SingleTrackKinematics(
                dist_front_hang = 0.5 * LENGTH - VEHICLE_MODEL[vehicle_type]['front_overhang'],
                dist_rear_hang = 0.5 * LENGTH - VEHICLE_MODEL[vehicle_type]['rear_overhang'],
                steer_range=tuple(VALID_STEER),
                speed_range=(-1,1))

class ActionMask():
    def __init__(self, VehicleBox=VehicleBox, n_iter=10) -> None:
        self.vehicle_box_base = VehicleBox
        self.n_iter = n_iter
        self.action_space = discrete_actions
        self.step_time = 0.5
        self.vehicle_boxes = self.init_vehicle_box()
        self.vehicle_lidar_base = 0
        self.lidar_num = 500
        self.distance_tolerance = 0.05
        self.vehicle_base = self.init_vehicle_base()

    def init_vehicle_box(self,):
        VehicleBox = self.vehicle_box_base
        car_coords = np.array(VehicleBox.coords)[:4] # (4,2)
        car_coords_x = car_coords[:,0].reshape(-1)
        car_coords_y = car_coords[:,1].reshape(-1) # (4)
        vehicle_boxes = []
        for action in self.action_space:
            state = State(0, 0, 0, 0, 0, 0, 0)
            for _ in range(self.n_iter):
                state, _ = physic_model.step(state, action, self.step_time/self.n_iter)
                x, y, heading = state.x, state.y, state.heading
                car_x_ = car_coords_x*np.cos(heading) - car_coords_y*np.sin(heading) + x # (4)
                car_y_ = car_coords_x*np.sin(heading) + car_coords_y*np.cos(heading) + y
                vehicle_coords = np.concatenate((np.expand_dims(car_x_, axis=-1), np.expand_dims(car_y_, axis=-1)), axis=-1) # (4, 2)
                vehicle_boxes.append(vehicle_coords)
        vehicle_boxes = np.array(vehicle_boxes).reshape(len(self.action_space), self.n_iter, 4, 2).transpose(1, 0, 2, 3) # (10, 42, 4, 2)
        return vehicle_boxes

    def init_vehicle_base(self, ):
        self.lidar_lines = []
        lidar_num = self.lidar_num
        lidar_range = 100.0
        for a in range(lidar_num):
            self.lidar_lines.append(LineString(((0,0), (np.cos(a*np.pi/lidar_num*2)*lidar_range,\
                 np.sin(a*np.pi/lidar_num*2)*lidar_range))))
        lidar_base = []
        ORIGIN = Point((0,0))
        for l in self.lidar_lines:
            distance = l.intersection(VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)

    
    def init_vehicle_box2(self,):
        VehicleBox = self.vehicle_box_base
        vehicle_boxes = []
        x,y,theta = 0,0,0#vehicle_state.loc.x, vehicle_state.loc.y, vehicle_state.heading
        actions = np.array(self.action_space)
        radius = 1/(np.tan(actions[:,0])/WHEEL_BASE) # TODO: the steer is 0
        car_coords = np.array(VehicleBox.coords)[:4] # (4,2)
        # print(car_coords)
        # car_coords[car_coords>0] += 0.1
        # car_coords[car_coords<0] -= 0.1
        # print(car_coords)
        # p
        car_coords_x = car_coords[:,0].reshape(1,-1)
        car_coords_y = car_coords[:,1].reshape(1,-1) # (1,4)
        # print(car_coords)
        # print(radius)
        Ox = x-radius*np.sin(theta)
        Oy = y+radius*np.cos(theta)
        delta_phi = 0.5*actions[:,1]/10/radius # (42) TODO: simu time
        ptheta = theta
        px, py = x,y

        for i in range(self.n_iter):
            ptheta = ptheta + delta_phi # (42)
            px = Ox + radius*np.sin(ptheta) # (42)
            py = Oy - radius*np.cos(ptheta) # (42)

            # coords transform
            cos_theta = np.cos(ptheta).reshape(-1,1) # (42)
            sin_theta = np.sin(ptheta).reshape(-1,1)
            vehicle_coords_x = cos_theta*car_coords_x - sin_theta*car_coords_y + px.reshape(-1,1) # (42,4)
            vehicle_coords_y = sin_theta*car_coords_x + cos_theta*car_coords_y + py.reshape(-1,1) # (42,4)
            vehicle_coords = np.concatenate((np.expand_dims(vehicle_coords_x, axis=-1), np.expand_dims(vehicle_coords_y, axis=-1)), axis=-1) # (42,4,2)
            # mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
            
            # vehicle_coords_ = vehicle_coords.reshape((vehicle_coords.shape[0], 1, vehicle_coords.shape[1], vehicle_coords.shape[2])) # (42,1,4,2)
            vehicle_boxes.append(vehicle_coords)
        
        return vehicle_boxes
    
    def draw_vehicle_box(self, ax, plt, num_iter, num_action):
        idx = len(self.action_space)*num_iter + num_action
        car_coords = np.array(self.vehicle_boxes).reshape(-1, 4, 2) # (10, 42, 4, 2) -> (420, 4, 2)
        car_edge_x1 = car_coords[:, :, 0].reshape(-1, 4, 1)
        car_edge_y1 = car_coords[:, :, 1].reshape(-1, 4, 1)
        shifted_car_coords = car_coords.copy()
        shifted_car_coords[:,:-1] = car_coords[:,1:]
        shifted_car_coords[:,-1] = car_coords[:,0]
        car_edge_x2 = shifted_car_coords[:, :, 0].reshape(-1, 4, 1)
        car_edge_y2 = shifted_car_coords[:, :, 1].reshape(-1, 4, 1) # (420, 4, 1)
        ax.add_line(plt.Line2D((car_edge_x1[idx,0,0], car_edge_x2[idx,0,0]), \
                                (car_edge_y1[idx,0,0], car_edge_y2[idx,0,0]), color='red'))
        ax.add_line(plt.Line2D((car_edge_x1[idx,1,0], car_edge_x2[idx,1,0]), \
                                (car_edge_y1[idx,1,0], car_edge_y2[idx,1,0]), color='red'))
        ax.add_line(plt.Line2D((car_edge_x1[idx,2,0], car_edge_x2[idx,2,0]), \
                                (car_edge_y1[idx,2,0], car_edge_y2[idx,2,0]), color='red'))
        ax.add_line(plt.Line2D((car_edge_x1[idx,3,0], car_edge_x2[idx,3,0]), \
                                (car_edge_y1[idx,3,0], car_edge_y2[idx,3,0]), color='red'))
    
    def get_idx(self, num_iter, num_action):
        return len(self.action_space)*num_iter + num_action

    
    def get_steps(self, raw_lidar_obs:np.ndarray):
        lidar_obs = np.maximum(self.vehicle_base, raw_lidar_obs-self.distance_tolerance)

        car_coords = np.array(self.vehicle_boxes).reshape(-1, 4, 2) # (10, 42, 4, 2) -> (420, 4, 2)
        car_edge_x1 = car_coords[:, :, 0].reshape(-1, 4, 1)
        car_edge_y1 = car_coords[:, :, 1].reshape(-1, 4, 1)
        shifted_car_coords = car_coords.copy()
        shifted_car_coords[:,:-1] = car_coords[:,1:]
        shifted_car_coords[:,-1] = car_coords[:,0]
        car_edge_x2 = shifted_car_coords[:, :, 0].reshape(-1, 4, 1)
        car_edge_y2 = shifted_car_coords[:, :, 1].reshape(-1, 4, 1) # (420, 4, 1)
        # for i in range(car_edge_x1.shape[0]):
        #     vehicle_coords1 = np.concatenate((car_edge_x2[i,:,:], car_edge_y2[i,:,:]), axis=1)
        #     import matplotlib.pyplot as plt
        #     fig=plt.figure()
        #     ax=fig.add_subplot(111)
        #     ax.add_patch(plt.Polygon(xy=vehicle_coords1, facecolor ='green'))
        #     plt.xlim(-10,10)
        #     plt.ylim(-10,10)
        #     plt.show()
        #     vehicle_coords2 = np.concatenate((car_edge_x1[i,:,:], car_edge_y1[i,:,:]), axis=1)
        #     print(vehicle_coords1)
        #     print(vehicle_coords2)
        #     import matplotlib.pyplot as plt
        #     fig=plt.figure()
        #     ax=fig.add_subplot(111)
        #     ax.add_patch(plt.Polygon(xy=vehicle_coords2, facecolor ='red'))
        #     plt.xlim(-10,10)
        #     plt.ylim(-10,10)
        #     plt.show()
        # Line 1: the edges of vehicle box, ax + by + c = 0
        a = (car_edge_y2 - car_edge_y1) # (420, 4, 1)
        b = (car_edge_x1 - car_edge_x2)
        c = (car_edge_y1*car_edge_x2 - car_edge_x1*car_edge_y2)
        
        lidar_obs = np.clip(lidar_obs, 0, 10) + self.vehicle_lidar_base
        lidar_num = len(lidar_obs)
        angle_vec = np.arange(lidar_num)*np.pi/lidar_num*2
        obstacle_range_x1 = np.cos(angle_vec)*lidar_obs # (N,)
        obstacle_range_y1 = np.sin(angle_vec)*lidar_obs
        obstacle_range_coords = np.concatenate(
            (np.expand_dims(obstacle_range_x1, 1), np.expand_dims(obstacle_range_y1, 1)), axis=1) # (N, 2)
        shifted_obstacle_coords = obstacle_range_coords.copy()
        shifted_obstacle_coords[:-1] = obstacle_range_coords[1:]
        shifted_obstacle_coords[-1] = obstacle_range_coords[0]
        obstacle_range_x2 = shifted_obstacle_coords[:, 0].reshape(1, 1, -1)
        obstacle_range_y2 = shifted_obstacle_coords[:, 1].reshape(1, 1, -1)
        obstacle_range_x1 = obstacle_range_x1.reshape(1, 1, -1)
        obstacle_range_y1 = obstacle_range_y1.reshape(1, 1, -1)
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (obstacle_range_y2 - obstacle_range_y1) # (1, 1, N)
        e = (obstacle_range_x1 - obstacle_range_x2)
        f = (obstacle_range_y1*obstacle_range_x2 - obstacle_range_x1*obstacle_range_y2)

        # calculate the intersections
        det = a*e - b*d # (420, 4, N)
        parallel_line_pos = (det==0) # (420, 4, N)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (420, 4, N)
        raw_y = (c*d - a*f)/det

        collide_map_x = np.ones_like(raw_x, dtype=np.uint8)
        collide_map_y = np.ones_like(raw_x, dtype=np.uint8)
        # print(np.around(raw_x[-len(self.action_space), :, :],2))
        # the false positive intersections on line L2(not on edge L2)
        tolerance_precision = 1e-4 # TODO fix bug in RL_PARKING!!
        collide_map_x[raw_x>np.maximum(obstacle_range_x1, obstacle_range_x2)+tolerance_precision] = 0
        collide_map_x[raw_x<np.minimum(obstacle_range_x1, obstacle_range_x2)-tolerance_precision] = 0
        collide_map_y[raw_y>np.maximum(obstacle_range_y1, obstacle_range_y2)+tolerance_precision] = 0
        collide_map_y[raw_y<np.minimum(obstacle_range_y1, obstacle_range_y2)-tolerance_precision] = 0
        # the false positive intersections on line L1(not on edge L1)
        collide_map_x[raw_x>np.maximum(car_edge_x1, car_edge_x2)+tolerance_precision] = 0
        collide_map_x[raw_x<np.minimum(car_edge_x1, car_edge_x2)-tolerance_precision] = 0
        collide_map_y[raw_y>np.maximum(car_edge_y1, car_edge_y2)+tolerance_precision] = 0
        collide_map_y[raw_y<np.minimum(car_edge_y1, car_edge_y2)-tolerance_precision] = 0

        collide_map = collide_map_x*collide_map_y # (420, 4, N)
        # print(collide_map_x[self.get_idx(7,1), :, :])
        # print(collide_map_y[self.get_idx(7,1), :, :])
        # import matplotlib.pyplot as plt
        # fig=plt.figure()
        # ax=fig.add_subplot(111)
        # tmp_x = raw_x[self.get_idx(7,1), :, :].reshape(-1)
        # tmp_y = raw_y[self.get_idx(7,1), :, :].reshape(-1)
        # tmp_p = collide_map[self.get_idx(7,1), :, :].reshape(-1)
        # print(collide_map[self.get_idx(7,1), :, :])
        # print("#")
        # for i in range(len(collide_map[self.get_idx(7,1), :, :].reshape(-1))):
        #     if tmp_p[i] == 0:
        #         # print('not', tmp_x[i:i+1], tmp_y[i:i+1])
        #         ax.scatter(tmp_x[i:i+1], tmp_y[i:i+1], c='green')
        # for i in range(len(collide_map[self.get_idx(7,1), :, :].reshape(-1))):
        #     if tmp_p[i] != 0:
        #         ax.scatter(tmp_x[i:i+1], tmp_y[i:i+1], c='red')
        #         print(tmp_x[i:i+1], tmp_y[i:i+1])
        # print(collide_map[-len(self.action_space), :, :])
        # print('*')
        collide_map[parallel_line_pos] = 0
        collides = np.sum(collide_map, axis=(1, 2)).reshape(self.n_iter, len(self.action_space)) # (420,) -> (10, 42)
        collides[collides!=0] = 1 # TODO fix bug in RL_PARKING!!
        # print(collides.transpose(1,0))
        # print(collides[:,0])
        collide_free_binary  = (np.sum(collides, axis=0)==0) # (42)
        step_len = np.argmax(collides, axis=0)
        step_len[collide_free_binary.astype(bool)] = self.n_iter
        # print('raw:\n',step_len, collide_free_binary)
        # ax.set_xlim(-10,10)
        # ax.set_ylim(-10,10)
        # ax.set_xlabel('x')
        # idx = 1
        # self.draw_vehicle_box(ax, plt, 9, 1)
        # self.draw_vehicle_box(ax, plt, 7, 1)
        # ax.add_line(plt.Line2D((car_edge_x1[-len(self.action_space)+idx,0,0], car_edge_x2[-len(self.action_space)+idx,0,0]), \
        #                         (car_edge_y1[-len(self.action_space)+idx,0,0], car_edge_y2[-len(self.action_space)+idx,0,0]), color='red'))
        # ax.add_line(plt.Line2D((car_edge_x1[-len(self.action_space)+idx,1,0], car_edge_x2[-len(self.action_space)+idx,1,0]), \
        #                         (car_edge_y1[-len(self.action_space)+idx,1,0], car_edge_y2[-len(self.action_space)+idx,1,0]), color='red'))
        # ax.add_line(plt.Line2D((car_edge_x1[-len(self.action_space)+idx,2,0], car_edge_x2[-len(self.action_space)+idx,2,0]), \
        #                         (car_edge_y1[-len(self.action_space)+idx,2,0], car_edge_y2[-len(self.action_space)+idx,2,0]), color='red'))
        # ax.add_line(plt.Line2D((car_edge_x1[-len(self.action_space)+idx,3,0], car_edge_x2[-len(self.action_space)+idx,3,0]), \
        #                         (car_edge_y1[-len(self.action_space)+idx,3,0], car_edge_y2[-len(self.action_space)+idx,3,0]), color='red'))
        # for i in range(len(lidar_obs)):
        #     # line=plt.Line2D((0,0),(math.cos(i*math.pi/180)*lidar_view[i],math.sin(i*math.pi/180)*lidar_view[i]))
        #     ax.add_line(plt.Line2D((obstacle_range_x1[0,0,i], obstacle_range_x2[0,0,i]), (obstacle_range_y1[0,0,i], obstacle_range_y2[0,0,i])))
        #     if i%1==0:
        #         ax.add_line(plt.Line2D((0,np.cos(i*np.pi/lidar_num*2)*lidar_obs[i]), (0,np.sin(i*np.pi/lidar_num*2)*lidar_obs[i])))
        # # # print(vehicle_coords[-5].shape)
        # # # ax.add_patch(plt.Polygon(xy=self.vehicle_boxes[-1][0], facecolor ='green'))
        # # # ax.add_patch(plt.Polygon(xy=self.vehicle_boxes[-1][1], facecolor ='green'))
        # # # ax.add_patch(plt.Polygon(xy=self.vehicle_boxes[-1][2], facecolor ='green'))
        # ax.add_line(plt.Line2D((VehicleBox.coords[1][0], VehicleBox.coords[2][0]), (VehicleBox.coords[1][1], VehicleBox.coords[2][1]), color='red'))
        # ax.add_line(plt.Line2D((VehicleBox.coords[2][0], VehicleBox.coords[3][0]), (VehicleBox.coords[2][1], VehicleBox.coords[3][1]), color='red'))
        # ax.add_line(plt.Line2D((VehicleBox.coords[3][0], VehicleBox.coords[4][0]), (VehicleBox.coords[3][1], VehicleBox.coords[4][1]), color='red'))
        # ax.add_line(plt.Line2D((VehicleBox.coords[4][0], VehicleBox.coords[1][0]), (VehicleBox.coords[4][1], VehicleBox.coords[1][1]), color='red'))
        # # # y = np.array([[1,1], [2,1], [2,2], [1,2], [0.5,1.5]])
        # # # p = Polygon(y, facecolor = 'k')
        # # ax.add_patch(p)
        # plt.xlim(-10,10)
        # plt.ylim(-10,10)
        # plt.show()
        
        action_mask = self.post_process(step_len)
        if np.sum(action_mask) == 0:
            return np.clip(action_mask, 0.01, 1)
        return action_mask

    def post_process(self, step_len:np.ndarray):
        kernel = 1
        forward_step_len = step_len[:len(step_len)//2]
        backward_step_len = step_len[len(step_len)//2:]
        forward_step_len[0] -= 1
        forward_step_len[-1] -= 1
        backward_step_len[0] -= 1
        backward_step_len[-1] -= 1
        forward_step_len_ = minimum_filter1d(forward_step_len, kernel)
        backward_step_len_ = minimum_filter1d(backward_step_len, kernel)
        return np.clip(np.concatenate((forward_step_len_, backward_step_len_)),0,10)/10
    
    
    def choose_action(self, action_mean, action_std, action_mask):
        if isinstance(action_mean, torch.Tensor):
            action_mean = action_mean.cpu().numpy()
            action_std = action_std.cpu().numpy()
        if isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.cpu().numpy()
        if len(action_mean.shape) == 2:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
        if len(action_mask.shape) == 2:
            action_mask = action_mask.squeeze(0)

        def calculate_probability(mean, std, values):
            z_scores = (values - mean) / std
            log_probabilities = -0.5 * z_scores ** 2 - np.log((np.sqrt(2 * np.pi) * std))
            # print(log_probabilities)
            return np.sum(np.clip(log_probabilities, -10, 10), axis=1)
        possible_actions = np.array(self.action_space)
        # deal the scaling
        action_mean[1] = 1 if action_mean[1]>0 else -1 # TODO
        scale_steer = VALID_STEER[1]
        scale_speed = 1
        possible_actions = possible_actions/np.array([scale_steer, scale_speed])
        # print(possible_actions)
        prob = calculate_probability(action_mean, action_std, possible_actions)
        # prob = np.clip(prob, -10, 10)
        # prob -= prob.min()
        exp_prob = np.exp(prob) * action_mask
        prob_softmax = exp_prob / np.sum(exp_prob)
        # print(prob)
        # print(np.round(prob_softmax, 3))
        actions = np.arange(len(possible_actions))
        action_chosen = np.random.choice(actions, p=prob_softmax)
        # action_chosen = np.argmax(prob_softmax)
        # possible_actions[action_chosen][1] = action_mean[1]
        # masked_speed = max(action_mean[1], action_mask[action_chosen][1]) if action_mask[action_chosen]<0 \
        #     else min(action_mean[1], action_mask[action_chosen])
        masked_speed = action_mask[action_chosen] if action_chosen<int(len(self.action_space)/2) else -action_mask[action_chosen]
        if action_mean[1]*masked_speed>0 and abs(action_mean[1])<abs(masked_speed):
            masked_speed = action_mean[1]
        # print('action', action_chosen, action_mask[action_chosen], masked_speed, possible_actions[action_chosen][0])
        masked_action = np.array([possible_actions[action_chosen][0], masked_speed])
        return masked_action
        # return possible_actions[action_chosen]
        
