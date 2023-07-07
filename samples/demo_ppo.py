from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from action_mask import ActionMask
from samples.tmp_config import *


OBS_SHAPE = {'lidar':(lidar_num,), 'action_mask':(42,), 'other':(6,)}
ACTOR_CONFIGS = {
    'lidar_shape':OBS_SHAPE['lidar'][0],
    'other_shape':6,
    'action_mask_shape':42,
    'output_size':2,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'use_tanh_output':True,
}

CRITIC_CONFIGS = {
    'lidar_shape':OBS_SHAPE['lidar'][0],
    'other_shape':6,
    'action_mask_shape':42,
    'output_size':1,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'use_tanh_output':False,
}

class StateNorm():
    def __init__(self, observation_shape:dict=OBS_SHAPE,) -> None:
        self.observation_shape = observation_shape
        self.n_state = 0
        self.state_mean, self.S, self.state_std = {}, {}, {}
        for obs_type in self.observation_shape.keys():
            self.state_mean[obs_type] = np.zeros(self.observation_shape[obs_type], dtype=np.float32)
            self.S[obs_type] = np.zeros(self.observation_shape[obs_type], dtype=np.float32)
            self.state_std[obs_type] = np.sqrt(self.S[obs_type])
    
    def init_state_norm(self, mean, std, S, n_state):
        self.n_state = n_state
        self.mean, self.std, self.S = mean, std, S

    def state_norm(self, observation: dict, update=True):
        if self.n_state == 0:
            self.n_state += 1
            for obs_type in self.observation_shape.keys():
                self.state_mean[obs_type] = observation[obs_type]
                self.state_std[obs_type] = observation[obs_type]
                observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
        elif update==True:
            self.n_state += 1
            for obs_type in self.observation_shape.keys():
                old_mean = self.state_mean[obs_type].copy()
                self.state_mean[obs_type] = old_mean + (observation[obs_type] - old_mean) / self.n_state
                self.S[obs_type] = self.S[obs_type] + (observation[obs_type] - old_mean) *\
                    (observation[obs_type] - self.state_mean[obs_type])
                self.state_std[obs_type] = np.sqrt(self.S[obs_type] / self.n_state)
                observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
        return observation

class ReplayMemory(object):
    def __init__(self, memory_size: int, extra_items: list = []):
        self.items = ["state", "action", "reward", "done"] + extra_items
        self.memory = {}
        for item in self.items:
            self.memory[item] = deque([], maxlen=memory_size)
    
    def push(self, observations:tuple):
        """Save a transition"""
        for i, item in enumerate(self.items):
            self.memory[item].append(observations[i])

    def get_items(self, idx_list: np.ndarray):
        batches = {}
        for item in self.items:
            batches[item] = []
        batches["next_state"] = []
        for idx in idx_list:
            for item in self.items:
                batches[item].append(self.memory[item][idx])
            if idx == self.__len__()-1 or self.memory["done"][idx]:
                batches["next_state"].append(None)#np.zeros(self.memory["state"][idx].shape))
            else:
                batches["next_state"].append(self.memory["state"][idx+1])
        for idx in batches.keys():
            if isinstance(batches[idx][0], np.ndarray):
                batches[idx] = np.array(batches[idx])
        return batches

    def sample(self, batch_size: int):
        idx_list = np.random.randint(self.__len__(), size=batch_size)
        return self.get_items(idx_list)

    def shuffle(self, idx_range: int = None):
        idx_range = self.__len__() if idx_range is None else idx_range
        idx_list = np.arange(idx_range)
        np.random.shuffle(idx_list)
        return self.get_items(idx_list)

    def clear(self):
        for item in self.items:
            self.memory[item].clear()

    def __len__(self):
        return len(self.memory["state"])

class Network(nn.Module):
    def __init__(self, configs):
        super().__init__()
        embed_size = configs['embed_size']
        hidden_size = configs['hidden_size']
        activate_func = nn.Tanh()
        n_model = 3

        if configs['n_hidden_layers'] == 1:
            layers = [nn.Linear(n_model*embed_size, configs['output_size'])]
        else:
            layers = [nn.Linear(n_model*embed_size, hidden_size)]
            for _ in range(configs['n_hidden_layers']-2):
                layers.append(activate_func)
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Linear(hidden_size, configs['output_size']))
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Tanh() if configs['use_tanh_output'] else None

        if configs['lidar_shape'] is not None:
            layers = [nn.Linear(configs['lidar_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_lidar = nn.Sequential(*layers)

        if configs['other_shape'] is not None:
            layers = [nn.Linear(configs['other_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_tgt = nn.Sequential(*layers)

        if configs['action_mask_shape'] is not None:
            layers = [nn.Linear(configs['action_mask_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_am = nn.Sequential(*layers)
        self.orthogonal_init()

    def orthogonal_init(self):
        i = 0
        for layer_name, layer in self.net.state_dict().items():
            # The output layer is specially dealt
            gain = 1 if i < len(self.net.state_dict()) - 2 else 0.01
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

        for layer_name, layer in self.embed_lidar.state_dict().items():
            gain = 1
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

        for layer_name, layer in self.embed_tgt.state_dict().items():
            gain = 1
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

        for layer_name, layer in self.embed_am.state_dict().items():
            # The output layer is specially dealt
            gain = 1
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

    def forward(self, x:dict):
        '''
            x: dictionary of different input modal. Includes:

            `img` : image with shape (n, c, w, h)
            `target` : tensor in shape (n, t)
            `lidar` : tensor in shape (n, l)

        '''
        feature_lidar = self.embed_lidar(x['lidar'])
        feature_target = self.embed_tgt(x['other'])
        feature_action_mask = self.embed_am(x['action_mask'])
        features = [feature_lidar, feature_target, feature_action_mask]

        embed = torch.cat(features, dim=1)
        out = self.net(embed)
        if self.output_layer is not None:
            out = self.output_layer(out)
        return out

class DemoPPO():
    def __init__(
        self, 
    ) -> None:
        
        # debug
        self.actor_loss_list = []
        self.critic_loss_list = []
        
        self.gamma: float = 0.95
        self.batch_size = 8192
        self.lr: float = 2e-6
        self.tau: float = 0.1
        self.lr_actor = self.lr
        self.lr_critic = self.lr*5
        self.mini_epoch = 10
        self.mini_batch = 32
        self.clip_epsilon = 0.2
        self.lambda_ = 0.95
        self.adam_epsilon = 1e-8
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.state_norm = True
        self.reward_norm = False
        self.use_gae = True
        self.adv_norm = True
        self.gradient_clip = False
        self.policy_entropy = False
        self.entropy_coef = 0.01
        self.observation_shape = OBS_SHAPE
        self.action_mask = ActionMask()

        # the networks
        self.actor_net = \
            Network(ACTOR_CONFIGS).to(self.device)
        self.log_std = \
            nn.Parameter(
                torch.zeros(1, ACTOR_CONFIGS['output_size']), requires_grad=False
            ).to(self.device)
        self.log_std.requires_grad = True
        self.actor_optimizer = \
            torch.optim.Adam(
                [{'params':self.actor_net.parameters()}, {'params': self.log_std}], 
                self.lr_actor, 
                eps=self.adam_epsilon
            )

        self.critic_net = \
            Network(CRITIC_CONFIGS).to(self.device)
        self.critic_optimizer = \
            torch.optim.Adam(
                self.critic_net.parameters(), 
                self.lr_critic,
                eps=self.adam_epsilon
            )
        self.critic_target = deepcopy(self.critic_net)

        # As a on-policy RL algorithm, PPO does not have memory, the self.memory represents
        # the buffer
        self.memory = ReplayMemory(self.batch_size, ["log_prob","next_obs"])

        # tricks
        if self.state_norm:
            self.state_normalize = StateNorm()

        # save and load
        self.check_list = [
            ("actor_net", self.actor_net, 1),
            ("actor_optimizer", self.actor_optimizer, 1),
            ("critic_net", self.critic_net, 1),
            ("critic_optimizer", self.critic_optimizer, 1),
            ("critic_target", self.critic_target, 1)
        ]
        self.check_list.append(("log_std", self.log_std, 0))

    def get_action(self, obs: np.ndarray):
        '''Take action based on one observation. 

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            action: If self.discrete, the action is an (int) index. 
                If the action space is continuous, the action is an (np.ndarray).
            log_prob(np.ndarray): the log probability of taken action.
        '''
        observation = deepcopy(obs)
        if self.state_norm:
            # observation = self.state_norm(observation)
            observation = self.state_normalize.state_norm(observation)
        observation = self.obs2tensor(observation)
        # if len(observation.shape) == len(self.configs.state_dim):
        #     observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            policy_dist = self.actor_net(observation)
            if len(policy_dist.shape) > 1 and policy_dist.shape[0] > 1:
                raise NotImplementedError
            mean =  torch.clamp(policy_dist,-1,1)  
            log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
            std = torch.exp(log_std)
            dist = Normal(mean, std)
                
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action)
        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return action, log_prob
    
    def choose_action(self, obs:np.ndarray):
        observation = deepcopy(obs)
        # self.action_mask.get_steps(obs['lidar']*10) # TODO scaling
        if self.state_norm:
            # observation = self.state_norm(observation)
            observation = self.state_normalize.state_norm(observation)
        observation = self.obs2tensor(observation)
        # if len(observation.shape) == len(self.configs.state_dim):
        #     observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            policy_dist = self.actor_net(observation)
            if len(policy_dist.shape) > 1 and policy_dist.shape[0] > 1:
                raise NotImplementedError
            mean =  torch.clamp(policy_dist,-1,1)  
            log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
            std = torch.exp(log_std)
            dist = Normal(mean, std)
                
        # action = dist.sample()
        action_mask = obs['action_mask'] 
        action = self.action_mask.choose_action(mean, std, action_mask)
        action = torch.FloatTensor(action).to(self.device)
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action)
        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return action, log_prob

    def get_log_prob(self, obs: np.ndarray, action: np.ndarray):
        '''get the log probability for given action based on current policy

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            log_prob(np.ndarray): the log probability of taken action.
        '''
        observation = deepcopy(obs)
        if self.state_norm:
            observation = self.state_normalize.state_norm(observation)
        observation = self.obs2tensor(observation)
        # if len(observation.shape) == len(self.configs.state_dim):
        #     observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            policy_dist = self.actor_net(observation)
            if len(policy_dist.shape) > 1 and policy_dist.shape[0] > 1:
                raise NotImplementedError
            mean =  torch.clamp(policy_dist,-1,1)  
            log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
            std = torch.exp(log_std)
            dist = Normal(mean, std)
        
        action = torch.FloatTensor(action).to(self.device)
        log_prob = dist.log_prob(action)
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return log_prob

    def push_memory(self, observations):
        '''
        Args:
            observations(tuple): (obs, action, reward, done, log_prob, next_obs)
        '''
        obs, action, reward, done, log_prob, next_obs = deepcopy(observations)
        if self.state_norm:
            obs = self.state_normalize.state_norm(obs)
            next_obs = self.state_normalize.state_norm(next_obs,update=True)
        observations = (obs, action, reward, done, log_prob, next_obs)
        self.memory.push(observations)

    def _reward_norm(self, reward):
        return (reward - reward.mean()) / (reward.std() + 1e-8)

    def obs2tensor(self, obs):
        if isinstance(obs, list):
            merged_obs = {}
            for obs_type in self.observation_shape.keys():
                merged_obs[obs_type] = []
                for o in obs:
                    merged_obs[obs_type].append(o[obs_type])
                merged_obs[obs_type] = torch.FloatTensor(np.array(merged_obs[obs_type])).to(self.device)
            obs = merged_obs 
        elif isinstance(obs, dict):
            for obs_type in self.observation_shape.keys():
                obs[obs_type] = torch.FloatTensor(obs[obs_type]).to(self.device).unsqueeze(0)
        else:
            raise NotImplementedError()
        return obs
    
    def get_obs(self, obs, ids):
        return {k:obs[k][ids] for k in obs }

    def update(self):
        # convert batches to tensors

        # GAE computation cannot use shuffled data
        # batches = self.memory.shuffle()
        batches = self.memory.get_items(np.arange(len(self.memory)))
        state_batch = self.obs2tensor(batches["state"])
        
        action_batch = torch.FloatTensor(batches["action"]).to(self.device)
        rewards = torch.FloatTensor(np.array(batches["reward"])).unsqueeze(1)
        reward_batch = self._reward_norm(rewards) \
            if self.reward_norm else rewards
        reward_batch = reward_batch.to(self.device)
        done_batch = torch.FloatTensor(batches["done"]).to(self.device).unsqueeze(1)
        old_log_prob_batch = torch.FloatTensor(batches["log_prob"]).to(self.device)
        next_state_batch = self.obs2tensor(batches["next_obs"])
        self.memory.clear()

        # GAE
        gae = 0
        adv = []

        with torch.no_grad():
            value = self.critic_net(state_batch)
            next_value = self.critic_net(next_state_batch)
            deltas = reward_batch + self.gamma * (1 - done_batch) * next_value - value
            if self.use_gae:
                for delta, done in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done_batch.cpu().flatten().numpy())):
                    gae = delta + self.gamma * self.lambda_ * gae * (1.0 - done)
                    adv.append(gae)
                adv.reverse()
                adv = torch.FloatTensor(adv).view(-1, 1).to(self.device)
            else:
                adv = deltas
            v_target = adv + self.critic_target(state_batch)
            if self.adv_norm: # advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        
        # apply multi update epoch
        for _ in range(self.mini_epoch):
            # use mini batch and shuffle data
            mini_batch = self.mini_batch
            batchsize = self.batch_size
            train_times = batchsize//mini_batch if batchsize%mini_batch==0 else batchsize//mini_batch+1
            random_idx = np.arange(batchsize)
            np.random.shuffle(random_idx)
            for i in range(train_times):
                if i == batchsize//mini_batch:
                    ri = random_idx[i*mini_batch:]
                else:
                    ri = random_idx[i*mini_batch:(i+1)*mini_batch]
                # state = state_batch[ri]
                state = self.get_obs(state_batch, ri)
                policy_dist = self.actor_net(state)
                mean = torch.clamp(policy_dist, -1, 1)
                log_std = self.log_std.expand_as(mean)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                dist_entropy = dist.entropy().sum(1, keepdim=True)
                log_prob = dist.log_prob(action_batch[ri])
                log_prob =torch.sum(log_prob,dim=1, keepdim=True)
                old_log_prob =torch.sum(old_log_prob_batch[ri],dim=1, keepdim=True)
                prob_ratio = (log_prob - old_log_prob).exp()

                loss1 = prob_ratio * adv[ri]
                loss2 = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv[ri]

                actor_loss = - torch.min(loss1, loss2)
                if self.policy_entropy:
                    actor_loss += - self.entropy_coef * dist_entropy
                critic_loss = F.mse_loss(v_target[ri], self.critic_net(state))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.mean().backward()
                critic_loss.mean().backward()
                
                self.actor_loss_list.append(actor_loss.mean().item())
                self.critic_loss_list.append(critic_loss.mean().item())
                if self.gradient_clip: # gradient clip
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                    nn.utils.clip_grad_norm(self.actor_net.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step() 

            self._soft_update(self.critic_target, self.critic_net)

        # for debug
        a = actor_loss.detach().cpu().numpy()[0][0]
        b = critic_loss.item()
        return a, b

    def _soft_update(self, target_net, current_net):
        for target, current in zip(target_net.parameters(), current_net.parameters()):
            target.data.copy_(current.data * self.tau + target.data * (1. - self.tau))

    def save(self, path: str = None, params_only: bool = None) -> None:
        """Store the model structure and corresponding parameters to a file.
        """
        if params_only is not None:
            self.save_params = params_only
        if self.save_params and len(self.check_list) > 0:
            checkpoint = dict()
            for name, item, save_state_dict in self.check_list:
                checkpoint[name] = item.state_dict() if save_state_dict else item
            # for PPO extra save
            checkpoint['log'] = self.log_std
            checkpoint['state_norm'] = self.state_normalize # (self.state_mean, self.state_std, self.S, self.n_state)
            checkpoint['optimizer'] = (self.actor_optimizer, self.critic_optimizer)
            torch.save(checkpoint, path)
        else:
            torch.save(self, path)
        

    def load(self, path: str = None, params_only: bool = None) -> None:
        """Load the model structure and corresponding parameters from a file.
        """
        if params_only is not None:
            self.load_params = params_only
        if self.load_params and len(self.check_list) > 0:
            checkpoint = torch.load(path)
            for name, item, save_state_dict in self.check_list:
                if save_state_dict:
                    item.load_state_dict(checkpoint[name])
                else:
                    item = checkpoint[name]

            self.log_std.data.copy_(checkpoint['log']) 
            
            self.state_normalize = checkpoint['state_norm'] 
            if 'optimizer' in checkpoint.keys():
                self.actor_optimizer, self.critic_optimizer = checkpoint['optimizer']
        else:
            torch.load(self, path)
        
            path =f"{path}/{name}_{id}.pth"
            state_dict = torch.load(path, map_location=self.device)
            object.load_state_dict(state_dict)
        print('load model!')