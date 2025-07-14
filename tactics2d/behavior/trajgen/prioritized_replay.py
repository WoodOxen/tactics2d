##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: prioritized_replay.py
# @Description: The prioritized replay buffer of TrajGen. Original code from https://github.com/gaoyinfeng/TrajGen/blob/main/python/interaction_rl/replay_buffer/prioritized_replay.py.
# @Author: Tactics2D Team
# @Version: 0.1.9


import logging
import os
import pickle
import random

import numpy as np


class SumTree:
    data_pointer = 0

    # initialise tree with all nodes = 0 and data with all values =0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            # overwrite
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class ReplayBuffer:
    def __init__(self, capacity, pretrain_length):
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.sum_tree = SumTree(capacity)
        self.capacity = capacity
        self.pretrain_length = pretrain_length

        self.absolute_error_upper = 1.0
        self.PER_e = 0.01
        self.PER_a = 0.6
        self.PER_b = 0.4
        self.PER_b_increment_per_sampling = 0.001

        buffer_dir = os.path.dirname(os.path.abspath(__file__))
        self.buffer_file = os.path.join(buffer_dir, "buffer.pkl")
        self.buffer_info = os.path.join(buffer_dir, "buffer_info.npy")

        self.check = True

    def store(self, experience):
        """
        Store a new experience in the tree with max_priority
        When training the priority is to be ajusted according with the prediction error
        """
        max_priority = np.max(self.sum_tree.tree[-self.capacity :])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.sum_tree.add(max_priority, experience)

    def sample(self, n):
        ids = np.empty((n,), dtype=np.int32)
        experiences = []
        weights = np.empty((n, 1), dtype=np.float32)

        segment = self.sum_tree.total_priority / n
        self.PER_b = np.min([1.0, self.PER_b + self.PER_b_increment_per_sampling])

        # calc max_Weights
        p_min = np.min(self.tree.tree[-self.tree.capacity :]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            a, b = segment * i, segment * (i + 1)
            value = np.random.uniform(a, b)
            id_, priority, data = self.sum_tree.get_leaf(value)

            prob = priority / self.sum_tree.total_priority
            weights[i, 0] = (n * prob) ** (-self.PER_b) / max_weight

            ids[i] = id_
            experiences.append(data)

        return ids, experiences, weights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def fill_buffer(self, env, control_steering=True):
        steps = 0
        logging.info("Starting to fill buffer using random action...")
        state = env.reset()

        while steps < self.pretrain_length:
            if steps % 500 == 0:
                logging.info(f"{steps} experiences stored")

            ego_id = list(state.keys())[0]
            if control_steering:
                action = [random.uniform(-1, 1), random.uniform(-1, 1)]
            else:
                action = [random.uniform(0, 1)]

            action_dict = {ego_id: action}
            next_state, reward, done, info = env.step(action_dict)

            experience = (
                state[ego_id],
                action_dict[ego_id],
                reward[ego_id],
                next_state[ego_id],
                done[ego_id],
            )
            self.store(experience)
            steps += 1

            if all(done.values()):
                logging.info(info[ego_id]["result"])
                state = env.reset()
            else:
                state = next_state

        logging.info(f"Finished filling buffer with {steps} experiences.")

    def save_buffer(self, buffer_data, episode_num=None, total_steps=None):
        with open(self.buffer_file, "wb") as f:
            pickle.dump(buffer_data, f)
        if episode_num is not None and total_steps is not None:
            info = {"episode_num": episode_num, "total_steps": total_steps}
            np.save(self.buffer_info, info)
        logging.info("Replay buffer saved.")

    def load_buffer(self, continue_training=False):
        with open(self.buffer_file, "rb") as f:
            buffer_data = pickle.load(f)

        if continue_training:
            info = np.load(self.buffer_info, allow_pickle=True).item()
            return buffer_data, info["total_steps"], info["episode_num"]

        return buffer_data

    def measure_utilization(self):
        if self.check:
            utilization = self.tree.data_pointer / self.tree.capacity
            if self.tree.data_pointer < self.pretrain_length:
                logging.info("Memory buffer is full")
                self.check = False
            else:
                logging.info("%s %% of the buffer has been filled" % round(utilization * 100, 2))
