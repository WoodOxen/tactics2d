##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: prioritized_replay.py
# @Description: Modified from https://github.com/gaoyinfeng/TrajGen/blob/main/python/interaction_rl/replay_buffer/prioritized_replay.py.
# @Author: Tactics2D Team
# @Version: 0.1.9


import logging
import os
import pickle
import random

import numpy as np


class SumTree:
    def __init__(self, capacity):
        assert capacity >= 1
        self.capacity = capacity
        self.leaf_count = 1 << (capacity - 1).bit_length()
        self.leaf_start = self.leaf_count - 1
        self.tree = np.zeros(2 * self.leaf_count - 1, dtype=float)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        tree_index = self.leaf_start + self.data_pointer
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent = 0
        while True:
            left = 2 * parent + 1
            if left >= len(self.tree):
                leaf = parent
                break
            right = left + 1
            if v <= self.tree[left]:
                parent = left
            else:
                v -= self.tree[left]
                parent = right
        data_index = leaf - self.leaf_start
        return leaf, self.tree[leaf], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size: int, alpha=0.6, beta=0.4, beta_increment=1e-3):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.sum_tree = SumTree(self.buffer_size)

        # max abs TD error cap — avoids excessively dominating samples
        self.upper_absolute_error = 1.0
        # small epsilon to avoid zero probability
        self.eps = 1e-3

        self._log_utilization_once = True

    def push(self, experience):
        """
        Store a new experience in the tree with max_priority and adjust the priority according to the prediction error.
        """
        start = self.sum_tree.leaf_start
        leaves = self.sum_tree.tree[start : start + self.buffer_size]
        max_priority = float(np.max(leaves)) if leaves.size else 0.0
        if max_priority <= 0:
            max_priority = self.upper_absolute_error
        self.sum_tree.add(max_priority, experience)

    def sample(self, n):
        total = self.sum_tree.total_priority
        if total <= 0.0:
            raise ValueError("Cannot sample from an empty buffer or when all priorities are zero.")

        ids = np.empty((n,), dtype=np.int32)
        experiences = []
        weights = np.empty((n, 1), dtype=np.float32)

        # anneal beta
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        start = self.sum_tree.leaf_start
        leaves = self.sum_tree.tree[start : start + self.buffer_size]
        p_min = np.min(leaves) / total
        if p_min <= 0.0:
            p_min = 1.0 / (self.buffer_size * 1e6)
        max_weight = (p_min * n) ** (-self.beta)

        segment = total / n
        for i in range(n):
            a, b = segment * i, segment * (i + 1)
            value = np.random.uniform(a, b - 1e-12)  # stay within [a, b)
            idx, priority, data = self.sum_tree.get_leaf(value)

            prob = priority / total
            weights[i, 0] = (n * prob) ** (-self.beta) / max_weight

            ids[i] = idx
            experiences.append(data)

        return ids, experiences, weights

    def update_priority(self, idx_list, abs_errors):
        abs_errors = np.asarray(abs_errors, dtype=np.float64).reshape(-1)
        priorities = np.power(
            np.minimum(abs_errors + self.eps, self.upper_absolute_error), self.alpha
        )
        for idx, p in zip(idx_list, priorities):
            self.sum_tree.update(int(idx), float(p))

    def save_buffer(self, path_file, path_info, buffer_data, episode_num=None, total_steps=None):
        with open(path_file, "wb") as f:
            pickle.dump(buffer_data, f)
        if episode_num is not None and total_steps is not None:
            info = {"episode_num": episode_num, "total_steps": total_steps}
            np.save(path_info, info)
        logging.info("Replay buffer saved.")

    def load_buffer(self, path_file, path_info, continue_training=False):
        with open(path_file, "rb") as f:
            buffer_data = pickle.load(f)

        if continue_training:
            info = np.load(path_info, allow_pickle=True).item()
            return buffer_data, info["total_steps"], info["episode_num"]

        return buffer_data

    def measure_utilization(self):
        """Log buffer utilization once after first wrap-around; else periodically log fill percentage."""
        pointer = self.sum_tree.data_pointer
        util = pointer / self.buffer_size
        percent = round(util * 100.0, 2)
        if self._log_utilization_once and pointer == 0:
            logging.info("Replay buffer wrapped: now continuously overwriting oldest samples.")
            self._log_utilization_once = False
        else:
            logging.info(f"{percent} % of the buffer has been filled")
