##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9

import logging
import math
import random
from typing import Callable

from numpy.typing import ArrayLike


class MCTS:
    """This class implements the Monte Carlo Tree Search algorithm."""

    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.total_reward = 0.0
            self.visits = 0

    def __init__(
        self,
        terminal_fn: Callable,
        expand_fn: Callable,
        reward_fn: Callable,
        simulate_fn: Callable,
        exploration_weight: float = 1.0,
    ):
        """
        Initialize the MCTS planner.

        Args:
            terminal_fn (Callable): Function to determine if a state is terminal. Should accept a state and return a bool.
            expand_fn (Callable): Function to generate possible next states from a given state. Should accept a state and return a list of states.
            reward_fn (Callable): Function to compute the reward for a given state. Should accept a state and return a float.
            simulate_fn (Callable): Function to simulate from a given state to a terminal state. Should accept a state and return a final state.
            exploration_weight (float, optional): Weight for the exploration term in UCB. Defaults to 1.0.
        """
        self.terminal_fn = terminal_fn
        self.expand_fn = expand_fn
        self.reward_fn = reward_fn
        self.simulate_fn = simulate_fn
        self.exploration_weight = exploration_weight

    def _get_best_child(self, node):
        best_value = float("-inf")
        best_node = []
        for child in node.children:
            if child.visits == 0:
                node_value = float("inf")
            else:
                exploit = child.total_reward / child.visits
                explore = self.exploration_weight * math.sqrt(
                    math.log(node.visits) / child.visits
                )  # UCB
                node_value = exploit + explore

            if node_value > best_value:
                best_value = node_value
                best_node = [child]
            elif node_value == best_value:
                best_node.append(child)

        if len(best_node) == 0:
            logging.info("MCTS: No best node found, returning None")
            return None

        return random.choice(best_node)

    def _select(self, node):
        while not self.terminal_fn(node.state):
            if len(node.children) == 0:
                return node
            else:
                node = self._get_best_child(node)
        return node

    def _expand(self, node):
        tried_states = [child.state for child in node.children]
        for state in self.expand_fn(node.state):
            if state not in tried_states:
                child = self.Node(state=state, parent=node)
                node.children.append(child)
                return child
        return node

    def _simulate(self, node):
        final_state = self.simulate_fn(node.state)
        return self.reward_fn(final_state)

    def _back_propagate(self, node, reward: float):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def plan(self, start: ArrayLike, max_try: int = 1e2):
        root = self.Node(state=start)
        for _ in range(int(max_try)):
            node = self._select(root)
            if not self.terminal_fn(node.state):
                child = self._expand(node)
                reward = self._simulate(child)
            else:
                child = node
                reward = self.reward_fn(child.state)
            self._back_propagate(child, reward)

        best_child = self._get_best_child(root)
        path = []
        node = best_child
        while node is not None:
            path.append(node.state)
            node = node.parent
        path = path[::-1]

        return path, root
