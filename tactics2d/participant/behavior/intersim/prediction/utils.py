import argparse
import inspect
import json
import math
import multiprocessing
import os
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict
from random import randint
from typing import Dict, List, Tuple, NamedTuple, Any, Union, Optional

import numpy as np
import torch
import yaml
from torch import Tensor

def get_from_mapping(mapping: List[Dict], key=None) -> List:
    """
    This function extracts values from each dictionary in a list of dictionaries based on a given key.
    
    If the key is not specified, this function attempts to infer the key from the calling context.
    
    Args:
        mapping (List[Dict]): A list of dictionaries to extract values from.
        key (str, optional): The key to use for extracting values. Defaults to None.
    
    Returns:
        List: A list of values extracted from each dictionary using the specified key.
    """
    if key is None:
        # Infer the key if not provided by examining the code context of the caller.
        line_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        key = line_context.split('=')[0].strip()
    return [each[key] for each in mapping]

def rotate(x: float, y: float, angle: float) -> Tuple[float, float]:
    """
    This function applies a rotation transformation to a given point (x, y) by a specified angle around the origin.
    
    Args:
        x (float): The x-coordinate of the point to rotate.
        y (float): The y-coordinate of the point to rotate.
        angle (float): The rotation angle in radians.
    
    Returns:
        Tuple[float, float]: The coordinates of the point after rotation.
    """
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y

def get_dis(points: np.ndarray, point_label: tuple) -> float:
    """
    This function calculates the Euclidean distance between each point in an array of points and a given point label.
    
    Args:
        points (np.ndarray): An array of points, where each point is [x, y].
        point_label (tuple): The label of the point to calculate the distance to, in the form (x, y).
    
    Returns:
        float: The Euclidean distance.
    """
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))

def merge_tensors(tensors: List[torch.Tensor], device, hidden_size: int = 128) -> Tuple[torch.Tensor, List[int]]:
    """
    This function merges a list of tensors into a single tensor, preserving the maximum length among them.
    
    Args:
        tensors (List[torch.Tensor]): A list of tensors to be merged.
        device (torch.device): The device to which the resulting tensor will be moved.
        hidden_size (int, optional): The size of the hidden dimension of the tensors. Defaults to 128.
    
    Returns:
        Tuple[torch.Tensor, List[int]]: The merged tensor and a list of lengths of the original tensors.
    """
    lengths = [tensor.shape[0] if tensor is not None else 0 for tensor in tensors]
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths

def merge_tensors_not_add_dim(tensor_list_list, module, sub_batch_size, device):
    """
    This function merges a list of tensors into a single tensor list using a given module, without adding an extra dimension.
    
    This method is intended for optimization purposes to save memory, merging tensors in sub-batches.
    
    Args:
        tensor_list_list (List[List[torch.Tensor]]): A list of lists of tensors to merge.
        module (torch.nn.Module): The module to apply to the merged tensors.
        sub_batch_size (int): The size of the sub-batches for merging.
        device (torch.device): The device to which the resulting tensor will be moved.
    
    Returns:
        List[torch.Tensor]: A list of tensors that are the result of merging and processing with the module.
    """
    # TODO(cyrushx): Can you add docstring and comments on what this function does?
    # this function is used to save memory in the past at the expense of readability,
    # it will be removed because it is too complicated for understanding
    batch_size = len(tensor_list_list)
    output_tensor_list = []
    for start in range(0, batch_size, sub_batch_size):
        end = min(batch_size, start + sub_batch_size)
        sub_tensor_list_list = tensor_list_list[start:end]
        sub_tensor_list = []
        for each in sub_tensor_list_list:
            sub_tensor_list.extend(each)
        inputs, lengths = merge_tensors(sub_tensor_list, device=device)
        outputs = module(inputs, lengths)
        sub_output_tensor_list = []
        sum = 0
        for each in sub_tensor_list_list:
            sub_output_tensor_list.append(outputs[sum:sum + len(each)])
            sum += len(each)
        output_tensor_list.extend(sub_output_tensor_list)
    return output_tensor_list

class Normalizer:
    """
    Normalizer is a class used for normalizing a set of points by rotating and translating them.
    It encapsulates the origin (x, y) coordinate and the yaw angle for the normalization transformation.
    """

    def __init__(self, x: float, y: float, yaw: float):
        """
        This function initializes the Normalizer with a specific origin and yaw angle.

        Args:
            x (float): The x-coordinate of the normalization origin.
            y (float): The y-coordinate of the normalization origin.
            yaw (float): The yaw angle in radians for the rotation transformation.
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        # Calculate the original point after applying the yaw rotation to the (-x, -y) point.
        self.origin = rotate(0.0 - x, 0.0 - y, yaw)

    def __call__(self, points: np.ndarray, reverse: bool = False) -> np.ndarray:
        """
        This function normalizes a set of points by rotating and translating them based on the Normalizer's origin and yaw.

        Args:
            points (np.ndarray): The points to be normalized. Can be a single point or an array of points.
            reverse (bool): If True, applies the reverse transformation.

        Returns:
            np.ndarray: The normalized points.

        This method applies the rotation and translation to the points and supports both single points 
        and arrays of points. It automatically adjusts its behavior based on the shape of the input array.
        """
        points = np.array(points)
        # Ensure points is of shape (N, 2) where N can be 1 for a single point.
        if points.shape == (2,):
            points.shape = (1, 2)
        # Assert that points is a two-dimensional array, which allows for processing multiple points.
        assert len(points.shape) <= 3
        if len(points.shape) == 3:
            # If points is a 3-dimensional array, apply the normalization to each array within it.
            for each in points:
                each[:] = self.__call__(each, reverse)
        else:
            # For a 2-dimensional array, normalize each point individually.
            assert len(points.shape) == 2
            for point in points:
                if reverse:
                    # If reverse is true, apply reverse transformation: rotate by -yaw and translate by origin.
                    point[0], point[1] = rotate(point[0] - self.origin[0],
                                                point[1] - self.origin[1], -self.yaw)
                else:
                    # Apply the forward transformation: rotate by yaw and translate by (x, y).
                    point[0], point[1] = rotate(point[0] - self.x,
                                                point[1] - self.y, self.yaw)
        return points