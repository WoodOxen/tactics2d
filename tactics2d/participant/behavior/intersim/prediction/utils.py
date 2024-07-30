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

def get_from_mapping(mapping: List[Dict], key=None):
    if key is None:
        line_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        key = line_context.split('=')[0].strip()
    return [each[key] for each in mapping]

def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y

def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))

def merge_tensors(tensors: List[torch.Tensor], device, hidden_size=None) -> Tuple[Tensor, List[int]]:
    """
    merge a list of tensors into a tensor
    """
    lengths = []
    hidden_size = 128
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths

def merge_tensors_not_add_dim(tensor_list_list, module, sub_batch_size, device):
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

class Args:
    data_dir = None
    data_kind = None
    debug = None
    train_batch_size = 64
    seed = None
    eval_batch_size = None
    distributed_training = None
    cuda_visible_device_num = None
    log_dir = None
    learning_rate = None
    do_eval = None
    hidden_size = 128
    sub_graph_depth = 3
    global_graph_depth = None
    train_batch_size = None
    num_train_epochs = None
    initializer_range = None
    sub_graph_batch_size = 4096
    temp_file_dir = None
    output_dir = None
    use_map = None
    reuse_temp_file = None
    old_version = None
    model_recover_path = None
    do_train = None
    max_distance = None
    no_sub_graph = None
    other_params: Dict = {
        "train_relation": True,
        "l1_loss": True,
        "densetnt": True,
        "goals_2D": True,
        "enhance_global_graph": True,
        "laneGCN": True,
        "point_sub_graph": True,
        "laneGCN-4": True,
        "stride_10_2": True,
        "raster": True,
    }
    eval_params = None
    train_params = None
    no_agents = None
    not_use_api = None
    core_num = 16
    visualize = None
    train_extra = None
    hidden_dropout_prob = None
    use_centerline = None
    autoregression = None
    lstm = None
    add_prefix = None
    attention_decay = True
    do_test = None
    placeholder = None
    multi = None
    method_span = None
    waymo = None
    argoverse = None
    nuscenes = None
    single_agent = None
    agent_type = None
    future_frame_num = 80
    no_cuda = None
    mode_num = None
    nms_threshold = None
    inter_agent_types = None
    config = None
    image = None

class Normalizer:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
