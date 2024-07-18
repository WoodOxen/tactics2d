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

class PredictionLoader:
    r"""

    Examples::

        >>> from prediction.predictor import Predictor
        >>> predictor = PredictionLoader()
        >>> from DataLoader import WaymoDL
        >>> loader = WaymoDL('./waymo_tf_example/training')
        >>> data = loader.get_next()
        >>> output = predictor(data)
        >>> print(output)
    """

    def __init__(self, all_agents=False):
        self.all_agents = all_agents

    def __call__(self, data, prediction_loaded, file_mode=0):
        """
        Args:
            data: the infos for the loading scenario
            prediction_loaded: loaded dictionary of the detection result
            file_mode: mode 0 - [scenario_id]: 'rst': np.array (n, 81, 2)

        Returns:
            agent_to_pred:
                a mapping which maps from agent_id to a dictionary {
                    'pred_trajectory': pred_trajectory,
                    'pred_yaw': pred_yaw
                }
                where pred_trajectory is a numpy array of shape (80, 2),
                pred_yaw is a numpy array of shape (80,).
        """

        scenario = data['scenario']
        vectors = []
        agent_to_pred = {}

        # with open(result_path, 'rb') as f:
        #     prediction_loaded = pickle.load(f)

        for agent_id in data['agent']:
            agent = data['agent'][agent_id]
            pose = agent['pose']
            speed = agent['speed']
            shape = agent['shape']
            type = int(agent['type'])
            to_predict = int(agent['to_predict'])

            if self.all_agents or to_predict:
                # import random
                # if random.random() < 0.7:
                #     continue
                pred_trajectory = np.zeros([6, 80, 2])
                assert len(pose) >= 11, len(pose)
                x, y = pose[10, 0], pose[10, 1]
                pred_yaw = np.zeros([6, 80])
                pred_scores = np.zeros(6)

                def get_angle(x, y):
                    return math.atan2(y, x)

                delta_list = []
                for i in range(6, 10):
                    delta = pose[i, 0] - pose[i - 1, 0], pose[i, 1] - pose[i - 1, 1]
                    delta_list.append(delta)
                delta_x, delta_y = np.array(delta_list).mean(axis=0)

                # for i in range(80):
                #     x, y = x + delta_x, y + delta_y
                #     pred_trajectory[:, i, 0], pred_trajectory[:, i, 1] = x, y

                if scenario in prediction_loaded:
                    if agent_id in prediction_loaded[scenario]:
                        if 'rst' in prediction_loaded[scenario][agent_id]:
                            # load without offset
                            pred_scores = np.exp(prediction_loaded[scenario][agent_id]['score'])
                            loaded_pred = prediction_loaded[scenario][agent_id]['rst']

                            for each_prediction in range(6):
                                # agent_index = prediction_loaded[scenario]['ids'].index(agent_id)
                                # pred_trajectory[each_prediction, :, :] = prediction_loaded[scenario]['rst'][each_prediction, agent_index, :, :]
                                pred_trajectory[each_prediction, :, :] = loaded_pred[each_prediction, :, :]
                                for i in range(80):
                                    if i > 0:
                                        x, y = pred_trajectory[each_prediction, i - 1, 0], pred_trajectory[each_prediction, i - 1, 1]
                                    else:
                                        x, y = pose[10, 0], pose[10, 1]
                                    pred_yaw[each_prediction, i] = get_angle(pred_trajectory[each_prediction, i, 0] - x,
                                                                             pred_trajectory[each_prediction, i, 1] - y)
                                    if pred_yaw[each_prediction, i] < 0:
                                        pred_yaw[each_prediction, i] += 2.0 * math.pi
                                    # TODO: delta_x not defined
                                    if abs(delta_x) + abs(delta_y) < 0.01:
                                        pred_yaw[each_prediction, i] = pose[10, -1]
                        else:
                            print(list(prediction_loaded[scenario][agent_id].keys()))
                            assert False, f'rst not in prediction result, is it with time offset? if so, use the predictor with time offset'

                else:
                    # skip scenarios not in prediction result file
                    print(f'scenario {scenario} not found in prediction result {prediction_loaded.keys()}')
                    return None

                agent_to_pred[agent_id] = {}
                agent_to_pred[agent_id]['pred_trajectory'] = pred_trajectory
                agent_to_pred[agent_id]['pred_yaw'] = pred_yaw
                if np.sum(pred_scores) > 0.01:
                    agent_to_pred[agent_id]['pred_scores'] = pred_scores / np.sum(pred_scores)
                else:
                    agent_to_pred[agent_id]['pred_scores'] = pred_scores

        return agent_to_pred