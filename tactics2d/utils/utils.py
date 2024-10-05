import inspect
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def mph_to_meterpersecond(mph):
    return mph * 0.4472222222


def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle


def get_current_pose_and_v(current_state, agent_id, current_frame_idx):
    my_current_pose = current_state["agent"][agent_id]["pose"][current_frame_idx - 1]
    if (
        current_state["agent"][agent_id]["pose"][current_frame_idx - 1, 0] == -1
        or current_state["agent"][agent_id]["pose"][current_frame_idx - 6, 0] == -1
    ):
        my_current_v_per_step = 0
        print("Past invalid for ", agent_id, " and setting v to 0")
    else:
        my_current_v_per_step = (
            utils.euclidean_distance(
                current_state["agent"][agent_id]["pose"][current_frame_idx - 1, :2],
                current_state["agent"][agent_id]["pose"][current_frame_idx - 6, :2],
            )
            / 5
        )
    return my_current_pose, my_current_v_per_step


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

def get_from_mapping(mapping: List[Dict], key=None):
    if key is None:
        # Infer the key if not provided by examining the code context of the caller.
        line_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        key = line_context.split("=")[0].strip()
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

def rotate(x, y, angle):
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


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(
        np.square(points[:, 0] - point_label[0]) + np.square(points[:, 1] - point_label[1])
    )


def merge_tensors(
    tensors: List[torch.Tensor], device, hidden_size=None
) -> Tuple[Tensor, List[int]]:
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
            res[i][: tensor.shape[0]] = tensor
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
            sub_output_tensor_list.append(outputs[sum : sum + len(each)])
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