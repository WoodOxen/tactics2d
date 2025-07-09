##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: config.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9


TRAJGEN_CONFIG = {
    # if control_steering is true
    "control_steering": True,
    "route_feature_num": 9,
    "ego_feature_num": 4,
    "action_size": 2,
    # if control_steering is false
    # "control_steering": False,
    # "route_feature_num": 8,
    # "ego_feature_num": 5,
    # "action_size": 1,
    # other environment features
    "npc_num": 5,
    "npc_feature_num": 7,
    # hyper-parameters of the RL network
    "encoder_size": [64, 64],
    "decoder_size": [256, 256],
    "features_per_head": 64,
    "feature_head": 1,
    # hyper-parameters of the RL agent
    "lr_actor": 1e-5,
    "lr_critic": 5e-5,
    "td3_delay": 2,
    "batch_size": 256,
    "tau": 1e-3,
    "gamma": 0.99,
}
