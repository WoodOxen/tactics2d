# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Learned relation-direction arbitration."""

from typing import Optional
from weakref import WeakKeyDictionary

import numpy as np
import torch

from tactics2d.geometry import spatial
from tactics2d.participant.element import Vehicle

from . import m2i_features as features
from .config import InterSimConfig
from .relation_model import RelationVectorNet

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.

# Process-wide cache: the checkpoints are read-only ~125 MB weight assets, so
# they are kept alive per path for the lifetime of the process.
_MODEL_CACHE = {}
_ROAD_SAMPLE_CACHE = WeakKeyDictionary()


def load_model(config: InterSimConfig):
    """Return the cached relation predictor and its automatically selected device."""

    if not config.relation_model_path:
        raise ValueError("relation_mode='nn' requires config.relation_model_path.")

    path = str(config.relation_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = (path, str(device))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = RelationVectorNet.from_checkpoint(path, device=str(device))
    return _MODEL_CACHE[key], device


def road_graph(map_, cx: float, cy: float):
    """Sample nearby map lanes into roadgraph point/type/id arrays.

    Returns:
        A tuple ``(points, types, ids)`` of ``(N, 3)`` road points, their lane
        subtype codes, and their lane indices.
    """

    type_map = {"road": 2, "highway": 1, "bicycle_lane": 3}
    cached = _ROAD_SAMPLE_CACHE.get(map_)
    if cached is None:
        points = []
        types = []
        ids = []
        lane_id = 0
        for lane in map_.lanes.values():
            centerline = lane.centerline()
            if centerline is None or len(centerline.coords) < 2:
                continue
            coords = np.asarray(centerline.coords, dtype=float)[::2]
            points.extend(np.column_stack([coords, np.zeros(len(coords))]))
            lane_type = type_map.get(lane.subtype, 2) if lane.subtype else 2
            types.extend([lane_type] * len(coords))
            ids.extend([lane_id] * len(coords))
            lane_id += 1
        cached = (
            np.asarray(points, dtype=np.float32).reshape(-1, 3),
            np.asarray(types, dtype=np.int32),
            np.asarray(ids, dtype=np.int32),
        )
        _ROAD_SAMPLE_CACHE[map_] = cached
    all_points, all_types, all_ids = cached
    if not len(all_points):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
    keep = (
        (np.abs(all_points[:, 0] - cx) <= 150.0)
        & (all_points[:, 1] - cy >= -60.0)
        & (all_points[:, 1] - cy <= 170.0)
    )
    return all_points[keep], all_types[keep], all_ids[keep]


def make_decider(config: InterSimConfig, poses, types, map_, ego_id, current: int):
    """Build the per-frame learned direction arbiter for relation_mode="nn".

    The closure only decides the direction of each vehicle-vehicle pair, and
    returns ``None`` when the predictor is not confident.
    """

    ego_pose = poses[ego_id][current]
    road_points, road_types, road_ids = road_graph(map_, float(ego_pose[0]), float(ego_pose[1]))
    is_vehicle = {agent_id: types[agent_id] is Vehicle for agent_id in poses}
    model, device = load_model(config)
    cache = {}

    def decide(reactor_id, influencer_id):
        if not (is_vehicle.get(reactor_id, True) and is_vehicle.get(influencer_id, True)):
            return None
        key = (reactor_id, influencer_id)
        if key not in cache:
            cache[key] = edge_yields(
                config,
                reactor_id,
                influencer_id,
                poses,
                current,
                is_vehicle,
                road_points,
                road_types,
                road_ids,
                model,
                device,
            )
        return cache[key]

    return decide


def edge_yields(
    config: InterSimConfig,
    reactor_id,
    influencer_id,
    poses,
    current: int,
    is_vehicle,
    road_points,
    road_types,
    road_ids,
    model,
    device,
) -> Optional[bool]:
    """Decide whether the reactor must yield, applying the rule prefilter first.

    Same-direction (< 30 deg) and non-vehicle pairs are decided by rule; the
    .bin predictor arbitrates the rest, and ``None`` marks low confidence.
    """

    reactor_pose = poses[reactor_id][current]
    influencer_pose = poses[influencer_id][current]
    if reactor_pose[0] == -1 or influencer_pose[0] == -1:
        return None

    yaw = float(reactor_pose[3])
    target_yaw = float(influencer_pose[3])
    yaw_diff = abs(spatial.normalize_angle(yaw - target_yaw))
    if yaw_diff < np.pi / 6.0:
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        offset = influencer_pose[:2] - reactor_pose[:2]
        return bool(float(np.dot(offset, heading)) > 0)
    if not is_vehicle.get(influencer_id, True):
        return True

    def feature_window(agent_id):
        out = np.zeros((11, 7), dtype=np.float32)
        for j in range(11):
            index = current + j
            if index >= config.scenario_steps or poses[agent_id][index, 0] == -1:
                continue
            out[j, 0] = poses[agent_id][index, 0]
            out[j, 1] = poses[agent_id][index, 1]
            out[j, 4] = poses[agent_id][index, 3]
        return out

    reactor = feature_window(reactor_id)
    influencer = feature_window(influencer_id)
    angle = -float(reactor[0, 4]) + np.pi / 2
    origin_x, origin_y = float(reactor[0, 0]), float(reactor[0, 1])
    stacked = np.stack([reactor, influencer], axis=0)
    mapping = features.build_mapping(
        stacked,
        np.ones(2, dtype=np.int32),
        road_points,
        road_types,
        road_ids,
        (origin_x, origin_y, angle),
        [reactor_id, influencer_id],
        str(current),
    )
    scores = model.forward(
        mapping["matrix"], mapping["polyline_spans"], mapping["map_start_polyline_idx"], device
    )[0]
    if float(np.max(scores)) <= 0.5:
        return None
    return bool(np.argmax(scores) == 1)
