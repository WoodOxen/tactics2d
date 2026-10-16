# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Learned relation-direction arbitration."""

import math
from dataclasses import dataclass
from typing import Dict, Tuple
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
_ROAD_GRID_SIZE_M = 64.0
_RELATION_BATCH_SIZE = 16


@dataclass(frozen=True)
class _RoadSamples:
    """Cached road points and a coarse spatial lookup grid."""

    points: np.ndarray
    types: np.ndarray
    ids: np.ndarray
    grid: Dict[Tuple[int, int], np.ndarray]


def _road_samples(map_) -> _RoadSamples:
    """Build the immutable road-point cache for one map."""

    cached = _ROAD_SAMPLE_CACHE.get(map_)
    if cached is not None:
        return cached

    type_map = {"road": 2, "highway": 1, "bicycle_lane": 3}
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

    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    buckets: Dict[Tuple[int, int], list] = {}
    for index, point in enumerate(point_array):
        key = (
            math.floor(float(point[0]) / _ROAD_GRID_SIZE_M),
            math.floor(float(point[1]) / _ROAD_GRID_SIZE_M),
        )
        buckets.setdefault(key, []).append(index)
    cached = _RoadSamples(
        points=point_array,
        types=np.asarray(types, dtype=np.int32),
        ids=np.asarray(ids, dtype=np.int32),
        grid={key: np.asarray(indices, dtype=np.int64) for key, indices in buckets.items()},
    )
    _ROAD_SAMPLE_CACHE[map_] = cached
    return cached


def _load_relation_model(config: InterSimConfig):
    """Return the cached relation predictor and its automatically selected device."""

    if not config.relation_model_path:
        raise ValueError("relation_mode='nn' requires config.relation_model_path.")

    path = str(config.relation_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = (path, str(device))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = RelationVectorNet.from_checkpoint(path, device=str(device))
    return _MODEL_CACHE[key], device


def sample_road_features(map_, cx: float, cy: float):
    """Sample nearby map lanes into roadgraph point/type/id arrays.

    Returns:
        A tuple ``(points, types, ids)`` of ``(N, 3)`` road points, their lane
        subtype codes, and their lane indices.
    """

    cached = _road_samples(map_)
    all_points, all_types, all_ids = cached.points, cached.types, cached.ids
    if not len(all_points):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
    min_x, max_x = cx - 150.0, cx + 150.0
    min_y, max_y = cy - 60.0, cy + 170.0
    indices = []
    for grid_x in range(
        math.floor(min_x / _ROAD_GRID_SIZE_M), math.floor(max_x / _ROAD_GRID_SIZE_M) + 1
    ):
        for grid_y in range(
            math.floor(min_y / _ROAD_GRID_SIZE_M), math.floor(max_y / _ROAD_GRID_SIZE_M) + 1
        ):
            bucket = cached.grid.get((grid_x, grid_y))
            if bucket is not None:
                indices.extend(bucket.tolist())
    if not indices:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
    candidate_indices = np.asarray(sorted(indices), dtype=np.int64)
    candidate_points = all_points[candidate_indices]
    keep = (
        (np.abs(candidate_points[:, 0] - cx) <= 150.0)
        & (candidate_points[:, 1] - cy >= -60.0)
        & (candidate_points[:, 1] - cy <= 170.0)
    )
    selected = candidate_indices[keep]
    return all_points[selected], all_types[selected], all_ids[selected]


def build_relation_decider(config: InterSimConfig, poses, types, map_, ego_id, current: int):
    """Build the per-frame learned direction arbiter for relation_mode="nn".

    The closure only decides the direction of each vehicle-vehicle pair, and
    returns ``None`` when the predictor is not confident.
    """

    ego_pose = poses[ego_id][current]
    road_points, road_types, road_ids = sample_road_features(
        map_, float(ego_pose[0]), float(ego_pose[1])
    )
    is_vehicle = {agent_id: types[agent_id] is Vehicle for agent_id in poses}
    model, device = _load_relation_model(config)
    return _RelationDecider(
        config, poses, current, is_vehicle, road_points, road_types, road_ids, model, device
    )


class _RelationDecider:
    """Cache and batch learned decisions within one replanning step."""

    def __init__(
        self, config, poses, current, is_vehicle, road_points, road_types, road_ids, model, device
    ):
        self.config = config
        self.poses = poses
        self.current = current
        self.is_vehicle = is_vehicle
        self.road_points = road_points
        self.road_types = road_types
        self.road_ids = road_ids
        self.model = model
        self.device = device
        self.cache = {}

    def __call__(self, reactor_id, influencer_id):
        key = (reactor_id, influencer_id)
        if key not in self.cache:
            self.decide_many([key])
        return self.cache[key]

    def decide_many(self, pairs):
        """Resolve uncached directed pairs with one neural-network batch."""

        mappings = []
        mapping_keys = []
        for key in pairs:
            if key in self.cache:
                continue
            reactor_id, influencer_id = key
            if not (self.is_vehicle[reactor_id] and self.is_vehicle[influencer_id]):
                self.cache[key] = None
                continue
            resolved, value = _rule_decision(reactor_id, influencer_id, self.poses, self.current)
            if resolved:
                self.cache[key] = value
                continue
            mappings.append(
                _relation_mapping(
                    self.config,
                    reactor_id,
                    influencer_id,
                    self.poses,
                    self.current,
                    self.road_points,
                    self.road_types,
                    self.road_ids,
                )
            )
            mapping_keys.append(key)

        if not mappings:
            return
        score_batches = [
            self.model.forward_many(mappings[start : start + _RELATION_BATCH_SIZE], self.device)
            for start in range(0, len(mappings), _RELATION_BATCH_SIZE)
        ]
        scores = np.concatenate(score_batches, axis=0)
        for key, mapping, pair_scores in zip(mapping_keys, mappings, scores):
            # Re-evaluate borderline pairs through the historical single-item
            # path so batching cannot flip a relation direction.
            if abs(float(pair_scores[0]) - float(pair_scores[1])) <= 1e-5:
                pair_scores = self.model.forward(
                    mapping["matrix"],
                    mapping["polyline_spans"],
                    mapping["map_start_polyline_idx"],
                    self.device,
                )[0]
            self.cache[key] = _scores_to_decision(pair_scores)


def _rule_decision(reactor_id, influencer_id, poses, current):
    """Return whether a pair is rule-resolved and its decision."""

    reactor_pose = poses[reactor_id][current]
    influencer_pose = poses[influencer_id][current]
    if reactor_pose[0] == -1 or influencer_pose[0] == -1:
        return True, None
    yaw = float(reactor_pose[3])
    yaw_diff = abs(spatial.normalize_angle(yaw - float(influencer_pose[3])))
    if yaw_diff < np.pi / 6.0:
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        offset = influencer_pose[:2] - reactor_pose[:2]
        return True, bool(float(np.dot(offset, heading)) > 0)
    return False, None


def _relation_mapping(
    config, reactor_id, influencer_id, poses, current, road_points, road_types, road_ids
):
    """Build one relation-model mapping after rule prefiltering."""

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
    return features.build_relation_model_input(
        stacked,
        np.ones(2, dtype=np.int32),
        road_points,
        road_types,
        road_ids,
        (origin_x, origin_y, angle),
        [reactor_id, influencer_id],
        str(current),
    )


def _scores_to_decision(scores):
    """Convert two-class relation scores to the released decision."""

    if float(np.max(scores)) <= 0.5:
        return None
    return bool(np.argmax(scores) == 1)
