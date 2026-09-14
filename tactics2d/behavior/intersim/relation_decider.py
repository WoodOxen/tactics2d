# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Learned relation-direction arbitration."""

from typing import Optional

import numpy as np

from tactics2d.geometry import spatial

from .config import InterSimConfig

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.

# The checkpoint is loaded once per process: it is a read-only weight asset, so
# a module-level cache keeps repeated decider builds from re-reading ~125 MB.
_MODEL_CACHE = None


def load_model(config: InterSimConfig):
    """Return the relation predictor, loading its checkpoint on first use."""

    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from .relation_model import RelationVectorNet

        if not config.relation_model_path:
            raise ValueError("relation_mode='nn' requires config.relation_model_path.")
        _MODEL_CACHE = RelationVectorNet.from_checkpoint(config.relation_model_path)
    return _MODEL_CACHE


def road_graph(map_, cx: float, cy: float):
    """Sample nearby map lanes into roadgraph point/type/id arrays.

    Returns:
        A tuple ``(points, types, ids)`` of ``(N, 3)`` road points, their lane
        subtype codes, and their lane indices.
    """

    type_map = {"road": 2, "highway": 1, "bicycle_lane": 3}
    points = []
    types = []
    ids = []
    lane_id = 0
    for lane in map_.lanes.values():
        centerline = lane.centerline()
        if centerline is None or len(centerline.coords) < 2:
            continue
        coords = np.asarray(centerline.coords, dtype=float)
        lane_type = type_map.get(lane.subtype, 2) if lane.subtype else 2
        for index in range(0, len(coords), 2):
            x, y = coords[index]
            if abs(x - cx) <= 150.0 and -60.0 <= y - cy <= 170.0:
                points.append([x, y, 0.0])
                types.append(lane_type)
                ids.append(lane_id)
        lane_id += 1
    if not points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(types, dtype=np.int32).reshape(-1),
        np.asarray(ids, dtype=np.int32).reshape(-1),
    )


def make_decider(config: InterSimConfig, poses, dims, types, map_, ego_id, current: int):
    """Build the per-frame learned direction arbiter for relation_mode="nn".

    The model's own geometric detector still builds the conflict graph; this
    closure only decides the direction of each vehicle-vehicle pair, holding
    the .bin predictor (with the upstream rule prefilter) and returning
    ``None`` when it is not confident so the geometric edge is kept.
    """

    import torch

    from tactics2d.participant.element import Vehicle

    ego_pose = poses[ego_id][current]
    road_points, road_types, road_ids = road_graph(map_, float(ego_pose[0]), float(ego_pose[1]))
    is_vehicle = {agent_id: types[agent_id] is Vehicle for agent_id in poses}
    model = load_model(config)
    device = torch.device("cpu")
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
    """Whether the reactor must yield, using the upstream rule prefilter.

    Returns ``True``/``False`` when a rule or a confident predictor decides,
    and ``None`` when the predictor is not confident (the caller then keeps
    the geometric edge). Mirrors upstream: the same-direction (< 30 deg) rule
    yields for whoever faces the other's body, a non-vehicle partner is never
    forced to yield, and the .bin predictor only arbitrates the rest.
    """

    from . import m2i_features as features

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

    def window_7(agent_id):
        out = np.zeros((11, 7), dtype=np.float32)
        for j in range(11):
            index = current + j
            if index >= config.scenario_steps or poses[agent_id][index, 0] == -1:
                continue
            out[j, 0] = poses[agent_id][index, 0]
            out[j, 1] = poses[agent_id][index, 1]
            out[j, 4] = poses[agent_id][index, 3]
        return out

    reactor = window_7(reactor_id)
    influencer = window_7(influencer_id)
    angle = -float(reactor[0, 4]) + np.pi / 2
    x0, y0 = float(reactor[0, 0]), float(reactor[0, 1])
    stacked = np.stack([reactor, influencer], axis=0)
    mapping = features.build_mapping(
        stacked,
        np.ones(2, dtype=np.int32),
        road_points,
        road_types,
        road_ids,
        (x0, y0, angle),
        [reactor_id, influencer_id],
        str(current),
    )
    scores = model.forward(
        mapping["matrix"], mapping["polyline_spans"], mapping["map_start_polyline_idx"], device
    )[0]
    if float(np.max(scores)) <= 0.5:
        return None
    return bool(np.argmax(scores) == 1)
