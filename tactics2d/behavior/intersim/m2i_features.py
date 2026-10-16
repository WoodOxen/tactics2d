# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure NumPy M2I relation vector features implementation."""

import math
from typing import Dict, List, Tuple

import numpy as np

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.

HISTORY_FRAME_NUM = 11
VECTOR_DIM = 128
ROAD_SCALE = 0.03
ROAD_STRIDE = 10
VISIBLE_Y = 30.0
MAX_DIS = 80.0
MAX_LANE_POINTS = 2500
MAX_LANE_NUM = 1000


def build_relation_model_input(
    agent_trajectories: np.ndarray,
    tracks_type: np.ndarray,
    road_points: np.ndarray,
    road_types: np.ndarray,
    road_ids: np.ndarray,
    normalizer: Tuple[float, float, float],
    all_agent_ids: List[object],
    scenario_id: str,
) -> Dict[str, object]:
    """Assemble the relation predictor ``mapping`` dict from agent + road data.

    Args:
        agent_trajectories: ``(A, T, 7)`` world-frame trajectories (only the
            first 11 history frames are consumed).
        tracks_type: per-agent integer type.
        road_points / road_types / road_ids: world lane sample arrays.
        normalizer: ``(x, y, yaw)`` origin in world coordinates.
        all_agent_ids: agent ids with the reactor first (index 0).
        scenario_id: opaque scenario label carried through to the output.

    Returns:
        A mapping dict understood by the relation VectorNet forward.
    """

    origin_x, origin_y, origin_yaw = normalizer
    normalized_trajectories = _normalize_trajectories(
        np.asarray(agent_trajectories, dtype=np.float32), origin_x, origin_y, origin_yaw
    )
    agent_rows, agent_spans = _agent_vectors(normalized_trajectories, tracks_type)
    road_rows, road_spans = _road_vectors(
        road_points, road_types, road_ids, origin_x, origin_y, origin_yaw
    )
    matrix = np.concatenate([agent_rows, road_rows], axis=0).astype(np.float32)
    map_start = len(agent_spans)
    agent_count = len(agent_rows)
    spans = list(agent_spans) + [
        slice(int(road_span.start) + agent_count, int(road_span.stop) + agent_count)
        for road_span in road_spans
    ]

    return {
        "matrix": matrix,
        "polyline_spans": spans,
        "map_start_polyline_idx": map_start,
        "all_agent_ids": list(all_agent_ids),
        "scenario_id": scenario_id,
    }


def _agent_vectors(
    trajectories: np.ndarray, tracks_type: np.ndarray
) -> Tuple[np.ndarray, List[slice]]:
    """Vectorize normalized agent trajectories into relation feature rows.

    Args:
        trajectories: ``(A, T, 7)`` already normalized (x/y in agent frame).
        tracks_type: per-agent integer type (inference uses all ones).
    Returns:
        A ``(A * 10, 128)`` float32 vector array and one ``slice`` span per
        agent covering its 10 frame-to-frame vectors.
    """

    trajectories = np.asarray(trajectories, dtype=np.float32)
    agent_num = trajectories.shape[0]
    vector_count = HISTORY_FRAME_NUM - 1
    vectors = np.zeros((agent_num * vector_count, VECTOR_DIM), dtype=np.float32)
    spans: List[slice] = []

    for agent_index in range(agent_num):
        row_start = agent_index * vector_count
        for frame in range(vector_count):
            row = vectors[row_start + frame]
            row[0:2] = trajectories[agent_index, frame, :2]
            row[2:7] = trajectories[agent_index, frame, 2:7]
            row[20:22] = trajectories[agent_index, frame + 1, :2]
            row[22:27] = trajectories[agent_index, frame + 1, 2:7]
            row[30] = frame
            row[31 + frame] = 1.0
            row[50] = tracks_type[agent_index]
            row[51 + int(tracks_type[agent_index])] = 1.0
        spans.append(slice(row_start, row_start + vector_count))

    return vectors, spans


def _road_vectors(
    road_points: np.ndarray,
    road_types: np.ndarray,
    road_ids: np.ndarray,
    x: float,
    y: float,
    yaw: float,
) -> Tuple[np.ndarray, List[slice]]:
    """Vectorize grouped road (lane) points.

    Args:
        road_points: ``(N, 3)`` world xyz of sampled lane points.
        road_types: ``(N,)`` integer lane type per point (type < 20).
        road_ids: ``(N,)`` integer lane id per point.
        x, y, yaw: normalizer origin (world) and rotation angle.
    Returns:
        road vector rows and one ``slice`` span per lane.
    """

    points = _normalize_points(road_points, x, y, yaw)

    lane_id_to_ids: Dict[int, List[int]] = {}
    for index in range(len(points)):
        px = float(points[index, 0])
        py = float(points[index, 1])
        if math.sqrt(px * px + (py - VISIBLE_Y) ** 2) < MAX_DIS:
            lane_id = int(road_ids[index])
            if 0 <= lane_id < MAX_LANE_NUM:
                bucket = lane_id_to_ids.setdefault(lane_id, [])
                if len(bucket) < MAX_LANE_POINTS:
                    bucket.append(index)

    vectors_all: List[np.ndarray] = []
    spans: List[slice] = []
    offset = 0
    for lane_id in sorted(lane_id_to_ids):
        point_ids = lane_id_to_ids[lane_id]
        length = len(point_ids)
        type_id = int(road_types[point_ids[0]])

        vector_count = (length + ROAD_STRIDE - 1) // ROAD_STRIDE
        lane_vectors = np.zeros((vector_count, VECTOR_DIM), dtype=np.float32)
        for vector_index in range(vector_count):
            row = lane_vectors[vector_index]
            start = vector_index * ROAD_STRIDE
            for k in range(12):
                point_id = point_ids[min(start + k, length - 1)]
                row[2 * k] = float(points[point_id, 0]) * ROAD_SCALE
                row[2 * k + 1] = float(points[point_id, 1]) * ROAD_SCALE
            row[30 + type_id] = 1.0
            row[40] = float(start)
            row[41] = float(start) / float(length)
        vectors_all.append(lane_vectors)
        spans.append(slice(offset, offset + len(lane_vectors)))
        offset += len(lane_vectors)

    if not vectors_all:
        single = np.zeros((1, VECTOR_DIM), dtype=np.float32)
        single[0, 30] = 1.0
        vectors_all = [single]
        spans = [slice(0, 1)]

    return np.concatenate(vectors_all, axis=0), spans


def _normalize_trajectories(trajectories: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Normalize a ``(A, T, 7)`` trajectory array into the agent frame.

    Only the x/y channels are rotated; the remaining feature channels are zeroed.
    """

    out = np.zeros_like(trajectories)
    out[..., :2] = _normalize_points(trajectories[..., :2], x, y, yaw)
    return out


def _normalize_points(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Rotate and translate points into the agent frame."""

    points = np.asarray(points, dtype=np.float32)
    dx = points[..., 0] - x
    dy = points[..., 1] - y
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    out = np.empty_like(points)
    out[..., 0] = dx * cos_yaw - dy * sin_yaw
    out[..., 1] = dx * sin_yaw + dy * cos_yaw
    return out
