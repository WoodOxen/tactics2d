# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure NumPy M2I relation vector features implementation."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab. Pure-NumPy port of the M2I relation
# feature builder (utils_cython get_normalized/get_agents/get_roads) so the
# learned relation predictor can run inside Tactics2D without the upstream
# Cython extension.

HISTORY_FRAME_NUM = 11
RASTER_SIZE = 224
RASTER_CHANNELS = 150
VECTOR_DIM = 128
AGENT_SCALE = 1.0
ROAD_SCALE = 0.03
ROAD_STRIDE = 10
VISIBLE_Y = 30.0
MAX_DIS = 80.0
MAX_LANE_POINTS = 2500
MAX_LANE_NUM = 1000
MAX_VECTOR_NUM = 10000


def round_half_up(value: float) -> int:
    """Round half away from zero (mirrors the Cython raster cast)."""

    return int(math.floor(value + 0.5))


def normalize_points(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
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


def normalize_trajectories(trajectories: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Normalize a ``(A, T, 7)`` trajectory array like ``get_normalized``.

    Only the x/y channels are rotated; the feature channels (length, width,
    bbox yaw, velocities) are zeroed, matching the training-time Cython op.
    """

    out = np.zeros_like(trajectories)
    out[..., :2] = normalize_points(trajectories[..., :2], x, y, yaw)
    return out


def _in_image(x: int, y: int) -> bool:
    return 0 <= x < RASTER_SIZE and 0 <= y < RASTER_SIZE


def agent_vectors_and_raster(
    trajectories: np.ndarray,
    tracks_type: np.ndarray,
    image: np.ndarray,
) -> Tuple[np.ndarray, List[slice]]:
    """Vectorize normalized agent trajectories into relation feature rows.

    Args:
        trajectories: ``(A, T, 7)`` already normalized (x/y in agent frame).
        tracks_type: per-agent integer type (inference uses all ones).
        image: ``(224, 224, 150)`` int8 raster, mutated in place.

    Returns:
        A ``(A * 10, 128)`` float32 vector array and one ``slice`` span per
        agent covering its 10 frame-to-frame vectors.
    """

    trajectories = np.asarray(trajectories, dtype=np.float32)
    agent_num = trajectories.shape[0]
    vectors = np.zeros((agent_num * 10, VECTOR_DIM), dtype=np.float32)
    spans: List[slice] = []

    for agent_index in range(agent_num):
        for frame in range(HISTORY_FRAME_NUM):
            x_int = round_half_up(float(trajectories[agent_index, frame, 0])) + RASTER_SIZE // 2
            y_int = round_half_up(float(trajectories[agent_index, frame, 1])) + 56
            if _in_image(x_int, y_int):
                channel = frame if agent_index == 0 else 20 + frame
                image[x_int, y_int, channel] = 1

        row_start = agent_index * 10
        for frame in range(10):
            row = vectors[row_start + frame]
            row[0:2] = trajectories[agent_index, frame, :2]
            row[2:7] = trajectories[agent_index, frame, 2:7]
            row[20:22] = trajectories[agent_index, frame + 1, :2]
            row[22:27] = trajectories[agent_index, frame + 1, 2:7]
            row[30] = frame
            row[31 + frame] = 1.0
            row[50] = tracks_type[agent_index]
            type_int = int(tracks_type[agent_index])
            if 0 <= type_int < 19:
                row[51 + type_int] = 1.0
        spans.append(slice(row_start, row_start + 10))

    return vectors, spans


def road_vectors_and_raster(
    road_points: np.ndarray,
    road_types: np.ndarray,
    road_ids: np.ndarray,
    x: float,
    y: float,
    yaw: float,
    image: np.ndarray,
) -> Tuple[np.ndarray, List[slice], List[np.ndarray]]:
    """Vectorize and rasterize grouped road (lane) points.

    Args:
        road_points: ``(N, 3)`` world xyz of sampled lane points.
        road_types: ``(N,)`` integer lane type per point (type < 20).
        road_ids: ``(N,)`` integer lane id per point.
        x, y, yaw: normalizer origin (world) and rotation angle.
        image: raster mutated in place.

    Returns:
        road vector rows, one span per lane, and per-lane world-normalized
        point arrays (``polygons``, unscaled metres).
    """

    points = normalize_points(road_points, x, y, yaw)

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
    polygons: List[np.ndarray] = []
    for lane_id in sorted(lane_id_to_ids):
        point_ids = lane_id_to_ids[lane_id]
        length = len(point_ids)
        type_id = int(road_types[point_ids[0]])

        for point_id in point_ids:
            x_int = round_half_up(float(points[point_id, 0])) + RASTER_SIZE // 2
            y_int = round_half_up(float(points[point_id, 1])) + 56
            if _in_image(x_int, y_int) and 0 <= 40 + type_id < RASTER_CHANNELS:
                image[x_int, y_int, 40 + type_id] = 1

        vector_count = (length + ROAD_STRIDE - 1) // ROAD_STRIDE
        lane_vectors = np.zeros((vector_count, VECTOR_DIM), dtype=np.float32)
        for vector_index in range(vector_count):
            row = lane_vectors[vector_index]
            start = vector_index * ROAD_STRIDE
            for k in range(12):
                point_id = point_ids[min(start + k, length - 1)]
                row[2 * k] = float(points[point_id, 0]) * ROAD_SCALE
                row[2 * k + 1] = float(points[point_id, 1]) * ROAD_SCALE
            if 0 <= 30 + type_id < VECTOR_DIM:
                row[30 + type_id] = 1.0
            row[40] = float(start)
            row[41] = float(start) / float(length)
        vectors_all.append(lane_vectors)
        lane_points = np.asarray([points[point_id, :2] for point_id in point_ids], dtype=np.float32)
        polygons.append(lane_points)
        span_start = sum(len(v) for v in vectors_all) - len(lane_vectors)
        spans.append(slice(span_start, span_start + len(lane_vectors)))

    if not vectors_all:
        single = np.zeros((1, VECTOR_DIM), dtype=np.float32)
        single[0, 30] = 1.0
        vectors_all = [single]
        spans = [slice(0, 1)]
        polygons = [np.zeros((1, 2), dtype=np.float32)]

    return np.concatenate(vectors_all, axis=0), spans, polygons


def build_mapping(
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
    image = np.zeros((RASTER_SIZE, RASTER_SIZE, RASTER_CHANNELS), dtype=np.int8)
    trajectory_7 = np.asarray(agent_trajectories, dtype=np.float32)
    trajectory_7[:, :, :] = normalize_trajectories(
        trajectory_7[:, :, :], origin_x, origin_y, origin_yaw
    )
    agent_vectors, agent_spans = agent_vectors_and_raster(trajectory_7, tracks_type, image)
    road_vectors, road_spans, _ = road_vectors_and_raster(
        road_points, road_types, road_ids, origin_x, origin_y, origin_yaw, image
    )
    matrix = np.concatenate([agent_vectors, road_vectors], axis=0).astype(np.float32)
    map_start = len(agent_spans)
    agent_count = len(agent_vectors)
    spans = list(agent_spans) + [
        slice(int(road_span.start) + agent_count, int(road_span.stop) + agent_count)
        for road_span in road_spans
    ]
    polyline_spans = [slice(int(s.start), int(s.stop)) for s in spans]

    return {
        "matrix": matrix,
        "polyline_spans": polyline_spans,
        "map_start_polyline_idx": map_start,
        "image": image,
        "all_agent_ids": list(all_agent_ids),
        "scenario_id": scenario_id,
    }
