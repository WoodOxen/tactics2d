# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Map polyline tokenizer for the SMART network."""

import math
from typing import List, Optional

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.linalg import get_blas_funcs
from shapely.geometry import Polygon

from tactics2d.map.query import SemanticMapQuery

from .config import SmartConfig, load_codebook
from .schema import LightType, PointType, PolygonType, SmartMapTokens

# WOMD ``Lane.Type`` shifted by one to SMART's polygon type.
_LANE_POLYGON_TYPE = {
    "highway": PolygonType.VEHICLE,
    "road": PolygonType.VEHICLE,
    "bicycle_lane": PolygonType.BIKE,
}

# Inverse of ``WOMDParser._LANE_TYPE_MAPPING``, applied before the table above.
# WOMD type 0 and type 2 both resolve to "road", so a type-0 lane reads as one.
_LANE_WOMD_TYPE = {"highway": 1, "road": 2, "bicycle_lane": 3}

# WOMD ``RoadLine`` type to SMART's point type. Upstream indexes a table that is
# scrambled against the enum names, so the codes below are the table entries.
_ROADLINE_POINT_TYPE = {
    ("virtual", None, None): PointType.UNKNOWN,
    ("line_thin", "dashed", "white"): PointType.DASHED_WHITE,
    ("line_thin", "solid", "white"): PointType.SOLID_WHITE,
    ("line_thin", "solid_solid", "white"): PointType.DOUBLE_DASH_WHITE,
    ("line_thin", "dashed", "yellow"): PointType.DASHED_YELLOW,
    ("line_thin", "dashed_dashed", "yellow"): PointType.DOUBLE_DASH_YELLOW,
    ("line_thin", "solid", "yellow"): PointType.SOLID_YELLOW,
    ("line_thin", "solid_solid", "yellow"): PointType.DOUBLE_SOLID_YELLOW,
    (None, "dashed", "yellow"): PointType.DASH_SOLID_YELLOW,
}

# ``WOMDParser._TRAFFIC_SIGNAL_STATE_MAPPING`` values to SMART's light table.
_LIGHT_TYPE_BY_STATE = {
    "stop": LightType.STOP,
    "arrow_stop": LightType.STOP,
    "flashing_stop": LightType.STOP,
    "go": LightType.GO,
    "arrow_go": LightType.GO,
    "caution": LightType.CAUTION,
    "arrow_caution": LightType.CAUTION,
    "flashing_caution": LightType.CAUTION,
    "unknown": LightType.UNKNOWN,
}

_ARC_DISTANCE = 0.5
# A token spans five metres, sampled every 0.5 m; tokens start five metres apart.
_POLYLINE_SIZE = int(5.0 / _ARC_DISTANCE)  # 10 sample steps per token
_MAP_TOKEN_POINTS = _POLYLINE_SIZE + 1  # 11 points per token window
_MAP_SAMPLE_COUNT = 3
# Index step between the window's three representative points (0, 5 and 10).
_WINDOW_STRIDE = _POLYLINE_SIZE // (_MAP_SAMPLE_COUNT - 1)
# Minimum run length; a leftover shorter than ``_MAP_SAMPLE_COUNT`` is dropped.
_MIN_RUN_POINTS = 2


# Resolved on first use.
_NRM2 = None


def _nrm2():
    """Return the cached BLAS 2-norm routine for float32 input.

    Returns:
        The BLAS ``nrm2`` routine for float32 input.
    """

    global _NRM2
    if _NRM2 is None:
        _NRM2 = get_blas_funcs("nrm2", dtype=np.float32)
    return _NRM2


def _cumulative_knots(positions: np.ndarray) -> np.ndarray:
    """Return the cumulative-distance knots of a polyline, upstream's rounding.

    Upstream sums ``scipy.spatial.distance.euclidean`` results, which route
    through BLAS; ``nrm2`` is called directly to keep the knots bit-equal.

    Args:
        positions (np.ndarray): Polyline points of shape ``(N, 2)``, float32.

    Returns:
        Cumulative distances of shape ``(N,)``, dtype ``float64``, from zero.
    """

    nrm2 = _nrm2()
    knots = [0.0]
    for i in range(1, positions.shape[0]):
        knots.append(knots[-1] + nrm2(positions[i] - positions[i - 1]))
    return np.array(knots, dtype=np.float64)


def _forward_headings(coords: np.ndarray) -> np.ndarray:
    """Return the heading of every segment but the last, wrapped as upstream wraps it.

    Callers pass every vertex; the heading attached to a point is the direction
    of the segment leaving it.

    Args:
        coords (np.ndarray): Every vertex of the polyline, shape ``(N, 2)``,
            dtype ``float32``.

    Returns:
        Headings of shape ``(N - 1,)``, dtype ``float32``.
    """

    deltas = np.diff(coords, axis=0)
    theta = np.arctan2(deltas[:, 1], deltas[:, 0])
    return np.float32(-math.pi) + (theta + np.float32(math.pi)) % np.float32(2 * math.pi)


def _split_runs(positions: np.ndarray, headings: np.ndarray) -> List[List[int]]:
    """Split a polyline into runs at upstream's heading breaks.

    Cut on sharp turns or big gaps; ``headings[1]`` is not a typo for
    ``headings[i - 1]`` -- upstream compares against the second point, and the
    pretrained split depends on it.

    Args:
        positions (np.ndarray): Polyline points of shape ``(N, 2)``, float32.
        headings (np.ndarray): Headings of shape ``(N,)``, float32, one per position.

    Returns:
        A list of point-index lists, one per run, in polyline order; the segment
        spanning a cut is dropped.
    """

    nrm2 = _nrm2()
    runs = [[0]]
    for i in range(1, positions.shape[0]):
        gap = nrm2(positions[i] - positions[i - 1])
        largest = max(headings[i], headings[i - 1])
        smallest = min(headings[1], headings[i - 1])
        turn = min(abs(largest - smallest), abs(largest - smallest + math.pi))
        if (turn > math.pi / 4 and gap > 3.0) or (turn > math.pi / 8 and gap > 3.0):
            runs.append([i])
        elif turn > 0.1 and gap > 3.0:
            runs.append([i])
        elif gap > 10.0:
            runs.append([i])
        else:
            runs[-1].append(i)
    return runs


def _resample_polyline(positions: np.ndarray, headings: np.ndarray) -> Optional[np.ndarray]:
    """Resample a polyline onto SMART's 0.5 m grid and cut it into tokens.

    Runs are resampled every 0.5 m with the end point appended, then cut into
    eleven-point windows stepping ten; each contributes points 0, 5 and 10 plus
    the first heading. ``torch.atan2`` rounds differently on the transposed
    layout, so the windowing mirrors upstream's tensor operations.

    Args:
        positions (np.ndarray): The polyline's points without its last vertex, ``(N, 2)`` float32.
        headings (np.ndarray): Headings from ``_forward_headings``, shape ``(N,)`` float32.

    Returns:
        Tokens of shape ``(K, 3, 3)``, dtype ``float64``, holding ``[x, y,
        theta]`` rows, or None when no window survives.
    """

    tokens = []
    for run in _split_runs(positions, headings):
        if len(run) < _MIN_RUN_POINTS:
            continue
        points = positions[run]
        knots = _cumulative_knots(points)
        query = np.arange(0.0, knots[-1], _ARC_DISTANCE)
        query = np.concatenate([query, knots[[-1]]])
        new_x = interp1d(knots, points[:, 0])(query)
        new_y = interp1d(knots, points[:, 1])(query)
        resampled = torch.from_numpy(np.vstack((new_x, new_y)).T)
        count = resampled.shape[0]

        theta = torch.arctan2(
            resampled[1:, 1] - resampled[:-1, 1], resampled[1:, 0] - resampled[:-1, 0]
        )
        theta = torch.cat([theta, theta[-1:]], dim=-1)[..., None]
        tagged = torch.cat([resampled, theta], dim=-1)

        if count >= _MAP_TOKEN_POINTS:
            padding = (count - _MAP_TOKEN_POINTS) % _POLYLINE_SIZE
            final_index = (count - _MAP_TOKEN_POINTS) // _POLYLINE_SIZE + 1
            windows = tagged.unfold(0, _MAP_TOKEN_POINTS, _POLYLINE_SIZE)
            windows = windows.transpose(1, 2)[:, ::_WINDOW_STRIDE, :]
        else:
            padding = count
            final_index = 0
            windows = None

        if padding >= _MAP_SAMPLE_COUNT:
            tail = tagged[final_index * _POLYLINE_SIZE :]
            picked = tail[torch.linspace(0, tail.shape[0] - 1, steps=_MAP_SAMPLE_COUNT).long()]
            picked = picked.unsqueeze(0)
            windows = picked if windows is None else torch.cat([windows, picked], dim=0)

        if windows is not None:
            tokens.append(windows.numpy())

    if not tokens:
        return None
    return np.concatenate(tokens, axis=0)


def _coordinates(geometry) -> np.ndarray:
    """Return a map element's vertices in the order the parser read them.

    Args:
        geometry: A shapely ``LineString`` or ``Polygon``.

    Returns:
        Vertices of shape ``(N, 2)``, dtype ``float32``.
    """

    if isinstance(geometry, Polygon):
        coords = np.asarray(geometry.exterior.coords, dtype=np.float32)
        if coords.shape[0] >= 2 and np.array_equal(coords[0], coords[-1]):
            coords = coords[:-1]
        return coords
    return np.asarray(geometry.coords, dtype=np.float32)


class MapTokenizer:
    """Quantize the polylines of a tactics2d map into the map codebook.

    Polylines are read from the native map, so a feature the parser drops cannot
    be recovered.

    Attributes:
        config (SmartConfig): The tokenizer configuration.
    """

    def __init__(self, config: Optional[SmartConfig] = None):
        """Initialize the tokenizer and load the map codebook.

        Args:
            config (Optional[SmartConfig], optional): Tokenizer configuration.
                Defaults to ``SmartConfig()``.

        Raises:
            FileNotFoundError: If the configured codebook does not exist.
        """

        self.config = config if config is not None else SmartConfig()
        self._load_codebook()

    def _load_codebook(self) -> None:
        """Load the map codebook's sampling points.

        ``sample_pt`` holds the three window points (indices 0, 5 and 10)
        upstream matches against, read as a contiguous float32 tensor.
        """

        codebook = load_codebook(self.config.map_codebook, "map codebook")

        sample_pt = np.asarray(codebook["sample_pt"], dtype=np.float32)
        if sample_pt.shape != (self.config.map_token_size, _MAP_SAMPLE_COUNT, 2):
            raise ValueError(
                f"Map codebook expects shape ({self.config.map_token_size}, "
                f"{_MAP_SAMPLE_COUNT}, 2), got {sample_pt.shape}."
            )
        self._sample_pt = torch.from_numpy(sample_pt)

    def _nearest_token(self, traj_pos: torch.Tensor, traj_theta: torch.Tensor) -> torch.Tensor:
        """Return the codebook entry closest to each window.

        Squared distances are summed in float32 with upstream's operand layout;
        the summing order decides ties between near-equal candidates.

        Args:
            traj_pos (torch.Tensor): Window points in the world frame, shape
                ``(N, points, 2)``, dtype ``float32``.
            traj_theta (torch.Tensor): Window headings, shape ``(N,)``, dtype
                ``float32``.

        Returns:
            Codebook indices of shape ``(N,)``, dtype ``int64``.
        """

        count = traj_pos.shape[0]
        cos, sin = traj_theta.cos(), traj_theta.sin()
        rotation = traj_theta.new_zeros(count, 2, 2)
        rotation[:, 0, 0] = cos
        rotation[:, 0, 1] = -sin
        rotation[:, 1, 0] = sin
        rotation[:, 1, 1] = cos

        local = torch.bmm(traj_pos - traj_pos[:, 0:1], rotation)
        sample_pt = self._sample_pt.to(traj_pos.device)
        distance = torch.sum((sample_pt[None] - local.unsqueeze(1)) ** 2, dim=(-2, -1))
        return torch.argmin(distance, dim=1)

    def _sources(self, map_, frame_ms: int) -> List[tuple]:
        """Collect the map's tokenizable polylines in upstream's group order.

        Order is lanes, road edges, road markings, then crosswalks. A lane whose
        light has no record at exactly ``frame_ms`` reads as unknown.

        Args:
            map_ (Map): The map to read.
            frame_ms (int): The scenario's own timestamp for frame
                ``history_steps``, in milliseconds.

        Returns:
            A list of ``(coordinates, point_type, polygon_type, light_type)``
            tuples, coordinates being ``(N, 2)`` float32.
        """

        query = SemanticMapQuery(map_)
        sources = []

        for lane in map_.lanes.values():
            centerline = lane.centerline()
            if centerline is None:
                continue
            state = query.get_traffic_light_state(lane.id_, frame_ms)
            light_type = LightType.UNKNOWN
            if state is not None and int(state.get("time_ms", -1)) == int(frame_ms):
                light_type = _LIGHT_TYPE_BY_STATE.get(state.get("state"), LightType.UNKNOWN)
            sources.append(
                (
                    _coordinates(centerline),
                    PointType.CENTERLINE,
                    _LANE_POLYGON_TYPE.get(lane.subtype, PolygonType.VEHICLE),
                    light_type,
                )
            )

        for want_edge in (True, False):
            for roadline in map_.roadlines.values():
                is_edge = roadline.type_ == "road_border"
                if is_edge != want_edge:
                    continue
                if is_edge:
                    point_type = PointType.EDGE
                else:
                    point_type = _ROADLINE_POINT_TYPE.get(
                        (roadline.type_, roadline.subtype, roadline.color), PointType.UNKNOWN
                    )
                sources.append(
                    (
                        _coordinates(roadline.geometry),
                        point_type,
                        PolygonType.VEHICLE,
                        LightType.UNKNOWN,
                    )
                )

        for area in map_.areas.values():
            if area.subtype != "crosswalk":
                continue
            sources.append(
                (
                    _coordinates(area.geometry),
                    PointType.CROSSWALK,
                    PolygonType.PEDESTRIAN,
                    LightType.UNKNOWN,
                )
            )

        return sources

    def build(self, map_, frame_ms: Optional[int] = None) -> SmartMapTokens:
        """Tokenize a map into windows of eleven points on a 0.5 m grid.

        The whole map is tokenized, as upstream does; callers wanting a bounded
        scene should prune ``map_``.

        Args:
            map_ (Map): The map to tokenize.
            frame_ms (int, optional): Timestamp whose traffic-light state is
                encoded, in milliseconds. Defaults to ``history_steps *
                step_ms``; pass the scenario's own timestamp if it is off the
                nominal 10 Hz grid.

        Returns:
            The map tokens, all empty when no polyline produced a window.
        """

        if frame_ms is None:
            frame_ms = self.config.history_steps * self.config.step_ms

        windows = []
        point_types = []
        polygon_types = []
        light_types = []
        polyline_ids = []
        for index, (coords, point_type, polygon_type, light_type) in enumerate(
            self._sources(map_, frame_ms)
        ):
            if coords.shape[0] <= 1:
                continue
            positions = coords[:-1]
            if positions.shape[0] <= _MIN_RUN_POINTS:
                continue
            tokens = _resample_polyline(positions, _forward_headings(coords))
            if tokens is None:
                continue
            windows.append(tokens)
            point_types.append(np.full(tokens.shape[0], point_type, dtype=np.int64))
            polygon_types.append(np.full(tokens.shape[0], polygon_type, dtype=np.int64))
            light_types.append(np.full(tokens.shape[0], light_type, dtype=np.int64))
            polyline_ids.append(np.full(tokens.shape[0], index, dtype=np.int64))

        if not windows:
            return self._empty_tokens()

        traj_pos = torch.from_numpy(np.concatenate(windows, axis=0))
        count = traj_pos.shape[0]
        # Match on the float32 cast; ``traj_pos`` stays float64 for the harness.
        local = traj_pos.to(torch.float32)
        # ``pt_position`` is (x, y, 0), so the window heading is not carried in.
        pt_position = torch.cat([local[:, 0, :2], torch.zeros(count, 1)], dim=-1)

        return SmartMapTokens(
            pt_position=pt_position,
            pt_orientation=local[:, 0, 2],
            pt_token_idx=self._nearest_token(local[:, :, :2], local[:, 0, 2]),
            pt_type=torch.from_numpy(np.concatenate(point_types)),
            pl_type=torch.from_numpy(np.concatenate(polygon_types)),
            pt_side=torch.zeros(count, dtype=torch.int64),
            light_type=torch.from_numpy(np.concatenate(light_types)),
            traj_pos=traj_pos[:, :, :2].contiguous(),
            traj_theta=traj_pos[:, 0, 2].contiguous(),
            token2pl=torch.stack(
                [torch.arange(count), torch.from_numpy(np.concatenate(polyline_ids))]
            ),
        )

    def _empty_tokens(self) -> SmartMapTokens:
        """Return the empty token set a map with no windows produces."""

        return SmartMapTokens(
            pt_position=torch.zeros(0, 3),
            pt_orientation=torch.zeros(0),
            pt_token_idx=torch.zeros(0, dtype=torch.int64),
            pt_type=torch.zeros(0, dtype=torch.int64),
            pl_type=torch.zeros(0, dtype=torch.int64),
            pt_side=torch.zeros(0, dtype=torch.int64),
            light_type=torch.zeros(0, dtype=torch.int64),
            traj_pos=torch.zeros(0, _MAP_SAMPLE_COUNT, 2),
            traj_theta=torch.zeros(0),
            token2pl=torch.zeros(2, 0, dtype=torch.int64),
        )
