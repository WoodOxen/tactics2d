# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preview helpers for the browser frontend."""

from __future__ import annotations

import logging
import re
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

LEVELX_DATASETS = ("highD", "inD", "rounD", "exiD", "uniD")
LEVELX_FRAME_STEP_MS = 40

_LEVELX_CANONICAL = {name.lower(): name for name in LEVELX_DATASETS}
_MAP_CONFIG_NAMES = (
    "HIGHD_MAP_CONFIG",
    "IND_MAP_CONFIG",
    "ROUND_MAP_CONFIG",
    "EXID_MAP_CONFIG",
    "UNID_MAP_CONFIG",
)


@dataclass
class DatasetPreviewResult:
    """Summary of a streamed dataset preview."""

    base_url: str
    sensor_id: str
    actual_time_range: tuple[int, int]
    sent_frames: int
    dropped_frames: int


def canonical_levelx_dataset(dataset: str) -> str:
    """Return the canonical LevelX dataset spelling."""

    try:
        return _LEVELX_CANONICAL[dataset.lower()]
    except KeyError as exc:
        raise KeyError(
            f"{dataset} is not available. Choose one of: {', '.join(LEVELX_DATASETS)}."
        ) from exc


def extract_levelx_file_id(file: str | int) -> int:
    """Extract a LevelX recording id from an integer or filename."""

    if isinstance(file, int):
        return file

    match = re.search(r"\d+", str(file))
    if match is None:
        raise ValueError(f"Cannot extract a LevelX file id from {file!r}.")

    return int(match.group(0))


def iter_map_configs() -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield registered map configs without importing them at module import time."""

    from tactics2d.map import map_config

    for config_name in _MAP_CONFIG_NAMES:
        configs = getattr(map_config, config_name)
        yield from configs.items()


def resolve_levelx_map_config(
    dataset: str, file: str | int, map_config: str | None = None, osm_path: Path | None = None
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve the map config for a LevelX recording when possible."""

    candidates = []
    if map_config:
        candidates.append(map_config)
    if osm_path is not None:
        candidates.append(osm_path.stem)

    all_configs = dict(iter_map_configs())
    lower_to_name = {name.lower(): name for name in all_configs}
    for candidate in candidates:
        name = candidate if candidate in all_configs else lower_to_name.get(candidate.lower())
        if name is not None:
            return name, all_configs[name]

    dataset_name = canonical_levelx_dataset(dataset)
    file_id = extract_levelx_file_id(file)
    for name, config in all_configs.items():
        if str(config.get("dataset", "")).lower() != dataset_name.lower():
            continue
        if file_id in config.get("trajectory_files", []):
            return name, config

    return None, None


def resolve_levelx_osm_path(
    dataset: str, folder: Path, configs: dict[str, Any] | None, osm_path: Path | None = None
) -> Path:
    """Resolve an OSM path from explicit input or the local dataset layout."""

    if osm_path is not None:
        return osm_path.expanduser().resolve()

    if configs is None or not configs.get("osm_file"):
        raise ValueError("An OSM path is required when no map config with `osm_file` is found.")

    dataset_name = canonical_levelx_dataset(dataset)
    osm_file = configs["osm_file"]
    folder = folder.expanduser()
    candidates = (
        Path.cwd() / "data" / f"{dataset_name}_map" / osm_file,
        Path.cwd() / "data" / dataset_name / osm_file,
        Path.cwd() / "tactics2d" / "data" / "map" / dataset_name / osm_file,
        folder / osm_file,
        folder.parent / osm_file,
        folder.parent / "map" / osm_file,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {osm_file}. Checked: {checked}")


def ensure_frontend_server(host: str, port: int, max_fps: int, open_browser: bool = False):
    """Return a renderer and start a background frontend server if needed."""

    from tactics2d.frontend import FrontendRenderer
    from tactics2d.frontend.renderer import start_server_process

    renderer = FrontendRenderer(host, port, max_fps=max_fps)
    if renderer.wait_until_ready(timeout=0.5):
        if open_browser:
            webbrowser.open(renderer.base_url)
        return renderer

    start_server_process(host, port, demo=False, max_fps=max_fps, open_browser=open_browser)
    if not renderer.wait_until_ready(timeout=5.0):
        raise RuntimeError(f"Tactics2D frontend did not start on {renderer.base_url}.")

    return renderer


def build_map_preview_sensor(
    osm_path: Path, lanelet2: bool, configs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build one frontend sensor payload for an OSM map preview."""

    from shapely.geometry import Point

    from tactics2d.map.parser import OSMParser
    from tactics2d.sensor import BEVCamera

    map_ = OSMParser(lanelet2=lanelet2).parse(str(osm_path), configs)
    camera = BEVCamera(id_=0, map_=map_)
    x_center = 0.5 * (map_.boundary[0] + map_.boundary[1])
    y_center = 0.5 * (map_.boundary[2] + map_.boundary[3])
    geometry_data, _, _ = camera.update(0, {}, [], set(), set(), Point(x_center, y_center), 0)

    coords = [
        point
        for element in geometry_data["map_data"]["road_elements"]
        for point in element["geometry"]
    ]
    if coords:
        x_values = [point[0] for point in coords]
        y_values = [point[1] for point in coords]
        x_center = 0.5 * (min(x_values) + max(x_values))
        y_center = 0.5 * (min(y_values) + max(y_values))
        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)
    else:
        x_span = map_.boundary[1] - map_.boundary[0]
        y_span = map_.boundary[3] - map_.boundary[2]

    preview_range = max(20.0, min(max(x_span, y_span) / 2, max(y_span * 3, x_span / 8)))

    return {
        "id": f"map-{osm_path.stem}",
        "perception_range": float(preview_range),
        "position": [x_center, y_center],
        "yaw": 0,
        "frame": 0,
        "map_data": geometry_data["map_data"],
        "participant_data": geometry_data["participant_data"],
    }


def _active_participant_ids(participants: dict[int, Any], frame: int) -> list[int]:
    return [
        participant_id
        for participant_id, participant in participants.items()
        if participant.trajectory.has_state(frame)
    ]


def _choose_camera_pose(
    participants: dict[int, Any],
    active_ids: list[int],
    frame: int,
    fallback_position: tuple[float, float],
    follow_id: int | None = None,
) -> tuple[tuple[float, float], float]:
    if follow_id in active_ids:
        state = participants[follow_id].trajectory.get_state(frame)
        return state.location, state.heading

    if active_ids:
        state = participants[active_ids[0]].trajectory.get_state(frame)
        return state.location, state.heading

    return fallback_position, 0


def _sensor_payload(
    sensor_id: str,
    frame: int,
    position: tuple[float, float],
    heading: float,
    perception_range: float,
    geometry_data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": sensor_id,
        "perception_range": perception_range,
        "position": list(position),
        "yaw": heading,
        "frame": frame,
        "map_data": geometry_data["map_data"],
        "participant_data": geometry_data["participant_data"],
    }


def stream_levelx_preview(
    dataset: str,
    folder: Path,
    file: str | int,
    osm_path: Path | None = None,
    map_config: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_fps: int = 30,
    open_browser: bool = True,
    lanelet2: bool = True,
    frames: int = 300,
    start_time_ms: int | None = None,
    ids: list[int] | None = None,
    follow_id: int | None = None,
    perception_range: float = 80.0,
) -> DatasetPreviewResult:
    """Stream a LevelX dataset slice into the browser frontend."""

    from shapely.geometry import Point

    from tactics2d.dataset_parser import LevelXParser
    from tactics2d.map.parser import OSMParser
    from tactics2d.sensor import BEVCamera

    dataset_name = canonical_levelx_dataset(dataset)
    file_id = extract_levelx_file_id(file)
    folder = folder.expanduser().resolve()
    config_name, configs = resolve_levelx_map_config(dataset_name, file_id, map_config, osm_path)
    osm_path = resolve_levelx_osm_path(dataset_name, folder, configs, osm_path)

    dataset_parser = LevelXParser(dataset_name)
    full_time_range = dataset_parser.get_time_range(file_id, str(folder))
    start = full_time_range[0] if start_time_ms is None else max(start_time_ms, full_time_range[0])
    end = min(full_time_range[1], start + (max(1, frames) - 1) * LEVELX_FRAME_STEP_MS)

    LOGGER.info(
        "Loading %s recording %02d from %s (%s-%s ms).", dataset_name, file_id, folder, start, end
    )
    participants, actual_time_range = dataset_parser.parse_trajectory(
        file_id, str(folder), time_range=(start, end), ids=ids
    )
    if not participants:
        raise RuntimeError(f"No participants found in {dataset_name} recording {file_id}.")

    LOGGER.info("Loading map %s%s.", osm_path, f" with config {config_name}" if config_name else "")
    map_ = OSMParser(lanelet2=lanelet2).parse(str(osm_path), configs)
    camera = BEVCamera(id_=0, map_=map_, perception_range=perception_range)
    renderer = ensure_frontend_server(host, port, max_fps, open_browser=open_browser)
    ack_timeout = 1.0 / max(1, min(max_fps, 100))

    fallback_position = (
        0.5 * (map_.boundary[0] + map_.boundary[1]),
        0.5 * (map_.boundary[2] + map_.boundary[3]),
    )
    prev_road_id_set = set()
    prev_participant_id_set = set()
    sent_frames = 0
    last_dropped_count = 0
    sensor_id = f"{dataset_name}-{file_id:02d}"

    for frame in range(
        int(actual_time_range[0]), int(actual_time_range[1]) + 1, LEVELX_FRAME_STEP_MS
    ):
        active_ids = _active_participant_ids(participants, frame)
        position, heading = _choose_camera_pose(
            participants, active_ids, frame, fallback_position, follow_id=follow_id
        )
        geometry_data, prev_road_id_set, prev_participant_id_set = camera.update(
            frame,
            participants,
            active_ids,
            prev_road_id_set,
            prev_participant_id_set,
            Point(*position),
            heading,
        )
        response = renderer.send_frame(
            [
                _sensor_payload(
                    sensor_id,
                    frame,
                    position,
                    heading,
                    float(camera.max_perception_distance),
                    geometry_data,
                )
            ],
            frame=frame,
            layout="grid",
            wait_ack=True,
            ack_timeout=ack_timeout,
            drop_if_busy=True,
        )
        last_dropped_count = response.get("dropped_frames", last_dropped_count)
        if response.get("status") == "dropped":
            continue

        sent_frames += 1

    LOGGER.info("Previewed %s frames at %s.", sent_frames, renderer.base_url)
    time.sleep(0.1)
    return DatasetPreviewResult(
        base_url=renderer.base_url,
        sensor_id=sensor_id,
        actual_time_range=(int(actual_time_range[0]), int(actual_time_range[1])),
        sent_frames=sent_frames,
        dropped_frames=int(last_dropped_count),
    )
