# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preview helpers for the browser frontend."""

from __future__ import annotations

import logging
import os
import re
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tactics2d.dataset_parser import LEVELX_DATASETS

LOGGER = logging.getLogger(__name__)

LEVELX_FRAME_STEP_MS = 40

NUPLAN_DATASET = "NuPlan"
# NuPlan lidar sweeps are nominally 20 Hz; observed stamps jitter by +-1 ms,
# so scenes iterate the actual stamp list instead of a fixed-step range.
NUPLAN_FRAME_STEP_MS = 50

DATA_ROOT_ENV = "TACTICS2D_DATA_ROOT"

_TRACKS_FILE_PATTERN = re.compile(r"^(\d+)_tracks\.csv$")
_MAX_DISCOVERED_MAPS = 200

_LEVELX_CANONICAL = {name.lower(): name for name in LEVELX_DATASETS}
_MAP_CONFIG_NAMES = (
    "HIGHD_MAP_CONFIG",
    "IND_MAP_CONFIG",
    "ROUND_MAP_CONFIG",
    "EXID_MAP_CONFIG",
    "UNID_MAP_CONFIG",
    "NUPLAN_MAP_CONFIG",
)

_NUPLAN_DIR_VARIANTS = ("nuPlan", "NuPlan", "nuplan")
# NuPlan maps are city scale (kilometers); cap the standalone map preview to a
# window around the map center so the payload stays browser-friendly.
_NUPLAN_PREVIEW_MAP_RANGE = 500.0


@dataclass
class DatasetPreviewResult:
    """Summary of a streamed dataset preview."""

    base_url: str
    sensor_id: str
    actual_time_range: tuple[int, int]
    sent_frames: int
    dropped_frames: int


@dataclass
class LevelXPreviewScene:
    """Loaded LevelX scene state used by CLI and server-side previews."""

    dataset_name: str
    file_id: int
    sensor_id: str
    actual_time_range: tuple[int, int]
    map_: Any
    camera: Any
    participants: dict[int, Any]
    fallback_position: tuple[float, float]
    follow_id: int | None = None
    prev_road_id_set: set = field(default_factory=set)
    prev_participant_id_set: set = field(default_factory=set)

    def iter_frames(self):
        """Yield frame ids in dataset timestamp units."""

        return range(
            int(self.actual_time_range[0]), int(self.actual_time_range[1]) + 1, LEVELX_FRAME_STEP_MS
        )

    def sensor_for_frame(self, frame: int) -> dict[str, Any]:
        """Build one frontend sensor payload for a dataset frame."""

        return _build_scene_sensor(self, frame)


@dataclass
class NuPlanPreviewScene:
    """Loaded NuPlan scene state used by CLI and server-side previews."""

    dataset_name: str
    file_id: str
    sensor_id: str
    actual_time_range: tuple[int, int]
    map_: Any
    camera: Any
    participants: dict[int, Any]
    fallback_position: tuple[float, float]
    frame_ids: list[int] = field(default_factory=list)
    follow_id: int | None = None
    prev_road_id_set: set = field(default_factory=set)
    prev_participant_id_set: set = field(default_factory=set)

    def iter_frames(self):
        """Yield the lidar sweep timestamps observed in the parsed window."""

        return list(self.frame_ids)

    def sensor_for_frame(self, frame: int) -> dict[str, Any]:
        """Build one frontend sensor payload for a dataset frame."""

        return _build_scene_sensor(self, frame)


def _build_scene_sensor(scene, frame: int) -> dict[str, Any]:
    """Build one frontend sensor payload for a loaded preview scene."""

    from shapely.geometry import Point

    active_ids = _active_participant_ids(scene.participants, frame)
    position, heading = _choose_camera_pose(
        scene.participants, active_ids, frame, scene.fallback_position, follow_id=scene.follow_id
    )
    geometry_data, scene.prev_road_id_set, scene.prev_participant_id_set = scene.camera.update(
        frame,
        scene.participants,
        active_ids,
        scene.prev_road_id_set,
        scene.prev_participant_id_set,
        Point(*position),
        heading,
    )
    return _sensor_payload(
        scene.sensor_id,
        frame,
        position,
        heading,
        float(scene.camera.max_perception_distance),
        geometry_data,
    )


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


def get_data_roots() -> list[Path]:
    """Return the existing dataset root directories.

    Roots are taken from the ``TACTICS2D_DATA_ROOT`` environment variable
    (multiple paths may be separated by ``os.pathsep``), followed by the
    repository convention ``./data`` relative to the working directory.
    Only directories that exist are returned.
    """

    roots: list[Path] = []
    for token in os.environ.get(DATA_ROOT_ENV, "").split(os.pathsep):
        token = token.strip()
        if token:
            roots.append(Path(token).expanduser())
    roots.append(Path.cwd() / "data")

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _list_recording_ids(folder: Path) -> list[int]:
    """Return recording ids found as ``<id>_tracks.csv`` files in a folder."""

    ids = set()
    try:
        for entry in folder.iterdir():
            match = _TRACKS_FILE_PATTERN.match(entry.name)
            if match:
                ids.add(int(match.group(1)))
    except OSError:
        return []
    return sorted(ids)


def discover_levelx_datasets() -> list[dict[str, Any]]:
    """Scan the data roots for LevelX datasets stored in the official layout.

    A dataset is detected when a directory named after it (case variants
    allowed) contains ``<id>_tracks.csv`` recordings either directly or in a
    ``data/`` subdirectory. The first data root containing a dataset wins.
    """

    discovered = []
    seen_datasets = set()
    for root in get_data_roots():
        for dataset in LEVELX_DATASETS:
            if dataset in seen_datasets:
                continue
            for candidate in (root / dataset, root / dataset.lower()):
                if not candidate.is_dir():
                    continue
                for folder in (candidate / "data", candidate):
                    ids = _list_recording_ids(folder)
                    if ids:
                        discovered.append({"dataset": dataset, "folder": str(folder), "files": ids})
                        seen_datasets.add(dataset)
                        break
                if dataset in seen_datasets:
                    break
    return discovered


def _list_nuplan_db_files(folder: Path) -> list[str]:
    """Return NuPlan ``.db`` logs below a folder, relative with split prefix."""

    files: set[Path] = set()
    try:
        for pattern in ("*.db", "*/*.db"):
            files.update(folder.glob(pattern))
    except OSError:
        return []
    return sorted(str(path.relative_to(folder)) for path in files)


def discover_nuplan_datasets() -> list[dict[str, Any]]:
    """Scan the data roots for NuPlan sqlite logs.

    Both the tactics2d convention ``nuPlan/data/cache/<split>/*.db`` and the
    nuplan-devkit convention ``nuplan/dataset/nuplan-v1.1/splits/<split>/*.db``
    are recognized. The first candidate folder containing logs wins.
    """

    for root in get_data_roots():
        for variant in _NUPLAN_DIR_VARIANTS:
            candidate = root / variant
            if not candidate.is_dir():
                continue
            for folder in (
                candidate / "data" / "cache",
                candidate / "dataset" / "nuplan-v1.1" / "splits",
                candidate,
            ):
                files = _list_nuplan_db_files(folder)
                if files:
                    return [{"dataset": NUPLAN_DATASET, "folder": str(folder), "files": files}]
    return []


def resolve_nuplan_maps_root(folder: Path) -> Path:
    """Resolve the NuPlan ``maps/`` root for a log folder.

    A candidate qualifies when it contains at least one registered
    ``<location>/<version>/map.gpkg`` from ``NUPLAN_MAP_CONFIG``.
    """

    from tactics2d.map.map_config import NUPLAN_MAP_CONFIG

    folder = folder.expanduser().resolve()
    candidates = [
        folder / "maps",
        folder.parent / "maps",
        folder.parent.parent / "maps",
        folder.parent.parent.parent / "maps",
        Path.cwd() / "tactics2d" / "data" / "map" / "NuPlan" / "maps",
        Path.cwd() / "tactics2d" / "data" / "map" / "NuPlan",
    ]
    for root in get_data_roots():
        for variant in _NUPLAN_DIR_VARIANTS:
            candidates.append(root / variant / "maps")

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for config in NUPLAN_MAP_CONFIG.values():
            if (candidate / config["folder"] / config["gpkg_file"]).exists():
                return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find a NuPlan maps folder. Checked: {checked}")


def discover_nuplan_maps(
    dataset_catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return registered NuPlan geopackage maps that exist on disk."""

    from tactics2d.map.map_config import NUPLAN_MAP_CONFIG

    folders = [
        Path(entry["folder"])
        for entry in (dataset_catalog or [])
        if entry.get("dataset") == NUPLAN_DATASET
    ]
    folders.append(Path.cwd() / "data" / NUPLAN_DATASET)

    maps_root = None
    for folder in folders:
        try:
            maps_root = resolve_nuplan_maps_root(folder)
            break
        except FileNotFoundError:
            continue
    if maps_root is None:
        return []

    maps = []
    for name, config in NUPLAN_MAP_CONFIG.items():
        path = maps_root / config["folder"] / config["gpkg_file"]
        if not path.exists():
            continue
        location = str(Path(config["folder"]).parts[0])
        maps.append(
            {
                "name": name,
                "dataset": NUPLAN_DATASET,
                "osm_path": str(path.resolve()),
                "description": location if location != name else "",
            }
        )
    return maps


def discover_maps(dataset_catalog: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return maps that are actually available on disk.

    Registered map configs are probed through :func:`resolve_levelx_osm_path`;
    unregistered ``.osm`` files found up to three levels below the data roots
    are appended after them.
    """

    folders = {entry["dataset"]: Path(entry["folder"]) for entry in (dataset_catalog or [])}
    maps: list[dict[str, Any]] = []
    known_paths: set[Path] = set()

    for name, config in iter_map_configs():
        dataset = config.get("dataset")
        if not dataset or not config.get("osm_file"):
            continue
        folder = folders.get(dataset, Path.cwd() / "data" / dataset)
        try:
            path = resolve_levelx_osm_path(dataset, folder, config)
        except (FileNotFoundError, ValueError):
            continue
        known_paths.add(path)
        maps.append(
            {
                "name": name,
                "dataset": dataset,
                "osm_path": str(path),
                "description": config.get("name", name),
            }
        )

    for entry in discover_nuplan_maps(dataset_catalog):
        known_paths.add(Path(entry["osm_path"]))
        maps.append(entry)

    for root in get_data_roots():
        for pattern in ("*.osm", "*/*.osm", "*/*/*.osm"):
            for osm in sorted(root.glob(pattern)):
                resolved = osm.resolve()
                if resolved in known_paths:
                    continue
                known_paths.add(resolved)
                maps.append(
                    {
                        "name": str(osm.relative_to(root)),
                        "dataset": None,
                        "osm_path": str(resolved),
                        "description": "",
                    }
                )
                if len(maps) >= _MAX_DISCOVERED_MAPS:
                    return maps
    return maps


def list_levelx_preview_options() -> dict[str, Any]:
    """Return lightweight option metadata for the browser controls."""

    map_configs = []
    for name, config in iter_map_configs():
        dataset = config.get("dataset")
        if dataset is None:
            continue

        map_configs.append(
            {
                "name": name,
                "dataset": dataset,
                "osm_file": config.get("osm_file"),
                "trajectory_files": config.get("trajectory_files", []),
                # NuPlan configs carry no display name; fall back to the key
                # (the UI already shows the key, so avoid repeating it).
                "description": config.get("name", ""),
            }
        )

    datasets = discover_levelx_datasets() + discover_nuplan_datasets()
    maps = discover_maps(datasets)
    first = datasets[0] if datasets else None

    return {
        "levelx_datasets": LEVELX_DATASETS,
        "map_configs": map_configs,
        "datasets": datasets,
        "maps": maps,
        "data_roots": [str(root) for root in get_data_roots()],
        "defaults": {
            "dataset": first["dataset"] if first else "highD",
            "folder": first["folder"] if first else "",
            "file": str(first["files"][0]) if first else "",
            "frames": 300,
            "max_fps": 30,
            "perception_range": 80,
        },
    }


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
        folder.parent / "maps" / "lanelet2" / osm_file,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Official levelXdata layout names maps <location>_<site>.osm under
    # maps/lanelet2/ (e.g. exiD's 0_cologne_butzweiler.osm), while registered
    # configs reference <dataset>_<location>.osm. Match by location number.
    maps_dir = folder.parent / "maps" / "lanelet2"
    location = re.search(r"(\d+)$", Path(osm_file).stem)
    if maps_dir.is_dir() and location is not None:
        prefix = f"{int(location.group(1))}_"
        for candidate in sorted(maps_dir.glob("*.osm")):
            if candidate.name.startswith(prefix):
                return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {osm_file}. Checked: {checked}")


def ensure_frontend_server(host: str, port: int, max_fps: int, open_browser: bool = False):
    """Return a renderer and start a background frontend server if needed."""

    from .renderer import FrontendRenderer, start_server_process

    renderer = FrontendRenderer(host, port, max_fps=max_fps)
    if renderer.wait_until_ready(timeout=0.5):
        if open_browser:
            webbrowser.open(renderer.base_url)
        return renderer

    start_server_process(host, port, demo=False, max_fps=max_fps, open_browser=open_browser)
    if not renderer.wait_until_ready(timeout=5.0):
        raise RuntimeError(f"Tactics2D frontend did not start on {renderer.base_url}.")

    return renderer


def _shift_map_origin(map_: Any) -> tuple[float, float]:
    """Translate a map's geometry to a local origin at its center.

    NuPlan maps carry UTM-scale coordinates (up to ~4.7e6 m). Three.js renders
    with float32, whose resolution at that magnitude is 0.5 m, so geometry must
    be shifted near the origin before it reaches the browser. Returns the
    subtracted ``(origin_x, origin_y)``.
    """

    from shapely import affinity

    boundary = map_.boundary
    origin_x = float(round(0.5 * (boundary[0] + boundary[1])))
    origin_y = float(round(0.5 * (boundary[2] + boundary[3])))

    def shifted(geometry):
        return affinity.translate(geometry, xoff=-origin_x, yoff=-origin_y)

    for roadline in map_.roadlines.values():
        if roadline.geometry is not None:
            roadline.geometry = shifted(roadline.geometry)
    for lane in map_.lanes.values():
        for attr in ("geometry", "left_side", "right_side"):
            geometry = getattr(lane, attr, None)
            if geometry is not None:
                setattr(lane, attr, shifted(geometry))
    for area in map_.areas.values():
        if area.geometry is not None:
            area.geometry = shifted(area.geometry)
    for junction in map_.junctions.values():
        if getattr(junction, "geometry", None) is not None:
            junction.geometry = shifted(junction.geometry)
        shape = (junction.custom_tags or {}).get("shape")
        if shape:
            junction.custom_tags["shape"] = [(x - origin_x, y - origin_y) for x, y in shape]

    # Keep the cached boundary consistent so later readers (e.g. a camera
    # constructed without an explicit perception range) see local coordinates.
    map_._boundary = (
        boundary[0] - origin_x,
        boundary[1] - origin_x,
        boundary[2] - origin_y,
        boundary[3] - origin_y,
    )
    return origin_x, origin_y


def _shift_participants(participants: dict[int, Any], origin: tuple[float, float]) -> None:
    """Translate all participant states by ``-origin`` in place."""

    origin_x, origin_y = origin
    for participant in participants.values():
        for state in participant.trajectory.history_states.values():
            state.x -= origin_x
            state.y -= origin_y


def load_nuplan_preview_scene(
    folder: Path,
    file: str,
    map_config: str | None = None,
    frames: int = 300,
    start_time_ms: int | None = None,
    follow_id: int | None = None,
    perception_range: float = 80.0,
) -> NuPlanPreviewScene:
    """Load a NuPlan scene and return a frame payload source.

    The map is resolved from the log's location via ``NUPLAN_MAP_CONFIG``
    unless ``map_config`` names a location explicitly. Map and trajectories are
    shifted to a local origin at the map center (see ``_shift_map_origin``).
    """

    import sqlite3

    from tactics2d.dataset_parser import NuPlanParser
    from tactics2d.display.sensor import BEVCamera
    from tactics2d.map.map_config import NUPLAN_MAP_CONFIG
    from tactics2d.participant.element import Vehicle

    folder = folder.expanduser().resolve()
    parser = NuPlanParser()

    location = map_config or parser.get_location(str(file), str(folder))
    if location not in NUPLAN_MAP_CONFIG:
        raise KeyError(
            f"{location} has no registered NuPlan map config. "
            f"Choose one of: {', '.join(NUPLAN_MAP_CONFIG)}."
        )
    configs = NUPLAN_MAP_CONFIG[location]

    # Query the sweep range cheaply so only the requested window is parsed.
    with sqlite3.connect(folder / file) as connection:
        row = connection.execute("SELECT MIN(timestamp), MAX(timestamp) FROM lidar_pc;").fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"No lidar sweeps found in NuPlan log {file}.")
    log_start_ms = int(row[0] / 1000 - NuPlanParser._DATETIME)

    start = log_start_ms if start_time_ms is None else max(start_time_ms, log_start_ms)
    # Half a step of margin absorbs sweep jitter; the frame list is truncated below.
    end = start + (max(1, frames) - 1) * NUPLAN_FRAME_STEP_MS + NUPLAN_FRAME_STEP_MS // 2

    LOGGER.info("Loading NuPlan log %s from %s (%s-%s ms).", file, folder, start, end)
    participants, _ = parser.parse_trajectory(str(file), str(folder), time_range=(start, end))
    if not participants:
        raise RuntimeError(f"No participants found in NuPlan log {file}.")

    maps_root = resolve_nuplan_maps_root(folder)
    map_path = str(Path(configs["folder"]) / configs["gpkg_file"])
    LOGGER.info("Loading map %s from %s.", map_path, maps_root)
    map_ = parser.parse_map(map_path, str(maps_root))

    origin = _shift_map_origin(map_)
    _shift_participants(participants, origin)

    frame_ids = sorted({frame for p in participants.values() for frame in p.trajectory.frames})
    frame_ids = frame_ids[: max(1, frames)]

    if follow_id is None:
        # Cones and barriers are participants too; follow the longest-lived vehicle.
        vehicles = [p for p in participants.values() if isinstance(p, Vehicle)]
        if vehicles:
            follow_id = max(vehicles, key=lambda p: len(p.trajectory.frames)).id_

    camera = BEVCamera(id_=0, map_=map_, perception_range=perception_range)
    boundary = map_.boundary
    fallback_position = (0.5 * (boundary[0] + boundary[1]), 0.5 * (boundary[2] + boundary[3]))

    return NuPlanPreviewScene(
        dataset_name=NUPLAN_DATASET,
        file_id=str(file),
        sensor_id=f"{NUPLAN_DATASET}-{Path(file).stem}",
        actual_time_range=(int(frame_ids[0]), int(frame_ids[-1])),
        map_=map_,
        camera=camera,
        participants=participants,
        fallback_position=fallback_position,
        frame_ids=[int(frame) for frame in frame_ids],
        follow_id=follow_id,
    )


def build_map_preview_sensor(
    osm_path: Path, lanelet2: bool, configs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build one frontend sensor payload for a map preview.

    ``.gpkg`` files are parsed as NuPlan geopackage maps and previewed in a
    capped window around the map center; anything else goes through
    ``OSMParser``.
    """

    from shapely.geometry import Point

    from tactics2d.display.sensor import BEVCamera

    if osm_path.suffix.lower() == ".gpkg":
        from tactics2d.dataset_parser import NuPlanParser

        map_ = NuPlanParser().parse_map(str(osm_path))
        _shift_map_origin(map_)
        camera_range = _NUPLAN_PREVIEW_MAP_RANGE
    else:
        from tactics2d.map.parser import OSMParser

        map_ = OSMParser(lanelet2=lanelet2).parse(str(osm_path), configs)
        camera_range = None
    camera = BEVCamera(id_=0, map_=map_, perception_range=camera_range)
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

    # Every NuPlan geopackage is named map.gpkg; label those by location dir.
    label = osm_path.stem if osm_path.stem != "map" else osm_path.parent.parent.name
    return {
        "id": f"map-{label}",
        "perception_range": float(preview_range),
        "viewport_aspect": float(max(1.0, x_span / (2 * preview_range))),
        "position": [float(x_center), float(y_center)],
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
    viewport_aspect: float = 16 / 9,
) -> dict[str, Any]:
    return {
        "id": sensor_id,
        "perception_range": perception_range,
        "viewport_aspect": viewport_aspect,
        "position": list(position),
        "yaw": heading,
        "frame": frame,
        "map_data": geometry_data["map_data"],
        "participant_data": geometry_data["participant_data"],
    }


def load_levelx_preview_scene(
    dataset: str,
    folder: Path,
    file: str | int,
    osm_path: Path | None = None,
    map_config: str | None = None,
    lanelet2: bool = True,
    frames: int = 300,
    start_time_ms: int | None = None,
    ids: list[int] | None = None,
    follow_id: int | None = None,
    perception_range: float = 80.0,
) -> LevelXPreviewScene:
    """Load a LevelX scene and return a frame payload source."""

    from tactics2d.dataset_parser import LevelXParser
    from tactics2d.display.sensor import BEVCamera
    from tactics2d.map.parser import OSMParser

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

    fallback_position = (
        0.5 * (map_.boundary[0] + map_.boundary[1]),
        0.5 * (map_.boundary[2] + map_.boundary[3]),
    )

    return LevelXPreviewScene(
        dataset_name=dataset_name,
        file_id=file_id,
        sensor_id=f"{dataset_name}-{file_id:02d}",
        actual_time_range=(int(actual_time_range[0]), int(actual_time_range[1])),
        map_=map_,
        camera=camera,
        participants=participants,
        fallback_position=fallback_position,
        follow_id=follow_id,
    )


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

    scene = load_levelx_preview_scene(
        dataset=dataset,
        folder=folder,
        file=file,
        osm_path=osm_path,
        map_config=map_config,
        lanelet2=lanelet2,
        frames=frames,
        start_time_ms=start_time_ms,
        ids=ids,
        follow_id=follow_id,
        perception_range=perception_range,
    )
    renderer = ensure_frontend_server(host, port, max_fps, open_browser=open_browser)
    ack_timeout = 1.0 / max(1, min(max_fps, 100))

    sent_frames = 0
    last_dropped_count = 0

    for frame in scene.iter_frames():
        response = renderer.send_frame(
            [scene.sensor_for_frame(frame)],
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
        sensor_id=scene.sensor_id,
        actual_time_range=scene.actual_time_range,
        sent_frames=sent_frames,
        dropped_frames=int(last_dropped_count),
    )
