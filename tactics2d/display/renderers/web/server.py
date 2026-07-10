# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""FastAPI server for the Tactics2D browser frontend."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

import orjson

try:
    from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    FastAPI = None
    Request = None
    Response = None
    WebSocket = None
    WebSocketDisconnect = None
    FileResponse = None
    JSONResponse = None
    StaticFiles = None

LOGGER = logging.getLogger(__name__)


async def _to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


class ConnectionManager:
    """Track browser websocket clients and broadcast frontend messages."""

    def __init__(self):
        self._clients = []
        self._ack_events = {}
        self._dropped_frames = 0
        self._last_ack = None
        self._last_frame_message = None
        self._last_snapshot_message = None
        self._last_frame_id = None
        self._sensor_snapshots = {}
        self._recording_file = None
        self._recording_path: Path | None = None
        self._recording_frames = 0

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def last_ack(self) -> Any:
        return self._last_ack

    @property
    def last_frame_id(self) -> Any:
        return self._last_frame_id

    @property
    def is_render_busy(self) -> bool:
        return (
            self.client_count > 0
            and self._last_frame_id is not None
            and self._last_ack != self._last_frame_id
        )

    async def connect(self, websocket):
        await websocket.accept()
        self._clients.append(websocket)

    async def disconnect(self, websocket):
        self._clients = [client for client in self._clients if client is not websocket]

    async def broadcast(self, message: dict) -> int:
        payload = orjson.dumps(message).decode("utf-8")
        clients = tuple(self._clients)

        delivered = 0
        stale_clients = []
        for websocket in clients:
            try:
                await websocket.send_text(payload)
                delivered += 1
            except Exception:
                stale_clients.append(websocket)

        if stale_clients:
            for websocket in stale_clients:
                self._clients = [client for client in self._clients if client is not websocket]

        return delivered

    async def publish_frame(
        self,
        payload: dict,
        frame_id: Any = None,
        wait_ack: bool = False,
        ack_timeout: float = 0.05,
        drop_if_busy: bool = False,
    ) -> dict:
        # Record before the busy check so the recording is not thinned by browser speed.
        if self._recording_file is not None:
            self._write_recording_line(frame_id, payload)

        if drop_if_busy and self.is_render_busy:
            self._dropped_frames += 1
            return {
                "status": "dropped",
                "delivered": 0,
                "acked": False,
                "frame_id": frame_id,
                "dropped_frames": self._dropped_frames,
            }

        self._last_frame_id = frame_id
        self._last_frame_message = {
            "type": "frame.update",
            "frame_id": frame_id,
            "payload": payload,
        }
        self._last_snapshot_message = {
            "type": "frame.update",
            "frame_id": frame_id,
            "payload": self._snapshot_payload(payload),
        }
        delivered = await self.broadcast(self._last_frame_message)
        acked = delivered == 0
        if acked:
            self.record_ack(frame_id)
        if wait_ack and delivered > 0:
            acked = await self.wait_for_ack(frame_id, ack_timeout)

        return {
            "status": "ok",
            "delivered": delivered,
            "acked": acked,
            "frame_id": frame_id,
            "dropped_frames": self._dropped_frames,
        }

    @property
    def is_recording(self) -> bool:
        return self._recording_file is not None

    @property
    def recording_name(self) -> str | None:
        return self._recording_path.stem if self._recording_path is not None else None

    def start_recording(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._recording_file = path.open("wb")
        self._recording_path = path
        self._recording_frames = 0
        # Seed with the merged snapshot so a replay starts with the complete scene.
        if self._last_snapshot_message is not None:
            self._write_recording_line(
                self._last_snapshot_message["frame_id"], self._last_snapshot_message["payload"]
            )

    def stop_recording(self) -> dict:
        info = {
            "name": self.recording_name,
            "path": str(self._recording_path),
            "frames": self._recording_frames,
        }
        self._recording_file.close()
        self._recording_file = None
        self._recording_path = None
        self._recording_frames = 0
        return info

    def _write_recording_line(self, frame_id: Any, payload: dict) -> None:
        record = {"frame_id": frame_id, "time": time.time(), "payload": payload}
        self._recording_file.write(orjson.dumps(record) + b"\n")
        self._recording_frames += 1

    def record_ack(self, frame_id: Any) -> None:
        self._last_ack = frame_id
        event = self._ack_events.pop(frame_id, None)
        if event is not None:
            event.set()

    async def wait_for_ack(self, frame_id: Any, timeout: float) -> bool:
        if self.client_count == 0 or self._last_ack == frame_id:
            return True

        event = self._ack_events.setdefault(frame_id, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout=max(timeout, 0))
            return True
        except asyncio.TimeoutError:
            return False

    def _snapshot_payload(self, payload: dict) -> dict:
        sensor_id_to_remove = set(payload.get("sensor_id_to_remove", []))
        for sensor_id in sensor_id_to_remove:
            self._sensor_snapshots.pop(sensor_id, None)

        sensors = payload.get("sensors", [])
        if payload.get("remove_missing_sensors", True):
            active_sensor_ids = {sensor.get("id") for sensor in sensors}
            for sensor_id in list(self._sensor_snapshots):
                if sensor_id not in active_sensor_ids:
                    self._sensor_snapshots.pop(sensor_id, None)

        for sensor in sensors:
            sensor_id = sensor.get("id")
            if sensor_id is None:
                continue

            snapshot = self._sensor_snapshots.get(
                sensor_id, {"sensor": {"id": sensor_id}, "roads": {}, "participants": {}}
            )
            sensor_snapshot = dict(snapshot["sensor"])
            for key, value in sensor.items():
                if key not in {"map_data", "participant_data"}:
                    sensor_snapshot[key] = value

            roads = dict(snapshot["roads"])
            map_data = sensor.get("map_data")
            if map_data is not None:
                for road_id in map_data.get("road_id_to_remove", []):
                    roads.pop(road_id, None)
                for road_element in map_data.get("road_elements", []):
                    roads[road_element["id"]] = road_element

            participants = dict(snapshot["participants"])
            participant_data = sensor.get("participant_data")
            if participant_data is not None:
                for participant_id in participant_data.get("participant_id_to_remove", []):
                    participants.pop(participant_id, None)
                for participant in participant_data.get("participants", []):
                    participants[participant["id"]] = participant

            sensor_snapshot["map_data"] = {
                "road_id_to_remove": [],
                "road_elements": list(roads.values()),
            }
            sensor_snapshot["participant_data"] = {
                "participant_id_to_create": list(participants.keys()),
                "participant_id_to_remove": [],
                "participants": list(participants.values()),
            }
            self._sensor_snapshots[sensor_id] = {
                "sensor": sensor_snapshot,
                "roads": roads,
                "participants": participants,
            }

        snapshot_payload = dict(payload)
        snapshot_payload["sensor_id_to_remove"] = []
        snapshot_payload["sensors"] = [
            snapshot["sensor"] for snapshot in self._sensor_snapshots.values()
        ]
        return snapshot_payload


class _PreviewPauseEvent:
    """Delay asyncio.Event creation until code is inside an event loop."""

    def __init__(self, is_set: bool = True):
        self._is_set = is_set
        self._event = None

    def is_set(self) -> bool:
        return self._event.is_set() if self._event is not None else self._is_set

    def set(self) -> None:
        self._is_set = True
        if self._event is not None:
            self._event.set()

    def clear(self) -> None:
        self._is_set = False
        if self._event is not None:
            self._event.clear()

    async def wait(self) -> None:
        if self._event is None:
            self._event = asyncio.Event()
            if self._is_set:
                self._event.set()
        await self._event.wait()


def _frontend_static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_path(value) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def _parse_ids(value) -> list[int] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


async def _cancel_task(task) -> None:
    if task is None or task.done():
        return

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _stop_preview_tasks(app) -> None:
    pause_event = getattr(app.state, "preview_pause_event", None)
    if pause_event is not None:
        pause_event.set()

    await _cancel_task(getattr(app.state, "demo_task", None))
    await _cancel_task(getattr(app.state, "preview_task", None))
    app.state.demo_task = None
    app.state.preview_task = None


def _recordings_dir() -> Path:
    env = os.environ.get("TACTICS2D_RECORD_DIR")
    if env:
        return Path(env).expanduser()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "tactics2d" / "recordings"


def _find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_to_mp4(ffmpeg: str, data: bytes, suffix: str) -> bytes:
    """Re-encode a MediaRecorder capture into a constant-frame-rate H.264 MP4.

    Raw MediaRecorder output has a variable frame rate (declared as 0/1),
    which strict players such as GNOME Videos refuse to play.
    """

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{suffix}"
        dst = Path(tmp) / "output.mp4"
        src.write_bytes(data)
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-r",
                "30",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(result.stderr.strip()[-500:] or "ffmpeg failed")
        return dst.read_bytes()


def _load_recording(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")

    frames = []
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if line:
                frames.append(orjson.loads(line))
    return frames


async def _run_recording_replay(
    manager: ConnectionManager, status: dict, payload: dict, pause_event: asyncio.Event
):
    max_fps = max(1, min(int(payload.get("max_fps", 30)), 100))
    interval = 1.0 / max_fps
    loop = bool(payload.get("loop", False))
    sent_frames = 0
    dropped_frames = 0
    loop_count = 0

    try:
        name = str(payload["name"])
        frames = await _to_thread(_load_recording, _recordings_dir() / f"{name}.jsonl")
        total_frames = len(frames)
        if not total_frames:
            raise ValueError(f"Recording {name} is empty.")

        status.update(
            {
                "status": "running",
                "source": "replay",
                "recording": name,
                "total_frames": total_frames,
                "frame_index": 0,
                "progress": 0,
                "paused": False,
                "loop": loop,
                "loop_count": loop_count,
                "sent_frames": 0,
                "dropped_frames": 0,
                "message": "replaying",
            }
        )

        while True:
            for frame_index, record in enumerate(frames, start=1):
                if not pause_event.is_set():
                    status.update({"status": "paused", "paused": True, "message": "paused"})
                await pause_event.wait()
                status.update({"status": "running", "paused": False, "message": "replaying"})

                started = asyncio.get_running_loop().time()
                result = await manager.publish_frame(
                    record.get("payload", {}),
                    frame_id=record.get("frame_id"),
                    wait_ack=True,
                    ack_timeout=interval,
                    drop_if_busy=True,
                )
                dropped_frames = int(result.get("dropped_frames", dropped_frames))
                if result.get("status") != "dropped":
                    sent_frames += 1

                status.update(
                    {
                        "frame": record.get("frame_id"),
                        "frame_index": frame_index,
                        "total_frames": total_frames,
                        "progress": frame_index / total_frames if total_frames else 0,
                        "sent_frames": sent_frames,
                        "dropped_frames": dropped_frames,
                        "loop_count": loop_count,
                    }
                )
                elapsed = asyncio.get_running_loop().time() - started
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)

            if not loop:
                break

            loop_count += 1

        status.update(
            {
                "status": "complete",
                "message": "complete",
                "paused": False,
                "progress": 1,
                "sent_frames": sent_frames,
                "dropped_frames": dropped_frames,
            }
        )
    except asyncio.CancelledError:
        status.update({"status": "stopped", "message": "stopped"})
    except Exception as exc:
        LOGGER.exception("Recording replay failed.")
        status.update({"status": "error", "message": str(exc)})


def _map_config_from_name(name: str | None) -> dict | None:
    if not name:
        return None

    from .preview import iter_map_configs

    configs = dict(iter_map_configs())
    lower_to_name = {config_name.lower(): config_name for config_name in configs}
    resolved_name = name if name in configs else lower_to_name.get(name.lower())
    if resolved_name is None:
        raise KeyError(f"Unknown map config: {name}")

    return configs[resolved_name]


async def _run_levelx_dataset_preview(
    manager: ConnectionManager, status: dict, payload: dict, pause_event: asyncio.Event
):
    from .preview import load_levelx_preview_scene

    max_fps = max(1, min(int(payload.get("max_fps", 30)), 100))
    interval = 1.0 / max_fps
    loop = bool(payload.get("loop", False))
    sent_frames = 0
    dropped_frames = 0
    loop_count = 0

    try:
        status.update({"status": "loading", "message": "loading dataset"})
        scene = await _to_thread(
            load_levelx_preview_scene,
            dataset=payload["dataset"],
            folder=Path(payload["folder"]),
            file=payload["file"],
            osm_path=_optional_path(payload.get("osm_path")),
            map_config=payload.get("map_config") or None,
            lanelet2=bool(payload.get("lanelet2", True)),
            frames=int(payload.get("frames", 300)),
            start_time_ms=_optional_int(payload.get("start_time_ms")),
            ids=_parse_ids(payload.get("ids")),
            follow_id=_optional_int(payload.get("follow_id")),
            perception_range=float(payload.get("perception_range", 80.0)),
        )
        frame_ids = list(scene.iter_frames())
        total_frames = len(frame_ids)
        status.update(
            {
                "status": "running",
                "source": "dataset",
                "sensor_id": scene.sensor_id,
                "actual_time_range": scene.actual_time_range,
                "total_frames": total_frames,
                "frame_index": 0,
                "progress": 0,
                "paused": False,
                "loop": loop,
                "loop_count": loop_count,
                "sent_frames": 0,
                "dropped_frames": 0,
                "message": "streaming",
            }
        )

        while True:
            for frame_index, frame in enumerate(frame_ids, start=1):
                if not pause_event.is_set():
                    status.update({"status": "paused", "paused": True, "message": "paused"})
                await pause_event.wait()
                status.update({"status": "running", "paused": False, "message": "streaming"})

                started = asyncio.get_running_loop().time()
                sensor = await _to_thread(scene.sensor_for_frame, frame)
                result = await manager.publish_frame(
                    {
                        "frame": frame,
                        "layout": "grid",
                        "sensors": [sensor],
                        "remove_missing_sensors": True,
                    },
                    frame_id=frame,
                    wait_ack=True,
                    ack_timeout=interval,
                    drop_if_busy=True,
                )
                dropped_frames = int(result.get("dropped_frames", dropped_frames))
                if result.get("status") != "dropped":
                    sent_frames += 1

                status.update(
                    {
                        "frame": frame,
                        "frame_index": frame_index,
                        "total_frames": total_frames,
                        "progress": frame_index / total_frames if total_frames else 0,
                        "sent_frames": sent_frames,
                        "dropped_frames": dropped_frames,
                        "loop_count": loop_count,
                    }
                )
                elapsed = asyncio.get_running_loop().time() - started
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)

            if not loop:
                break

            loop_count += 1

        status.update(
            {
                "status": "complete",
                "message": "complete",
                "paused": False,
                "progress": 1,
                "sent_frames": sent_frames,
                "dropped_frames": dropped_frames,
            }
        )
    except asyncio.CancelledError:
        status.update({"status": "stopped", "message": "stopped"})
    except Exception as exc:
        LOGGER.exception("Dataset preview failed.")
        status.update({"status": "error", "message": str(exc)})


def _demo_map_data() -> dict:
    road = {
        "id": 1001,
        "shape": "polygon",
        "geometry": [[-70, -8], [70, -8], [70, 8], [-70, 8]],
        "color": "road",
        "type": "road",
        "line_width": 0,
    }
    lane_lines = [
        {
            "id": 1100 + index,
            "shape": "line",
            "geometry": [[-70, y], [70, y]],
            "color": "roadline",
            "type": "roadline",
            "line_style": "dashed" if y == 0 else "solid",
            "line_width": 1,
        }
        for index, y in enumerate([-8, 0, 8])
    ]
    return {"road_id_to_remove": [], "road_elements": [road, *lane_lines]}


def _vehicle_element(id_: int, x: float, y: float, heading: float, color: str) -> dict:
    return {
        "id": id_,
        "shape": "polygon",
        "geometry": [[-2.2, -0.9], [2.2, -0.9], [2.2, 0.9], [-2.2, 0.9]],
        "position": [x, y],
        "rotation": heading,
        "color": color,
        "type": "vehicle",
        "line_width": 1,
    }


def _demo_sensor(frame_id: int, sensor_id: str, offset: float, include_map: bool) -> dict:
    t = frame_id / 30.0
    x = ((t * 12 + offset) % 120) - 60
    lead_x = ((t * 10 + offset + 28) % 120) - 60
    yaw = 0.02 * math.sin(t)
    map_data = _demo_map_data() if include_map else {"road_id_to_remove": [], "road_elements": []}
    participants = [
        _vehicle_element(1, x, -4, yaw, "vehicle"),
        _vehicle_element(2, lead_x, 4, 0, "light-blue"),
    ]
    return {
        "id": sensor_id,
        "perception_range": 50,
        "viewport_aspect": 16 / 9,
        "position": [0, 0],
        "yaw": 0,
        "frame": frame_id,
        "map_data": map_data,
        "participant_data": {
            "participant_id_to_create": [1, 2],
            "participant_id_to_remove": [],
            "participants": participants,
        },
    }


def _demo_frame(frame_id: int) -> dict:
    return {
        "frame": frame_id,
        "layout": "grid",
        "sensors": [
            _demo_sensor(frame_id, "camera-overview", 0, True),
            _demo_sensor(frame_id, "camera-follow", 25, True),
        ],
    }


async def _run_demo(
    manager: ConnectionManager, max_fps: int, pause_event: asyncio.Event | None = None
):
    frame_id = 0
    interval = 1.0 / max(1, min(max_fps, 100))
    while True:
        if pause_event is not None:
            await pause_event.wait()

        result = await manager.publish_frame(
            _demo_frame(frame_id),
            frame_id=frame_id,
            wait_ack=True,
            ack_timeout=interval,
            drop_if_busy=True,
        )
        if result["status"] != "dropped":
            frame_id += 1
        await asyncio.sleep(interval)


def create_app(demo: bool = False, max_fps: int = 30):
    """Create the FastAPI app lazily so importing Tactics2D stays lightweight."""

    if FastAPI is None:
        raise RuntimeError(
            "The frontend server requires FastAPI and Uvicorn. "
            "Install them with `pip install fastapi uvicorn[standard]`."
        )

    manager = ConnectionManager()
    static_dir = _frontend_static_dir()

    @asynccontextmanager
    async def lifespan(app):
        if demo:
            app.state.demo_task = asyncio.create_task(_run_demo(manager, max_fps))
        try:
            yield
        finally:
            await _stop_preview_tasks(app)

    app = FastAPI(lifespan=lifespan)
    app.state.connection_manager = manager
    app.state.demo_task = None
    app.state.preview_task = None
    app.state.preview_pause_event = _PreviewPauseEvent(is_set=True)
    app.state.preview_status = {
        "status": "running",
        "source": "live",
        "paused": False,
        "sensor_count": 0,
        "message": "waiting live stream",
    }

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health():
        return {
            "status": "running",
            "clients": manager.client_count,
            "last_ack": manager.last_ack,
            "last_frame_id": manager.last_frame_id,
            "render_busy": manager.is_render_busy,
            "dropped_frames": manager.dropped_frames,
        }

    @app.post("/api/frame")
    async def publish_frame(request: Request):
        payload = await request.json()
        wait_ack = bool(payload.pop("wait_ack", False))
        ack_timeout = float(payload.pop("ack_timeout", 0.05))
        drop_if_busy = bool(payload.pop("drop_if_busy", False))
        frame_id = payload.get("frame", payload.get("frame_id"))

        if app.state.preview_status.get("source") != "live":
            await _stop_preview_tasks(app)
            app.state.preview_pause_event = asyncio.Event()
            app.state.preview_pause_event.set()

        if not app.state.preview_pause_event.is_set():
            app.state.preview_status.update(
                {
                    "status": "paused",
                    "source": "live",
                    "paused": True,
                    "frame": frame_id,
                    "message": "paused",
                }
            )
            return {
                "status": "paused",
                "delivered": 0,
                "acked": False,
                "frame_id": frame_id,
                "dropped_frames": manager.dropped_frames,
            }

        result = await manager.publish_frame(
            payload,
            frame_id=frame_id,
            wait_ack=wait_ack,
            ack_timeout=ack_timeout,
            drop_if_busy=drop_if_busy,
        )
        app.state.preview_status.update(
            {
                "status": "running",
                "source": "live",
                "paused": False,
                "frame": frame_id,
                "sensor_count": len(payload.get("sensors", [])),
                "dropped_frames": result.get("dropped_frames", manager.dropped_frames),
                "message": "live streaming",
            }
        )
        return result

    @app.post("/api/layout")
    async def set_layout(request: Request):
        payload = await request.json()
        layout = payload.get("layout", "grid")
        delivered = await manager.broadcast({"type": "layout.set", "layout": layout})
        return {"status": "ok", "delivered": delivered, "layout": layout}

    @app.get("/api/preview/options")
    async def preview_options():
        from .preview import list_levelx_preview_options

        return list_levelx_preview_options()

    @app.get("/api/preview/status")
    async def preview_status():
        return app.state.preview_status

    @app.post("/api/preview/stop")
    async def stop_preview():
        await _stop_preview_tasks(app)
        app.state.preview_status = {"status": "stopped", "message": "stopped"}
        return app.state.preview_status

    @app.post("/api/preview/pause")
    async def pause_preview():
        app.state.preview_pause_event.clear()
        app.state.preview_status.update({"status": "paused", "paused": True, "message": "paused"})
        return app.state.preview_status

    @app.post("/api/preview/resume")
    async def resume_preview():
        app.state.preview_pause_event.set()
        app.state.preview_status.update(
            {"status": "running", "paused": False, "message": "streaming"}
        )
        return app.state.preview_status

    @app.post("/api/preview/demo")
    async def start_demo(request: Request):
        payload = await request.json()
        await _stop_preview_tasks(app)
        requested_fps = int(payload.get("max_fps", max_fps))
        app.state.preview_pause_event = asyncio.Event()
        app.state.preview_pause_event.set()
        app.state.demo_task = asyncio.create_task(
            _run_demo(manager, requested_fps, app.state.preview_pause_event)
        )
        app.state.preview_status = {
            "status": "running",
            "source": "demo",
            "paused": False,
            "message": "demo running",
        }
        return app.state.preview_status

    @app.post("/api/preview/live")
    async def start_live_preview():
        previous_status = dict(app.state.preview_status)
        await _stop_preview_tasks(app)
        app.state.preview_pause_event = asyncio.Event()
        app.state.preview_pause_event.set()
        app.state.preview_status = {
            "status": "running",
            "source": "live",
            "paused": False,
            "sensor_count": (
                previous_status.get("sensor_count", 0)
                if previous_status.get("source") == "live"
                else 0
            ),
            "message": "waiting live stream",
        }
        if previous_status.get("source") == "live" and "frame" in previous_status:
            app.state.preview_status["frame"] = previous_status["frame"]
        return app.state.preview_status

    @app.post("/api/preview/map")
    async def preview_map(request: Request):
        from .preview import build_map_preview_sensor

        payload = await request.json()
        await _stop_preview_tasks(app)
        try:
            osm_path = payload.get("osm_path")
            if not osm_path:
                raise ValueError("osm_path is required")
            configs = _map_config_from_name(payload.get("map_config"))
            sensor = await _to_thread(
                build_map_preview_sensor,
                Path(osm_path),
                bool(payload.get("lanelet2", True)),
                configs,
            )
        except Exception as exc:
            LOGGER.warning("Map preview failed: %s", exc)
            app.state.preview_status = {
                "status": "error",
                "source": "map",
                "message": str(exc),
            }
            return JSONResponse(status_code=400, content=app.state.preview_status)
        result = await manager.publish_frame(
            {"frame": 0, "layout": "grid", "sensors": [sensor], "remove_missing_sensors": True},
            frame_id=0,
            wait_ack=False,
            drop_if_busy=False,
        )
        app.state.preview_status = {
            "status": "complete",
            "source": "map",
            "sensor_id": sensor["id"],
            "message": "map loaded",
            "result": result,
        }
        return app.state.preview_status

    @app.post("/api/preview/dataset")
    async def preview_dataset(request: Request):
        payload = await request.json()
        await _stop_preview_tasks(app)
        app.state.preview_status = {
            "status": "loading",
            "source": "dataset",
            "message": "loading dataset",
        }
        app.state.preview_pause_event = asyncio.Event()
        app.state.preview_pause_event.set()
        app.state.preview_task = asyncio.create_task(
            _run_levelx_dataset_preview(
                manager, app.state.preview_status, payload, app.state.preview_pause_event
            )
        )
        return app.state.preview_status

    async def _request_json(request: Request) -> dict:
        try:
            return await request.json()
        except Exception:
            return {}

    @app.post("/api/record/start")
    async def record_start(request: Request):
        payload = await _request_json(request)
        if manager.is_recording:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "recording",
                    "name": manager.recording_name,
                    "message": "already recording",
                },
            )
        raw_name = str(payload.get("name") or time.strftime("record-%Y%m%d-%H%M%S"))
        name = re.sub(r"[^A-Za-z0-9_\-一-鿿]", "-", raw_name)
        manager.start_recording(_recordings_dir() / f"{name}.jsonl")
        return {"status": "recording", "name": name, "message": "recording"}

    @app.post("/api/record/stop")
    async def record_stop():
        if not manager.is_recording:
            return {"status": "idle", "message": "not recording"}
        info = manager.stop_recording()
        return {"status": "saved", "message": f"saved {info['frames']} frames", **info}

    @app.post("/api/record/export")
    async def record_export(request: Request):
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            return JSONResponse(
                status_code=501,
                content={"status": "error", "message": "ffmpeg is not available on the server"},
            )
        data = await request.body()
        if not data:
            return JSONResponse(
                status_code=400, content={"status": "error", "message": "empty body"}
            )
        content_type = request.headers.get("content-type") or ""
        suffix = ".webm" if "webm" in content_type else ".mp4"
        try:
            output = await _to_thread(_transcode_to_mp4, ffmpeg, data, suffix)
        except Exception as exc:
            LOGGER.warning("Screen recording export failed: %s", exc)
            return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
        return Response(content=output, media_type="video/mp4")

    @app.get("/api/recordings")
    async def list_recordings():
        directory = _recordings_dir()
        recordings = []
        if directory.is_dir():
            for item in directory.glob("*.jsonl"):
                stat = item.stat()
                recordings.append(
                    {"name": item.stem, "size": stat.st_size, "modified": stat.st_mtime}
                )
        recordings.sort(key=lambda item: item["modified"], reverse=True)
        return {"recordings": recordings, "recording": manager.recording_name}

    @app.post("/api/preview/replay")
    async def preview_replay(request: Request):
        payload = await _request_json(request)
        if not payload.get("name"):
            return JSONResponse(
                status_code=400, content={"status": "error", "message": "name is required"}
            )
        await _stop_preview_tasks(app)
        app.state.preview_status = {
            "status": "loading",
            "source": "replay",
            "message": "loading recording",
        }
        app.state.preview_pause_event = asyncio.Event()
        app.state.preview_pause_event.set()
        app.state.preview_task = asyncio.create_task(
            _run_recording_replay(
                manager, app.state.preview_status, payload, app.state.preview_pause_event
            )
        )
        return app.state.preview_status

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        await websocket.send_text(
            orjson.dumps({"type": "client.count", "clients": manager.client_count}).decode("utf-8")
        )
        if manager._last_snapshot_message is not None:
            await websocket.send_text(orjson.dumps(manager._last_snapshot_message).decode("utf-8"))
        try:
            while True:
                raw_message = await websocket.receive_text()
                message = orjson.loads(raw_message)
                if message.get("type") == "render.ack":
                    manager.record_ack(message.get("frame_id"))
        except WebSocketDisconnect:
            await manager.disconnect(websocket)
            await manager.broadcast({"type": "client.count", "clients": manager.client_count})

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    demo: bool = False,
    max_fps: int = 30,
    open_browser: bool = False,
) -> None:
    """Run the frontend app in the foreground."""

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "The frontend server requires Uvicorn. Install it with "
            "`pip install uvicorn[standard]`."
        ) from exc

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}/")).start()

    uvicorn.run(create_app(demo=demo, max_fps=max_fps), host=host, port=port, log_level="info")


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run the Tactics2D frontend server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--max-fps", type=int, default=30)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    parser.add_argument("--data-root", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.data_root:
        os.environ["TACTICS2D_DATA_ROOT"] = args.data_root
    run_server(args.host, args.port, args.demo, args.max_fps, args.open_browser)


if __name__ == "__main__":
    main()
