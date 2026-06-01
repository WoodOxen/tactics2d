# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""FastAPI server for the Tactics2D browser frontend."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import orjson

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    FastAPI = None
    Request = None
    WebSocket = None
    WebSocketDisconnect = None
    FileResponse = None
    StaticFiles = None

LOGGER = logging.getLogger(__name__)


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


def _frontend_static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


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


async def _run_demo(manager: ConnectionManager, max_fps: int):
    frame_id = 0
    interval = 1.0 / max(1, min(max_fps, 100))
    while True:
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
            demo_task = getattr(app.state, "demo_task", None)
            if demo_task is not None:
                demo_task.cancel()

    app = FastAPI(lifespan=lifespan)
    app.state.connection_manager = manager

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

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
        return await manager.publish_frame(
            payload,
            frame_id=frame_id,
            wait_ack=wait_ack,
            ack_timeout=ack_timeout,
            drop_if_busy=drop_if_busy,
        )

    @app.post("/api/layout")
    async def set_layout(request: Request):
        payload = await request.json()
        layout = payload.get("layout", "grid")
        delivered = await manager.broadcast({"type": "layout.set", "layout": layout})
        return {"status": "ok", "delivered": delivered, "layout": layout}

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_server(args.host, args.port, args.demo, args.max_fps, args.open_browser)


if __name__ == "__main__":
    main()
