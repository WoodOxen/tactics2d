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
        self._last_ack = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def last_ack(self) -> Any:
        return self._last_ack

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

    def record_ack(self, frame_id: Any) -> None:
        self._last_ack = frame_id


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
        await manager.broadcast(
            {"type": "frame.update", "frame_id": frame_id, "payload": _demo_frame(frame_id)}
        )
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
        return {"status": "running", "clients": manager.client_count, "last_ack": manager.last_ack}

    @app.post("/api/frame")
    async def publish_frame(request: Request):
        payload = await request.json()
        frame_id = payload.get("frame", payload.get("frame_id"))
        delivered = await manager.broadcast(
            {"type": "frame.update", "frame_id": frame_id, "payload": payload}
        )
        return {"status": "ok", "delivered": delivered, "frame_id": frame_id}

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
