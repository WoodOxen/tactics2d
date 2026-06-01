# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Python-side client for the Tactics2D browser frontend."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request


def _default_pid_file() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "tactics2d" / "frontend.pid"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "x") and hasattr(value, "y"):
        return [value.x, value.y]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


class FrontendRenderer:
    """Send sensor frames to a running Tactics2D frontend server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_fps: int = 60,
        timeout: float = 1.0,
        wait_ack: bool = True,
        ack_timeout: float = 0.05,
        drop_if_busy: bool = True,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_fps = min(max(1, int(max_fps)), 100)
        self.wait_ack = wait_ack
        self.ack_timeout = ack_timeout
        self.drop_if_busy = drop_if_busy
        self._last_send_time = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, default=_json_default).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, path: str) -> dict[str, Any]:
        with request.urlopen(f"{self.base_url}{path}", timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.health()
                return True
            except (OSError, error.URLError):
                time.sleep(0.1)
        return False

    def set_layout(self, layout: str) -> dict[str, Any]:
        return self._post("/api/layout", {"layout": layout})

    def preview_status(self) -> dict[str, Any]:
        return self._get("/api/preview/status")

    def pause_preview(self) -> dict[str, Any]:
        return self._post("/api/preview/pause", {})

    def resume_preview(self) -> dict[str, Any]:
        return self._post("/api/preview/resume", {})

    def stop_preview(self) -> dict[str, Any]:
        return self._post("/api/preview/stop", {})

    def preview_demo(self, max_fps: int | None = None) -> dict[str, Any]:
        payload = {}
        if max_fps is not None:
            payload["max_fps"] = max_fps
        return self._post("/api/preview/demo", payload)

    def preview_map(
        self, osm_path: Path | str, lanelet2: bool = True, map_config: str | None = None
    ) -> dict[str, Any]:
        return self._post(
            "/api/preview/map",
            {"osm_path": osm_path, "lanelet2": lanelet2, "map_config": map_config},
        )

    def preview_dataset(
        self,
        dataset: str,
        folder: Path | str,
        file: str | int,
        osm_path: Path | str | None = None,
        map_config: str | None = None,
        lanelet2: bool = True,
        frames: int = 300,
        start_time_ms: int | None = None,
        ids: Iterable[int] | str | None = None,
        follow_id: int | None = None,
        perception_range: float = 80.0,
        max_fps: int | None = None,
        loop: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "dataset": dataset,
            "folder": folder,
            "file": file,
            "osm_path": osm_path,
            "map_config": map_config,
            "lanelet2": lanelet2,
            "frames": frames,
            "start_time_ms": start_time_ms,
            "ids": ids,
            "follow_id": follow_id,
            "perception_range": perception_range,
            "max_fps": self.max_fps if max_fps is None else max_fps,
            "loop": loop,
        }
        return self._post("/api/preview/dataset", payload)

    def send_frame(
        self,
        sensors: Iterable[dict[str, Any]],
        frame: int | None = None,
        layout: str | None = None,
        sensor_id_to_remove: Iterable[str] | None = None,
        remove_missing_sensors: bool = True,
        wait_ack: bool | None = None,
        ack_timeout: float | None = None,
        drop_if_busy: bool | None = None,
    ) -> dict[str, Any]:
        min_interval = 1.0 / self.max_fps
        elapsed = time.monotonic() - self._last_send_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        payload = {"sensors": list(sensors)}
        if frame is not None:
            payload["frame"] = frame
        if layout is not None:
            payload["layout"] = layout
        if sensor_id_to_remove is not None:
            payload["sensor_id_to_remove"] = list(sensor_id_to_remove)

        payload["remove_missing_sensors"] = remove_missing_sensors
        payload["wait_ack"] = self.wait_ack if wait_ack is None else wait_ack
        payload["ack_timeout"] = self.ack_timeout if ack_timeout is None else ack_timeout
        payload["drop_if_busy"] = self.drop_if_busy if drop_if_busy is None else drop_if_busy

        response = self._post("/api/frame", payload)
        self._last_send_time = time.monotonic()
        return response


class FrontendServer:
    """Context manager that owns a background frontend server process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_fps: int = 60,
        open_browser: bool = False,
        pid_file: Path | None = None,
    ):
        self.host = host
        self.port = port
        self.max_fps = max_fps
        self.open_browser = open_browser
        self.pid_file = pid_file
        self.process = None
        self.renderer = FrontendRenderer(host, port, max_fps=max_fps)

    def __enter__(self) -> FrontendRenderer:
        self.process = start_server_process(
            self.host,
            self.port,
            demo=False,
            max_fps=self.max_fps,
            open_browser=self.open_browser,
            pid_file=self.pid_file,
        )
        if not self.renderer.wait_until_ready(timeout=5.0):
            raise RuntimeError(f"Tactics2D frontend did not start on {self.renderer.base_url}.")

        return self.renderer

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        stop_server_process(self.pid_file)


def start_server_process(
    host: str = "127.0.0.1",
    port: int = 8765,
    demo: bool = False,
    max_fps: int = 30,
    open_browser: bool = False,
    pid_file: Path | None = None,
) -> subprocess.Popen:
    """Start the frontend server in a background process and write its PID."""

    command = [
        sys.executable,
        "-m",
        "tactics2d.frontend.server",
        "--host",
        host,
        "--port",
        str(port),
        "--max-fps",
        str(max_fps),
    ]
    if demo:
        command.append("--demo")
    if open_browser:
        command.append("--open")

    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    pid_file = pid_file or _default_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(process.pid), encoding="utf-8")
    return process


def stop_server_process(pid_file: Path | None = None) -> int:
    """Stop a background frontend server process and return the stopped PID."""

    pid_file = pid_file or _default_pid_file()
    if not pid_file.exists():
        raise FileNotFoundError(f"No frontend PID file found at {pid_file}.")

    pid = int(pid_file.read_text(encoding="utf-8").strip())
    os.kill(pid, signal.SIGTERM)
    pid_file.unlink(missing_ok=True)
    return pid
