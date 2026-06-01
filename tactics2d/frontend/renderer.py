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
        self, host: str = "127.0.0.1", port: int = 8765, max_fps: int = 60, timeout: float = 1.0
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_fps = min(max(1, int(max_fps)), 100)
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

    def health(self) -> dict[str, Any]:
        with request.urlopen(f"{self.base_url}/health", timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

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

    def send_frame(
        self, sensors: Iterable[dict[str, Any]], frame: int | None = None, layout: str | None = None
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

        response = self._post("/api/frame", payload)
        self._last_send_time = time.monotonic()
        return response


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
