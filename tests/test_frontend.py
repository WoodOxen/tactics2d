# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frontend server smoke tests."""

import pytest


def test_frontend_app_health_endpoint():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "running"

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json() == {"type": "client.count", "clients": 1}
        websocket.send_json({"type": "render.ack", "frame_id": 3})

    assert client.get("/health").json()["last_ack"] == 3


def test_demo_frame_contains_sensor_payloads():
    from tactics2d.frontend.server import _demo_frame

    frame = _demo_frame(0)

    assert frame["frame"] == 0
    assert len(frame["sensors"]) == 2
    assert frame["sensors"][0]["map_data"]["road_elements"]
