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


def test_frontend_frame_drop_when_browser_is_busy():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    client = TestClient(create_app())
    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "client.count"

        response = client.post("/api/frame", json={"frame": 10, "sensors": []})
        assert response.json()["status"] == "ok"
        assert response.json()["delivered"] == 1
        assert client.get("/health").json()["render_busy"] is True

        response = client.post(
            "/api/frame", json={"frame": 11, "sensors": [], "drop_if_busy": True}
        )
        assert response.json()["status"] == "dropped"
        assert response.json()["dropped_frames"] == 1

        assert websocket.receive_json()["frame_id"] == 10
        websocket.send_json({"type": "render.ack", "frame_id": 10})

    health = client.get("/health").json()
    assert health["last_ack"] == 10
    assert health["render_busy"] is False


def test_frontend_replays_latest_frame_to_new_browser():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    client = TestClient(create_app())
    response = client.post("/api/frame", json={"frame": 15, "sensors": []})
    assert response.json()["delivered"] == 0

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "client.count"
        cached_frame = websocket.receive_json()

    assert cached_frame["type"] == "frame.update"
    assert cached_frame["frame_id"] == 15


def test_demo_frame_contains_sensor_payloads():
    from tactics2d.frontend.server import _demo_frame

    frame = _demo_frame(0)

    assert frame["frame"] == 0
    assert len(frame["sensors"]) == 2
    assert frame["sensors"][0]["map_data"]["road_elements"]


def test_cli_preview_map_arguments():
    from tactics2d.cli import parse_args

    args = parse_args(["preview", "map", "tests/runtime/net2osm_net.osm", "--no-open"])

    assert args.command == "preview"
    assert args.preview_command == "map"
    assert args.open_browser is False


def test_frontend_renderer_frame_controls_are_sent():
    from tactics2d.frontend import FrontendRenderer

    renderer = FrontendRenderer(wait_ack=True, drop_if_busy=True, ack_timeout=0.01)
    payloads = []

    def fake_post(path, payload):
        payloads.append((path, payload))
        return {"status": "ok"}

    renderer._post = fake_post
    renderer.send_frame(
        [{"id": "camera-1"}], frame=5, layout="hierarchical", sensor_id_to_remove=["camera-0"]
    )

    assert payloads[0][0] == "/api/frame"
    assert payloads[0][1]["wait_ack"] is True
    assert payloads[0][1]["drop_if_busy"] is True
    assert payloads[0][1]["sensor_id_to_remove"] == ["camera-0"]
