# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frontend server smoke tests."""

from pathlib import Path

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


def test_frontend_replays_snapshot_to_new_browser():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    first_frame = {
        "frame": 1,
        "sensors": [
            {
                "id": "camera-1",
                "perception_range": 50,
                "position": [0, 0],
                "map_data": {
                    "road_id_to_remove": [],
                    "road_elements": [
                        {
                            "id": 10,
                            "shape": "line",
                            "geometry": [[0, 0], [1, 1]],
                            "color": "roadline",
                            "type": "roadline",
                        }
                    ],
                },
                "participant_data": {
                    "participant_id_to_create": [1],
                    "participant_id_to_remove": [],
                    "participants": [{"id": 1, "shape": "circle", "position": [0, 0]}],
                },
            }
        ],
    }
    next_frame = {
        "frame": 2,
        "sensors": [
            {
                "id": "camera-1",
                "perception_range": 50,
                "position": [1, 0],
                "map_data": {"road_id_to_remove": [], "road_elements": []},
                "participant_data": {
                    "participant_id_to_create": [],
                    "participant_id_to_remove": [],
                    "participants": [{"id": 1, "shape": "circle", "position": [1, 0]}],
                },
            }
        ],
    }

    client = TestClient(create_app())
    client.post("/api/frame", json=first_frame)
    client.post("/api/frame", json=next_frame)

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "client.count"
        cached_frame = websocket.receive_json()

    sensor = cached_frame["payload"]["sensors"][0]
    assert cached_frame["frame_id"] == 2
    assert sensor["map_data"]["road_elements"][0]["id"] == 10
    assert sensor["participant_data"]["participants"][0]["position"] == [1, 0]


def test_preview_options_endpoint_contains_levelx_defaults():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    client = TestClient(create_app())
    response = client.get("/api/preview/options")

    assert response.status_code == 200
    assert "highD" in response.json()["levelx_datasets"]
    assert response.json()["defaults"]["folder"] == "/mnt/server_data/Datasets/highD/data"


def test_preview_map_endpoint_publishes_frame():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    client = TestClient(create_app())
    response = client.post(
        "/api/preview/map", json={"osm_path": "tests/runtime/net2osm_net.osm", "lanelet2": True}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert client.get("/health").json()["last_frame_id"] == 0


def test_preview_pause_and_resume_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    client = TestClient(create_app())

    response = client.post("/api/preview/pause", json={})
    assert response.status_code == 200
    assert response.json()["status"] == "paused"
    assert response.json()["paused"] is True

    response = client.post("/api/preview/resume", json={})
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["paused"] is False


def test_live_preview_endpoint_sets_live_status():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    client = TestClient(create_app())
    response = client.post("/api/preview/live", json={})

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["source"] == "live"
    assert client.get("/api/preview/status").json()["source"] == "live"


def test_live_preview_endpoint_preserves_existing_live_snapshot():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    frame = {
        "frame": 7,
        "sensors": [
            {
                "id": "camera-1",
                "map_data": {
                    "road_id_to_remove": [],
                    "road_elements": [
                        {
                            "id": 10,
                            "shape": "line",
                            "geometry": [[0, 0], [1, 1]],
                            "color": "roadline",
                            "type": "roadline",
                        }
                    ],
                },
                "participant_data": {
                    "participant_id_to_create": [],
                    "participant_id_to_remove": [],
                    "participants": [],
                },
            }
        ],
    }

    client = TestClient(create_app())
    client.post("/api/frame", json=frame)
    response = client.post("/api/preview/live", json={})

    assert response.json()["sensor_count"] == 1
    assert response.json()["frame"] == 7

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "client.count"
        cached_frame = websocket.receive_json()

    sensor = cached_frame["payload"]["sensors"][0]
    assert sensor["map_data"]["road_elements"][0]["id"] == 10


def test_frame_endpoint_marks_programmatic_stream_as_live():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    client = TestClient(create_app())
    response = client.post("/api/frame", json={"frame": 42, "sensors": [{"id": "camera"}]})

    assert response.status_code == 200
    status = client.get("/api/preview/status").json()
    assert status["status"] == "running"
    assert status["source"] == "live"
    assert status["frame"] == 42
    assert status["sensor_count"] == 1


def test_paused_live_preview_drops_programmatic_frames():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tactics2d.frontend.server import create_app

    client = TestClient(create_app())
    client.post("/api/preview/live", json={})
    client.post("/api/preview/pause", json={})
    response = client.post("/api/frame", json={"frame": 43, "sensors": [{"id": "camera"}]})

    assert response.status_code == 200
    assert response.json()["status"] == "paused"
    assert client.get("/api/preview/status").json()["paused"] is True


def test_demo_frame_contains_sensor_payloads():
    from tactics2d.frontend.server import _demo_frame

    frame = _demo_frame(0)

    assert frame["frame"] == 0
    assert len(frame["sensors"]) == 2
    assert frame["sensors"][0]["map_data"]["road_elements"]
    assert frame["sensors"][0]["viewport_aspect"] == 16 / 9


def test_frontend_programmatic_entrypoints_are_exported():
    import tactics2d.frontend as frontend

    assert frontend.FrontendRenderer
    assert frontend.FrontendServer
    assert frontend.ensure_frontend_server
    assert frontend.start_server_process
    assert frontend.stop_server_process


def test_cli_preview_map_arguments():
    from tactics2d.cli import parse_args

    args = parse_args(
        ["preview", "map", "tests/runtime/net2osm_net.osm", "--map-config", "highD_1", "--no-open"]
    )

    assert args.command == "preview"
    assert args.preview_command == "map"
    assert args.map_config == "highD_1"
    assert args.open_browser is False


def test_cli_preview_dataset_arguments():
    from tactics2d.cli import parse_args

    args = parse_args(
        [
            "preview",
            "dataset",
            "--dataset",
            "highD",
            "--folder",
            "data/highD",
            "--file",
            "11",
            "--no-open",
            "--frames",
            "10",
            "--loop",
        ]
    )

    assert args.command == "preview"
    assert args.preview_command == "dataset"
    assert args.dataset == "highD"
    assert args.folder == Path("data/highD")
    assert args.file == "11"
    assert args.open_browser is False
    assert args.frames == 10
    assert args.loop is True


def test_levelx_preview_resolves_map_config_from_recording():
    from tactics2d.frontend.preview import resolve_levelx_map_config

    name, config = resolve_levelx_map_config("highD", "11")

    assert name == "highD_1"
    assert config["osm_file"] == "highD_1.osm"


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


def test_frontend_renderer_preview_dataset_payload():
    from tactics2d.frontend import FrontendRenderer

    renderer = FrontendRenderer(max_fps=75)
    payloads = []

    def fake_post(path, payload):
        payloads.append((path, payload))
        return {"status": "loading"}

    renderer._post = fake_post
    renderer.preview_dataset(
        dataset="highD",
        folder=Path("data/highD"),
        file="11",
        osm_path=Path("data/highD_map/highD_1.osm"),
        map_config="highD_1",
        frames=120,
        ids=[1, 2],
        follow_id=1,
        loop=True,
    )

    assert payloads[0][0] == "/api/preview/dataset"
    assert payloads[0][1]["folder"] == Path("data/highD")
    assert payloads[0][1]["max_fps"] == 75
    assert payloads[0][1]["loop"] is True
    assert payloads[0][1]["ids"] == [1, 2]


def test_frontend_renderer_preview_controls_call_endpoints():
    from tactics2d.frontend import FrontendRenderer

    renderer = FrontendRenderer()
    calls = []

    def fake_get(path):
        calls.append(("GET", path, None))
        return {"status": "idle"}

    def fake_post(path, payload):
        calls.append(("POST", path, payload))
        return {"status": "ok"}

    renderer._get = fake_get
    renderer._post = fake_post

    assert renderer.preview_status()["status"] == "idle"
    renderer.pause_preview()
    renderer.resume_preview()
    renderer.stop_preview()
    renderer.preview_demo(max_fps=45)
    renderer.preview_map("data/map.osm", lanelet2=False, map_config="plain")

    assert calls == [
        ("GET", "/api/preview/status", None),
        ("POST", "/api/preview/pause", {}),
        ("POST", "/api/preview/resume", {}),
        ("POST", "/api/preview/stop", {}),
        ("POST", "/api/preview/demo", {"max_fps": 45}),
        (
            "POST",
            "/api/preview/map",
            {"osm_path": "data/map.osm", "lanelet2": False, "map_config": "plain"},
        ),
    ]
