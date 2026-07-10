# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frontend server smoke tests."""

import asyncio
import json
import re
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib import error as url_error

import pytest

# tactics2d internal module imports (lifted from test function bodies)
import tactics2d.cli as cli_module
import tactics2d.dataset_parser as dataset_parser
import tactics2d.display.renderers.web as frontend
import tactics2d.display.renderers.web.preview as preview
import tactics2d.display.renderers.web.renderer as renderer_module
import tactics2d.display.renderers.web.server as server_module
import tactics2d.display.sensor as sensor_module
import tactics2d.map.parser as map_parser


def test_frontend_app_health_endpoint():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(server_module.create_app())
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

    client = TestClient(server_module.create_app())
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

    client = TestClient(server_module.create_app())
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

    client = TestClient(server_module.create_app())
    client.post("/api/frame", json=first_frame)
    client.post("/api/frame", json=next_frame)

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "client.count"
        cached_frame = websocket.receive_json()

    sensor = cached_frame["payload"]["sensors"][0]
    assert cached_frame["frame_id"] == 2
    assert sensor["map_data"]["road_elements"][0]["id"] == 10
    assert sensor["participant_data"]["participants"][0]["position"] == [1, 0]


def test_preview_options_endpoint_derives_defaults_from_discovery(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    folder = tmp_path / "rounD" / "data"
    folder.mkdir(parents=True)
    (folder / "00_tracks.csv").touch()
    monkeypatch.setenv(preview.DATA_ROOT_ENV, str(tmp_path))

    client = TestClient(server_module.create_app())
    response = client.get("/api/preview/options")

    assert response.status_code == 200
    payload = response.json()
    assert "highD" in payload["levelx_datasets"]
    assert str(tmp_path.resolve()) in payload["data_roots"]
    assert payload["defaults"]["dataset"] == "rounD"
    assert payload["defaults"]["folder"] == str(folder)
    assert payload["defaults"]["file"] == "0"


def test_get_data_roots_reads_environment(monkeypatch, tmp_path):
    root = tmp_path / "roots"
    root.mkdir()
    monkeypatch.setenv(preview.DATA_ROOT_ENV, str(root))

    assert preview.get_data_roots()[0] == root.resolve()


def test_get_data_roots_skips_missing_directories(monkeypatch, tmp_path):
    monkeypatch.setenv(preview.DATA_ROOT_ENV, str(tmp_path / "missing"))

    assert (tmp_path / "missing").resolve() not in preview.get_data_roots()


def test_discover_levelx_datasets_finds_official_layout(monkeypatch, tmp_path):
    folder = tmp_path / "highD" / "data"
    folder.mkdir(parents=True)
    (folder / "07_tracks.csv").touch()
    (folder / "12_tracks.csv").touch()
    (folder / "12_tracksMeta.csv").touch()
    (folder / "notes.txt").touch()
    monkeypatch.setenv(preview.DATA_ROOT_ENV, str(tmp_path))

    discovered = preview.discover_levelx_datasets()

    entry = next(item for item in discovered if item["dataset"] == "highD")
    assert entry["folder"] == str(folder)
    assert entry["files"] == [7, 12]


def test_discover_levelx_datasets_supports_flat_layout(monkeypatch, tmp_path):
    folder = tmp_path / "uniD"
    folder.mkdir(parents=True)
    (folder / "03_tracks.csv").touch()
    monkeypatch.setenv(preview.DATA_ROOT_ENV, str(tmp_path))

    discovered = preview.discover_levelx_datasets()

    entry = next(item for item in discovered if item["dataset"] == "uniD")
    assert entry["folder"] == str(folder)
    assert entry["files"] == [3]


def test_record_endpoints_write_jsonl(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("TACTICS2D_RECORD_DIR", str(tmp_path))
    client = TestClient(server_module.create_app())

    response = client.post("/api/record/start", json={"name": "test-rec"})
    assert response.status_code == 200
    assert response.json()["status"] == "recording"

    client.post("/api/frame", json={"frame": 1, "sensors": []})
    client.post("/api/frame", json={"frame": 2, "sensors": []})

    response = client.post("/api/record/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "saved"
    assert response.json()["frames"] == 2

    recording_file = tmp_path / "test-rec.jsonl"
    assert recording_file.is_file()
    lines = [line for line in recording_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["frame_id"] == 1

    response = client.get("/api/recordings")
    assert any(item["name"] == "test-rec" for item in response.json()["recordings"])


def test_record_start_twice_returns_conflict(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("TACTICS2D_RECORD_DIR", str(tmp_path))
    client = TestClient(server_module.create_app())

    assert client.post("/api/record/start", json={"name": "first"}).status_code == 200
    response = client.post("/api/record/start", json={"name": "second"})
    assert response.status_code == 409
    assert response.json()["name"] == "first"


def test_record_export_transcodes_to_mp4(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    ffmpeg = server_module._find_ffmpeg()
    if ffmpeg is None:
        pytest.skip("ffmpeg is not available")

    source = tmp_path / "input.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.4:size=64x64:rate=10",
            str(source),
        ],
        check=True,
    )

    client = TestClient(server_module.create_app())
    response = client.post(
        "/api/record/export",
        content=source.read_bytes(),
        headers={"Content-Type": "video/mp4"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content[4:8] == b"ftyp"


def test_publish_frame_serializes_numpy_payload(tmp_path):
    """Parser-built payloads (e.g. inD pedestrians) may carry numpy scalars."""
    np = pytest.importorskip("numpy")

    manager = server_module.ConnectionManager()
    manager.start_recording(tmp_path / "numpy.jsonl")
    payload = {
        "frame": np.int64(1600),
        "sensors": [
            {
                "id": "cam",
                "position": [np.float64(1.5), np.float64(-2.5)],
                "participant_data": {
                    "participants": [{"id": 9, "position": np.array([1.0, 2.0])}]
                },
            }
        ],
    }
    asyncio.run(manager.publish_frame(payload, frame_id=np.int64(1600)))
    manager.stop_recording()

    lines = [json.loads(line) for line in (tmp_path / "numpy.jsonl").read_text().splitlines()]
    sensor = lines[-1]["payload"]["sensors"][0]
    assert lines[-1]["frame_id"] == 1600
    assert sensor["position"] == [1.5, -2.5]
    assert sensor["participant_data"]["participants"][0]["position"] == [1.0, 2.0]


def test_record_export_pads_to_aligned_dimensions(tmp_path):
    """Widths that are not a multiple of 4 crash buggy hardware H.264 decoders."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    ffmpeg = server_module._find_ffmpeg()
    if ffmpeg is None:
        pytest.skip("ffmpeg is not available")

    source = tmp_path / "input.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.4:size=62x30:rate=10",
            str(source),
        ],
        check=True,
    )

    client = TestClient(server_module.create_app())
    response = client.post(
        "/api/record/export",
        content=source.read_bytes(),
        headers={"Content-Type": "video/mp4"},
    )
    assert response.status_code == 200

    output = tmp_path / "output.mp4"
    output.write_bytes(response.content)
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(output)], capture_output=True, text=True
    )
    match = re.search(r" (\d{2,5})x(\d{2,5})[ ,]", probe.stderr)
    assert match is not None, probe.stderr
    width, height = int(match.group(1)), int(match.group(2))
    assert width % 4 == 0 and height % 4 == 0
    assert (width, height) == (64, 32)


def test_record_export_rejects_invalid_video():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    if server_module._find_ffmpeg() is None:
        pytest.skip("ffmpeg is not available")

    client = TestClient(server_module.create_app())
    response = client.post(
        "/api/record/export",
        content=b"not a video",
        headers={"Content-Type": "video/mp4"},
    )

    assert response.status_code == 400


def test_replay_endpoint_requires_name(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("TACTICS2D_RECORD_DIR", str(tmp_path))
    client = TestClient(server_module.create_app())

    response = client.post("/api/preview/replay", json={})
    assert response.status_code == 400


def test_replay_task_streams_recording(monkeypatch, tmp_path):
    monkeypatch.setenv("TACTICS2D_RECORD_DIR", str(tmp_path))
    recording_file = tmp_path / "clip.jsonl"
    frames = [
        {"frame_id": index, "time": 0.0, "payload": {"frame": index, "sensors": []}}
        for index in range(3)
    ]
    recording_file.write_text("\n".join(json.dumps(frame) for frame in frames) + "\n")

    manager = server_module.ConnectionManager()
    status = {}
    pause_event = asyncio.Event()
    pause_event.set()

    asyncio.run(
        server_module._run_recording_replay(
            manager, status, {"name": "clip", "max_fps": 100}, pause_event
        )
    )

    assert status["status"] == "complete"
    assert status["total_frames"] == 3
    assert status["sent_frames"] == 3


def test_discover_maps_includes_scanned_osm_files(monkeypatch, tmp_path):
    osm = tmp_path / "custom" / "my_map.osm"
    osm.parent.mkdir(parents=True)
    osm.touch()
    monkeypatch.setenv(preview.DATA_ROOT_ENV, str(tmp_path))

    maps = preview.discover_maps()

    entry = next(item for item in maps if item["osm_path"] == str(osm.resolve()))
    assert entry["dataset"] is None
    assert entry["name"] == str(Path("custom") / "my_map.osm")


def test_preview_map_endpoint_publishes_frame():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(server_module.create_app())
    response = client.post(
        "/api/preview/map", json={"osm_path": "tests/cases/OsmSamples/cross.osm", "lanelet2": True}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert client.get("/health").json()["last_frame_id"] == 0


def test_preview_pause_and_resume_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(server_module.create_app())

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

    client = TestClient(server_module.create_app())
    response = client.post("/api/preview/live", json={})

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["source"] == "live"
    assert client.get("/api/preview/status").json()["source"] == "live"


def test_live_preview_endpoint_preserves_existing_live_snapshot():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

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

    client = TestClient(server_module.create_app())
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

    client = TestClient(server_module.create_app())
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

    client = TestClient(server_module.create_app())
    client.post("/api/preview/live", json={})
    client.post("/api/preview/pause", json={})
    response = client.post("/api/frame", json={"frame": 43, "sensors": [{"id": "camera"}]})

    assert response.status_code == 200
    assert response.json()["status"] == "paused"
    assert client.get("/api/preview/status").json()["paused"] is True


def test_demo_frame_contains_sensor_payloads():
    frame = server_module._demo_frame(0)

    assert frame["frame"] == 0
    assert len(frame["sensors"]) == 2
    assert frame["sensors"][0]["map_data"]["road_elements"]
    assert frame["sensors"][0]["viewport_aspect"] == 16 / 9


def test_frontend_programmatic_entrypoints_are_exported():

    assert frontend.FrontendRenderer
    assert frontend.FrontendServer
    assert frontend.ensure_frontend_server
    assert frontend.start_server_process
    assert frontend.stop_server_process


def test_cli_preview_map_arguments():

    args = cli_module.parse_args(
        [
            "preview",
            "map",
            "tests/cases/OsmSamples/cross.osm",
            "--map-config",
            "highD_1",
            "--no-open",
        ]
    )

    assert args.command == "preview"
    assert args.preview_command == "map"
    assert args.map_config == "highD_1"
    assert args.open_browser is False


def test_cli_preview_dataset_arguments():

    args = cli_module.parse_args(
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


def test_cli_command_handlers_call_frontend(monkeypatch, capsys, tmp_path):
    calls = []

    class FakeRenderer:
        def __init__(self, host="127.0.0.1", port=8765, max_fps=60):
            self.host = host
            self.port = port
            self.max_fps = max_fps
            self.base_url = f"http://{host}:{port}"

        def wait_until_ready(self, timeout):
            calls.append(("wait", self.host, self.port, timeout))
            return True

        def health(self):
            return {"status": "running", "port": self.port}

        def preview_demo(self, **kwargs):
            calls.append(("preview_demo", kwargs))

        def preview_map(self, *args, **kwargs):
            calls.append(("preview_map", args, kwargs))

        def preview_dataset(self, **kwargs):
            calls.append(("preview_dataset", kwargs))

    monkeypatch.setattr(frontend, "FrontendRenderer", FakeRenderer)
    monkeypatch.setattr(
        renderer_module,
        "start_server_process",
        lambda *args: calls.append(("start_process", args)) or SimpleNamespace(pid=42),
    )
    monkeypatch.setattr(frontend, "run_server", lambda *args: calls.append(("run_server", args)))
    monkeypatch.setattr(
        renderer_module,
        "stop_server_process",
        lambda pid_file: calls.append(("stop", pid_file)) or 42,
    )
    monkeypatch.setattr(
        preview,
        "ensure_frontend_server",
        lambda host, port, max_fps, open_browser: calls.append(
            ("ensure", host, port, max_fps, open_browser)
        )
        or FakeRenderer(host, port, max_fps),
    )

    cli_module._start(
        SimpleNamespace(
            background=True,
            host="127.0.0.1",
            port=8765,
            demo=True,
            max_fps=30,
            open_browser=False,
            pid_file=tmp_path / "frontend.pid",
        )
    )
    cli_module._start(
        SimpleNamespace(
            background=False,
            host="0.0.0.0",
            port=9000,
            demo=False,
            max_fps=60,
            open_browser=True,
            pid_file=None,
        )
    )
    cli_module._stop(SimpleNamespace(pid_file=tmp_path / "frontend.pid"))
    cli_module._status(SimpleNamespace(host="127.0.0.1", port=8765))
    assert json.loads(capsys.readouterr().out)["status"] == "running"

    cli_module._preview(
        SimpleNamespace(
            preview_command="demo", host="127.0.0.1", port=8765, max_fps=30, open_browser=False
        )
    )
    cli_module._preview(
        SimpleNamespace(
            preview_command="map",
            host="127.0.0.1",
            port=8765,
            max_fps=30,
            open_browser=False,
            osm=Path("map.osm"),
            lanelet2=False,
            map_config="plain",
        )
    )
    cli_module._preview(
        SimpleNamespace(
            preview_command="dataset",
            host="127.0.0.1",
            port=8765,
            max_fps=30,
            open_browser=False,
            dataset="highD",
            folder=Path("data/highD"),
            file="11",
            osm=Path("map.osm"),
            map_config="highD_1",
            lanelet2=True,
            frames=5,
            start_time_ms=100,
            ids=[1],
            follow_id=1,
            perception_range=50,
            loop=True,
        )
    )
    with pytest.raises(SystemExit):
        cli_module._preview(SimpleNamespace(preview_command=None))

    assert (
        "start_process",
        ("127.0.0.1", 8765, True, 30, False, tmp_path / "frontend.pid"),
    ) in calls
    assert ("run_server", ("0.0.0.0", 9000, False, 60, True)) in calls
    assert ("stop", tmp_path / "frontend.pid") in calls
    assert any(call[0] == "preview_demo" for call in calls)
    assert any(call[0] == "preview_map" for call in calls)
    assert any(call[0] == "preview_dataset" for call in calls)


def test_cli_main_dispatch(monkeypatch):

    calls = []
    handlers = {
        "start": lambda args: calls.append("start"),
        "stop": lambda args: calls.append("stop"),
        "status": lambda args: calls.append("status"),
        "preview": lambda args: calls.append("preview"),
    }
    monkeypatch.setattr(cli_module, "_start", handlers["start"])
    monkeypatch.setattr(cli_module, "_stop", handlers["stop"])
    monkeypatch.setattr(cli_module, "_status", handlers["status"])
    monkeypatch.setattr(cli_module, "_preview", handlers["preview"])

    commands = iter(["start", "stop", "status", "preview", None, None])

    def fake_parse_args(argv=None):
        command = next(commands)
        calls.append(("parse", argv, command))
        return SimpleNamespace(command=command)

    monkeypatch.setattr(cli_module, "parse_args", fake_parse_args)

    cli_module.main([])
    cli_module.main([])
    cli_module.main([])
    cli_module.main([])
    cli_module.main([])

    assert calls == [
        ("parse", [], "start"),
        "start",
        ("parse", [], "stop"),
        "stop",
        ("parse", [], "status"),
        "status",
        ("parse", [], "preview"),
        "preview",
        ("parse", [], None),
        ("parse", ["--help"], None),
    ]


def test_levelx_preview_resolves_map_config_from_recording():
    name, config = preview.resolve_levelx_map_config("highD", "11")

    assert name == "highD_1"
    assert config["osm_file"] == "highD_1.osm"


def test_levelx_preview_option_and_path_helpers(monkeypatch, tmp_path):

    fake_configs = [
        ("ignored", {"name": "no dataset"}),
        (
            "highD_1",
            {
                "dataset": "highD",
                "osm_file": "highD_1.osm",
                "trajectory_files": [11],
                "name": "HighD location 1",
            },
        ),
    ]
    monkeypatch.setattr(preview, "iter_map_configs", lambda: iter(fake_configs))

    assert preview.canonical_levelx_dataset("HIGHd") == "highD"
    with pytest.raises(KeyError):
        preview.canonical_levelx_dataset("missing")

    assert preview.extract_levelx_file_id(7) == 7
    assert preview.extract_levelx_file_id("recording_011_tracks.csv") == 11
    with pytest.raises(ValueError):
        preview.extract_levelx_file_id("tracks.csv")

    options = preview.list_levelx_preview_options()
    assert options["map_configs"] == [
        {
            "name": "highD_1",
            "dataset": "highD",
            "osm_file": "highD_1.osm",
            "trajectory_files": [11],
            "description": "HighD location 1",
        }
    ]

    name, config = preview.resolve_levelx_map_config("highD", "11", map_config="highd_1")
    assert name == "highD_1"
    assert config["osm_file"] == "highD_1.osm"

    name, _ = preview.resolve_levelx_map_config("highD", "11", osm_path=Path("HIGHD_1.osm"))
    assert name == "highD_1"

    assert preview.resolve_levelx_map_config("highD", "99") == (None, None)

    explicit_osm = tmp_path / "explicit.osm"
    explicit_osm.write_text("<osm />", encoding="utf-8")
    assert preview.resolve_levelx_osm_path("highD", tmp_path, None, explicit_osm) == explicit_osm

    folder = tmp_path / "recordings"
    folder.mkdir()
    discovered_osm = tmp_path / "map" / "local_test.osm"
    discovered_osm.parent.mkdir()
    discovered_osm.write_text("<osm />", encoding="utf-8")
    assert (
        preview.resolve_levelx_osm_path("highD", folder, {"osm_file": "local_test.osm"})
        == discovered_osm.resolve()
    )

    with pytest.raises(ValueError):
        preview.resolve_levelx_osm_path("highD", folder, None)
    with pytest.raises(FileNotFoundError):
        preview.resolve_levelx_osm_path("highD", folder, {"osm_file": "missing.osm"})


def test_levelx_preview_scene_uses_follow_vehicle_pose():

    class FakeTrajectory:
        def __init__(self, states):
            self.states = states

        def has_state(self, frame):
            return frame in self.states

        def get_state(self, frame):
            return self.states[frame]

    class FakeCamera:
        max_perception_distance = 88

        def update(
            self,
            frame,
            participants,
            active_ids,
            prev_road_id_set,
            prev_participant_id_set,
            point,
            heading,
        ):
            assert frame == 40
            assert active_ids == [1, 2]
            assert point.x == 8
            assert point.y == 9
            assert heading == 1.2
            return (
                {
                    "map_data": {"road_id_to_remove": [], "road_elements": []},
                    "participant_data": {
                        "participant_id_to_create": [],
                        "participant_id_to_remove": [],
                        "participants": [],
                    },
                },
                {"road"},
                {"participant"},
            )

    participants = {
        1: SimpleNamespace(
            trajectory=FakeTrajectory({40: SimpleNamespace(location=(1, 2), heading=0.5)})
        ),
        2: SimpleNamespace(
            trajectory=FakeTrajectory({40: SimpleNamespace(location=(8, 9), heading=1.2)})
        ),
    }
    scene = preview.LevelXPreviewScene(
        dataset_name="highD",
        file_id=11,
        sensor_id="sensor",
        actual_time_range=(0, 80),
        map_=SimpleNamespace(),
        camera=FakeCamera(),
        participants=participants,
        fallback_position=(3, 4),
        follow_id=2,
    )

    assert list(scene.iter_frames()) == [0, 40, 80]
    sensor = scene.sensor_for_frame(40)
    assert sensor["position"] == [8, 9]
    assert sensor["yaw"] == 1.2
    assert sensor["perception_range"] == 88
    assert scene.prev_road_id_set == {"road"}
    assert scene.prev_participant_id_set == {"participant"}
    assert preview._choose_camera_pose(participants, [], 40, (3, 4)) == ((3, 4), 0)


def test_load_levelx_preview_scene_uses_parser_outputs(monkeypatch, tmp_path):

    calls = {}

    class FakeLevelXParser:
        def __init__(self, dataset_name):
            calls["dataset_name"] = dataset_name

        def get_time_range(self, file_id, folder):
            calls["get_time_range"] = (file_id, folder)
            return (1000, 2000)

        def parse_trajectory(self, file_id, folder, time_range, ids):
            calls["parse_trajectory"] = (file_id, folder, time_range, ids)
            return ({3: SimpleNamespace()}, time_range)

    class FakeOSMParser:
        def __init__(self, lanelet2):
            calls["lanelet2"] = lanelet2

        def parse(self, path, configs):
            calls["osm_parse"] = (path, configs)
            return SimpleNamespace(boundary=(0, 10, 0, 20))

    class FakeBEVCamera:
        def __init__(self, id_, map_, perception_range):
            calls["camera"] = (id_, map_, perception_range)
            self.max_perception_distance = perception_range

    monkeypatch.setattr(dataset_parser, "LevelXParser", FakeLevelXParser)
    monkeypatch.setattr(map_parser, "OSMParser", FakeOSMParser)
    monkeypatch.setattr(sensor_module, "BEVCamera", FakeBEVCamera)
    monkeypatch.setattr(preview, "resolve_levelx_map_config", lambda *args: ("custom", {"k": "v"}))

    osm_path = tmp_path / "map.osm"
    osm_path.write_text("<osm />", encoding="utf-8")
    scene = preview.load_levelx_preview_scene(
        dataset="highD",
        folder=tmp_path,
        file="11",
        osm_path=osm_path,
        lanelet2=False,
        frames=3,
        start_time_ms=900,
        ids=[3],
        follow_id=3,
        perception_range=33,
    )

    assert scene.sensor_id == "highD-11"
    assert scene.actual_time_range == (1000, 1080)
    assert scene.fallback_position == (5, 10)
    assert scene.follow_id == 3
    assert calls["dataset_name"] == "highD"
    assert calls["parse_trajectory"][2] == (1000, 1080)
    assert calls["parse_trajectory"][3] == [3]
    assert calls["lanelet2"] is False
    assert calls["camera"][2] == 33


def test_load_levelx_preview_scene_rejects_empty_recording(monkeypatch, tmp_path):

    class EmptyLevelXParser:
        def __init__(self, dataset_name):
            pass

        def get_time_range(self, file_id, folder):
            return (0, 40)

        def parse_trajectory(self, file_id, folder, time_range, ids):
            return ({}, time_range)

    monkeypatch.setattr(dataset_parser, "LevelXParser", EmptyLevelXParser)
    monkeypatch.setattr(preview, "resolve_levelx_map_config", lambda *args: (None, None))
    monkeypatch.setattr(preview, "resolve_levelx_osm_path", lambda *args: tmp_path / "map.osm")

    with pytest.raises(RuntimeError):
        preview.load_levelx_preview_scene("highD", tmp_path, 11)


def test_stream_levelx_preview_reports_sent_and_dropped_frames(monkeypatch, tmp_path):

    class FakeScene:
        sensor_id = "highD-11"
        actual_time_range = (0, 80)

        def iter_frames(self):
            return [0, 40, 80]

        def sensor_for_frame(self, frame):
            return {"id": "camera", "frame": frame}

    class FakeRenderer:
        base_url = "http://127.0.0.1:8765"

        def __init__(self):
            self.calls = []

        def send_frame(self, sensors, **kwargs):
            self.calls.append((sensors, kwargs))
            if kwargs["frame"] == 40:
                return {"status": "dropped", "dropped_frames": 1}
            return {"status": "ok", "dropped_frames": 2}

    fake_scene = FakeScene()
    fake_renderer = FakeRenderer()
    monkeypatch.setattr(preview, "load_levelx_preview_scene", lambda **kwargs: fake_scene)
    monkeypatch.setattr(
        preview, "ensure_frontend_server", lambda host, port, max_fps, open_browser: fake_renderer
    )
    monkeypatch.setattr(preview.time, "sleep", lambda seconds: None)

    result = preview.stream_levelx_preview(
        "highD", tmp_path, 11, host="0.0.0.0", port=9000, max_fps=20, open_browser=False
    )

    assert result.base_url == fake_renderer.base_url
    assert result.sent_frames == 2
    assert result.dropped_frames == 2
    assert fake_renderer.calls[0][1]["ack_timeout"] == 0.05


def test_frontend_renderer_frame_controls_are_sent():

    renderer = frontend.FrontendRenderer(wait_ack=True, drop_if_busy=True, ack_timeout=0.01)
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

    renderer = frontend.FrontendRenderer(max_fps=75)
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

    renderer = frontend.FrontendRenderer()
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


def test_frontend_renderer_http_helpers_and_ready_loop(monkeypatch, tmp_path):

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert renderer_module._default_pid_file() == tmp_path / "tactics2d" / "frontend.pid"

    class PointLike:
        x = 1
        y = 2

    class ArrayLike:
        def tolist(self):
            return [3, 4]

    assert renderer_module._json_default(Path("map.osm")) == "map.osm"
    assert renderer_module._json_default(PointLike()) == [1, 2]
    assert renderer_module._json_default(ArrayLike()) == [3, 4]
    assert sorted(renderer_module._json_default({2, 1})) == [1, 2]
    with pytest.raises(TypeError):
        renderer_module._json_default(object())

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    requests = []

    def fake_urlopen(http_request, timeout):
        requests.append((http_request, timeout))
        if isinstance(http_request, str):
            return FakeResponse({"status": "running"})

        return FakeResponse({"payload": json.loads(http_request.data.decode("utf-8"))})

    monkeypatch.setattr(renderer_module.request, "urlopen", fake_urlopen)
    renderer = frontend.FrontendRenderer("localhost", 9999, max_fps=500, timeout=0.2)

    assert renderer.health() == {"status": "running"}
    response = renderer._post(
        "/api/test",
        {"path": Path("map.osm"), "point": PointLike(), "array": ArrayLike(), "ids": {1, 2}},
    )
    assert response["payload"]["path"] == "map.osm"
    assert response["payload"]["point"] == [1, 2]
    assert requests[0] == ("http://localhost:9999/health", 0.2)

    ready_renderer = frontend.FrontendRenderer()
    ready_renderer.health = lambda: {"status": "running"}
    assert ready_renderer.wait_until_ready(timeout=0.1) is True

    failing_renderer = frontend.FrontendRenderer()

    def fail_health():
        raise url_error.URLError("offline")

    failing_renderer.health = fail_health
    times = iter([0, 0.05, 0.2])
    monkeypatch.setattr(renderer_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(renderer_module.time, "sleep", lambda seconds: None)
    assert failing_renderer.wait_until_ready(timeout=0.1) is False


def test_frontend_renderer_process_helpers(monkeypatch, tmp_path):

    popen_calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return FakeProcess()

    pid_file = tmp_path / "frontend.pid"
    monkeypatch.setattr(renderer_module.subprocess, "Popen", fake_popen)

    process = renderer_module.start_server_process(
        "0.0.0.0", 9000, demo=True, max_fps=45, open_browser=True, pid_file=pid_file
    )

    assert process.pid == 12345
    assert "--demo" in popen_calls[0][0]
    assert "--open" in popen_calls[0][0]
    assert popen_calls[0][1]["start_new_session"] is True
    assert pid_file.read_text(encoding="utf-8") == "12345"

    killed = []
    monkeypatch.setattr(renderer_module.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    assert renderer_module.stop_server_process(pid_file) == 12345
    assert killed == [(12345, signal.SIGTERM)]
    assert not pid_file.exists()

    with pytest.raises(FileNotFoundError):
        renderer_module.stop_server_process(pid_file)


def test_frontend_server_context_manager(monkeypatch, tmp_path):

    started = []
    stopped = []
    monkeypatch.setattr(
        renderer_module,
        "start_server_process",
        lambda *args, **kwargs: started.append((args, kwargs)) or SimpleNamespace(pid=9),
    )
    monkeypatch.setattr(
        renderer_module, "stop_server_process", lambda pid_file=None: stopped.append(pid_file)
    )
    monkeypatch.setattr(
        renderer_module.FrontendRenderer, "wait_until_ready", lambda self, timeout: True
    )

    pid_file = tmp_path / "frontend.pid"
    with renderer_module.FrontendServer(port=9001, pid_file=pid_file) as renderer:
        assert renderer.port == 9001

    assert started
    assert stopped == [pid_file]

    monkeypatch.setattr(
        renderer_module.FrontendRenderer, "wait_until_ready", lambda self, timeout: False
    )
    with pytest.raises(RuntimeError):
        renderer_module.FrontendServer(port=9002).__enter__()


def test_connection_manager_stale_clients_and_ack_timeout():

    class GoodClient:
        def __init__(self):
            self.messages = []

        async def send_text(self, payload):
            self.messages.append(payload)

    class BrokenClient:
        async def send_text(self, payload):
            raise RuntimeError("closed")

    async def exercise():
        manager = server_module.ConnectionManager()
        good_client = GoodClient()
        manager._clients = [good_client, BrokenClient()]

        delivered = await manager.broadcast({"type": "test"})
        assert delivered == 1
        assert manager.client_count == 1
        assert good_client.messages
        assert await manager.wait_for_ack("late-frame", 0) is False

        wait_task = asyncio.create_task(manager.wait_for_ack("frame", 0.1))
        await asyncio.sleep(0)
        manager.record_ack("frame")
        assert await wait_task is True

    asyncio.run(exercise())


def test_server_helper_parsers_and_map_config_errors(monkeypatch):

    assert server_module._optional_int(None) is None
    assert server_module._optional_int("") is None
    assert server_module._optional_int("12") == 12
    assert server_module._optional_path(None) is None
    assert server_module._optional_path("") is None
    assert server_module._optional_path("data") == Path("data")
    assert server_module._parse_ids(None) is None
    assert server_module._parse_ids("") is None
    assert server_module._parse_ids([]) is None
    assert server_module._parse_ids("1, 2,") == [1, 2]
    assert server_module._parse_ids([3, "4"]) == [3, 4]

    monkeypatch.setattr(
        "tactics2d.display.renderers.web.preview.iter_map_configs",
        lambda: iter([("highD_1", {"dataset": "highD"})]),
    )
    assert server_module._map_config_from_name(None) is None
    assert server_module._map_config_from_name("highd_1") == {"dataset": "highD"}
    with pytest.raises(KeyError):
        server_module._map_config_from_name("missing")


def test_server_preview_control_endpoints(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    async def fake_run_dataset_preview(manager, status, payload, pause_event):
        status.update({"status": "complete", "source": "dataset", "payload_file": payload["file"]})

    monkeypatch.setattr(server_module, "_run_levelx_dataset_preview", fake_run_dataset_preview)
    client = TestClient(server_module.create_app())

    assert client.get("/").status_code == 200
    assert client.post("/api/layout", json={"layout": "focus"}).json()["layout"] == "focus"

    demo = client.post("/api/preview/demo", json={"max_fps": 15}).json()
    assert demo["source"] == "demo"

    stopped = client.post("/api/preview/stop", json={}).json()
    assert stopped["status"] == "stopped"

    dataset = client.post(
        "/api/preview/dataset", json={"dataset": "highD", "folder": "data/highD", "file": "11"}
    ).json()
    assert dataset["source"] == "dataset"


def test_server_run_server_and_main(monkeypatch):
    import threading
    import webbrowser

    calls = []

    class FakeTimer:
        def __init__(self, interval, callback):
            calls.append(("timer", interval))
            self.callback = callback

        def start(self):
            calls.append(("timer_start",))
            self.callback()

    fake_uvicorn = SimpleNamespace(
        run=lambda app, host, port, log_level: calls.append(("uvicorn", app, host, port, log_level))
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(threading, "Timer", FakeTimer)
    monkeypatch.setattr(webbrowser, "open", lambda url: calls.append(("open", url)))
    monkeypatch.setattr(
        server_module,
        "create_app",
        lambda demo=False, max_fps=30: {"demo": demo, "max_fps": max_fps},
    )

    server_module.run_server("0.0.0.0", 9000, demo=True, max_fps=55, open_browser=True)
    assert ("open", "http://0.0.0.0:9000/") in calls
    assert ("uvicorn", {"demo": True, "max_fps": 55}, "0.0.0.0", 9000, "info") in calls

    args = server_module._parse_args(
        ["--host", "0.0.0.0", "--port", "9001", "--demo", "--max-fps", "44", "--open"]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9001
    assert args.demo is True
    assert args.max_fps == 44
    assert args.open_browser is True

    monkeypatch.setattr(server_module, "run_server", lambda *args: calls.append(("main", args)))
    server_module.main(["--host", "127.0.0.1", "--port", "9002"])
    assert ("main", ("127.0.0.1", 9002, False, 30, False)) in calls


def test_run_levelx_dataset_preview_updates_status(monkeypatch):

    class FakeScene:
        sensor_id = "highD-11"
        actual_time_range = (0, 40)

        def iter_frames(self):
            return [0, 40]

        def sensor_for_frame(self, frame):
            return {"id": "camera", "frame": frame}

    class FakeManager:
        def __init__(self):
            self.frames = []

        async def publish_frame(self, payload, **kwargs):
            self.frames.append((payload, kwargs))
            if kwargs["frame_id"] == 40:
                return {"status": "dropped", "dropped_frames": 1}
            return {"status": "ok", "dropped_frames": 0}

    async def exercise_success():
        status = {}
        pause_event = asyncio.Event()
        pause_event.set()
        manager = FakeManager()
        await server_module._run_levelx_dataset_preview(
            manager,
            status,
            {
                "dataset": "highD",
                "folder": "data/highD",
                "file": "11",
                "frames": 2,
                "max_fps": 100,
                "ids": "1,2",
                "start_time_ms": "",
            },
            pause_event,
        )
        assert status["status"] == "complete"
        assert status["sent_frames"] == 1
        assert status["dropped_frames"] == 1
        assert len(manager.frames) == 2

    monkeypatch.setattr(preview, "load_levelx_preview_scene", lambda **kwargs: FakeScene())
    asyncio.run(exercise_success())

    async def exercise_error():
        status = {}
        pause_event = asyncio.Event()
        pause_event.set()
        await server_module._run_levelx_dataset_preview(
            FakeManager(),
            status,
            {"dataset": "highD", "folder": "data/highD", "file": "11"},
            pause_event,
        )
        assert status["status"] == "error"
        assert "boom" in status["message"]

    monkeypatch.setattr(
        preview,
        "load_levelx_preview_scene",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(server_module.LOGGER, "exception", lambda *args, **kwargs: None)
    asyncio.run(exercise_error())
