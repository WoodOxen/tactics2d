# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for display module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tactics2d.display import (
    CameraMetadata,
    DisplayBackend,
    FrameCollector,
    FrameExporter,
    GifRecorder,
    MatplotlibBackend,
    NullBackend,
    ParticipantElement,
    PointCloudElement,
    PygameBackend,
    RoadElement,
    SceneSnapshot,
    TrafficLightState,
    create_display_backend,
)

# ---------------------------------------------------------------------------
# SceneSnapshot
# ---------------------------------------------------------------------------


class TestSceneSnapshot:
    def test_minimal_creation(self):
        snapshot = SceneSnapshot()
        assert snapshot.version == "1.0"
        assert snapshot.frame == 0
        assert len(snapshot.road_elements) == 0
        assert len(snapshot.participants) == 0

    def test_full_creation(self):
        road = RoadElement(
            id_="lane_1", shape="polygon", geometry=[(0, 0), (1, 0), (1, 1), (0, 1)], type_="lane"
        )
        participant = ParticipantElement(
            id_=1,
            shape="polygon",
            geometry=[(0, 0), (1, 0), (1, 1), (0, 1)],
            position=(5, 5),
            rotation=0.0,
            type_="vehicle",
        )
        camera = CameraMetadata(id_="cam0", position=(0, 0), yaw=0.0, perception_range=50.0)
        light = TrafficLightState(id_="tl_1", position=(10, 10), state="green")
        pc = PointCloudElement(id_="pc1", points=[(1.0, 2.0), (3.0, 4.0)])

        snapshot = SceneSnapshot(
            version="1.0",
            frame=42,
            timestamp_ms=12345,
            scene_name="test_scene",
            road_elements={"lane_1": road},
            participants={1: participant},
            cameras=[camera],
            traffic_lights={"tl_1": light},
            point_clouds=[pc],
            ego_participant_id=1,
        )

        assert snapshot.frame == 42
        assert snapshot.scene_name == "test_scene"
        assert snapshot.road_elements["lane_1"].type_ == "lane"
        assert snapshot.participants[1].type_ == "vehicle"
        assert snapshot.cameras[0].id_ == "cam0"
        assert snapshot.traffic_lights["tl_1"].state == "green"
        assert len(snapshot.point_clouds) == 1

    def test_to_dict_round_trip(self):
        road = RoadElement(id_="lane_1", shape="polygon", geometry=[(0, 0), (1, 0)], type_="lane")
        participant = ParticipantElement(
            id_=1,
            shape="polygon",
            geometry=[(0, 0), (1, 0)],
            position=(5, 5),
            rotation=0.0,
            type_="vehicle",
        )

        original = SceneSnapshot(
            frame=1,
            road_elements={"lane_1": road},
            participants={1: participant},
            ego_participant_id=1,
        )

        d = original.to_dict()
        restored = SceneSnapshot.from_dict(d)

        assert restored.frame == original.frame
        assert restored.ego_participant_id == original.ego_participant_id
        assert len(restored.road_elements) == 1
        assert "lane_1" in restored.road_elements
        assert restored.road_elements["lane_1"].type_ == "lane"

    def test_to_json_valid(self):
        snapshot = SceneSnapshot(frame=5, scene_name="json_test")
        json_str = snapshot.to_json()
        parsed = json.loads(json_str)
        assert parsed["frame"] == 5
        assert parsed["scene_name"] == "json_test"

    def test_incremental_fields(self):
        snapshot = SceneSnapshot(
            participant_ids_to_create=[1, 2],
            participant_ids_to_remove=[3],
            road_ids_to_remove=["old_road"],
        )
        assert snapshot.participant_ids_to_create == [1, 2]
        assert snapshot.participant_ids_to_remove == [3]
        assert snapshot.road_ids_to_remove == ["old_road"]

    def test_from_dict(self):
        data = {
            "frame": 10,
            "road_elements": {
                1: {
                    "id_": 1,
                    "shape": "polygon",
                    "geometry": [[0.0, 0.0], [1.0, 0.0]],
                    "type_": "lane",
                }
            },
            "participants": {},
            "participant_ids_to_create": [],
            "participant_ids_to_remove": [],
            "road_ids_to_remove": [],
            "point_clouds": [],
            "traffic_lights": {},
            "cameras": [],
            "debug_overlays": {},
            "extra": {},
        }
        snapshot = SceneSnapshot.from_dict(data)
        assert snapshot.frame == 10
        assert 1 in snapshot.road_elements
        assert snapshot.road_elements[1].type_ == "lane"


# ---------------------------------------------------------------------------
# Backend ABC and NullBackend
# ---------------------------------------------------------------------------


class TestDisplayBackend:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DisplayBackend()  # abstract methods not implemented


class TestNullBackend:
    def test_lifecycle(self):
        backend = NullBackend()
        assert backend.backend_name == "none"
        assert not backend.supports_rgb_array
        assert backend.is_headless

        backend.reset()
        result = backend.render(SceneSnapshot())
        assert result is None
        backend.close()

    def test_class_attributes(self):
        backend = NullBackend()
        assert not backend.supports_interactive
        assert backend.is_headless
        assert not backend.supports_rgb_array


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateDisplayBackend:
    def test_none(self):
        backend = create_display_backend("none")
        assert isinstance(backend, NullBackend)
        backend.close()

    def test_none_via_none_arg(self):
        backend = create_display_backend(None)
        assert isinstance(backend, NullBackend)
        backend.close()

    def test_pygame_human(self):
        backend = create_display_backend("human")
        assert isinstance(backend, PygameBackend)
        backend.close()

    def test_pygame_rgb_array(self):
        backend = create_display_backend("rgb_array")
        assert isinstance(backend, PygameBackend)
        backend.close()

    def test_browser(self):
        backend = create_display_backend("browser")
        assert backend.backend_name == "browser"
        backend.close()

    def test_matplotlib(self):
        backend = create_display_backend("matplotlib")
        assert backend.backend_name == "matplotlib"
        backend.close()

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_display_backend("unknown_mode")


# ---------------------------------------------------------------------------
# Recorder tests (mock backend)
# ---------------------------------------------------------------------------


class _MockBackend(DisplayBackend):
    """Minimal backend that returns a dummy RGB array for recorder tests."""

    backend_name = "mock"
    supports_rgb_array = True
    is_headless = True

    def __init__(self):
        self.reset_count = 0
        self.close_count = 0

    def reset(self, snapshot=None):
        self.reset_count += 1

    def render(self, snapshot):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self):
        self.close_count += 1


class TestFrameCollector:
    def test_collect_records_frames(self):
        backend = _MockBackend()
        collector = FrameCollector(backend, "test_output")

        snapshot = SceneSnapshot()
        for _ in range(5):
            collector.render(snapshot)

        assert len(collector) == 5

    def test_non_array_not_recorded(self):
        backend = NullBackend()
        collector = FrameCollector(backend, "test_output")

        for _ in range(5):
            collector.render(SceneSnapshot())

        assert len(collector) == 0  # NullBackend returns None

    def test_reset_clears_frames(self):
        backend = _MockBackend()
        collector = FrameCollector(backend, "test_output")

        for _ in range(3):
            collector.render(SceneSnapshot())
        assert len(collector) == 3

        collector.reset()
        assert len(collector) == 0
        assert backend.reset_count >= 1

    def test_save_not_implemented(self):
        collector = FrameCollector(_MockBackend(), "test_output")
        with pytest.raises(NotImplementedError):
            collector.save()

    def test_close_closes_backend(self):
        backend = _MockBackend()
        collector = FrameCollector(backend, "test_output")

        collector.close()
        assert backend.close_count == 1


class TestGifRecorder:
    def test_no_frames_warns(self):
        backend = _MockBackend()
        recorder = GifRecorder(backend, "test.gif")
        recorder.save()  # should warn but not raise

    def test_import_error_when_imageio_missing(self):
        backend = _MockBackend()
        recorder = GifRecorder(backend, "test.gif")
        recorder.render(SceneSnapshot())

        try:
            import imageio  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError):
                recorder.save()
        else:
            # imageio is available; just verify save succeeds
            with tempfile.TemporaryDirectory() as tmpdir:
                recorder._output_path = Path(tmpdir) / "test.gif"
                recorder.save()
                assert (Path(tmpdir) / "test.gif").exists()


class TestFrameExporter:
    def test_no_frames_warns(self):
        backend = _MockBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FrameExporter(backend, tmpdir)
            exporter.save()

    def test_export_png_files(self):
        backend = _MockBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FrameExporter(backend, tmpdir)

            for _ in range(3):
                exporter.render(SceneSnapshot())

            exporter.save()

            files = sorted(Path(tmpdir).glob("*.png"))
            assert len(files) == 3
            assert files[0].name == "frame_00000.png"
            assert files[1].name == "frame_00001.png"
            assert files[2].name == "frame_00002.png"

    def test_import_error_when_pil_missing(self):
        backend = _MockBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FrameExporter(backend, tmpdir)
            exporter.render(SceneSnapshot())

            try:
                from PIL import Image  # noqa: F401
            except ImportError:
                with pytest.raises(ImportError):
                    exporter.save()


# ---------------------------------------------------------------------------
# MatplotlibBackend (headless, works in CI)
# ---------------------------------------------------------------------------


class TestMatplotlibBackend:
    def test_create_and_lifecycle(self):
        backend = MatplotlibBackend()
        assert backend.backend_name == "matplotlib"
        assert backend.supports_rgb_array
        assert backend.is_headless

        backend.reset()
        backend.close()

    def test_render_returns_array(self):
        backend = MatplotlibBackend()
        backend.reset()

        snapshot = SceneSnapshot(
            frame=0,
            cameras=[CameraMetadata(id_="cam", position=(0, 0), yaw=0.0, perception_range=50.0)],
        )
        result = backend.render(snapshot)
        backend.close()

        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB

    def test_render_before_reset_returns_none(self):
        backend = MatplotlibBackend()
        result = backend.render(SceneSnapshot())
        assert result is None
        backend.close()

    def test_multiple_reset_cycles(self):
        backend = MatplotlibBackend()
        for _ in range(3):
            backend.reset()
            result = backend.render(
                SceneSnapshot(
                    frame=0,
                    cameras=[
                        CameraMetadata(id_="cam", position=(0, 0), yaw=0.0, perception_range=50.0)
                    ],
                )
            )
            assert isinstance(result, np.ndarray)
        backend.close()


# ---------------------------------------------------------------------------
# PygameBackend (lifecycle only — full render needs display)
# ---------------------------------------------------------------------------


class TestPygameBackend:
    def test_create_and_close_off_screen(self):
        backend = PygameBackend(off_screen=True)
        assert backend.backend_name == "pygame"
        backend.reset()
        backend.close()

    def test_create_and_close_on_screen(self):
        backend = PygameBackend(off_screen=False)
        assert not backend._off_screen
        backend.reset()
        backend.close()
