# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unified display backend module.

Provides a common :class:`DisplayBackend` interface and concrete
implementations (pygame, browser, matplotlib, null) so that all
``tactics2d.envs`` environments share a single rendering pipeline.

Typical usage::

    from tactics2d.display import create_display_backend

    backend = create_display_backend("rgb_array", window_size=(800, 600))
    snapshot = SceneSnapshot(...)
    rgb = backend.render(snapshot)   # (H, W, 3) numpy array
    backend.close()
"""

from .backend import DisplayBackend, NullBackend
from .backends.matplotlib import MatplotlibBackend
from .backends.pygame import PygameBackend
from .backends.web import BrowserBackend
from .converter import SceneSnapshotConverter
from .factory import create_display_backend
from .recorder import FrameCollector, FrameExporter, GifRecorder
from .renderers import MatplotlibRenderer, PygameRenderer, RenderManager
from .renderers.web import FrontendRenderer, FrontendServer, run_server
from .renderers.web.preview import (
    build_map_preview_sensor,
    ensure_frontend_server,
    stream_levelx_preview,
)
from .snapshot import (
    CameraMetadata,
    ParticipantElement,
    PointCloudElement,
    RoadElement,
    SceneSnapshot,
    TrafficLightState,
)

__all__ = [
    "BrowserBackend",
    "CameraMetadata",
    "DisplayBackend",
    "FrameCollector",
    "FrameExporter",
    "FrontendRenderer",
    "FrontendServer",
    "GifRecorder",
    "MatplotlibBackend",
    "MatplotlibRenderer",
    "NullBackend",
    "ParticipantElement",
    "PointCloudElement",
    "PygameBackend",
    "PygameRenderer",
    "RenderManager",
    "RoadElement",
    "SceneSnapshot",
    "SceneSnapshotConverter",
    "TrafficLightState",
    "build_map_preview_sensor",
    "create_display_backend",
    "ensure_frontend_server",
    "run_server",
    "stream_levelx_preview",
]
