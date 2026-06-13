# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frame recorder for GIF / video / frame-sequence export.

Recorders wrap :class:`~tactics2d.display.DisplayBackend` instances and
capture rendered frames for later export.

Usage::

    from tactics2d.display import create_display_backend
    from tactics2d.display.recorder import GifRecorder

    backend = create_display_backend("matplotlib")
    recorder = GifRecorder(backend, output_path="demo.gif", fps=10)

    for step in range(100):
        snapshot = ...
        recorder.render(snapshot)   # renders + records

    recorder.save()
    recorder.close()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .backend import DisplayBackend
from .snapshot import SceneSnapshot

LOGGER = logging.getLogger(__name__)


class FrameCollector:
    """Records frames from a backend and saves them as an image sequence.

    This is the base class for frame-based recorders.  It stores rendered
    RGB arrays and provides ``save()`` to write them to files.

    Args:
        backend: The display backend to capture frames from.
        output_path: Path template for output files.  For frame sequences
            this may contain a ``{frame:05d}`` placeholder.
    """

    def __init__(self, backend: DisplayBackend, output_path: str):
        self._backend = backend
        self._output_path = Path(output_path)
        self._frames: list[np.ndarray] = []

    def render(self, snapshot: SceneSnapshot) -> np.ndarray | None:
        """Render the snapshot and record the frame.

        Args:
            snapshot: The scene snapshot to render.

        Returns:
            The RGB array from the backend, or ``None``.
        """
        result = self._backend.render(snapshot)
        if isinstance(result, np.ndarray):
            self._frames.append(result)
        return result

    def reset(self, snapshot: SceneSnapshot | None = None) -> None:
        """Reset the backend and clear recorded frames."""
        self._frames.clear()
        self._backend.reset(snapshot)

    def close(self) -> None:
        """Save pending frames (if any) and close the backend."""
        if self._frames:
            self.save()
        self._backend.close()

    def save(self) -> None:
        """Save recorded frames to ``self._output_path``.

        Subclasses override this to produce the specific output format.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._frames)


class GifRecorder(FrameCollector):
    """Records frames and saves them as an animated GIF.

    Requires ``imageio`` (optional dependency).

    Args:
        backend: The display backend to capture frames from.
        output_path: Path for the output GIF file.
        fps: Frames per second for the GIF animation.
        loop: Number of loops (0 = infinite).
    """

    def __init__(self, backend: DisplayBackend, output_path: str, fps: int = 10, loop: int = 0):
        super().__init__(backend, output_path)
        self._fps = fps
        self._loop = loop

    def save(self) -> None:
        """Save the recorded frames as an animated GIF."""
        if not self._frames:
            LOGGER.warning("No frames to save for GIF %s.", self._output_path)
            return

        try:
            import imageio
        except ImportError:
            raise ImportError("GifRecorder requires imageio. Install with: pip install imageio")

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(self._output_path), self._frames, fps=self._fps, loop=self._loop)
        LOGGER.info("Saved GIF with %d frames to %s.", len(self._frames), self._output_path)


class FrameExporter(FrameCollector):
    """Records frames and saves them as a numbered PNG sequence.

    Args:
        backend: The display backend to capture frames from.
        output_dir: Directory for the output PNG files.
        format_template: Filename template with ``{frame:05d}`` placeholder.
    """

    def __init__(
        self,
        backend: DisplayBackend,
        output_dir: str,
        format_template: str = "frame_{frame:05d}.png",
    ):

        super().__init__(backend, output_dir)
        self._format_template = format_template
        self._output_dir = Path(output_dir)

    def save(self) -> None:
        """Save all recorded frames as individual PNG files."""
        if not self._frames:
            LOGGER.warning("No frames to export.")
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image
        except ImportError:
            raise ImportError("FrameExporter requires Pillow. Install with: pip install Pillow")

        for i, frame in enumerate(self._frames):
            filename = self._format_template.format(frame=i)
            filepath = self._output_dir / filename
            Image.fromarray(frame).save(filepath)

        LOGGER.info("Exported %d frames to %s.", len(self._frames), self._output_dir)
