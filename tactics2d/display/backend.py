# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Display backend abstract base class and null backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .snapshot import SceneSnapshot


class DisplayBackend(ABC):
    """Abstract interface for all display backends.

    Each backend renders a ``SceneSnapshot`` to a specific output target:

    * **PygameBackend** — local pygame window or off-screen surface
    * **BrowserBackend** — remote browser over HTTP + WebSocket
    * **MatplotlibBackend** — matplotlib figure (RGB array or file)
    * **NullBackend** — no-op (for ``render_mode="none"``)

    Lifecycle
    ---------
    ``reset()`` → ``render()`` × *N* → ``close()``

    Implementations must be safe for multiple reset/render/close cycles.
    """

    backend_name: str = ""
    """Globally unique backend identifier (``"pygame"``, ``"browser"``, ...)."""

    supports_rgb_array: bool = False
    """Whether ``render()`` may return an (H, W, 3) numpy array."""

    supports_interactive: bool = False
    """Whether the backend shows an interactive window."""

    is_headless: bool = True
    """Whether the backend can run without a physical display."""

    @abstractmethod
    def reset(self, snapshot: SceneSnapshot | None = None) -> None:
        """Reset backend state, optionally rendering an initial snapshot.

        Called when the environment is reset.  The backend should re-initialise
        any internal state and optionally display the provided snapshot.

        Args:
            snapshot: Optional initial scene snapshot to render.
        """

    @abstractmethod
    def render(self, snapshot: SceneSnapshot) -> np.ndarray | None:
        """Render one frame from the given snapshot.

        Args:
            snapshot: The current scene snapshot to render.

        Returns:
            An (H, W, 3) RGB array if ``supports_rgb_array`` is ``True``,
            otherwise ``None``.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources (window, server process, connections, ...).

        After ``close()`` the backend must be ``reset()`` before it can be
        used again.
        """

    def save_frame(self, path: str) -> None:
        """Save the currently displayed frame to a file (optional)."""
        pass

    def set_layout(self, layout: str) -> None:
        """Set multi-sensor layout (optional, used by PygameBackend)."""
        pass


class NullBackend(DisplayBackend):
    """No-op backend for ``render_mode="none"``.

    All methods are no-ops.  ``render()`` always returns ``None``.
    """

    backend_name = "none"
    supports_rgb_array = False
    supports_interactive = False
    is_headless = True

    def reset(self, snapshot: SceneSnapshot | None = None) -> None:
        pass

    def render(self, snapshot: SceneSnapshot) -> np.ndarray | None:
        return None

    def close(self) -> None:
        pass
