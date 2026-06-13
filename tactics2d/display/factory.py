# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Display backend factory — maps render_mode strings to backend instances."""

from __future__ import annotations

from typing import Any

from .backend import DisplayBackend, NullBackend


def create_display_backend(render_mode: str, **kwargs: Any) -> DisplayBackend:
    """Create a :class:`DisplayBackend` from a render mode string.

    Args:
        render_mode: One of ``"human"``, ``"pygame"``, ``"rgb_array"``,
            ``"browser"``, ``"matplotlib"``, ``"none"``, or ``None``.
        **kwargs: Backend-specific keyword arguments forwarded to the
            constructor (e.g. ``host``, ``port``, ``window_size``, ``fps``).

    Returns:
        A :class:`DisplayBackend` instance.

    Raises:
        ValueError: If ``render_mode`` is not recognised.
    """
    mode = (render_mode or "none").lower()

    if mode in ("human", "pygame"):
        from .backends.pygame import PygameBackend

        return PygameBackend(off_screen=False, **kwargs)

    elif mode == "rgb_array":
        from .backends.pygame import PygameBackend

        return PygameBackend(off_screen=True, **kwargs)

    elif mode == "browser":
        from .backends.web import BrowserBackend

        return BrowserBackend(**kwargs)

    elif mode == "matplotlib":
        from .backends.matplotlib import MatplotlibBackend

        return MatplotlibBackend(**kwargs)

    elif mode == "none":
        return NullBackend()

    else:
        raise ValueError(
            f"Unknown render_mode: {render_mode!r}. "
            f"Expected one of: human, pygame, rgb_array, browser, matplotlib, none."
        )
