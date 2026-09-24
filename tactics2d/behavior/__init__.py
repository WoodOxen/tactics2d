# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Behavior-model public API with optional models loaded on first access."""

from importlib import import_module

from .base import BehaviorModelBase

_LAZY_EXPORTS = {
    "BitsBehaviorModel": (".bits.model", "BitsBehaviorModel"),
    "BitsConfig": (".bits.config", "BitsConfig"),
    "InterSimBehaviorModel": (".intersim.model", "InterSimBehaviorModel"),
    "InterSimConfig": (".intersim.config", "InterSimConfig"),
    "LimSimBehaviorModel": (".limsim.model", "LimSimBehaviorModel"),
    "LimSimConfig": (".limsim.config", "LimSimConfig"),
    "SmartBehaviorModel": (".smart.model", "SmartBehaviorModel"),
    "SmartConfig": (".smart.config", "SmartConfig"),
}


def __getattr__(name: str):
    """Load model implementations lazily."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "BehaviorModelBase",
    "BitsBehaviorModel",
    "BitsConfig",
    "InterSimBehaviorModel",
    "InterSimConfig",
    "LimSimBehaviorModel",
    "LimSimConfig",
    "SmartBehaviorModel",
    "SmartConfig",
]
