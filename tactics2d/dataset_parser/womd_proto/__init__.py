# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Womd proto module with support for multiple protobuf versions."""


try:
    from packaging.version import Version
except ImportError:
    # Fallback for simple version comparison
    class Version:
        def __init__(self, version_str):
            self.parts = tuple(map(int, version_str.split(".")))

        def __le__(self, other):
            return self.parts <= other.parts


import google.protobuf

# Choose the appropriate generated code based on protobuf version
if Version(google.protobuf.__version__) <= Version("3.20.3"):
    from .pb2 import scenario_pb2 as scenario_pb
else:
    from .pb3 import scenario_pb2 as scenario_pb

__all__ = ["scenario_pb"]
