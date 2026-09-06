# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Internal lane-semantics helpers shared by vehicle route consumers."""

_NON_VEHICLE_LANE_SUBTYPES = frozenset(
    {
        "bicycle_lane",
        "crosswalk",
        "cycleway",
        "exit",
        "footway",
        "pedestrian",
        "shared_walkway",
        "sidewalk",
        "stairs",
        "stairway",
        "walkway",
    }
)


def is_vehicle_lane(lane) -> bool:
    """Return whether a map lane is suitable for vehicle routing."""

    if lane is None:
        return False
    subtype = str(getattr(lane, "subtype", "") or "").strip().lower()
    return subtype not in _NON_VEHICLE_LANE_SUBTYPES
