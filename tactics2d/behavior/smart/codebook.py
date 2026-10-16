# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Load SMART codebook assets."""

import pickle
from typing import Dict, Optional

MOTION_CODEBOOK_NAME = "motion_codebook.pkl"
MAP_CODEBOOK_NAME = "map_codebook.pkl"


def load_codebook(path: Optional[str], name: str) -> Dict[str, object]:
    """Load one of SMART's downloadable codebooks."""

    if path is None:
        raise ValueError(
            f"The SMART {name} is not redistributed with the package. Download it "
            f"together with the checkpoint and set SmartConfig(asset_root=...) or pass "
            f"the path directly."
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)
