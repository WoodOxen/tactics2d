# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output data types for BITS inference and scoring."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class BitsPrediction:
    """BITS prediction in the current ego frame."""

    positions: np.ndarray
    yaws: np.ndarray
    availabilities: np.ndarray
    scores: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "positions": self.positions,
            "yaws": self.yaws,
            "availabilities": self.availabilities,
            "scores": self.scores,
        }


@dataclass(frozen=True)
class BitsPlan:
    """Spatial plan candidates in the current ego frame."""

    positions: np.ndarray
    yaws: np.ndarray
    availabilities: np.ndarray
    scores: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "positions": self.positions,
            "yaws": self.yaws,
            "availabilities": self.availabilities,
            "scores": self.scores,
        }


@dataclass(frozen=True)
class BitsAgentPrediction:
    """Neighbouring-agent predictions aligned with BITS plan candidates."""

    positions: np.ndarray
    yaws: np.ndarray
    availabilities: np.ndarray


@dataclass(frozen=True)
class BitsPlanScoreBreakdown:
    """Rule-based score terms for spatial plan candidates."""

    total: np.ndarray
    likelihood: np.ndarray
    progress: np.ndarray
    lane: np.ndarray
    collision: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "total": self.total,
            "likelihood": self.likelihood,
            "progress": self.progress,
            "lane": self.lane,
            "collision": self.collision,
        }
