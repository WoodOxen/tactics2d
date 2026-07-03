# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""BITS-style bi-level imitation behavior model (inference)."""

from .config import BitsConfig
from .inference import (
    BitsCheckpointCompatibilityReport,
    BitsCheckpointMetadata,
    BitsCheckpointShapeMismatch,
    BitsInferenceLoadResult,
    load_bits_checkpoint,
    load_bits_inference_model,
    load_tbsim_bits_inference_weights,
    map_tbsim_bits_planner_state_dict,
    map_tbsim_bits_predictor_state_dict,
    merge_tbsim_bits_state_dicts,
)
from .model import BitsBehaviorModel

__all__ = [
    "BitsBehaviorModel",
    "BitsCheckpointCompatibilityReport",
    "BitsCheckpointMetadata",
    "BitsCheckpointShapeMismatch",
    "BitsConfig",
    "BitsInferenceLoadResult",
    "load_bits_checkpoint",
    "load_bits_inference_model",
    "load_tbsim_bits_inference_weights",
    "map_tbsim_bits_planner_state_dict",
    "map_tbsim_bits_predictor_state_dict",
    "merge_tbsim_bits_state_dicts",
]
